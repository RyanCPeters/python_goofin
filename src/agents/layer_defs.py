from src.agents.cuda_kernels.forward_kernels import *
from src.agents.cuda_kernels.update_kernels import *
from src.agents.cuda_kernels.backprop_kernels import *
from src.agents.cuda_kernels.cuda_kernels import *
from src.agents import *
from collections import deque

init_default = np.empty((0,))

class BaseLayer:

    def __init__(self, name:str, input_shape:list, input_layer:"BaseLayer"=None,
                 output_layer:"BaseLayer"=None,use_momentum:bool=False, batch_size:int=1) -> None:
        """

        :param input_shape: A 3-tuple that expresses the height, width, and filter depth of the
                            inputs given to this layer
        :type input_shape: tuple[int,int,int]

        :param input_layer: [OPTIONAL] a reference to the network layer that precedes this one in a forward pass.
        :type input_layer: Any subclass of BaseLayer

        :param output_layer: [OPTIONAL] a reference to the network layer that follows this one in a forward pass.
        :type output_layer: Any subclass of BaseLayer
        """
        self.name=name
        self.is_input = False # set to true for the network's input layer.
        self.is_output = False # set to true for the network's output layer.
        self.input_layer = input_layer
        self.output_layer = output_layer
        self._input_shape = input_shape
        self.input = init_default
        self.W = init_default
        self.b = init_default
        self.out = init_default
        self.blame_in = init_default
        self.blame_out = init_default
        self.net = init_default
        self.s = init_default
        self.output = init_default
        self.last_update_dumps = deque(maxlen=3)
        self.last_forward_dumps = deque(maxlen=3)
        self.last_backprop_dumps = deque(maxlen=3)
        self.forward_net = None
        self.forward_xfr = None
        self.forward_pool = None
        self.batched_forward_net = None
        self.batched_forward_xfr = None
        self.batched_forward_pool = None
        self.backprop_sense = None
        self.backprop_blame = None
        self.batched_backprop_sense = None
        self.batched_backprop_blame = None
        self.update_bias = None
        self.update_weights = None
        self.tpb = None
        self.bpg = None
        self.use_momentum = use_momentum
        self.batch_size = batch_size

    @property
    def trainable_parameter_count(self):
        return self.W.size + self.b.size

    def dump_to_host(self, samp_idx):
        """A diagnostic debugging function that snapshots all of the layer's array components from the gpu and "dumps"
        the data to host (cpu) memory for inspection."""
        try:
            keys = ["input","out","s","W","b"]
            if self.blame_out is not None:
                keys.append("blame_out")
            if self.blame_in is not None:
                keys.append("blame_in")
            keys.append("output")
            try:
                ret = {k:getattr(self,k)[samp_idx].copy_to_host() for k in keys}
            except BaseException as be:
                root_error_logger(f"{type(be)}: {be.args}",exc_info=True)
                ret = {k: getattr(self, k) for k in keys}
            try:
                _ret = {k:(arr.shape,arr,*(arr.argmax(i) for i in range(len(arr.shape)))) for k,arr in ret.items()}
            except BaseException as be:
                root_error_logger(f"{type(be)}: {be.args}",exc_info=True)
                _ret = ret
            return _ret
        except BaseException as be:
            root_error_logger(f"{type(be)}: {be.args}",exc_info=True)

    @property
    def layer_type(self):
        return "BaseLayer"

    def compile(self, input_data_mean):
        """
        Once the network structure has been built, and each layer's respective input_layer/output_layer references
        have been assigned, call compile on the network's first hidden layer to finalize all array component references
        and gpu kernel compilations.

        :return:
        :rtype:
        """
        if self.input_layer is None:
            self.is_input = True
        if self.output_layer is None:
            self.is_output = True
        if self.is_input:
            self.input = np.empty((self.batch_size,*self._input_shape), GLOBAL_DTYPE)
        if self.output_layer:
            self.output_layer.compile(input_data_mean)
        self.output = self.out

    def forward(self, input_p: np.ndarray, forward_stream, cleanup_stream, samp_idx: int = 0):
        self.input = input_p
        if self.output_layer:
            self.output_layer._forward(forward_stream)

    def _forward(self, forward_stream, cleanup_stream, samp_idx: int = 0):
        if self.output_layer:
            self.output_layer._forward(forward_stream)

    def batched_forward(self, input_p: np.ndarray, forward_stream, cleanup_stream):
        self.input = input_p
        if self.output_layer:
            self.output_layer._batched_forward(forward_stream,cleanup_stream)

    def _batched_forward(self, forward_stream, cleanup_stream):
        if self.output_layer:
            self.output_layer._batched_forward(forward_stream,cleanup_stream)

    def backprop(self,err:np.ndarray, backprop_stream, samp_idx):
        if not self.input_layer:
            self.input_layer._backprop(backprop_stream, samp_idx)

    def _backprop(self, backprop_stream, samp_idx):
        if not self.input_layer:
            self.input_layer._backprop(backprop_stream, samp_idx)

    def batched_backprop(self,err:np.ndarray, backprop_stream):
        if not self.input_layer:
            self.input_layer._batched_backprop(backprop_stream)

    def _batched_backprop(self, backprop_stream):
        if not self.input_layer:
            self.input_layer._batched_backprop(backprop_stream)

    def batch_sum_reduce(self, lr:float, stream_a, stream_b, gamma:float=.95):
        ib,ih,iw,*fi = self.input.shape
        if fi:
            fi = fi[0]
            batch_sum_4d[self.bpg, self.tpb, stream_b](self.input, ib, ih, iw, fi)
        else:
            batch_sum_3d[self.bpg, self.tpb, stream_b](self.input, ib, ih, iw)
        sb,sh,sw,*sf = self.s.shape
        if sf:
            sf = sf[0]
            batch_sum_4d[self.bpg, self.tpb, stream_a](self.s, sb, sh, sw, sf)
        else:
            batch_sum_3d[self.bpg, self.tpb, stream_a](self.s, sb, sh, sw)

        if self.is_input:
            self.update(lr, stream_a, stream_b, GLOBAL_DTYPE(gamma))
        else:
            self.input_layer.batch_sum_reduce(lr, stream_a, stream_b, gamma)

    def update(self,lr:float,update_stream,cleanup_stream,gamma:float=.95,):
        if self.output_layer:
            self.output_layer.update(lr,update_stream,cleanup_stream,gamma)

    def __str__(self) -> str:
        inpt_shape = self.input.shape if self.input is not init_default else "None"
        out_shape = self.output.shape if self.output is not init_default else "None"
        return f"{self.layer_type}/{self.name}:{{in shape:{inpt_shape}; out shape:{out_shape}}}"

    def __repr__(self) -> str:
        inpt_shape = self.input.shape if self.input is not init_default else "None"
        out_shape = self.output.shape if self.output is not init_default else "None"
        return f"{self.layer_type}/{self.name}:{{in shape:{inpt_shape}; out shape:{out_shape}}}"

class VectorOutputLayer(BaseLayer):
    def __init__(self,name:str, output_shape:list, input_shape:list=None,
                 kh:int=None, kw:int=None,
                 Fmult:int=1, input_layer:BaseLayer=None, output_layer:BaseLayer=None,use_momentum:bool=False,batch_size:int=1) -> None:
        """

        :param input_shape: A 3-tuple that expresses the height, width, and filter depth of the
                            inputs given to this layer
        :type input_shape: tuple[int,int,int]

        :param kh:
        :type kh:

        :param kw:
        :type kw:


        :param Fmult:
        :type Fmult:

        :param input_layer: [OPTIONAL] a reference to the network layer that precedes this one in a forward pass.
        :type input_layer: Any subclass of BaseLayer

        :param output_layer: [OPTIONAL] a reference to the network layer that follows this one in a forward pass.
        :type output_layer: Any subclass of BaseLayer
        """
        assert isinstance(Fmult,int) and Fmult>1, f"The filter multiplier, Fmult, must be an integer and Fmult>=1: {Fmult=}"
        if kh is None:
            kh = MUTABLE_GLOBALS.KERNEL_HW
        if kw is None:
            kw = MUTABLE_GLOBALS.KERNEL_HW
        super().__init__(name,input_shape, input_layer, output_layer,use_momentum,batch_size)
        self.output_shape = output_shape

        self.tpb = [*MUTABLE_GLOBALS.TPB_VECT_FORWARD]
        self.tpb_1d = np.prod(MUTABLE_GLOBALS.TPB_VECT_FORWARD)
        self.bpg = [1, 1]
        self.bpg_1d = 1
        self.kh = kh
        self.kw = kw
        self.filters_in = 1
        self.filters_out = 11

    def dump_to_host(self, samp_idx,stream):
        """A diagnostic debugging function that snapshots all of the layer's array components from the gpu and "dumps"
        the data to host (cpu) memory for inspection."""
        keys = ["input","out","s","W","b"]
        if self.blame_out is not None:
            keys.append("blame_out")
        if self.blame_in is not None:
            keys.append("blame_in")
        keys.append("output")
        ret = {k:getattr(self,k).copy_to_host(stream=stream)[samp_idx] for k in keys}
        ret = {k:(arr.shape,arr,*(arr.argmax(i) for i in range(len(arr.shape)))) for k,arr in ret.items()}
        return ret

    def compile(self,input_data_mean):
        if self.output_layer is None:
            self.is_output = True
        if self.input_layer is None:
            self.is_input = True
            self.input = np.empty([self.batch_size] + self._input_shape, GLOBAL_DTYPE)
        else:
            self.input = self.input_layer.output
            self._input_shape = self.input.shape
            self.blame_out = self.input_layer.blame_in
        inpt = self.input
        ih, iw, f_in = inpt.shape[1:]
        self.tpb_1d = self.tpb[0]*self.tpb[1]
        for i, (t,ax) in enumerate(zip(self.tpb,inpt.shape[-2:])):
            self.bpg[i] = (ax + t - 1) // t
        self.bpg_1d = (ih+self.tpb_1d-1)//self.tpb_1d
        self.filters_in = f_in
        self.filters_out = f_in * MUTABLE_GLOBALS.FMULT
        self.W = cuda.to_device(random_initializer(input_data_mean, self.output_shape[0], np.prod(self.input.shape[1:])))
        self.b = cuda.to_device(random_initializer(input_data_mean, self.output_shape[0]))
        self.s = cuda.device_array(shape=(self.batch_size,*self.output_shape), dtype=GLOBAL_DTYPE)
        self.out = cuda.device_array_like(self.s)
        self.net = cuda.device_array_like(self.s)
        self.output = self.out
        MUTABLE_GLOBALS.POOL_STRIDES = 1 # must be called before compiling gpu kernels
        self._compile_forward_kernels()
        self._compile_backprop_kernels()
        self._compile_update_kernels()
        if not self.is_output:
            self.blame_in = cuda.device_array_like(self.s)
            self.output_layer.compile(input_data_mean)

    def _compile_forward_kernels(self):
        kernels = forward_vector_wrapper(self.tpb,MUTABLE_GLOBALS.FMULT)
        self.forward_net = cuda.jit(func_or_sig=kernels["inline"]["net"],fastmath=True)
        self.forward_xfr = cuda.jit(func_or_sig=kernels["inline"]["xfr"],fastmath=True)
        self.batched_forward_net = cuda.jit(func_or_sig=kernels["batched"]["net"],fastmath=True)
        self.batched_forward_xfr = cuda.jit(func_or_sig=kernels["batched"]["xfr"],fastmath=True)

    def _compile_backprop_kernels(self):
        kernels = backprop_vector_wrapper()
        if self.is_output:
            self.backprop_sense = cuda.jit(func_or_sig=kernels["inline"]["fsense"],fastmath=True)
            self.batched_backprop_sense = cuda.jit(func_or_sig=kernels["batched"]["fsense"],fastmath=True)
        else:
            self.backprop_sense = cuda.jit(func_or_sig=kernels["inline"]["sense"], fastmath=True)
            self.batched_backprop_sense = cuda.jit(func_or_sig=kernels["batched"]["sense"], fastmath=True)
        self.backprop_blame = cuda.jit(func_or_sig=kernels["inline"]["blame"],fastmath=True)
        self.batched_backprop_blame = cuda.jit(func_or_sig=kernels["batched"]["blame"], fastmath=True)

    def _compile_update_kernels(self):
        if self.use_momentum:
            self.update_bias = cuda.jit(func_or_sig=momentum_update_bias_vector_kernel,fastmath=True)
            self.update_weights = cuda.jit(func_or_sig=momentum_update_weights_vector_kernel,fastmath=True)
        else:
            self.update_bias = cuda.jit(func_or_sig=update_bias_vector_kernel,fastmath=True)
            self.update_weights = cuda.jit(func_or_sig=update_weights_vector_kernel,fastmath=True)

    def forward(self, input_p: np.ndarray, forward_stream, cleanup_stream, samp_idx: int = 0):
        """Starts the forward pass through the network. input_p should be a 3d; height:width:filter-channels
        :param cleanup_stream:
        :type cleanup_stream:
        """
        # root_info_logger(f"starting forward pass for {samp_idx=}")
        self.input = input_p
        self._forward(forward_stream,cleanup_stream,samp_idx)

    def _forward(self, forward_stream, cleanup_stream, samp_idx: int = 0):
        if len(self.s.shape) == 4:
            cuda_batched_reset_to_zero_hwd[self.bpg, self.tpb, cleanup_stream](self.s)
        else:
            cuda_batched_reset_to_zero_hw[self.bpg, self.tpb, cleanup_stream](self.s)
        self.forward_net[self.bpg, self.tpb, forward_stream](self.input[samp_idx], self.W, self.b, self.net[samp_idx])
        self.forward_xfr[self.bpg_1d, self.tpb_1d, forward_stream](self.net[samp_idx],self.out[samp_idx])
        if self.output_layer:
            self.output_layer._forward(forward_stream,samp_idx)

    def batched_forward(self, input_p: np.ndarray, forward_stream, cleanup_stream):
        self.input = input_p
        if self.output_layer:
            self.output_layer._batched_forward(forward_stream,cleanup_stream)

    def _batched_forward(self, forward_stream, cleanup_stream):
        if len(self.s.shape) == 4:
            cuda_batched_reset_to_zero_hwd[self.bpg, self.tpb, cleanup_stream](self.s)
        else:
            cuda_batched_reset_to_zero_hw[self.bpg, self.tpb, cleanup_stream](self.s)
        self.batched_forward_net[self.bpg, self.tpb, forward_stream](self.input, self.W, self.b, self.net)
        self.batched_forward_xfr[self.bpg_1d, self.tpb_1d, forward_stream](self.net,self.out)
        if self.output_layer:
            self.output_layer._batched_forward(forward_stream,cleanup_stream)

    def backprop(self, err: np.ndarray, backprop_stream,samp_idx):
        self.backprop_sense[self.bpg_1d, self.tpb_1d, backprop_stream](self.s[samp_idx], self.out[samp_idx], err)
        self.backprop_blame[self.bpg, self.tpb, backprop_stream](self.blame_out[samp_idx], self.s[samp_idx], self.W)
        if self.input_layer:
            self.input_layer._backprop(backprop_stream,samp_idx)

    def _backprop(self, backprop_stream,samp_idx):
        # root_info_logger(f"continuing backprop phase at layer: {self.name} for {samp_idx=}")
        # self.last_backprop_dumps.append(self.dump_to_host)
        self.backprop_sense[self.bpg_1d, self.tpb_1d, backprop_stream](self.blame_in[samp_idx], self.s[samp_idx], self.out[samp_idx])
        # self.last_backprop_dumps.append(self.dump_to_host)
        self.backprop_blame[self.bpg, self.tpb, backprop_stream](self.blame_out[samp_idx], self.s[samp_idx], self.W)
        # self.last_backprop_dumps.append(self.dump_to_host)
        # check = [(k,*(np.any(c1[k][1]!=c2[k][1]) for c1,c2 in combinations(self.last_backprop_dumps, 2)))
        #          for k in self.last_backprop_dumps[0].keys()]
        # check = [tpl for tpl in check if any(tpl[1:])]
        # last_dumps = self.last_backprop_dumps
        # b_in = last_dumps[-1]["blame_in"]
        # s = last_dumps[-1]["s"]
        # b_out = last_dumps[-1]["blame_out"] if not self.is_input else [[1],[1]]
        # if (np.count_nonzero(b_in[1])>0 \
        #         and np.count_nonzero(s[1])==0) \
        #         or np.count_nonzero(b_out[1])==0:
        #     dbg_break = 0
        if self.input_layer:
            self.input_layer._backprop(backprop_stream,samp_idx)

    def batched_backprop(self, err: np.ndarray, backprop_stream):
        self.batched_backprop_sense[self.bpg_1d, self.tpb_1d, backprop_stream](self.s, self.out, err)
        self.batched_backprop_blame[self.bpg, self.tpb, backprop_stream](self.blame_out, self.s, self.W)
        if self.input_layer:
            self.input_layer._batched_backprop(backprop_stream)

    def _batched_backprop(self, backprop_stream):
        self.batched_backprop_sense[self.bpg_1d, self.tpb_1d, backprop_stream](self.blame_in, self.s, self.out)
        self.batched_backprop_blame[self.bpg, self.tpb, backprop_stream](self.blame_out, self.s, self.W)
        if self.input_layer:
            self.input_layer._batched_backprop(backprop_stream)

    def update(self,lr:float,update_stream,cleanup_stream,gamma:GLOBAL_DTYPE=.95):
        if not self.is_input:
            if len(self.blame_out.shape)==4:
                cuda_batched_reset_to_zero_hwd[self.bpg, self.tpb, cleanup_stream](self.blame_out)
            else:
                cuda_batched_reset_to_zero_hw[self.bpg, self.tpb, cleanup_stream](self.blame_out)
        self.update_bias[self.bpg, self.tpb,update_stream](gamma,lr,self.s[0],self.b)
        self.update_weights[self.bpg, self.tpb,update_stream](gamma,lr,self.input[0],self.s[0],self.W)
        if not self.is_output:
            self.output_layer.update(lr,update_stream,cleanup_stream,gamma)

    @property
    def layer_type(self):
        return "VectorOutputLayer"


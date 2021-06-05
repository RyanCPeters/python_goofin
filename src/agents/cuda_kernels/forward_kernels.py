from src.agents.cuda_kernels.cuda_kernels import *

########################################################################################################################
## forward pass prediction kernels
########################################################################################################################

###########################################################
## forward net and transfer kernels for final network layer
###########################################################
def forward_vector_wrapper(tpb_vect,fmult, tfunc = cuda_device_sigmoid):
    tpb_vect = tuple(tpb_vect)
    def forward_vector_net_kernel(layer_input:np.ndarray, layer_W:np.ndarray,
                                  layer_b:np.ndarray, layer_net:np.ndarray):
        """Perform matrix multiplication of layer_out = layer_w * layer_input

        layer_input would normally be given with dimensions
            h:w:Fin
        however, we will transparently flatten layer_input to be:
            h*w*Fin:1
        then we can permit the assumption that:
            layer_w has dims
                M:h*w*Fin
            layer_out has dims
                M:1
        """
        jinit, iinit = cuda.grid(2)  # gets the current thread's position. here it's the i'th row and j'th column
        jstep, istep = cuda.gridsize(2)
        tstart = jinit*istep+iinit # each thread's starting position in a flattened iteration
        tstep = istep*jstep # grid sized flattened step size for all threads.
        h, w = layer_W.shape
        # Each element of layer_net is the sum of sums of products... As in layer_out = sum(layer_W*layer_input)+layer_b
        # so we initialize layer_out with the values from layer_b to handle that the outer sum.
        for i in range(tstart,h,tstep):
            layer_net[i, 0] = layer_b[i]
        cuda.threadfence()
        # We will be iterating over the flattened input array
        f_in,_h,_w = layer_input.shape
        hw_in = _h*_w
        for i in range(h):
            tmp = 0
            for j in range(tstart, w, tstep):
                ij_w = layer_W[i,j]
                inpt_j = j % _w
                inpt_i = (j // _w) % _h
                f = (j // hw_in) % f_in
                inpt = layer_input[f, inpt_i, inpt_j]
                # using fused-multiply-add to update tmp
                tmp = cuda.fma(inpt,ij_w,tmp)
            cuda.atomic.add(layer_net,(i,0),tmp)

    def batched_forward_vector_net_kernel(layer_input:np.ndarray, layer_W:np.ndarray,
                                          layer_b:np.ndarray, layer_net:np.ndarray):
        """Perform matrix multiplication of layer_out = layer_w * layer_input

        NOTE: layer_W and layer_b do not have batched duplicates of themselves

        layer_input would normally be given with dimensions
            h:w:Fin
        however, we will transparently flatten layer_input to be:
            h*w*Fin:1
        then we can permit the assumption that:
            layer_w has dims
                M:h*w*Fin
            layer_out has dims
                M:1
        """
        smb = cuda.shared.array(fmult,GLOBAL_DTYPE)
        jinit, iinit = cuda.grid(2)  # gets the current thread's position. here it's the i'th row and j'th column
        jstep, istep = cuda.gridsize(2)
        jbstrt,ibstrt = cuda.threadIdx.x,cuda.threadIdx.y
        jbstep,ibstep = cuda.blockDim.x,cuda.blockDim.y
        blockstart = jbstrt*ibstep+ibstrt
        blockstep = jbstep*ibstep
        bh = layer_b.shape[0]
        for i in range(blockstart,bh,blockstep):
            smb[i] = layer_b[i]
        cuda.syncthreads()
        gridstart = jinit*istep+iinit # each thread's starting position in a flattened iteration
        gridstep = istep*jstep # grid sized flattened step size for all threads.
        h, w = layer_W.shape
        batch = layer_net.shape[0]
        # Each element of layer_net is the sum of sums of products... As in layer_out = sum(layer_W*layer_input)+layer_b
        # so we initialize layer_out with the values from layer_b to handle that the outer sum.
        for b in range(batch):
            for i in range(gridstart,h,gridstep):
                layer_net[b,i, 0] = smb[i]
        cuda.threadfence()
        # We will be iterating over the flattened input array
        f_in,_h,_w = layer_input.shape[1:]
        hw_in = _h*_w
        for b in range(batch):
            for i in range(h):
                tmp = 0.
                # The best, and only practical way to ensure proper distribution of workload
                # across all threads in device grid for all possible array dimensions is to
                # perform a flattened iteration of the array elements and map the flattened
                # index back to the f:i:j indices.
                for j in range(gridstart, w, gridstep):
                    ij_w = layer_W[i,j]
                    inpt_j = j % _w
                    inpt_i = (j // _w) % _h
                    f = (j // hw_in) % f_in
                    inpt = layer_input[b,f, inpt_i, inpt_j]
                    # using fused-multiply-add to update tmp
                    tmp = cuda.fma(inpt,ij_w,tmp)
                cuda.syncthreads()
                cuda.atomic.add(layer_net,(b,i,0),tmp)

    def forward_vector_transfer_kernel(layer_net:np.ndarray,layer_out:np.ndarray):
        sm = cuda.shared.array(tpb_vect,GLOBAL_DTYPE)
        start_i = cuda.grid(1)  # gets the current thread's position. here it's the i'th row and j'th column
        i_step = cuda.gridsize(1)
        tj,ti = cuda.threadIdx.x,cuda.threadIdx.y
        h = layer_out.shape[0]
        stop = 0
        for _i in range(0, h-i_step, i_step):
            i = _i + start_i
            sm[tj,ti] = layer_net[i, 0]
            cuda.syncthreads()
            sm[tj,ti] = tfunc(sm[tj,ti])
            cuda.syncthreads()
            layer_out[i, 0] = sm[tj,ti]
            stop = i
        stop += i_step
        if stop<h:
            layer_out[stop,0] = tfunc(layer_net[i, 0])

    def batched_forward_vector_transfer_kernel(layer_net:np.ndarray,layer_out:np.ndarray):
        sm = cuda.shared.array(tpb_vect,GLOBAL_DTYPE)
        start_i = cuda.grid(1)  # gets the current thread's position. here it's the i'th row and j'th column
        i_step = cuda.gridsize(1)
        tj,ti = cuda.threadIdx.x,cuda.threadIdx.y
        batch,h = layer_out.shape[:2]
        for b in range(batch):
            for _i in range(0, h - i_step, i_step):
                i = _i + start_i
                sm[tj, ti] = layer_net[b,i, 0]
                cuda.syncthreads()
                sm[tj, ti] = tfunc(sm[tj, ti])
                cuda.syncthreads()
                layer_out[b,i, 0] = sm[tj, ti]
                stop = i
            stop += i_step
            if stop < h:
                layer_out[b,stop, 0] = tfunc(layer_net[b,i, 0])

    ret = {"batched": {"net": batched_forward_vector_net_kernel,
                       "xfr": batched_forward_vector_transfer_kernel,
                       },
           "inline": {"net": forward_vector_net_kernel,
                      "xfr": forward_vector_transfer_kernel,
                      }}
    return ret

#############################################################
## forward net and transfer kernels for hidden network layers
#############################################################
def forward_conv_wrapper(tpb_conv, tfunc = cuda_device_relu):
    tpb_conv = tuple(map(int,tpb_conv))
    def forward_conv_net_kernel(layer_input:np.ndarray, layer_W:np.ndarray,
                                layer_b:np.ndarray, layer_net:np.ndarray):
        """Perform convolutional multiplication between layer_w and layer_input

        In simplest terms::

            net  = input * W + b # for each filter channel


        where::

            layer_b     has dims ( Fmult )
            layer_input has dims ( h    : w     : Fin )
            layer_W     has dims ( K1   : K2    : Fmult )
            layer_out   has dims ( h-K1 : w-K2  : FinFmult )

        h, K1, w, and K2 are heights and widths in terms of pixels.

        Fin is the number of filters in the layer_input.

        Fmult is the number of filters generated per input filter.

        Fin*Fmult is the total number of resulting filters in the output.

        The K1,K2 dimensions of layer_w (... : K1 : K2 : ...) represent pixel height and pixel width of our
        convolutional kernel for each "node" in the this layer.


        Note 1: we assume that the I/O for will only be grayscale single channel images.

        Note 2: We do not bother with the 180 degree rotation of the kernel as it is assumed the kernel is built using
                randomly selected weights. Implying that regardless of orientation, the network will learn the proper
                final weights, and their locations, as the network converges.

        see the following for a brief on convolutional forward pass:
        https://towardsdatascience.com/forward-and-backward-propagations-for-2d-convolutional-layers-ed970f8bf602

         Quick Note: We use cuda.shared.array memory to allow the gpu to do batched memory caching from global shared
             memory to thread-block specific cached memory. This reduces the amount of global memory accesses
             by a factor of approximately (tpbj * tpbi * bpgj * bpgi).
             This reduces load on the memory bus, thus allowing us to use much larger and more numerous hidden
             layers, and/or to run more concurrent input evaluations.

             For the value of increasing hidden layer size/count see the publication:
               "Sensitivity and Generalization in Neural Networks: an Empirical Study"
               found here: https://arxiv.org/abs/1802.08760
               cited on 3/5/21

        :param layer_input:
        :type layer_input: np.ndarray[GLOBAL_DTYPE]

        :param layer_W: An array of node-wise kernels that sums select nodes from the input to produce this layer's net
        :type layer_W: np.ndarray[GLOBAL_DTYPE]

        :param layer_b:
        :type layer_b: np.ndarray[GLOBAL_DTYPE]

        :param layer_out:
        :type layer_out: np.ndarray[GLOBAL_DTYPE]

        :return: Cuda kernels have no return value; however, the results of the operations done in this function are
                 stored in the layer_out argument.
        :rtype: None
        """
        smw = cuda.shared.array(tpb_conv, GLOBAL_DTYPE)
        jinit, iinit = cuda.grid(2) # gets the current thread's position. here it's the i'th row and j'th column
        jstep,istep= cuda.gridsize(2)
        tj,ti = cuda.threadIdx.x,cuda.threadIdx.y
        if_len,ih,iw = layer_input.shape
        of_len,oh,ow = layer_net.shape
        kf,kh,kw = layer_W.shape
        half_kh = int(kh / 2 + .5) # faster than calling python's `round` builtin function for unsigned numbers.
        half_kw = int(kw / 2 + .5)
        # initialize layer_out to the values of layer_b
        # this is accounts for the final summation between the product of W*inpt and b -> W * inpt + b
        for fm in range(kf):
            b = layer_b[fm]
            for f1 in range(if_len):
                f2 = fm*if_len+f1
                for i in range(iinit, oh, istep):
                    for j in range(jinit, ow, jstep):
                        layer_net[f2,i,j] = b
        # wait for all threads in current block to finish updating the region of the output under their control.
        cuda.syncthreads()
        # preprocessing some boundary checks.
        _iinit = iinit+half_kh
        _jinit = jinit+half_kw
        _half_kh = 1-half_kh
        _half_kw = 1-half_kw
        for fm in range(kf):
            _f2 = fm*if_len
            smw[ti,tj] = layer_W[fm,ti,tj]
            cuda.syncthreads()
            m = ti
            n = tj
            # for m in range(kh):
            #     for n in range(kw):
            #         w = layer_W[fm, m, n]
            for f1 in range(if_len):
                f2 = _f2 + f1
                for i in range(_iinit, oh, istep):
                    i_inpt = m
                    i_inpt += i
                    i_inpt += _half_kh
                    inp = layer_input[f1,i_inpt]
                    for j in range(_jinit, ow, jstep):
                        j_inpt = n
                        j_inpt += j
                        j_inpt += _half_kw
                        inpt = inp[j_inpt]
                        tmp = layer_net[f2,i,j]
                        tmp = cuda.fma(inpt,smw[ti,tj],tmp)
                        layer_net[f2, i, j] = tmp

    def batched_forward_conv_net_kernel(layer_input:np.ndarray, layer_W:np.ndarray,
                                        layer_b:np.ndarray, layer_net:np.ndarray):
        """Perform convolutional multiplication between layer_w and layer_input

        In simplest terms::

            net  = input * W + b # for each filter channel


        where::

            layer_b     has dims ( Fmult )
            layer_input has dims ( h    : w     : Fin )
            layer_W     has dims ( K1   : K2    : Fmult )
            layer_out   has dims ( h-K1 : w-K2  : FinFmult )

        h, K1, w, and K2 are heights and widths in terms of pixels.

        Fin is the number of filters in the layer_input.

        Fmult is the number of filters generated per input filter.

        Fin*Fmult is the total number of resulting filters in the output.

        The K1,K2 dimensions of layer_w (... : K1 : K2 : ...) represent pixel height and pixel width of our
        convolutional kernel for each "node" in the this layer.


        Note 1: we assume that the I/O for will only be grayscale single channel images.

        Note 2: We do not bother with the 180 degree rotation of the kernel as it is assumed the kernel is built using
                randomly selected weights. Implying that regardless of orientation, the network will learn the proper
                final weights, and their locations, as the network converges.

        see the following for a brief on convolutional forward pass:
        https://towardsdatascience.com/forward-and-backward-propagations-for-2d-convolutional-layers-ed970f8bf602

         Quick Note: We use cuda.shared.array memory to allow the gpu to do batched memory caching from global shared
             memory to thread-block specific cached memory. This reduces the amount of global memory accesses
             by a factor of approximately (tpbj * tpbi * bpgj * bpgi).
             This reduces load on the memory bus, thus allowing us to use much larger and more numerous hidden
             layers, and/or to run more concurrent input evaluations.

             For the value of increasing hidden layer size/count see the publication:
               "Sensitivity and Generalization in Neural Networks: an Empirical Study"
               found here: https://arxiv.org/abs/1802.08760
               cited on 3/5/21

        :param layer_input:
        :type layer_input: np.ndarray[GLOBAL_DTYPE]

        :param layer_W: An array of node-wise kernels that sums select nodes from the input to produce this layer's net
        :type layer_W: np.ndarray[GLOBAL_DTYPE]

        :param layer_b:
        :type layer_b: np.ndarray[GLOBAL_DTYPE]

        :param layer_out:
        :type layer_out: np.ndarray[GLOBAL_DTYPE]

        :return: Cuda kernels have no return value; however, the results of the operations done in this function are
                 stored in the layer_out argument.
        :rtype: None
        """
        smw = cuda.shared.array(tpb_conv, GLOBAL_DTYPE)
        jinit, iinit = cuda.grid(2) # gets the current thread's position. here it's the i'th row and j'th column
        jstep,istep= cuda.gridsize(2)
        tj,ti = cuda.threadIdx.x,cuda.threadIdx.y
        batch,if_len,ih,iw = layer_input.shape
        of_len,oh,ow = layer_net.shape[1:]
        kf,kh,kw = layer_W.shape
        half_kh = int(kh / 2 + .5) # faster than calling python's `round` builtin function for unsigned numbers.
        half_kw = int(kw / 2 + .5)
        # initialize layer_out to the values of layer_b
        # this is accounts for the final summation between the product of W*inpt and b -> W * inpt + b
        for fm in range(kf):
            bias = layer_b[fm]
            _f2 = fm*if_len
            for b in range(batch):
                for f1 in range(if_len):
                    f2 = _f2+f1
                    for i in range(iinit, oh, istep):
                        for j in range(jinit, ow, jstep):
                            layer_net[b,f2,i,j] = bias
        # wait for all threads in current block to finish updating the region of the output under their control.
        cuda.syncthreads()
        # preprocessing some boundary checks.
        if half_kh>oh:
            half_kh = oh
        if half_kw>ow:
            half_kw = ow
        _iinit = iinit+half_kh
        _jinit = jinit+half_kw
        _half_kh = 1-half_kh
        _half_kw = 1-half_kw
        # now we compute the products of the input and weight kernels then add those products to the output.
        for fm in range(kf):
            _f2 = fm*if_len
            smw[ti,tj] = layer_W[fm,ti,tj]
            cuda.syncthreads()
            m = ti
            n = tj
            for b in range(batch):
                for f1 in range(if_len):
                    f2 = _f2 + f1
                    for i in range(_iinit, oh, istep):
                        i_inpt = m
                        i_inpt += i
                        i_inpt += _half_kh
                        inp = layer_input[b,f1,i_inpt]
                        for j in range(_jinit, ow, jstep):
                            j_inpt = n
                            j_inpt += j
                            j_inpt += _half_kw
                            inpt = inp[j_inpt]
                            tmp = layer_net[b,f2,i,j]
                            tmp = cuda.fma(inpt,smw[ti,tj],tmp)
                            layer_net[b, f2, i, j] = tmp

    def forward_transfer_kernel(layer_net:np.ndarray, layer_out:np.ndarray):
        """
            layer_out   has dims (h : w : Fin*Fmult)

        :param layer_out: the computed net sum of products for the current input, weights and bias. We will now
                          update these values by applying the sigmoid transfer function to them.
        :type layer_out: np.ndarray[GLOBAL_DTYPE]

        :return: Cuda kernels have no return value; however, the results of the operations done in this function are
                 stored in the layer_out argument.
        :rtype: None
        """
        sm_batch = cuda.shared.array(tpb_conv, dtype=GLOBAL_DTYPE)
        start_j,start_i = cuda.grid(2)
        j_step,i_step = cuda.gridsize(2)
        tj,ti = cuda.threadIdx.x,cuda.threadIdx.y
        f_len,h,w = layer_out.shape
        total_out = h*w*f_len
        flattened_start = start_j*i_step+start_i
        flat_step = i_step*j_step
        hw = h*w
        # by flattening the arrays, we ensure the most even distribution of work across our threads.
        for _v in range(0,total_out,flat_step):
            v = _v + flattened_start
            _w = v%w
            _h = (v//w)%h
            f = (v//hw)%f_len
            sm_batch[tj,ti]= layer_net[f,_h,_w]
            cuda.syncthreads()
            sm_batch[tj,ti]= tfunc(sm_batch[tj,ti])
            cuda.syncthreads()
            layer_out[f,_h,_w] = sm_batch[tj,ti]

    def batched_forward_transfer_kernel(layer_net:np.ndarray, layer_out:np.ndarray):
        """
            layer_out   has dims (h : w : Fin*Fmult)

        :param layer_out: the computed net sum of products for the current input, weights and bias. We will now
                          update these values by applying the sigmoid transfer function to them.
        :type layer_out: np.ndarray[GLOBAL_DTYPE]

        :return: Cuda kernels have no return value; however, the results of the operations done in this function are
                 stored in the layer_out argument.
        :rtype: None
        """
        # sm_d = cuda.shared.array((tpbj,tpbi,FinFmult), dtype=GLOBAL_DTYPE)
        sm_batch = cuda.shared.array(tpb_conv, dtype=GLOBAL_DTYPE)
        start_j,start_i = cuda.grid(2)
        j_step,i_step = cuda.gridsize(2)
        tj,ti = cuda.threadIdx.x,cuda.threadIdx.y
        batch,f_len,h,w = layer_out.shape
        total_out = h*w*f_len*batch
        flattened_start = start_j*i_step+start_i
        flat_step = i_step*j_step
        hw = h*w
        fhw = f_len*hw
        # by flattening the arrays, we ensure the most even distribution of work across our threads.
        # for v in range(flattened_start,total_out,flat_step):
        for _v in range(0,total_out,flat_step):
            v = _v + flattened_start
            _w = v%w
            _h = (v//w)%h
            f = (v//hw)%f_len
            b = (v//fhw)%batch
            sm_batch[tj,ti]= layer_net[b,f,_h,_w]
            cuda.syncthreads()
            sm_batch[tj,ti]= tfunc(sm_batch[tj,ti])
            cuda.syncthreads()
            layer_out[b,f,_h,_w] = sm_batch[tj,ti]


    def forward_maxpool_kernel(layer_out:np.ndarray, layer_pool:np.ndarray,
                               layer_net:np.ndarray, pool_stride):
        """
            layer_out           has dims: (h              :   w               :   Fin*Fmult)
            layer_net           has dims: (h              :   w               :   Fin*Fmult)
            layer_pool          has dims: (h//pool_stride :   w//pool_stride  :   Fin*Fmult)

        :param layer_out: The output after the layer's transfer function has been applied to the layer's net.
        :type layer_out: np.ndarray[GLOBAL_DTYPE]

        :param layer_pool: A np.ndarray with dtyp: GLOBAL_DTYPE;
                           shape: (h//pool_stride : w//pool_stride : Fin*Fmult)
        :type layer_pool: np.ndarray[GLOBAL_DTYPE]

        :param layer_net: A np.ndarray with dtyp: np.bool_;
                                shape: (h//pool_stride : w//pool_stride : Fin*Fmult)
                                Serves as a value mask where nodes not chosen for the pool layer are set to 0, thus
                                establishing layer_net as a sparse matrix that we can use in backprop without
                                having to do any further mapping.
        :type layer_net: np.ndarray[GLOBAL_DTYPE]

        :return: Cuda kernels have no return value; however, the results of the operations done in this function are
                 stored in the layer_pool and layer_net arguments.
        :rtype: None
        """
        start_j, start_i = cuda.grid(2)
        F, h, w = layer_out.shape
        pf, ph, pw = layer_pool.shape
        j_step, i_step = cuda.gridsize(2)
        # we iterate over the entire output, scaling steps and starting positions by pool_stride
        # to ensure non-overlapping regions for each thread, and select the max value from these regions
        # which we then store in layer_pool, and then mark the positions of the max nodes with a 1 in the
        # layer_pool_sparse array.
        if ph==1 and pw==1:
            tstart = start_j*i_step+start_i
            tstep = j_step*i_step
            for f in range(tstart,pf,tstep):
                fmaxo = -0xffffffff
                fmaxn = 0
                mi = 0
                mj = 0
                for i in range(h):
                    for j in range(w):
                        n = layer_net[f,i,j]
                        layer_net[f,i,j] *= 0
                        o = layer_out[f,i,j]
                        if fmaxo<o:
                            fmaxo = o
                            fmaxn = n
                            mi = i
                            mj = j
                layer_pool[f,0,0] = fmaxo # we only get here if layer_pool has dimensions F:1:1
                layer_net[f,mi,mj] = fmaxn
            return # in this condition all threads will reach the same outcome.
        si = start_i*pool_stride
        sj = start_j*pool_stride
        h_ = h-pool_stride
        w_ = w-pool_stride
        sistep = i_step*pool_stride
        sjstep = j_step*pool_stride
        for f in range(F):
            for i in range(si,h_, sistep):
                pi = i//pool_stride
                for j in range(sj,w_, sjstep):
                    pj = j//pool_stride
                    # block-wide preload (caching) of layer_out for the region we wish to down-sample to max value
                    # block-wide computations, using only shared memory, that select the max pool winner
                    fmaxo = -0xffffffff
                    fmaxn = -0xffffffff
                    mi = i
                    mj = j
                    for fi in range(i,i+pool_stride):
                        for fj in range(j,j+pool_stride):
                            n = layer_net[f, fi, fj]
                            layer_net[f, fi, fj] *= 0 # we also need to clear the pool region to 0 before saving the max.
                            o = layer_out[f, fi, fj]
                            if fmaxo<o:
                                fmaxo = o
                                fmaxn = n
                                mi = fi
                                mj = fj
                    layer_pool[f,pi,pj] = fmaxo
                    layer_net[f,mi,mj] = fmaxn


    def batched_forward_maxpool_kernel(layer_out:np.ndarray, layer_pool:np.ndarray,
                               layer_net:np.ndarray, pool_stride):
        """
            layer_out           has dims: (h              :   w               :   Fin*Fmult)
            layer_net           has dims: (h              :   w               :   Fin*Fmult)
            layer_pool          has dims: (h//pool_stride :   w//pool_stride  :   Fin*Fmult)

        :param layer_out: The output after the layer's transfer function has been applied to the layer's net.
        :type layer_out: np.ndarray[GLOBAL_DTYPE]

        :param layer_pool: A np.ndarray with dtyp: GLOBAL_DTYPE;
                           shape: (h//pool_stride : w//pool_stride : Fin*Fmult)
        :type layer_pool: np.ndarray[GLOBAL_DTYPE]

        :param layer_net: A np.ndarray with dtyp: np.bool_;
                                shape: (h//pool_stride : w//pool_stride : Fin*Fmult)
                                Serves as a value mask where nodes not chosen for the pool layer are set to 0, thus
                                establishing layer_net as a sparse matrix that we can use in backprop without
                                having to do any further mapping.
        :type layer_net: np.ndarray[GLOBAL_DTYPE]

        :return: Cuda kernels have no return value; however, the results of the operations done in this function are
                 stored in the layer_pool and layer_net arguments.
        :rtype: None
        """
        jstrt, istrt = cuda.grid(2)
        jstep, istep = cuda.gridsize(2)
        batch,f_len, h, w = layer_out.shape
        pf, ph, pw = layer_pool.shape[1:]
        # we iterate over the entire output, scaling steps and starting positions by pool_stride
        # to ensure non-overlapping regions for each thread, and select the max value from these regions
        # which we then store in layer_pool, and then mark the positions of the max nodes with a 1 in the
        # layer_pool_sparse array.
        if ph==1 and pw==1:
            tstart = jstrt*istep+istrt
            tstep = jstep*istep
            for b in range(batch):
                for f in range(tstart,pf,tstep):
                    fmaxo = -0xffffffff
                    fmaxn = 0
                    mi = 0
                    mj = 0
                    for i in range(h):
                        for j in range(w):
                            n = layer_net[b,f,i,j]
                            layer_net[b,f,i,j] *= 0
                            o = layer_out[b,f,i,j]
                            if fmaxo<o:
                                fmaxo = o
                                fmaxn = n
                                mi = i
                                mj = j
                    layer_pool[b,f,0,0] = fmaxo # we only get here if layer_pool has dimensions F:1:1
                    layer_net[b,f,mi,mj] = fmaxn
            return # in this condition all threads will reach the same outcome.
        si = istrt*pool_stride
        sj = jstrt*pool_stride
        h_ = h-pool_stride
        w_ = w-pool_stride
        sistep = istep*pool_stride
        sjstep = jstep*pool_stride
        for b in range(batch):
            for f in range(f_len):
                for i in range(si,h_, sistep):
                    pi = i//pool_stride
                    for j in range(sj,w_, sjstep):
                        # block-wide preload (caching) of layer_out for the region we wish to down-sample to max value
                        # block-wide computations, using only shared memory, that select the max pool winner
                        fmaxo = -0xffffffff
                        fmaxn = -0xffffffff
                        mi = i
                        mj = j
                        for pooli in range(i,i+pool_stride):
                            for poolj in range(j,j+pool_stride):
                                n = layer_net[b,f, pooli, poolj]
                                layer_net[b,f, pooli, poolj] *= 0 # we also need to clear the pool region to 0 before saving the max.
                                o = layer_out[b,f, pooli, poolj]
                                if fmaxo<o:
                                    fmaxo = o
                                    fmaxn = n
                                    mi = pooli
                                    mj = poolj
                        pj = j//pool_stride
                        layer_pool[b,f,pi,pj] = fmaxo
                        layer_net[b,f,mi,mj] = fmaxn

    ret = {"batched":{"net":batched_forward_conv_net_kernel,
                      "xfr":batched_forward_transfer_kernel,
                      "pool":batched_forward_maxpool_kernel,
                      },
           "inline":{"net":forward_conv_net_kernel,
                      "xfr":forward_transfer_kernel,
                      "pool":forward_maxpool_kernel,
                     }}
    return ret
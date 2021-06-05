from src.fashion_code.kernels.cuda_kernels import *
from numba import cuda


########################################################################################################################
## sensitivity and blame kernels
########################################################################################################################


############################################
## sense and blame kernels for vector layers
############################################
def backprop_vector_wrapper():
    def batched_backprop_sense_final_kernel(layer_s: np.ndarray, layer_out: np.ndarray, err: np.ndarray):
        """A 1-d kernel that computes the sensitivity from the blame passed to a given layer.

        returns nothing, but will update the values stored in final_s
        """
        # make the base assumption that the final output layer is a fully connected one-hot column vector
        start_i = cuda.grid(1)  # gets the current thread's position within the grid of blocks of threads.
        batch,h = layer_s.shape[:2]
        i_step = cuda.gridsize(1)
        for b in range(batch):
            for i in range(start_i, h, i_step):
                layer_s[b, i, 0] = cuda_device_dsigmoid(layer_out[b, i, 0])
                layer_s[b, i, 0] *= err[b, i, 0]

    def batched_backprop_sense_vector_kernel(blame_in: np.ndarray, layer_s: np.ndarray, layer_out: np.ndarray):
        """A 1-d kernel that computes the sensitivity from the blame passed to a given layer.

        returns nothing, but will update the values stored in layer_s
        """
        start_i = cuda.grid(1)  # gets the current thread's position. here it's the i'th row and j'th column
        b,h = layer_s.shape[:2]
        i_step = cuda.gridsize(1)
        for i in range(start_i, h, i_step):
            layer_s[b,i, 0] = cuda_device_dsigmoid(layer_out[b,i, 0])
            layer_s[b,i, 0] *= blame_in[b, i, 0]

    def batched_backprop_blame_vector_kernel(blame_out: np.ndarray, layer_s: np.ndarray, layer_W: np.ndarray):
        """Computes the W'*s (blame) associated with the final output layer.
        The logical flow of this function is based on the assumption that the final output layer is a one-hot encoding,
        meaning we expect layer_s to be a column vector, not a matrix.

        we also expect that blame_out takes the shape of the layer's input, for which we map the layer_W width dimension
        to in duplicate for each row in layer_W.



        NOTE: We leave it to the caller to reset blame_out to all zeros before passing it to this function kernel.

        :param blame_out:
        :type blame_out:
        :param layer_s:
        :type layer_s:
        :param layer_W:
        :type layer_W:
        :return:
        :rtype:
        """
        jinit, iinit = cuda.grid(2)  # gets the current thread's position. here it's the i'th row and j'th column
        jstep, istep = cuda.gridsize(2)
        tstart = jinit * istep + iinit
        tstep = jstep * istep
        h, w = layer_W.shape
        batch, bf, bh, bw = blame_out.shape
        bhw = bw * bh
        for b in range(batch):
            for j in range(tstart, w, tstep):
                tmp = 0.
                for i in range(h):
                    tmp = cuda.fma(layer_W[i, j], layer_s[b, i, 0], tmp)
                bj = j % bw
                bi = (j // bw) % bh
                f = (j // bhw) % bf
                blame_out[b, f, bi, bj] = tmp

    def backprop_sense_final_kernel(layer_s:np.ndarray, layer_out:np.ndarray, err:np.ndarray):
        """A 1-d kernel that computes the sensitivity from the blame passed to a given layer.

        returns nothing, but will update the values stored in final_s
        """
        # make the base assumption that the final output layer is a fully connected one-hot column vector
        start_i = cuda.grid(1)  # gets the current thread's position within the grid of blocks of threads.
        h = layer_s.shape[0]
        i_step = cuda.gridsize(1)
        for i in range(start_i, h, i_step):
            layer_s[i, 0] = cuda_device_dsigmoid(layer_out[i, 0])
            layer_s[i, 0] *= err[i, 0]

    def backprop_sense_vector_kernel(blame_in: np.ndarray, layer_s: np.ndarray, layer_out: np.ndarray):
        """A 1-d kernel that computes the sensitivity from the blame passed to a given layer.

        returns nothing, but will update the values stored in layer_s
        """
        start_i = cuda.grid(1)  # gets the current thread's position. here it's the i'th row and j'th column
        h = layer_s.shape[0]
        i_step = cuda.gridsize(1)
        for i in range(start_i, h, i_step):
            layer_s[i, 0] = cuda_device_dsigmoid(layer_out[i, 0])
            layer_s[i, 0] *= blame_in[i, 0]

    def backprop_blame_vector_kernel(blame_out:np.ndarray, layer_s:np.ndarray, layer_W:np.ndarray):
        """Computes the W'*s (blame) associated with the final output layer.
        The logical flow of this function is based on the assumption that the final output layer is a one-hot encoding,
        meaning we expect layer_s to be a column vector, not a matrix.

        we also expect that blame_out takes the shape of the layer's input, for which we map the layer_W width dimension
        to in duplicate for each row in layer_W.



        NOTE: We leave it to the caller to reset blame_out to all zeros before passing it to this function kernel.

        :param blame_out:
        :type blame_out:
        :param layer_s:
        :type layer_s:
        :param layer_W:
        :type layer_W:
        :return:
        :rtype:
        """
        jinit, iinit = cuda.grid(2)  # gets the current thread's position. here it's the i'th row and j'th column
        jstep, istep = cuda.gridsize(2)
        tstart = jinit*istep+iinit
        tstep = jstep*istep
        h, w = layer_W.shape
        bf,bh,bw = blame_out.shape
        bhw = bw*bh
        for j in range(tstart, w, tstep):
            tmp = 0.
            for i in range(h):
                tmp = cuda.fma(layer_W[i, j], layer_s[i, 0], tmp)
            bj = j % bw
            bi = (j // bw) % bh
            f = (j // bhw) % bf
            blame_out[f,bi,bj] = tmp

    ret = {"batched":{"fsense":batched_backprop_sense_final_kernel,
                      "sense":batched_backprop_sense_vector_kernel,
                      "blame":batched_backprop_blame_vector_kernel,
                      },
           "inline":{"fsense":backprop_sense_final_kernel,
                     "sense":backprop_sense_vector_kernel,
                     "blame":backprop_blame_vector_kernel,
                     }}
    return ret


############################################
## sense and blame kernels for conv layers
############################################
def backprop_conv_wrapper():

    def batched_backprop_conv_sense_kernel(blame_in: np.ndarray, layer_s: np.ndarray, layer_net_sparse: np.ndarray,
                                   pool_stride):
        """Computes the sensitivity for the current layer to the error rate in the network's output.

        This kernel should be launched for layers that performed a convolutional 2d forward.

        :param blame_in: h:w:Fmult
        :type blame_in:

        :param layer_s: h:w:Fmult layer_s will be computed as a sparse 3D array, that contains filter-wise
                        sensitivities mapped to the i,j cooordinates of the first 2 dimensions, and per filter on the
                        third dimension.
        :type layer_s:

        :param layer_net_sparse: h:w:Fmult
                                  This is the upsample mapping for how to distribute sensitivities through the
                                  layer's output.
        :type layer_net_sparse:

        :param layer_W: K1:K2:Fmult
        :type layer_W:
        :return:
        :rtype:
        """
        start_j, start_i = cuda.grid(2)
        step_j, step_i = cuda.gridsize(2)
        batch,sparse_f, h, w = layer_net_sparse.shape
        istart = start_i * pool_stride
        jstart = start_j * pool_stride
        h_ = h - pool_stride
        w_ = w - pool_stride
        istep = step_i * pool_stride
        jstep = step_j * pool_stride
        for b in range(batch):
            for f in range(sparse_f):
                for i in range(istart, h_, istep):
                    bi = i // pool_stride  # implicitly repeats the elements of blame_in as needed
                    for j in range(jstart, w_, jstep):
                        bj = j // pool_stride  # implicitly repeats the elements of blame_in as needed
                        # compute derivative on output at i,j for each non-zero filter and multiply those values,
                        # element-wise, with blame_in
                        nsparse = layer_net_sparse[b, f, i, j]
                        tmp = cuda_device_delu(nsparse)
                        tmp *= blame_in[b, f, bi, bj]
                        layer_s[b, f, i, j] = tmp

    def batched_backprop_conv_blame_kernel(blame_out: np.ndarray, layer_s: np.ndarray, layer_W: np.ndarray, pool_stride):
        """ computes the W'*s (the blame) component needed by the next layer up the chain.

        The caller is responsible for ensuring the blame_out is initialized to the same size and dtype as the
        the layer's given inputs. This implicitly relies on the fact that we are handling the upsampling process
        in the backprop_conv_sense_kernel function.

        we assume that the caller has zeroed all elements of blame_out before calling this function.

        :param blame_out:
        :type blame_out:
        :param layer_s:
        :type layer_s:
        :param layer_W:
        :type layer_W:
        :return:
        :rtype:
        """
        start_j, start_i = cuda.grid(2)
        step_j, step_i = cuda.gridsize(2)
        batch, sf, sh, sw = layer_s.shape
        wf, wh, ww = layer_W.shape
        bf, bh, bw = blame_out.shape[1:]
        wh_half = wh // 2
        ww_half = ww // 2
        # for i in range(start_i+wh_half, sh-wh_half, step_i):
        for b in range(batch):
            for bi in range(start_i, bh, step_i):
                i = bi // pool_stride + wh_half
                # for j in range(start_j+ww_half, sw-ww_half, step_j):
                for bj in range(start_j, bw, step_j):
                    j = bj // pool_stride + ww_half
                    # perform convolution at each filter layer of the layer_W and layer_s
                    for m in range(wh):
                        wi = i + m - wh_half
                        if 0 <= wi < sh:
                            for n in range(ww):
                                wj = j + n - ww_half
                                if 0 <= wj < sw:
                                    for f in range(sf):
                                        _f = f // bf
                                        s = layer_s[b, f, wi, wj]
                                        W = layer_W[_f, -m, -n]
                                        tmp = s * W
                                        if tmp:
                                            for p1 in range(pool_stride):
                                                p1 += bi
                                                if p1 < bh:
                                                    for p2 in range(pool_stride):
                                                        p2 += bj
                                                        if p2 < bw:
                                                            blame_out[b, _f, p1, p2] = tmp

    def backprop_conv_sense_kernel(blame_in:np.ndarray, layer_s:np.ndarray, layer_net_sparse:np.ndarray, pool_stride):
        """Computes the sensitivity for the current layer to the error rate in the network's output.

        This kernel should be launched for layers that performed a convolutional 2d forward.

        :param blame_in: h:w:Fmult
        :type blame_in:

        :param layer_s: h:w:Fmult layer_s will be computed as a sparse 3D array, that contains filter-wise
                        sensitivities mapped to the i,j cooordinates of the first 2 dimensions, and per filter on the
                        third dimension.
        :type layer_s:

        :param layer_net_sparse: h:w:Fmult
                                  This is the upsample mapping for how to distribute sensitivities through the
                                  layer's output.
        :type layer_net_sparse:

        :param layer_W: K1:K2:Fmult
        :type layer_W:
        :return:
        :rtype:
        """
        start_j,start_i = cuda.grid(2)
        step_j,step_i = cuda.gridsize(2)
        sparse_f,h,w = layer_net_sparse.shape
        for f in range(sparse_f):
            for i in range(start_i*pool_stride,h-pool_stride,step_i*pool_stride):
                bi = i//pool_stride # implicitly repeats the elements of blame_in as needed
                for j in range(start_j*pool_stride,w-pool_stride,step_j*pool_stride):
                    bj = j//pool_stride # implicitly repeats the elements of blame_in as needed
                    # compute derivative on output at i,j for each non-zero filter and multiply those values,
                    # element-wise, with blame_in
                    nsparse = layer_net_sparse[f, i, j]
                    if nsparse:
                        tmp = cuda_device_delu(nsparse)
                        tmp *= blame_in[f,bi,bj]
                        layer_s[f,i,j] = tmp
                    else:
                        # multiplying by 0 requires fewer operations than assigning 0 :^\
                        layer_s[f,i,j] *= 0.


    def backprop_conv_blame_kernel(blame_out:np.ndarray, layer_s:np.ndarray, layer_W:np.ndarray, pool_stride):
        """ computes the W'*s (the blame) component needed by the next layer up the chain.

        The caller is responsible for ensuring the blame_out is initialized to the same size and dtype as the
        the layer's given inputs. This implicitly relies on the fact that we are handling the upsampling process
        in the backprop_conv_sense_kernel function.

        we assume that the caller has zeroed all elements of blame_out before calling this function.

        :param blame_out:
        :type blame_out:
        :param layer_s:
        :type layer_s:
        :param layer_W:
        :type layer_W:
        :return:
        :rtype:
        """
        start_j, start_i = cuda.grid(2)
        step_j, step_i = cuda.gridsize(2)
        sf, sh, sw = layer_s.shape
        wf, wh, ww = layer_W.shape
        bf, bh, bw = blame_out.shape
        wh_half = wh//2
        ww_half = ww//2
        # for i in range(start_i+wh_half, sh-wh_half, step_i):
        for bi in range(start_i, bh, step_i):
            i = bi//pool_stride+wh_half
            # for j in range(start_j+ww_half, sw-ww_half, step_j):
            for bj in range(start_j, bw, step_j):
                j = bj//pool_stride+ww_half
                # perform convolution at each filter layer of the layer_W and layer_s
                for m in range(wh):
                    wi = i+m-wh_half
                    if 0<=wi<sh:
                        for n in range(ww):
                            wj = j+n-ww_half
                            if 0<=wj<sw:
                                for f in range(sf):
                                    _f = f//bf
                                    s = layer_s[f, wi, wj]
                                    W = layer_W[_f, -m, -n]
                                    tmp = s*W
                                    if tmp:
                                        for p1 in range(pool_stride):
                                            p1+=bi
                                            if p1<bh:
                                                for p2 in range(pool_stride):
                                                    p2 += bj
                                                    if p2<bw:
                                                        blame_out[_f,p1,p2] = tmp

    ret = {"batched":{"sense":batched_backprop_conv_sense_kernel,
                      "blame":batched_backprop_conv_blame_kernel,
                      },
           "inline":{"sense":backprop_conv_sense_kernel,
                     "blame":backprop_conv_blame_kernel,
                     }}
    return ret


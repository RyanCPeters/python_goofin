from src.fashion_code.kernels.cuda_kernels import *

########################################################################################################################
## update kernels
########################################################################################################################
def update_bias_conv_kernel(gamma:float,lr:float,
                            layer_s:np.ndarray,layer_b:np.ndarray):
    bf = layer_b.shape[0]
    f_len,h,w = layer_s.shape
    wh = h*w
    s_total = f_len*wh
    jinit,iinit = cuda.grid(2)
    jstep,istep = cuda.gridsize(2)
    lr = -lr
    # jblck = cuda.blockDim()
    tstart = jinit*istep+iinit
    tstep = jstep*istep
    fscale = f_len//bf
    for v in range(tstart,s_total,tstep):
        j = v%w
        i = (v//w)%h
        sf = (v//wh)%f_len
        s = layer_s[sf, i, j]
        if s!=0:
            s *= lr
            cuda.atomic.add(layer_b,sf//fscale,s)

def update_weights_conv_kernel(gamma:float,lr:float,
                               layer_input:np.ndarray, layer_s:np.ndarray, layer_W:np.ndarray):
    """Updates the layer's weights array (W) loosly conforming to the following expression:
        W_new = W_old - alpha * s * input.T

    :param gamma: argument placeholder, not used in this function
    :type gamma:
    :param lr:
    :type lr:
    :param layer_input:
    :type layer_input:
    :param layer_s:
    :type layer_s:
    :param layer_W:
    :type layer_W:
    :return:
    :rtype:
    """
    jinit, iinit = cuda.grid(2)
    jstep, istep = cuda.gridsize(2)
    f_in,ih,iw = layer_input.shape
    sf,sh,sw = layer_s.shape
    fmult,kh,kw = layer_W.shape
    half_kh = kh//2
    half_kw = kw//2
    lr = -lr
    tih = ih-half_kh
    tiw = iw-half_kw
    _half_kh = 1-half_kh
    _half_kw = 1-half_kw
    tiinit = iinit+half_kh
    tjinit = jinit+half_kw
    for f1 in range(f_in):
        _f2 = f1 * fmult
        for fm in range(fmult):
            f2 =  _f2 + fm
            for k1 in range(kh):
                _k1 = k1 + _half_kh
                for k2 in range(kw):
                    _k2 = k2 + _half_kw
                    tmp = 0.
                    for i in range(tiinit,tih,istep):
                        _si = i + _k1
                        if 0<=_si<sh:
                            for j in range(tjinit,tiw,jstep):
                                _sj = j + _k2
                                if 0<=_sj<sw:
                                    inpt = layer_input[f1,i, j]
                                    s = layer_s[f2,_si, _sj]
                                    s *= lr
                                    tmp = cuda.fma(inpt,s,tmp) # same result as: tmp += inpt*s
                    if tmp!=0:
                        cuda.atomic.add(layer_W,(fm,k1,k2),tmp)

def momentum_update_bias_conv_kernel(gamma: float, lr: float,
                                     layer_s: np.ndarray, layer_b: np.ndarray):
    j_start,i_start = cuda.grid(2)
    j_step,i_step = cuda.gridsize(2)
    lr = -lr * (1-gamma)
    flat_step = j_step*i_step
    # assign specific threads to perform update to layer_b
    h,w,f_len = layer_s.shape
    bf = layer_b.shape[0]
    for f in range(j_start*i_step+i_start,f_len,flat_step):
        layer_b[f] *= gamma
    cuda.syncthreads()
    # now each thread will compute a partial sum of each filter layer
    f_span = f_len//bf
    for i in range(i_start,h,i_step):
        for j in range(j_start,w,j_step):
            for f1 in range(bf):
                tmp = 0.
                fpos = bf*f_span
                for f2 in range(fpos,fpos+f_span):
                    s = layer_s[i,j,f2]
                    if s!=0:
                        s *= lr
                        tmp += s
                cuda.atomic.add(layer_b,f1,tmp)

def momentum_update_weights_conv_kernel(gamma:float, lr:float,
                                        layer_input:np.ndarray, layer_s:np.ndarray, layer_W:np.ndarray):
    """Momentum weight updates is temporarily broken while we figure out what's wrong with calling cuda.shared.array"""
    # layer_s.shape should equal layer_sparse.shape and the height,width of layer_s/layer_sparse should equal
    # layer_input height-kh, width-kw
    # ToDo: figure out why cuda shared mem arrays are crapping out on launch.
    start_j, start_i = cuda.grid(2)
    j_step, i_step = cuda.gridsize(2)
    ih,iw,f_in = layer_input.shape
    kh,kw,fmult = layer_W.shape
    half_kh = kh//2
    half_kw = kw//2
    lr = -lr*(1-gamma)
    flat_step = i_step*j_step
    total_w = kh*kw*fmult
    # ToDo: This implementation assumes that every filter channel will see an update but we are not assured that this
    #  will always be the case. Especially as we get towards the end of training and changes become very small, and
    #  some layers may not see any updates. This will result in iteratively pushing the weights towards 0... dis bad...
    for v in range(start_j*i_step+start_i,total_w,flat_step):
        f = v%fmult
        k2 = (v//fmult)%kw
        k1 = (v//(kw*fmult))%kh
        layer_W[k1,k2,f] *= gamma
    cuda.syncthreads()
    for f1 in range(f_in):
        for fm in range(fmult):
            f2 = f1 * fmult + fm
            for k1 in range(kh):
                for k2 in range(kw):
                    tmp = 0.
                    for i in range(i_step+half_kh,ih-half_kh,i_step):
                        _i = i - half_kh
                        for j in range(start_j+half_kw,iw-half_kw,j_step):
                            _j = j - half_kw
                            dw = layer_s[_i, _j, f2]
                            if dw!=0:
                                dw *= layer_input[i,j,f1]
                                dw *= lr
                                tmp += dw
                    if tmp!=0:
                        cuda.atomic.add(layer_W,(k1,k2,fm),tmp)

########################
## vector update kernels
########################

def update_bias_vector_kernel(gamma:float,lr:float,
                              layer_s:np.ndarray,layer_b:np.ndarray):
    start_i = cuda.grid(1)
    h = layer_b.shape[0]
    i_step = cuda.gridsize(1)
    lr = -lr
    for i in range(start_i, h, i_step):
        lb = layer_b[i]
        s = layer_s[i, 0]
        lb = cuda.fma(s,lr,lb)
        layer_b[i] = lb


def update_weights_vector_kernel(gamma:float,lr:float,
                                 layer_input:np.ndarray, layer_s:np.ndarray, layer_W:np.ndarray):
    jinit, iinit = cuda.grid(2)
    jstep, istep = cuda.gridsize(2)
    tstep = jstep*istep
    tstart = jinit*istep+iinit
    h,w = layer_W.shape[:2]
    f_in,ih,iw = layer_input.shape
    fw = f_in*iw
    lr = -lr
    # w_hits = 0
    for i in range(h):
        s = layer_s[i,0]
        s *= lr # remember, lr is already negative
        if s!=0:
            for j in range(tstart,w,tstep): # note, w == prod(layer_input.shape)
                # we perform a flattened mapping of the input here
                f = j%f_in
                _j = (j//f_in)%iw
                _i = (j//fw)%ih
                li = layer_input[f,_i,_j]
                if li!=0:
                    # because we made lr negative, the transitive property of
                    # scalr multiplication turns this fused-multiply-add
                    # into fused-multiply-subtraction
                    layer_W[i, j] = cuda.fma(s, li, layer_W[i, j])


def momentum_update_bias_vector_kernel(gamma: GLOBAL_DTYPE, lr: GLOBAL_DTYPE,
                                       layer_s: np.ndarray, layer_b: np.ndarray):
    start_i = cuda.grid(1)
    h = layer_b.shape[0]
    i_step = cuda.gridsize(1)
    lr = -lr * ( 1.- gamma )
    for i in range(start_i, h, i_step):
        lb = layer_b[i]
        lb *= gamma
        s = layer_s[i, 0]
        lb = cuda.fma(s,lr,lb)
        layer_b[i] = lb

def momentum_update_weights_vector_kernel(gamma:GLOBAL_DTYPE, lr:GLOBAL_DTYPE,
                                          layer_input:np.ndarray, layer_s:np.ndarray, layer_W:np.ndarray):
    jinit, iinit = cuda.grid(2)
    jstep, istep = cuda.gridsize(2)
    tstep = jstep * istep
    tstart = jinit * istep + iinit
    h, w = layer_W.shape[:2]
    f_in, ih, iw = layer_input.shape
    fw = f_in * iw
    lr = -lr * (1.-gamma)
    for i in range(h):
        s = layer_s[i,0]
        s *= lr # remember, lr is already negative
        if s!=0:
            for j in range(tstart,w,tstep): # note, w == prod(layer_input.shape)
                # we perform a flattened mapping of the input here
                f = j%f_in
                _j = (j//f_in)%iw
                _i = (j//fw)%ih
                li = layer_input[f,_i,_j]
                if li!=0:
                    # because we made lr negative, the transitive property of
                    # scalr multiplication turns this fused-multiply-add
                    # into fused-multiply-subtraction
                    layer_W[i, j] = cuda.fma(s, li, layer_W[i, j]*gamma)


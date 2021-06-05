"""
By: Ryan Peters; petersryn84@gmail.com
Date: 3/1/21; March 1, 2021
keywords: cuda; gpu; artificial-neural-net; 2d-convolution; mnist-fashion; forward-convolution; backprop-convolution
"""

from numba import cuda
from src.agents import GLOBAL_DTYPE
import numpy as np
from math import exp, sqrt, log
# most of my understanding for how the math of convolutional forward/backprop networks actually works came from:
#   https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/
#   http://ufldl.stanford.edu/tutorial/supervised/ConvolutionalNeuralNetwork/

ELU_ALPHA = GLOBAL_DTYPE(.01)
HUBER_DELTA = GLOBAL_DTYPE(.4)
HUBER_DELTA2 = HUBER_DELTA * HUBER_DELTA
HUBER_DELTA2_INV = 1. / HUBER_DELTA2
HUBER_NEG_HALF_DELTA = GLOBAL_DTYPE(-.5) * HUBER_DELTA
LOG_LOSS_EPSILON = GLOBAL_DTYPE(np.finfo(GLOBAL_DTYPE).eps)


########################################################################################################################
## element-wise device-only transfer functions and their derivatives
########################################################################################################################
@cuda.jit(device=True, inline=True)
def cuda_device_sigmoid(n:GLOBAL_DTYPE):
    """Accepts a single float for which we compute the log-sigmoid output as a float.

    :param n: A GLOBAL_DTYPE, the net output of applying weights and bias to a given input value.
    :type n: GLOBAL_DTYPE
    :return: The computed GLOBAL_DTYPE output of the log-sigmoid function with n as the input
    :rtype: GLOBAL_DTYPE
    """
    o = exp(-n)
    o += 1.
    o = 1./o
    return o


@cuda.jit(device=True,inline=True)
def cuda_device_dsigmoid(o:GLOBAL_DTYPE):
    """
    accepts a single float (the output of the log-sigmoid transfer function), and returns the derivative of the
    log-sigmoid computed at o.
    for computing derivative at o, an output of the log-sigmoid transfer function:
        result = o * (1-o)

    :param o: A GLOBAL_DTYPE, and the computed output of the log-sigmoid function of n.
    :return: GLOBAL_DTYPE
    """
    result = 1.
    result -= o
    result *= o
    return result


@cuda.jit(device=True,inline=True)
def cuda_device_relu(n:GLOBAL_DTYPE):
    return GLOBAL_DTYPE(0<n)*n


@cuda.jit(device=True,inline=True)
def cuda_device_drelu(n:GLOBAL_DTYPE):
    return GLOBAL_DTYPE(0<n)


# ToDo: look into creating a monitor for max values of the layer's net values to see if we are getting exploding values
#   that would benefit from using relu6
@cuda.jit(device=True,inline=True)
def cuda_device_relu6(n:GLOBAL_DTYPE):
    ret = GLOBAL_DTYPE(0<n<=6)
    ret *= n
    ret2 = GLOBAL_DTYPE(n<6)
    ret2 *= 6.
    return ret + ret2

@cuda.jit(device=True,inline=True)
def cuda_device_drelu6(n:GLOBAL_DTYPE):
    return GLOBAL_DTYPE(0<n<=6)

@cuda.jit(device=True,inline=True)
def cuda_device_lrelu6(n:GLOBAL_DTYPE):
    ret1 = -GLOBAL_DTYPE(n<=0)
    ret1 *= ELU_ALPHA

    ret2 = GLOBAL_DTYPE(0. < n <= 6)
    ret2 *= n

    ret3 = GLOBAL_DTYPE(6<n)
    ret3 *= 6.
    return ret1+ret2+ret3

@cuda.jit(device=True,inline=True)
def cuda_device_dlrelu6(n:GLOBAL_DTYPE):
    ret = GLOBAL_DTYPE(0. < n <= 6)
    ret2 = -GLOBAL_DTYPE(n <= 0)
    ret2 *= ELU_ALPHA
    # if n>=6 then the derivative is zero, so we don't need to add that as a computation to the expression
    return  ret + ret2


@cuda.jit(device=True,inline=True)
def cuda_device_elu(n:GLOBAL_DTYPE):
    """Exponential Linear Unit (elu) function:
    if n<=0:
        return alpha*((e^n)-1) # where alpha ~= .1
    else:
        return n
    """
    ret = exp(n)
    ret -= 1.
    ret *= GLOBAL_DTYPE(n<=0)
    ret2 = GLOBAL_DTYPE(0<n)
    ret2 *= n
    return ret + ret2

@cuda.jit(device=True,inline=True)
def cuda_device_delu(n:GLOBAL_DTYPE):
    ret = GLOBAL_DTYPE(n<=0)
    ret *= exp(n)
    ret *= ELU_ALPHA
    ret2 = GLOBAL_DTYPE(0<n)
    return ret + ret2


########################################################################################################################
## utility functions
########################################################################################################################

@cuda.jit(fastmath=True)
def cuda_reset_to_zero_h(arr:np.ndarray):
    strt = cuda.grid(1)
    step = cuda.gridsize(1)
    h = arr.shape[0]
    for i in range(strt,h,step):
        arr[i,0] *= 0.


@cuda.jit(fastmath=True)
def cuda_reset_to_zero_hwd(arr:np.ndarray):
    jinit,iinit = cuda.grid(2)
    jstep,istep = cuda.gridsize(2)
    f_len,h,w = arr.shape
    hw = h*w
    ttote = f_len*hw
    tstart =  jinit*istep+iinit
    tstep = jstep*istep
    for v in range(tstart,ttote,tstep):
        j = v%w
        i = (v//w)%h
        f = (v//hw)%f_len
        arr[f, i, j] *= 0.


@cuda.jit(fastmath=True)
def cuda_batched_reset_to_zero_hw(arr:np.ndarray):
    jinit,iinit = cuda.grid(2)
    jstep,istep = cuda.gridsize(2)
    batch,h,w = arr.shape
    hw = h*w
    ttote = batch*hw
    tstart =  jinit*istep+iinit
    tstep = jstep*istep
    for v in range(tstart,ttote,tstep):
        j = v%w
        i = (v//w)%h
        b = (v//hw)%batch
        arr[b, i, j] *= 0.


@cuda.jit(fastmath=True)
def cuda_batched_reset_to_zero_hwd(arr:np.ndarray):
    jinit,iinit = cuda.grid(2)
    jstep,istep = cuda.gridsize(2)
    batch,f_len,h,w = arr.shape
    hw = h*w
    hwf = hw*f_len
    ttote = batch*hwf
    tstart =  jinit*istep+iinit
    tstep = jstep*istep
    for v in range(tstart,ttote,tstep):
        j = v%w
        i = (v//w)%h
        f = (v//hw)%f_len
        b = (v//hwf)%batch
        arr[b, f, i, j] *= 0.


@cuda.jit(fastmath=True)
def batch_sum_4d(arr: np.ndarray, batch,h,w,f_len):
    jstart, istart = cuda.grid(2)
    jstep, istep = cuda.gridsize(2)
    inv_batch = 1./batch
    for f in range(f_len):
        for i in range(istart,h,istep):
            for j in range(jstart, w, jstep):
                tmp = 0.
                # we are batch-wise accumulating all values
                # into the first element on the batch axis.
                # So we need to iterate over the batch last
                # to keep the f,i,j coordinates straight
                for b in range(1, batch):
                    tmp += arr[b, f, i, j]
                arr[0,f,i,j] += tmp # sums all samples in batch for coordinate f,i,j
                arr[0,f, i, j] *= inv_batch # gives the average for all samples.


@cuda.jit(fastmath=True)
def batch_sum_3d(arr: np.ndarray, batch,h,w):
    jstart, istart = cuda.grid(2)
    jstep, istep = cuda.gridsize(2)
    inv_batch = 1./batch
    for i in range(istart,h,istep):
        for j in range(jstart, w, jstep):
            tmp = 0.
            for b in range(1, batch):
                tmp += arr[b, i, j]
            arr[0,i,j] += tmp
            arr[0, i, j] *= inv_batch

########################################################################################################################
## error functions
########################################################################################################################
@cuda.jit(fastmath=True,inline=True)
def error_MSE_kernel_1d(t:np.ndarray,a:np.ndarray,result:np.ndarray):
    start = cuda.grid(1)
    step = cuda.gridsize(1)
    h = t.shape[0]
    for i in range(start,h,step):
        result[i,0] = t[i,0]-a[i,0]
        result[i,0] *= result[i,0]
        result[i,0] *= .5


@cuda.jit(fastmath=True)
def error_MSE_kernel(t:np.ndarray,a:np.ndarray,result:np.ndarray):
    jstart,istart = cuda.grid(2)
    jstep,istep = cuda.gridsize(2)
    f_len,h,w = t.shape
    for f in range(f_len):
        for i in range(istart,h,istep):
            for j in range(jstart,w,jstep):
                val = t[f,i,j]-a[f,i,j]
                val *= val
                val *= .5 # averaged as the squared error over 2
                result[f,i,j] = val


@cuda.jit(fastmath=True)
def error_simple(t:np.ndarray,a:np.ndarray,result:np.ndarray):
    jstart,istart = cuda.grid(2)
    jstep,istep = cuda.gridsize(2)
    tstart = jstep*istep+istart
    tstep = jstep*istep
    h,w = t.shape
    for i in range(tstart,h,tstep):
        for j in range(w):
            result[i,j] = t[i,j]-a[i,j]


@cuda.jit(fastmath=True)
def batch_error_MSE_kernel(t:np.ndarray,a:np.ndarray,result:np.ndarray):
    jtinit = cuda.threadIdx.x
    itinit = cuda.threadIdx.y
    jtstep = cuda.blockDim.x
    itstep = cuda.blockDim.y
    jbinit = cuda.blockIdx.x
    jgrid,igrid = cuda.gridsize(2)
    jbstep = jgrid//jtstep
    ibstep = igrid//itstep
    ibinit = cuda.blockIdx.y
    bstart = jbinit*ibstep+ibinit
    bstep = jbstep*ibstep
    tstart = jtinit*itstep+itinit
    tstep = jtstep*itstep
    batch,h,w = t.shape
    for b in range(bstart,batch,bstep):
        for i in range(tstart,h,tstep):
            for j in range(w):
                val = t[b,i,j]-a[b,i,j]
                val *= val
                val *= .5 # averaged as the squared error over 2
                result[i,j] = val


@cuda.jit(fastmath=True)
def batch_error_simple(t:np.ndarray,a:np.ndarray,result:np.ndarray):
    jtinit = cuda.threadIdx.x
    itinit = cuda.threadIdx.y
    jtstep = cuda.blockDim.x
    itstep = cuda.blockDim.y
    jbinit = cuda.blockIdx.x
    jgrid, igrid = cuda.gridsize(2)
    jbstep = jgrid // jtstep
    ibstep = igrid // itstep
    ibinit = cuda.blockIdx.y
    bstart = jbinit * ibstep + ibinit
    bstep = jbstep * ibstep
    tstart = jtinit * itstep + itinit
    tstep = jtstep * itstep
    batch, h, w = t.shape
    for b in range(bstart, batch, bstep):
        for i in range(tstart, h, tstep):
            for j in range(w):
                result[b,i,j] = t[b,i,j]-a[b,i,j]


@cuda.jit(fastmath=True)
def batch_error_pseudo_huber(t:np.ndarray, a:np.ndarray, result:np.ndarray):
    """An attempt at implementing the pseudo-huber loss function as described on this wiki page:
        https://en.wikipedia.org/wiki/Huber_loss

    The Pseudo-Huber loss function can be used as a smooth approximation of the Huber loss function.
    It combines the best properties of L2 squared loss and L1 absolute loss by being strongly convex when close to the
    target/minimum and less steep for extreme values. This steepness can be controlled by the
    `delta`  value. The Pseudo-Huber loss function ensures that derivatives are continuous
    for all degrees. It is defined as

    L(a) = (delta^2)*(sqrt(1+(a/d)^2) - 1)


    As such, this function approximates (`a`^2)/2 for small values of `a`, and approximates a straight line with
    slope `delta` for large values of `a`.

    While the above is the most common form, other smooth approximations of the Huber loss function also exist.

    for the time being, delta is being hard-coded as .4

    NOTE: I originally implemented this function for another long past project

    :param t: the ground-truth target
    :type t:

    :param a: the model's output prediction
    :type a:

    :param result: the receptical for the results of the difference
    :type result:

    :return:
    :rtype:
    """
    jtinit = cuda.threadIdx.x
    itinit = cuda.threadIdx.y
    jtstep = cuda.blockDim.x
    itstep = cuda.blockDim.y
    jbinit = cuda.blockIdx.x
    jgrid, igrid = cuda.gridsize(2)
    jbstep = jgrid // jtstep
    ibstep = igrid // itstep
    ibinit = cuda.blockIdx.y
    bstart = jbinit * ibstep + ibinit
    bstep = jbstep * ibstep
    tstart = jtinit * itstep + itinit
    tstep = jtstep * itstep
    batch, h, w = t.shape
    for b in range(bstart, batch, bstep):
        for i in range(tstart, h, tstep):
            for j in range(w):
                val = t[b, i, j] - a[b, i, j]
                # now we perform set up th series of instructions to compute:
                #       result = DELTA2*(sqrt(1+(val**2/DELTA2))-1)
                val *= val
                val *= HUBER_DELTA2_INV # covers (val*val / (DELTA*DELTA))
                val += 1.
                val = sqrt(val)
                val -= 1.
                val *= HUBER_DELTA2 # DELTA2 == DELTA**2
                result[b,i,j] = val


@cuda.jit(fastmath=True)
def batch_error_huber_loss(t:np.ndarray, a:np.ndarray, result:np.ndarray):
    jtinit = cuda.threadIdx.x
    itinit = cuda.threadIdx.y
    jtstep = cuda.blockDim.x
    itstep = cuda.blockDim.y
    jbinit = cuda.blockIdx.x
    jgrid, igrid = cuda.gridsize(2)
    jbstep = jgrid // jtstep
    ibstep = igrid // itstep
    ibinit = cuda.blockIdx.y
    bstart = jbinit * ibstep + ibinit
    bstep = jbstep * ibstep
    tstart = jtinit * itstep + itinit
    tstep = jtstep * itstep
    batch, h, w = t.shape
    for b in range(bstart, batch, bstep):
        for i in range(tstart, h, tstep):
            for j in range(w):
                val = t[b, i, j] - a[b, i, j]
                abs_val = abs(val)
                # now we do a non-processor branching piecewise calculation for:
                #   result = .5*(val*val) if abs_val<=DELTA else DELTA*(abs_val-.5*DELTA)
                # first part of conditional: .5*(val*val)
                v1 = GLOBAL_DTYPE(abs_val <= HUBER_DELTA)
                v1 *= .5
                val *= val
                v1 *= val
                # second part of conditional: DELTA*(abs_val-.5*DELTA)
                v2 = GLOBAL_DTYPE(abs_val > HUBER_DELTA)
                v2 *= abs_val
                v2 += HUBER_NEG_HALF_DELTA
                v2 *= HUBER_DELTA
                result[b, i, j] = v1 + v2


@cuda.jit(device=True,inline=True)
def _weighted_binary_crossentropy(target, prediction, punishment_weight):
    f1 = target * log(prediction + LOG_LOSS_EPSILON)
    f2 = (1. - target) * log(1. + LOG_LOSS_EPSILON - prediction)
    ret = f1+f2
    ret *= punishment_weight
    ret *= .5
    ret -= 1.
    return ret


@cuda.jit(fastmath=True)
def batch_false_pos_log_loss(t:np.ndarray, a:np.ndarray, result:np.ndarray):
    ''' Log loss that weights false positives more.
        Punish the false negatives if you care about making sure all the neurons
        are found and don't mind some false positives. Vice versa for punishing
        the false positives. Concept taken from the UNet paper where they
        use_weights boundary errors to get cleaner boundaries.

        This code was taken from:
        https://gist.github.com/alexklibisz/34d4865c721d3047b8f124195b225ffb

        I originally implemented this code for a long past project.

        :param ytruth:  A tensor who's values represent the base truth for the sample being considered.
        :param ypred:  A tensor who's values represent the model's prediction of the truth for the
                    sample being considered.
        :return:
        the product of computing the log-loss associated with the model's prediction (ypred)  to
        more heavily punish false positive predictions. That is to say, it scales the loss_name value
        higher when the model erroneously predicts a true value for a given sample when it should
        have predicted a false value.
    '''
    # [0,1] -> [-1,0]
    # [-1,0] -> [1,0]
    # [1, 0] -> [m-1, 0]
    # [m-1,0] -> [m,1]
    # w = 1 + ((1 - ytruth) * (num_classes - 1))
    # return _weighted_binary_crossentropy(ytruth, ypred, w)
    jtinit = cuda.threadIdx.x
    itinit = cuda.threadIdx.y
    jtstep = cuda.blockDim.x
    itstep = cuda.blockDim.y
    jbinit = cuda.blockIdx.x
    jgrid, igrid = cuda.gridsize(2)
    jbstep = jgrid // jtstep
    ibstep = igrid // itstep
    ibinit = cuda.blockIdx.y
    bstart = jbinit * ibstep + ibinit
    bstep = jbstep * ibstep
    tstart = jtinit * itstep + itinit
    tstep = jtstep * itstep
    batch, h, w = t.shape
    h_ = h-1.
    for b in range(bstart, batch, bstep):
        for i in range(tstart, h, tstep):
            for j in range(w):
                # result[b, i, j] = t[b, i, j] - a[b, i, j]
                w = 1. + ((1.-t[b, i, j])*h_) # h also represents the number of classes we are classifying against
                result[b, i, j] = _weighted_binary_crossentropy(t[b, i, j], a[b, i, j], w)
    # Below return statement would match the keras binary_crossentropy value.
    # return -1. * K.mean((a + b))


@cuda.jit(fastmath=True)
def batch_false_neg_log_loss(t:np.ndarray, a:np.ndarray, result:np.ndarray):
    ''' Log loss that weights false negatives more.
        Punish the false negatives if you care about making sure all the neurons
        are found and don't mind some false positives. Vice versa for punishing
        the false positives. Concept taken from the UNet paper where they
        use_weights boundary errors to get cleaner boundaries.

        This code was taken from:
        https://gist.github.com/alexklibisz/34d4865c721d3047b8f124195b225ffb

        :param ytruth:  A tensor who's values represent the base truth for the sample being
        considered.
        :param ypred:  A tensor who's values represent the model's prediction of the truth for the
                    sample being considered.
        :return:
        the product of computing the loss_name associated with the model's prediction (ypred)
        use_weights to more heavily punish false negative predictions. That is to say, it scales the
        loss_name value higher when the model erroneously predicts a false value for a given sample when
        it should have predicted a true value.
        '''
    # [0,1] -> [0,m].
    # w = ytruth * num_classes
    # return _weighted_binary_crossentropy(ytruth, ypred, w)
    # Below return statement would match the keras binary_crossentropy value.
    # return -1. * K.mean((a + b))
    jtinit = cuda.threadIdx.x
    itinit = cuda.threadIdx.y
    jtstep = cuda.blockDim.x
    itstep = cuda.blockDim.y
    jbinit = cuda.blockIdx.x
    jgrid, igrid = cuda.gridsize(2)
    jbstep = jgrid // jtstep
    ibstep = igrid // itstep
    ibinit = cuda.blockIdx.y
    bstart = jbinit * ibstep + ibinit
    bstep = jbstep * ibstep
    tstart = jtinit * itstep + itinit
    tstep = jtstep * itstep
    batch, h, w = t.shape
    for b in range(bstart, batch, bstep):
        for i in range(tstart, h, tstep):
            for j in range(w):
                # result[b, i, j] = t[b, i, j] - a[b, i, j]
                w = t[b, i, j]*h # h also represents the number of classes we are classifying against
                result[b, i, j] = _weighted_binary_crossentropy(t[b, i, j], a[b, i, j], w)

# @cuda.jit(fastmath=True)
# def batch_dice_loss(t:np.ndarray,a:np.ndarray,result:np.ndarray):
#     jtinit = cuda.threadIdx.x
#     itinit = cuda.threadIdx.y
#     jtstep = cuda.blockDim.x
#     itstep = cuda.blockDim.y
#     jbinit = cuda.blockIdx.x
#     jgrid, igrid = cuda.gridsize(2)
#     jbstep = jgrid // jtstep
#     ibstep = igrid // itstep
#     ibinit = cuda.blockIdx.y
#     bstart = jbinit * ibstep + ibinit
#     bstep = jbstep * ibstep
#     tstart = jtinit * itstep + itinit
#     tstep = jtstep * itstep
#     batch, h, w = t.shape
#     for b in range(bstart, batch, bstep):
#         for i in range(tstart, h, tstep):
#             for j in range(w):
#                 val = GLOBAL_DTYPE(t[b,i,j]==a[b,i,j])
#
#                 result[b,i,j] = 0.

BATCH_ERROR_MAP = {
    "MSE":batch_error_MSE_kernel,
    "SIMPLE":batch_error_simple,
    "PSEUDO_HUBER":batch_error_pseudo_huber,
    "HUBER":batch_error_huber_loss,
    "FALSE_POS_LOG":batch_false_pos_log_loss,
    "FALSE_NEG_LOG":batch_false_neg_log_loss,
}
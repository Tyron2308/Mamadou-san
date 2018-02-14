from cs231n.fast_layers import conv_forward_fast, conv_backward_fast
from time import time
import numpy as np
from img2col import *

def conv_forward_naive(x, w, b, conv_param):

    out = None
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    S = conv_param['stride']
    P = conv_param['pad']

    # Add padding to each image
    x_pad = np.pad(x, ((0,), (0,), (P,), (P,)), 'constant')
    # Size of the output
    Hh = 1 + (H + 2 * P - HH) / S
    Hw = 1 + (W + 2 * P - WW) / S

    out = np.zeros((N, F, Hh, Hw))

    for n in range(N):  # First, iterate over all the images
        for f in range(F):  # Second, iterate over all the kernels
            for k in range(Hh):
                for l in range(Hw):
                    out[n, f, k, l] = np.sum(
                        x_pad[n, :, k * S:k * S + HH, l * S:l * S + WW] * w[f, :]) + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


def eval_numerical_gradient_array(f, x, df, h=1e-5):
    """
    Evaluate a numeric gradient for a function that accepts a numpy
    array and returns a numpy array.
    """
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        ix = it.multi_index

        oldval = x[ix]
        x[ix] = oldval + h
        pos = f(x).copy()
        x[ix] = oldval - h
        neg = f(x).copy()
        x[ix] = oldval

        grad[ix] = np.sum((pos - neg) * df) / (2 * h)
        it.iternext()
    return grad


if __name__ == "__main__":

    x = np.random.randn(100, 1, 31, 31)
    w = np.random.randn(25, 1, 3, 3)
    b = np.random.randn(25,)
    conv_param = {'stride': 2, 'pad': 1}

    t0 = time()
    out_naive, cache_naive = conv_forward_naive(x, w, b, conv_param)
    t1 = time()

    x = np.random.randn(100, 31, 31)
    w = np.random.randn(25, 3, 3)
    res = []
    for i in range(x):
        for k in range(w):
            out_fast = convolution_layer(i, k)
            res.append(out_fast)

    t2 = time()

    print 'Testing conv_forward_fast:'
    print 'Naive: %fs' % (t1 - t0)
    print 'Fast: %fs' % (t2 - t1)
    print 'Speedup: %fx' % ((t1 - t0) / (t2 - t1))
    print 'Difference: ', rel_error(out_naive, out_fast)

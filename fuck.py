from scipy.misc import imread, imresize
import numpy as np
import os
import skimage.io as io
from scipy import ndimage
import matplotlib.pyplot as plt
import time


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


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.
    Inputs:
    - dout: Upstream derivatives of shape (N, F, H', W')
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive
    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################

    # Extract shapes and constants
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    # Construct output
    dx_pad = np.zeros_like(x_pad)
    dx = np.zeros_like(x)
    dw = np.zeros_like(w)
    db = np.zeros_like(b)
    # Naive Loops
    for n in range(N):
        for f in range(F):
            db[f] += dout[n, f].sum()
            for j in range(0, H_prime):
                for i in range(0, W_prime):
                    dw[f] += x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] * dout[n, f, j, i]
                    dx_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f] * dout[n, f, j, i]
    # Extract dx from dx_pad
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]

    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    print ('io===', i0.shape)

    i0 = np.tile(i0, C)
    print ('io===', i0.shape)

    i1 = stride * np.repeat(np.arange(out_height), out_width)

    print ('io===', i1.shape)

    j0 = np.tile(np.arange(field_width), field_height * C)

    print ('io===', j0.shape)

    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    print('io===', j.shape)
    print('io===', i.shape, k.shape)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)
    cols = x_padded[:, k, i, j]

    print('cols===', cols.shape)
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0)
    print('ok===', cols.shape) #.reshape(field_height * field_width * C, -1)
    return cols.reshape(field_height * field_width * C, -1)


def run_img(arr, w, b, conv_param, img_size, to_benchmark):
    start = time.time()

    dir = os.listdir(arr[0])
    x = np.zeros((len(dir), 3, img_size, img_size))
    for idx, img in enumerate(dir):
        im = io.imread(arr[0] + "/" + img)
        x[idx, :, :, :] = imresize(im, (img_size, img_size)).transpose((2, 0, 1))

    print('xx====', x.shape)
    out, cache = to_benchmark(x, w, b, conv_param)

    end = time.time()
    print((end - start) // 60)
    return x, out, cache


def conv_forward_naive2(x, w, b, conv_param):
    print("VECTORIZE")
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    print('shapeeee=====', x.shape, w.shape)
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    # Check for parameter sanity
    assert (H + 2 * pad - HH) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'

    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   'constant', constant_values=0)
    # Construct output
    X_col = im2col_indices(x, w.shape[2], w.shape[3],
                           conv_param['stride'], conv_param['pad'])


    print('xcol==', X_col.shape)
    W_col = w.reshape(F, -1)

    print('wcol', W_col.shape)
    out = X_col.T @ W_col.T
    out = out.reshape(2, H_prime, W_prime, N)
    out = out.transpose(3, 0, 1, 2)
    kernel = np.flipud(np.fliplr(kernel))    # Flip the kernel
    output = np.zeros_like(image)            # convolution output
    # Add zero padding to the input image
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image
    for x in range(image.shape[1]):     # Loop over every pixel of the image
       for y in range(image.shape[0]):
           # element-wise multiplication of the kernel and the image
          output[y,x]=(kernel*image_padded[y:y+3,x:x+3]).sum()
    cache = (x, w, b, conv_param)
    return out, cache


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.
    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.
    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.
    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """

    N, C, H, W = x.shape
    F, _, HH, WW = w.shape

    print('shapeeee=====', x.shape, w.shape)
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    assert (H + 2 * pad - HH) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    out = np.zeros((N, F, H_prime, W_prime))

    for n in range(N):
        for f in range(F):
            for j in range(0, H_prime):
                for i in range(0, W_prime):
                    out[n, f, j, i] = (x_pad[n, :, j*stride:j*stride+HH,
                                       i*stride:i*stride+WW] * w[f, :, :, :]).sum() + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    print('img===', img.shape)
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        print('shape', img_max.shape, img_min.shape)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')


if __name__ == "__main__":
    print('new test')
    arr = np.array(["data/train/beauty-personal_care-hygiene",
                    "data/train/clothing",
                    "data/train/communications",
                    "data/train/footwear",
                    "data/train/household-furniture",
                    "data/train/kitchen_merchandise",
                    "data/train/personal_accessories",
                    "data/train/sports_equipment",
                    "data/train/toys-games"])

    w = np.zeros((2, 3, 3, 3))

    # The first filter converts the image to grayscale.
    # Set up the red, green, and blue channels of the filter.
    w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
    w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]

    # Second filter detects horizontal edges in the blue channel.
    w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]

    # Vector of biases. We don't need any bias for the grayscale
    #    filter, but for the edge detection filter we want to add 128
    # to each output so that nothing is negative.
    b = np.array([0, 128])

    from skimage import io, color
    import matplotlib.pyplot as plt
    import numpy as np
    from skimage import exposure
    import pylab

    img = io.imread('image.png')    # Load the image
    img = color.rgb2gray(img)       # Convert the image to grayscale (1 channel)
    # Adjust the contrast of the image by applying Histogram Equalization
    image_equalized = exposure.equalize_adapthist(img/np.max(np.abs(img)), clip_limit=0.03)
    plt.imshow(image_equalized, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    x2, lool2, c = run_img(arr, w, b, {'stride': 1, 'pad': 1},
                           400, conv_forward_naive2)

    print('ok', x2.shape, lool2.shape)
    plt.subplot(2, 3, 1)

    b = np.random.randn(2,)
    dout = np.random.randn(4, 2, 5, 5)
    conv_param = {'stride': 1, 'pad': 1}

    #dx_num = eval_numerical_gradient_array(lambda x: conv_forward_naive(x2, w, b, conv_param)[0], x, dout)
    #dw_num = eval_numerical_gradient_array(lambda w: conv_forward_naive(x2, w, b, conv_param)[0], w, dout)
    #db_num = eval_numerical_gradient_array(lambda b: conv_forward_naive(x2, w, b, conv_param)[0], b, dout)

   # out, cache = conv_forward_naive(x2, w, b, conv_param)
   # dx, dw, db = conv_backward_naive(dout, cache)

# Your errors should be around 1e-9'
    #print ('Testing conv_backward_naive function', dx.shape, dw.shape, db.shape)
   # print ('dx error: ', rel_error(dx, dx_num))
   # print ('dw error: ', rel_error(dw, dw_num))
   # print ('db error: ', rel_error(db, db_num))
    plt.subplot(2, 3, 2)
    imshow_noax(lool2[0, 1])
    plt.subplot(2, 3, 3)
    imshow_noax(lool2[0, 0])
    plt.show()

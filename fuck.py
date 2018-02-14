from scipy.misc import imread, imresize
import numpy as np
import os
import skimage.io as io
from scipy import ndimage
import matplotlib.pyplot as plt
import time




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
    N, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    print ('io===', i0.shape)

    i0 = np.tile(i0, 1)
    print ('io===', i0.shape)

    i1 = stride * np.repeat(np.arange(out_height), out_width)

    print ('io===', i1.shape)

    j0 = np.tile(np.arange(field_width), field_height * 1)

    print ('io===', j0.shape)

    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(1), field_height * field_width).reshape(-1, 1)
    print('io===', j.shape)
    print('io===', i.shape, k.shape)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
#    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    print(x.shape, i.shape, j.shape, k.shape)
    cols = x[:, j, i]

    print('cols===', cols.shape)
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0)
    print('ok===', cols.shape) #.reshape(field_height * field_width * C, -1)
    return cols.reshape(field_height * field_width * C, -1)


def conv_forward_naive2(x, w, b, conv_param):
    print("VECTORIZE", x.shape)
    N, H, W = x.shape
    m, HH, WW = w.shape

    print('shapeeee=====', x.shape, w.shape)
    stride = conv_param.get('stride', 1)
    pad = conv_param.get('pad', 0)
    # Check for parameter sanity
    assert (H + 2 * pad - HH) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'

    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride

    kernel = np.flipud(np.fliplr(w))
    kernel2 = np.flipud(np.fliplr(w[1])) # Flip the kernel
    output = np.zeros_like(x)            # convolution output
    # Add zero padding to the input image
    print('output ====', output.shape)
    #image_padded = np.zeros((x.shape[0], x.shape[1] + 2, x.shape[2] + 2))

    #print('padded ===', image_padded.shape)
   # image_padded[:, 1:-1, 1:-1] = x

    W_col = kernel.reshape(2, 9)

    print("wcol=", W_col.shape)

    X_col = x.reshape(2, x.shape[1] * x.shape[2] * x.shape[0])

    print("lol==", X_col.shape)
    print("lol==", W_col.shape)
    test = W_col.T @ X_col
    print("test==", test.shape)
    out = test.reshape(9, x.shape[1], x.shape[2], N)
    out = out.transpose(3, 0, 1, 2)


    print('test==', test.shape, out.shape)
    #
    # for idx, img in enumerate(x):
    #     for l in range(x.shape[1]):     # Loop over every pixel of the image
    #         for y in range(400):
    #             output[idx, y, l] =\
    #                 (kernel*image_padded[idx, y:y+3, l:l+3]).sum()

    cache = (x, w, b, conv_param)

    print('out==', output.shape)
    return output, cache, out


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

    w = np.zeros((2, 3, 3))
    w = np.zeros((2, 3, 3, 3))

    # The first filter converts the image to grayscale.
    # Set up the red, green, and blue channels of the filter.
    w[0, 0, :, :] = [[0, 0, 0], [0, 0.3, 0], [0, 0, 0]]
    w[0, 1, :, :] = [[0, 0, 0], [0, 0.6, 0], [0, 0, 0]]
    w[0, 2, :, :] = [[0, 0, 0], [0, 0.1, 0], [0, 0, 0]]
    w[1, 2, :, :] = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
    b = np.array([0, 128])
    w2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    w3 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    t = np.array((w2, w3))
    print('w3===', w3.shape)
    from skimage import color
    from skimage import exposure

    #x = run_img(arr, 400)
    conv_param = {'stride': 1, 'pad': 1}

    x2, l, x3 = conv_forward_naive2(x, t, b, conv_param)

#    edge = exposure.equalize_adapthist(x2[0]/np.max(np.abs(x2)),clip_limit=0.03)

    #plt.imshow(x2[0], cmap=plt.cm.gray)
    plt.imshow(x3[0, 1], cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

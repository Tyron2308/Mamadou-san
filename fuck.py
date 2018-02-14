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
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   'constant', constant_values=0)
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
                    dw[f] += x_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW]\
                             * dout[n, f, j, i]
                    dx_pad[n, :, j * stride:j * stride + HH, i * stride:i * stride + WW] += w[f]\
                            * dout[n, f, j, i]
    # Extract dx from dx_pad
    dx = dx_pad[:, :, pad:pad+H, pad:pad+W]
    return dx, dw, db


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


#    edge = exposure.equalize_adapthist(x2[0]/np.max(np.abs(x2)),clip_limit=0.03)


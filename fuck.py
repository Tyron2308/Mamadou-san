from scipy.misc import imread, imresize
import numpy as np
import os
import skimage.io as io
from scipy import ndimage
import matplotlib.pyplot as plt
import time


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
    print ('io===', j.shape)
    print ('io===', i.shape, k.shape)

    return (k, i, j)


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

    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)),
                   'constant', constant_values=0)
    # Construct output
    t = im2col_indices(x, w.shape[2], w.shape[3], conv_param['stride'], conv_param['pad'])

    print('???', t.shape)
    W = w.reshape((w.shape[0], (w.shape[1] * w.shape[2] * w.shape[3])))
    X = np.resize(x, (27, 8320000))

    print(X.shape)

    ret = np.matmul(X.T, W.T) + b
    print('first test ===', W.shape, t.shape, ret.shape)
    s = ret.T.reshape(w.shape[0], x.shape[0], x.shape[2], x.shape[3])
    #s = ret.T.reshape(w.shape[0], x.shape[2], x.shape[3], x.shape[0])
   # s = s.transpose(3, 0, 1, 2)
    print('first test ===', s.shape)
    cache = (x, w, b, conv_param)
    return s, cache


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
    # Check for parameter sanity
    assert (H + 2 * pad - HH) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Height'
    assert (W + 2 * pad - WW) % stride == 0, 'Sanity Check Status: Conv Layer Failed in Width'
    H_prime = 1 + (H + 2 * pad - HH) // stride
    W_prime = 1 + (W + 2 * pad - WW) // stride
    # Padding
    x_pad = np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), 'constant', constant_values=0)
    # Construct output
    out = np.zeros((N, F, H_prime, W_prime))


    # Naive Loops
    for n in range(N):
        for f in range(F):
            for j in range(0, H_prime):
                for i in range(0, W_prime):
                    out[n, f, j, i] = (x_pad[n, :, j*stride:j*stride+HH,
                                       i*stride:i*stride+WW] * w[f, :, :, :]).sum() + b[f]

    cache = (x, w, b, conv_param)
    return out, cache


def conv_layer(input_map, w, b, conv):
    bank_filter = np.random.randn(2, 3, 3, 3).astype(np.float64)
    matrix_filter = np.matmul(bank_filter[0], bank_filter[1]) + b
    print('shape====>', matrix_filter.shape)
    mat = [matrix_filter]
    s = ndimage.convolve(input_map, mat, mode='constant', cval=1.0)

    print('conv shappp === ', conv.shape)
    cache = (input_map, w, b, conv)
    return s, cache


# kitten is wide, and puppy is already square
#d = first.shape[1] -.shape[0]
#kitten_cropped = kitten[:, d/2:-d/2, :]

def imshow_noax(img, normalize=True):
    """ Tiny helper to show images as uint8 and remove axis labels """
    print('img===', img.shape)
    if normalize:
        img_max, img_min = np.max(img), np.min(img)
        print('shape', img_max.shape, img_min.shape)
        img = 255.0 * (img - img_min) / (img_max - img_min)
    plt.imshow(img.astype('uint8'))
    plt.gca().axis('off')

# Show the original images and the results of the conv operation


if __name__=="__main__":
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

    #x, lool, c = run_img(arr, w, b, {'stride': 1, 'pad': 1}, 400, conv_forward_naive)
   # x2, lool18, c = run_img(arr, w, b, {'stride': 1, 'pad': 1}, 400, conv_forward_naive)
    x2, lool2, c = run_img(arr, w, b, {'stride': 1, 'pad': 1}, 400, conv_forward_naive2)

    #print('ok', x.shape, lool.shape)
    plt.subplot(2, 3, 1)
    #imshow_noax(lool2[1, 0], normalize=False)

    #
    # imshow_noax(lool2[0, 1])
    # plt.subplot(2, 3, 2)
    #
    # imshow_noax(lool2[0, 2])
    #
    # plt.subplot(2, 3, 3)
    # imshow_noax(lool2[0, 3])
    #
    # plt.subplot(2, 3, 4)
    #
    # imshow_noax(lool2[0, 4])
    #
    # plt.subplot(2, 3, 5)
    #
    # imshow_noax(lool2[0, 7])
    #
    # plt.subplot(2, 3, 6)
    #
    # imshow_noax(lool2[0, 6])
    #
    # plt.subplot(2, 4, 6)
    #
    # imshow_noax(lool2[0, 5])

    imshow_noax(lool2[1, 1])
    plt.subplot(2, 3, 2)

    imshow_noax(lool2[1, 2])

    plt.subplot(2, 3, 3)
    imshow_noax(lool2[1, 3])

    plt.subplot(2, 3, 4)

    imshow_noax(lool2[1, 4])

    plt.subplot(2, 3, 5)

    imshow_noax(lool2[1, 7])

    plt.subplot(2, 3, 6)

    imshow_noax(lool2[1, 6])

    plt.subplot(2, 4, 6)

    imshow_noax(lool2[1, 5])


#for idx, l in enumerate(lool2):
       # plt.subplot(2, 3, 2)
      #  plt.title('Grayscale')
    #
    # plt.subplot(2, 3, 3)
    # imshow_noax(out[0, 0])
    # plt.title('Grayscale 2 ')

  #  plt.subplot(2, 3, 4)
 #   imshow_noax(lool18[0, 1])
#    plt.title('Edge')

    # plt.subplot(2, 3, 5)
    # imshow_noax(out[0, 1])
    # plt.title('Edge2')

    plt.show()

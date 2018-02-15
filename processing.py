from scipy import linalg
import numpy as np
import os
import skimage.io as io
from skimage import color
from scipy.misc import imresize
from multiprocessing.pool import ThreadPool


def run_img(arr, img_size):

    dir = os.listdir(arr[0])
    x = np.zeros((len(dir), img_size, img_size))
    for idx, img in enumerate(dir):
        im = io.imread(arr[0] + "/" + img)
        img = color.rgb2gray(im)
        x[idx, :, :] = imresize(img, (img_size, img_size))
    return x


def reelu_activation(x):
    return np.maximum(0, x)


def naive_max_pooling(input_image, stride, windows):
    H, W = input_image.shape
    H1 = int(H - windows / stride + 1)
    W1 = int(W - windows / stride + 1)

    print('size h` et w ===', H1, W1)
    out = np.zeros((H1, W1))
    for i in range(H1):
        for j in range(W1):
            out[i, j] = np.max(input_image[i*stride:i*stride+windows,
                               j*stride:j*stride+windows])
    return out


def convolution_layer(image, kernel):
    padding = np.zeros((image.shape[0] - 9), kernel.dtype)
    padding2 = np.zeros((image.shape[0] - 1), kernel.dtype)

    first_col = np.r_[kernel.flatten(), padding]
    first_row = np.r_[kernel[0][0], padding2]
    output = linalg.toeplitz(first_col, first_row)
    y = np.dot(image, output)
    return y


if __name__ == "__main__":

    pool = ThreadPool(5)
    arr = np.array(["data/train/beauty-personal_care-hygiene",
                    "data/train/clothing",
                    "data/train/communications",
                    "data/train/footwear",
                    "data/train/household-furniture",
                    "data/train/kitchen_merchandise",
                    "data/train/personal_accessories",
                    "data/train/sports_equipment",
                    "data/train/toys-games"])
    t = run_img(arr, 400)
    w2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    import matplotlib.pyplot as plt

    p = os.listdir(arr[0])
    img = io.imread(arr[0] + "/" + p[10])
    plt.imshow(img, cmap=plt.cm.gray)
    x = t[0]

    output = [] * len(t)
    curr_thread = [] * (len(t) + 1)
    for idx, val in enumerate(t):
        curr_thread.append(pool.apply_async(convolution_layer, (val, w2)))

    #TODO: gerer thread safe
    for idx, thread in enumerate(curr_thread):
        output.append(thread.get())

    r  = reelu_activation(output[1])
    test = naive_max_pooling(output[0], 1, 32)

    print('shape apres pooling --', test.shape)
    plt.imshow(output[10], cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    #print ("y(mult):", y)



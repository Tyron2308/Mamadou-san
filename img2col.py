from scipy import linalg
from scipy import signal
import numpy as np
import os
import skimage.io as io
from skimage import color
from scipy.misc import imread, imresize
from scipy import ndimage
from multiprocessing.pool import ThreadPool
import threading


def run_img(arr, img_size):

    dir = os.listdir(arr[0])
    x = np.zeros((len(dir), img_size, img_size))
    for idx, img in enumerate(dir):
        im = io.imread(arr[0] + "/" + img)
        img = color.rgb2gray(im)
        x[idx, :, :] = imresize(img, (img_size, img_size))
    return x


def convolution_layer(image, kernel):
    padding = np.zeros((400 - 9), w2.dtype)
    padding2 = np.zeros((400 - 1), w2.dtype)

    first_col = np.r_[kernel.flatten(), padding]
    first_row = np.r_[kernel[0][0], padding2]
    output = linalg.toeplitz(first_col, first_row)

    print(output)
    y = np.dot(image, output)
    return y


def f(x):
    return x*x


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

    plt.imshow(output[10], cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    #print ("y(mult):", y)



from scipy import linalg
from scipy import signal
import numpy as np
import os
import skimage.io as io
from skimage import color
from scipy.misc import imread, imresize
from scipy import ndimage


def run_img(arr, img_size):

    dir = os.listdir(arr[0])
    x = np.zeros((len(dir), img_size, img_size))
    for idx, img in enumerate(dir):
        im = io.imread(arr[0] + "/" + img)
        img = color.rgb2gray(im)
        x[idx, :, :] = imresize(img, (img_size, img_size))
    return x


if __name__ == "__main__":

    

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

    x = t[0]

    #h = np.zeros(((x.shape[0]-w2.shape[0] + 1)**2, x.shape[0])
    #print(w2.shape)
    padding = np.zeros((400 - 9), w2.dtype)
    padding2 = np.zeros((400 - 1), w2.dtype)


    print(padding.shape)
    first_col = np.r_[w2.flatten(), padding]
    first_row = np.r_[w2[0][0], padding2]
    p = linalg.toeplitz(first_col, first_row)



    print('HH===', p , p.shape)


    #x = x.reshape([x.shape[0], 1, x.shape[1]])
    #x = np.tile(x, [1, x.shape[1], 1])
    # # x = x.flatten()

    y = np.dot(x, p)
    # print('x====', p, p.shape, )
    #
    # p.reshape((400, 400, 3, 3))
    #
    #
    #
    import matplotlib.pyplot as plt
    plt.imshow(y, cmap=plt.cm.gray)
    plt.axis('off')
    plt.show()

    #print ("y(mult):", y)



from scipy import linalg
import numpy as np
import os
import skimage.io as io
from skimage import color
from scipy.misc import imresize
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot as plt


def run_img(img_size, array=np.array(["data/train/beauty-personal_care-hygiene",
                                      "data/train/clothing",
                                      "data/train/communications",
                                      "data/train/footwear",
                                           "data/train/household-furniture",
                                           "data/train/kitchen_merchandise",
                                           "data/train/personal_accessories",
                                           "data/train/sports_equipment",
                                           "data/train/toys-games"])):

    out = np.zeros((800, img_size, img_size))
    for id, a in enumerate(array):
        dir = os.listdir(array[id])
        for idx, img in enumerate(dir):
           im = io.imread(array[id] + "/" + img)
           img = color.rgb2gray(im)
           out[idx, :, :] = imresize(img, (img_size, img_size))
    return out


def naive_max_pooling(input_image, stride=1, windows=32):

    #input_image = input_image[0:2]
    print("img shape", input_image.shape)
    x_num, height, width = input_image.shape
    print("naive max pooling ===> i", input_image.shape, windows, height, stride)

    H1 = int(height - windows / stride + 1)
    W1 = int(width - windows / stride + 1)

    print('size h` et w ===', H1, W1, input_image[0])
    out = np.zeros((x_num, H1, W1))
    for n in range(x_num):
        print("image numero", n)
        for i in range(0, H1, stride):
            for j in range(0, W1, stride):
                out[n, i, j] = np.max(input_image[n, i*stride:i * stride + windows,
                                      j * stride:j * stride + windows])

    return out


def convolution_layer(image, kernel):

    N, H, W = image.shape
    out = np.zeros((N, H, W))

    padding = np.zeros((image.shape[1] - 9), kernel.dtype)
    padding2 = np.zeros((image.shape[1] - 1), kernel.dtype)
    first_col = np.r_[kernel.flatten(), padding]
    first_row = np.r_[kernel[0][0], padding2]
    output = linalg.toeplitz(first_col, first_row)

    for i in range(N):
        out[i, :, :] = np.dot(image[i], output)
    print("shape out ==", out.shape)
    return out


def use_pool(pool, func_to_use, helper, inpute,  params):
    curr_thread = []
    start = 0
    print("conv layer")
    for val in range(5):
        curr_thread.append(pool.apply_async(func_to_use,
                              (inpute[start:start+int(inpute.shape[0]/5)],
                              helper[params])))
        start += int(inpute.shape[0] / 5)
        start = 0
    output = np.zeros((inpute.shape[0]+200, 400, 400))
    for thread in curr_thread:
        arr_concat = thread.get()
        print("arrr ", arr_concat.shape)
        for count, img in enumerate(arr_concat):
            output[start + count, :, :] = img
            #print("start", start, start + 160, count,  arr_concat.shape)
        start += int(inpute.shape[0]/5)
    return output


if __name__ == "__main__":

    p = ThreadPool(5)
    arr = np.array(["data/train/beauty-personal_care-hygiene",
                    "data/train/clothing",
                    "data/train/communications",
                    "data/train/footwear",
                    "data/train/household-furniture",
                    "data/train/kitchen_merchandise",
                    "data/train/personal_accessories",
                    "data/train/sports_equipment",
                    "data/train/toys-games"])
    t = run_img(400)
    w2 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

    dictio = {}
    dictio["w2"] = w2
    dictio["stride"] = 1
    dictio["windows"] = 32

    print("size depart =>", t.shape)
    out = use_pool(p, convolution_layer, dictio, t, "w2")

    print("ok")
    test = out[1:3]

    print("test shape", test.shape)
    res = naive_max_pooling(test, 1, 32)

    print("res ===> ", res.shape)

    out = use_pool(p, convolution_layer, dictio, res, "w2")
    print("out shape cnv 2 ==> ", out.shape)
    plt.axis('off')
    plt.show()

    #print ("y(mult):", y)



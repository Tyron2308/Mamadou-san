import numpy
from scipy import ndimage
import pickle
import os
import traceback
import sys
import skimage.io as io
import matplotlib.pyplot as plt
from skimage import color
from scipy.misc import imresize


class OnehotEncoder:

    def __init__(self, num_label):
        self.num_label = num_label

    def to_onehot(self, vector):
        stock = [] * (len(vector) + 1)
        stock_img = [] * (len(vector) + 1)
        for label_path in vector:
            for img in label_path:
                z = numpy.zeros(self.num_label)
                z[int(img[0])] = 1
                stock.append(z)
                stock_img.append(img[1])
        return numpy.array(stock), numpy.array(stock_img)


class Opti:
    @staticmethod
    def reelu_activation(input_map):
        z = numpy.zeros_like(input_map, dtype=numpy.float64)
        r = numpy.where(input_map > z, input_map, z)
        return r.reshape((z.shape[1], z.shape[0], z.shape[2]))

    @staticmethod
    def reelu_activation2(input_map):
        z = numpy.zeros_like(input_map, dtype=numpy.float64)
        return numpy.where(input_map > z, input_map, z)

    @staticmethod
    def sigmoid(x):
        slope = 0.2
        shift = 0.5
        x = (x * slope) + shift
        x = numpy.clip(x, 0, 1)
        return x


class Metric:

    @staticmethod
    def rmse(prediction, label):
        return numpy.sqrt(numpy.sum((prediction - label) ** 2))


class Helper:

    @staticmethod
    def rgb2gray(rgb):
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray

    def transform_map(self, array_path, dictionary_categorical):
        if array_path.size > 0:
            for key in dictionary_categorical:
                if not dictionary_categorical[key]:
                    dictionary_categorical[key] = array_path[0]
                    break
            return self.transform_map(array_path[1:len(array_path)],
                                      dictionary_categorical)
        return dictionary_categorical

    def vectorize_features(self, path_arr, dictionary, count, vector_feature):
        if path_arr.size > 0:
            try:
                arr_txt = os.listdir(path_arr[0])
                vector_input = [] * (len(arr_txt) + 1)
                g = lambda x, data: x.get(data)
                for idx, img in enumerate(arr_txt):
                    img = io.imread(path_arr[0] + "/" + img)
                    im = color.rgb2gray(img)
                    vector_input.append((g(dictionary, path_arr[0]), im))
                vector_feature.append(vector_input)
                return self.vectorize_features(path_arr[1:], dictionary, count + 1,
                                               vector_feature)
            except OSError:
                tb = sys.exc_info()[-1]
                stk = traceback.extract_tb(tb, 1)
                name = stk[0][2]
                print('The failing function was', tb, name)
        else:
            return numpy.array(vector_feature)

    @staticmethod
    def reverse_dictionnary(dictionary):
        for k, v in dictionary.items():
            yield v, k

    def __init__(self, array_path):
        valv = dict.fromkeys((range(len(array_path))))
        dictionary = self.transform_map(array_path, valv)
        reverse = self.reverse_dictionnary(dictionary)
        self.vector_input = self.vectorize_features(array_path, dict(reverse),
                                                    0, [] * (len(array_path) + 1))
        print(self.vector_input.shape)


class LightCNN:

    def __init__(self, h, w, c):
        self.to_cache = {}
        self.bank_filter = numpy.random.randn(5, 3, 3).astype(numpy.float64)
        self.weight = []
        self.bias = []
        self.stride = 1
        self.neural = LightNN(h * w * c, 2)
        self.tmp = [] * 2

    def assert_size(self, h_size):
        assert((h_size - self.bank_filter.shape[1]+2*1)/self.stride) + 1, 'output shape is incorrect'

    def pool_layer(self, input_map, check):
        if check:
            window = 2
            print('input===', input_map.shape)
            n_h = int(1 + (input_map.shape[0] - window) / self.stride)
            n_w = int(1 + (input_map.shape[1] - window) / self.stride)
            pooled_features = numpy.zeros_like((input_map.shape[0], n_h, n_w))
            print('pooled_', pooled_features.shape, n_h, n_w)
            for i in range(len(input_map[0])):
                for h in range(n_h):                # loop on the vertical axis of the output volume
                    for w in range(n_w):           # loop on the horizontal axis of the output volume
                        #for c in range (3):        # loop over the channels of the output volume
                        vert_start = h * 1
                        vert_end = vert_start + 2
                        horiz_start = w * 1
                        horiz_end = horiz_start + 2
                        a_prev_slice = input_map[vert_start:vert_end, horiz_start:horiz_end]
                        pooled_features[i, h, w] = numpy.max(a_prev_slice)
        return pooled_features

    def conv_layer(self, input_map, activation_function):
        matrix_filter = numpy.matmul(self.bank_filter[0], self.bank_filter[1])
        conv = ndimage.convolve(input_map, matrix_filter, mode='constant', cval=1.0)
        return activation_function(conv)

    def dense_layer(self, input_map, label, activation_function, learning_rate, lamba):
        print('dense')
        forward_res = self.neural.forward_propagation(input_map, activation_function, self.tmp)
        self.tmp.append(forward_res)
        output, cost_vec = self.neural.cost_gradient(forward_res, label, learning_rate, lamba)
        return output, cost_vec

    @staticmethod
    def softmax(input_map):
        return input_map.argmax(axis=-1)

    @staticmethod
    def drop_out(input_map, seed):
        input_map = input_map * (1. - seed)
        return input_map


if __name__ == '__main__':

    arr = numpy.array(["data/train/beauty-personal_care-hygiene",
                     "data/train/clothing",
                     "data/train/communications",
                     "data/train/footwear",
                     "data/train/household-furniture",
                     "data/train/kitchen_merchandise",
                     "data/train/personal_accessories",
                     "data/train/sports_equipment",
                     "data/train/toys-games"])
    helper = Helper(arr)
    label, img = OnehotEncoder(len(arr)).to_onehot(helper.vector_input)

    print(len(label))

import numpy
from scipy import ndimage
import pickle
import os
import traceback
import sys
import skimage.io as io
import matplotlib.pyplot as plt


class Opti:
    @staticmethod
    def reelu_activation(input_map):
        z = numpy.zeros_like(input_map, dtype=numpy.float64)
        r = numpy.where(input_map > z, input_map, z)
        return r.reshape((z.shape[1], z.shape[0], z.shape[2]))

    @staticmethod
    def sigmoid(x):
        #return 1/1 + numpy.exp(-input_map)
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
        if array_path and array_path[0]:
            for key in dictionary_categorical:
                if not dictionary_categorical[key]:
                    dictionary_categorical[key] = array_path[0]
                    break
            return self.transform_map(array_path[1:len(array_path)], dictionary_categorical)
        return dictionary_categorical

    def vectorize_features(self, path_arr, dictionary, count, vector_feature):
        if count < len(path_arr) and path_arr[count]:
            try:
                vector_input = list()
                arr_txt = os.listdir(path_arr[count])
                for img in arr_txt:
                    img = numpy.array(io.imread(path_arr[count] + "/" + img), dtype=float)
                    g = lambda x, data: x.get(data)
                    vector_input.append((g(dict(dictionary), path_arr[count]), img))
                vector_feature.append(vector_input)
                return self.vectorize_features(path_arr[1:len(path_arr)], dictionary, count + 1, vector_feature)
            except OSError:
                tb = sys.exc_info()[-1]
                stk = traceback.extract_tb(tb, 1)
                name = stk[0][2]
                print('The failing function was', name)
        else:
            return vector_feature

    @staticmethod
    def reverse_dictionnary(dictionary):
        for k,v in dictionary.items():
            yield v, k

    def __init__(self, array_path):
        valv = dict.fromkeys((range(len(array_path))))
        dictionary = self.transform_map(array_path, valv)
        reverse = self.reverse_dictionnary(dictionary)
        self.vector_input = self.vectorize_features(array_path, reverse, 0, list())


class Mamadou:

    def __init__(self):
        self.mamadou = LightCNN(5, 5, 3)

    @staticmethod
    def load_weight(filename="weight.pkl"):
        try:
            file = open(filename, 'wb')
            weight = pickle.load(file)
        except OSError:
            tb = sys.exc_info()[-1]
            stk = traceback.extract_tb(tb, 1)
            name = stk[0][2]
            print('The failing function was', name)
            raise RuntimeError
        return weight

    @staticmethod
    def save_weight(filename):
        try:
            pickle.dump(filename)
        except OSError:
            tb = sys.exc_info()[-1]
            stk = traceback.extract_tb(tb, 1)
            name = stk[0][2]
            print('The failing function was', name)
            raise RuntimeError

    def train(self, train_test, iter_max, pass_to_execute, label):
        idx = 0
        output = numpy.array(train_test, dtype=numpy.float64)
        while idx < iter_max:
            for count, img in enumerate(train_test):
                print('image number', count)
                output[count], cost = numpy.reshape(pass_to_execute(img, label[count]),
                                              img.shape)
            idx = idx + 1
        
        plt.plot(cost)
        plt.ylabel('test')
        plt.show()

    def predict(self, input_map):
        weight_tmp = self.mamadou.neural.weight
        reshaped = numpy.reshape(weight_tmp, (2, 5, 5, 3))
        a1 = numpy.dot(numpy.transpose(reshaped[0]), input_map)
        activated = Opti.sigmoid(a1)
        a1 = numpy.dot(numpy.transpose(reshaped[1]), activated)
        probability = Opti.sigmoid(a1)
        v = self.mamadou.softmax(probability)
        return numpy.max(v)

    def forward_pass(self, input_map, labels):
        output = self.mamadou.conv_layer(input_map, Opti.reelu_activation)
        output = self.mamadou.conv_layer(output, Opti.reelu_activation)
        output = self.mamadou.pool_layer(output, True)
        output = self.mamadou.drop_out(output, True)
        output, cost = self.mamadou.dense_layer(output, labels, Opti.sigmoid, 0.01, 0.01)
        return output, cost

    ###TODO: miss implem
    def backward_pass(self, input_map, lab):
        print('ok')
        return input_map


class LightNN:

    def __init__(self, sze_input, number_hidden):
        self.weight = numpy.random.random_sample((number_hidden, sze_input))

        print(self.weight.shape, self.weight)
        return

    @staticmethod
    def replace_zeroes(data):
        min_nonzero = numpy.zeros_like(data)
        data = numpy.where(data > 0, data, min_nonzero)
        data[data == 0] = 0.0000000001
        return data

    @staticmethod
    def numpy_minmax(x):
        xmin = x.min(axis=0)
        return (x - xmin) / (x.max(axis=0) - xmin)

    def cost_gradient(self, input_map, label, learning_rate, lamba):
        vec = []
        input_map = self.replace_zeroes(input_map)
        flat = self.flatten_layer(label)
        reg = (lamba / len(input_map)) * sum(self.weight ** 2)
        loss = input_map - flat
        cost = numpy.sum(loss ** 2) / (2 * len(input_map))
        cost2 = numpy.sum(numpy.multiply(-flat, numpy.log(input_map))
                          - numpy.multiply((1 - flat), numpy.log(1 - input_map)))
        vec.append((cost, cost2))
        #print (cost, cost2)
        grad = (numpy.multiply(numpy.transpose(input_map), loss) / len(input_map))

        print('COST ====> ', cost, cost2)
        print(grad.shape, )
        self.weight -= learning_rate * grad
        return input_map, vec

    def back_propagation(self, input_map, label, lamba, reverse):
        print ("backpropagation", self.weight)
        input_map = numpy.reshape(input_map, label.shape)
        delta_scrip = label - input_map
        index = 2
        delta_curr = delta_scrip
        i = 0
        weight_grad = numpy.zeros_like(self.weight)
        delta_erra = []
        while index > 1:
            v = numpy.reshape(reverse[index - 1], self.weight[0].shape)
            c = numpy.reshape(delta_curr, self.weight[0].shape)
            t = numpy.dot(self.weight[index - 1], c)
            e = numpy.dot(t, v)
            print('z score activation====', reverse[index - 1])
            print('weight shape weight ==>', self.weight, self.weight[index - 1])
            print('t', e.shape, self.weight[index - 1].shape, c.shape, v.shape)
            delta_erra.append(e)
            print('reverse shape', len(reverse[index - 1]))
            print('len delta===', delta_erra[i].shape)
            weight_grad[index - 1] = weight_grad[index - 1] + delta_erra[i] * reverse[index - 1]
            i += 1
            weight_grad[index - 1] /= len(input_map)
            weight_grad[index - 1] += lamba / len(input_map) * self.weight[index - 1]
            index -= 1
        self.weight = weight_grad
        return self.weight

    @staticmethod
    def flatten_layer(X):
        return numpy.reshape(X, (numpy.prod(X.shape[:])))

    def forward_propagation(self, input_, activation_function, tmp, bias):
        print('forward propagation weight shape and value ', self.weight.shape, self.weight)
        weight_tmp = self.weight + bias                                                                              
        a1 = numpy.dot(numpy.transpose(weight_tmp[0]), input_)
        activated = activation_function(a1)
        tmp.append(activated)
        a1 = numpy.dot(numpy.transpose(weight_tmp[1]), activated)
        return activation_function(a1)


class LightCNN:

    def __init__(self, h, w, c):
        self.to_cache = {}
        self.bank_filter = numpy.random.randn(5, 3, 3, 3).astype(numpy.float64)
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

            n_h = int(1 + (input_map.shape[1] - window) / self.stride)
            n_w = int(1 + (input_map.shape[2] - window) / self.stride)
            pooled_features = numpy.zeros((len(input_map[0]), n_h, n_w, 3))
            for i in range(len(input_map[0])):
                for h in range(n_h):                # loop on the vertical axis of the output volume
                    for w in range(n_w):           # loop on the horizontal axis of the output volume
                        for c in range (3):        # loop over the channels of the output volume
                            vert_start = h * 1
                            vert_end = vert_start + 2
                            horiz_start = w * 1
                            horiz_end = horiz_start + 2
                            a_prev_slice = input_map[vert_start:vert_end, horiz_start:horiz_end, c]
                            pooled_features[i, h, w, c] = numpy.max(a_prev_slice)
        return input_map

    def conv_layer(self, input_map, activation_function):
        matrix_filter = numpy.matmul(self.bank_filter[0], self.bank_filter[1])
        conv = ndimage.convolve(input_map, matrix_filter, mode='constant', cval=1.0)
        return activation_function(conv)

    def dense_layer(self, input_map, label, activation_function, learning_rate, lamba):
        flat = self.neural.flatten_layer(input_map)
        forward_res = self.neural.forward_propagation(flat, activation_function, self.tmp,
                                                      numpy.ones((1, self.neural.weight.shape[1])))
        self.tmp.append(forward_res)
        output, cost_vec = self.neural.cost_gradient(forward_res, label, learning_rate, lamba)

        self.neural.back_propagation(output, label, lamba, self.tmp)
        return output, cost_vec

    @staticmethod
    def softmax(input_map):
        return input_map.argmax(axis=-1)

    @staticmethod
    def drop_out(input_map, seed):
        input_map = input_map * (1. - seed)
        return input_map


if __name__ == '__main__':
    Mamadou = Mamadou()

    helper = Helper(["data/train/beauty-personal_care-hygiene", "data/train/clothing"
                     "data/train/communications", "data/train/footwear", "data/train/household-furniture",
                     "data/train/kitchen_merchandise", "data/train/personal_accessories", "data/train/sports_equipment",
                     "data/train/toys-game"])

    labels = helper.vector_input
    print(len(helper.vector_input[1]), len(helper.vector_input[2]))
    Mamadou.train(numpy.random.randn(10, 5, 5, 3).astype(numpy.float64), 400,
                  Mamadou.forward_pass, numpy.random.randn(10, 5, 5, 3).astype(numpy.float64))

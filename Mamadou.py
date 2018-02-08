import numpy
from scipy import ndimage
import pickle
import os
import traceback
import sys
import skimage.io as io
import matplotlib.pyplot as plt


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
                for img in arr_txt:
                    img = numpy.array(io.imread(path_arr[0] + "/" + img),
                                      dtype=numpy.float64)
                    vector_input.append((g(dictionary, path_arr[0]), img))
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
        print('LABEL ', label.shape)
        output = []
        c = []
        while idx < iter_max:
            for count, img in enumerate(train_test):
                print('image number', count)
                test = numpy.matrix(Helper.rgb2gray(img))
                print('test===', test, 'imgg====', img)
                print(test.shape)
                output1, cost = pass_to_execute(test, label[count])
                output[count] = numpy.reshape(output1, img.shape)
                c.append(cost)
            idx = idx + 1
    #    c1 = self.mamadou.neural.replace_zeroes(numpy.array(c))
    #    c2 = numpy.reshape(c1, (2, 25000))

        print('weight===', self.mamadou.neural.weight)
        #print('cost====', c2, c2.shape)
        #plt.plot(c2)
        #plt.ylabel(train_test[0])
        #plt.show()

    def predict(self, input_map):
        output = self.mamadou.neural.replace_zeroes(input_map)
        weight_tmp = self.mamadou.neural.weight
        reshaped = numpy.reshape(weight_tmp, (2, 5, 5, 3))
        a1 = numpy.dot(numpy.transpose(reshaped[0]), output)
        output = self.mamadou.neural.replace_zeroes(a1)
        activated = Opti.reelu_activation2(output)
        a1 = numpy.dot(numpy.transpose(reshaped[1]), activated)
        output = self.mamadou.neural.replace_zeroes(a1)
        probability = Opti.reelu_activation2(output)
        v = self.mamadou.softmax(probability)

        print('response softmax', v)
        return numpy.max(v)

    def forward_pass(self, input_map, labels):
        output = self.mamadou.conv_layer(input_map, Opti.reelu_activation2)
        output = self.mamadou.conv_layer(output, Opti.reelu_activation2)
        output = self.mamadou.pool_layer(output, True)
        output, cost = self.mamadou.dense_layer(output, labels, Opti.sigmoid, 0.01, 0.01)
        return output, cost

    ###TODO: miss implem
    def backward_pass(self, input_map, lab):
        print('ok')
        return input_map


class LightNN:

    def __init__(self, sze_input, number_hidden):
        params = (numpy.random.random(number_hidden * (sze_input + 1) + 10 * (number_hidden +1)))

        print(params.shape)
        self.weight = numpy.random.random_sample((number_hidden, sze_input))

        self.theta1 = numpy.random.rand(5, 26)
        self.theta2 = numpy.random.rand(9, 26)


        return

    @staticmethod
    def replace_zeroes(data):
        min_nonzero = numpy.zeros_like(data)
        where_nan = numpy.isnan(data)
        data[where_nan] = 0
        data = numpy.where(data > 0, data, min_nonzero)
        data[data == 0] = 0.0000000001
        max_number = numpy.ones(data.shape)
        data = numpy.where(data>1, data, max_number)
        return data

    @staticmethod
    def numpy_minmax(x):
        xmin = x.min(axis=0)
        return (x - xmin) / (x.max(axis=0) - xmin)

    def cost_gradient(self, input_map, label, learning_rate, lamba):
        vec = [] * 400
        reg = (lamba / len(input_map)) * sum(self.weight ** 2)
        loss = [] * 100
        grad = []

        print ('LEN', len(label), label.shape)
        for isx, img in enumerate(input_map):
            loss.append(img - label)
            cost = numpy.sum(numpy.array(loss) ** 2) / 2 / len(input_map)
            print('labe===l ', label.shape, input_map.shape, numpy.array(loss).shape, input_map)
            cost2 = numpy.sum(numpy.multiply(-label[isx], numpy.log(img))
                          - numpy.multiply((1 - label[isx]), numpy.log(1 - img)))
            vec.append(cost * 1000)
            print('COST====', cost, 'COST2====', cost2)
            grad.append(numpy.sum(numpy.dot(input_map, numpy.array(loss))) / len(input_map) + reg)
            print('grad shape', numpy.array(grad).shape)
        self.weight -= learning_rate * numpy.reshape(numpy.array(grad), input_map.shape)
        return input_map, vec

    def pute(self, i, label, lamba, learning):
        import scipy.optimize as opt
        result = opt.fmin_tnc(func=self.cost_gradient, x0=self.weight,
                              fprime=self.cost_gradient,
                              args=(i, label, lamba))
        print('RESULT===>', result[0])

    def back_propagation(self, input_map, label, lamba, reverse):
        #print ("backpropagation", self.weight)
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
            #print('z score activation====', reverse[index - 1])
            #print('weight shape weight ==>', self.weight, self.weight[index - 1])
            #print('t', e.shape, self.weight[index - 1].shape, c.shape, v.shape)
            delta_erra.append(e)
            #print('reverse shape', len(reverse[index - 1]))
            #print('len delta===', delta_erra[i].shape)
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

    def forward_propagation(self, input_, activation_function, tmp):
        print('fwd')
        m = input_.shape[0]

        a1prime = numpy.dot(self.theta1.T, input_)


        print('ok===',self.theta1.shape, self.theta2.shape, input_.shape, a1prime.shape)
        activatedprime = activation_function(a1prime)
        tmp.append(activatedprime)

        print(self.theta2)
       # test = numpy.insert(self.theta2, 0, 1, axis=1)

      #  print(test.shape)
        a1prime = numpy.dot(self.theta2, activatedprime)
        t1 = activation_function(a1prime)
        print(t1.shape)
        return t1


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
    mamadou = Mamadou()

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
    mamadou.train(img, 500, mamadou.forward_pass, label)

import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from processing import *
from Mamadou import *


class NeuralNet:
    def __init__(self, hidden_size, input_size, num_label=9):
        self.params = {}
        parameter = (np.random.random(size=hidden_size * (input_size + 1) +
                                      num_label * (hidden_size + 1)) - 0.5) * 0.25
        print("hidden size : ", hidden_size, "input_size ",input_size)
        theta1 = 0.25 * np.random.rand(hidden_size, (input_size*input_size + 1))
        theta2 = 0.25 * np.random.rand(hidden_size + 1, num_label)

        print("theta", theta2.shape, theta1.shape)
        self.params['W1'] = theta1
        self.params['W2'] = theta2


def reelu_activation(x):
    return np.maximum(0.000000001, x)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(x, theta1, theta2):
    x = np.matrix(x)
    print("xsh", x.shape, "theta2", theta1.shape, "thetha3=", theta2.shape)
    a1 = np.insert(x, 0, np.ones(1), axis=1)

    print("x===", x.shape, a1.shape, a1.flatten().shape, theta2.shape, theta1.shape)
    z2 = np.dot(a1, theta1.T)

    print("multiplication edge * theta1", z2.shape)
    hidden_layer = reelu_activation(z2)
    print("hidden activer", hidden_layer.shape)

    biased = np.insert(hidden_layer, 0, np.ones(1), axis=0)

    print("biases ===", biased.shape)
    last_layer = biased * theta2.T
    print("last activer", last_layer.shape)

    out = reelu_activation(last_layer)

    print("out ===", out.shape)
    return hidden_layer, last_layer, out


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def backprop(params, X, y, learning_rate, reg):
    m = X.shape[0]
    #X = np.matrix(X)
    #y = np.matrix(y)

    for k in range(X.shape[0]):
        hidden_l, last_l, output = forward_propagate(X[k], params['W1'], params['W2'])

        print("hidden_layer === ", hidden_l.shape, "last_layer", last_l.shape)

        j = 0

        first_term = np.multiply(-y[k, :], np.log(output[k, :]))
        second_term = np.multiply((1 - y[k, :]), np.log(1 - output[k, :]))
        j += np.sum(first_term - second_term) / m

        j += (float(learning_rate) / (2 * m)) *\
            (np.sum(np.power(params['W1'], 2)) + np.sum(np.power(params['W2'], 2)))

        error_sortie = output - y

        out_delta = sigmoid_gradient(output.T) * error_sortie
        print("out delta = ", out_delta.shape)

        err_layer_hidden = out_delta.dot(params["W2"].T)
        print("err layer hidden ", err_layer_hidden.shape)

        p = np.insert(hidden_l, 0, np.ones(1), axis=1)
        print(p.shape)
        hid_delta = np.dot(err_layer_hidden, sigmoid_gradient(p.T))

        print("shape====", params["W2"].shape, params["W1"].shape,
          out_delta.shape, hid_delta.shape,
          hidden_l.shape , last_l.shape)

        params['W2'] += np.multiply(learning_rate,  np.dot(hidden_l, out_delta).T)
        params['W1'] += np.multiply(learning_rate, np.dot(X.T, hid_delta.T))

    return j, params


def train(x, y_onehot, learning_rate=1e-3,
          learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200):
    num_train = x.shape[0]
    hidden_size = 25
    num_labels = 10

    mamadou = NeuralNet(hidden_size, x.shape[1],  num_labels)
    iterations_per_epoch = int(max(num_train / batch_size, 1))
    loss_history = []

    for iteration in range(iterations_per_epoch):
        for it in range(num_iters):
            sample_indices = np.random.choice(np.arange(num_train), batch_size)
            x_batch = x[sample_indices]
            y_batch = y_onehot[sample_indices]
            loss, grad = backprop(mamadou.params, x_batch, y_batch,
                                  learning_rate, reg)
            loss_history.append(loss)

        learning_rate *= learning_rate_decay
        print("iteration epoch", iteration, "/", iterations_per_epoch, "/n")

    return loss_history, mamadou


def predict(x, params):
    #x_ = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    a1 = np.dot(x, params["W1"].T) + params["b1"]
    a2 = np.maximum(a1, 0)
    score = np.dot(a2, params["W2"])
    return np.argmax(score, axis=1)


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
    #X = run_img(400)

    label, img = OnehotEncoder(len(arr)).to_onehot(helper.vector_input)

    print(img.shape, label.shape)

    losses, neural = train(img, label)

    y_pred = predict(img.ravel(), neural.params)
    correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]
    valid = correct[correct == 1]
    false = correct[correct == 0]

    print("valid and false ==", valid, false, correct)
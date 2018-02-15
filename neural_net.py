import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


class NeuralNet:
    def __init__(self, hidden_size, input_size, num_label=9):
        self.params = {}
        parameter = (np.random.random(size=hidden_size * (input_size + 1) +
                                      num_label * (hidden_size + 1)) - 0.5) * 0.25
        theta1 = np.matrix(np.reshape(parameter[:hidden_size * (input_size + 1)],
                                      (hidden_size, (input_size + 1))))
        theta2 = np.matrix(np.reshape(parameter[hidden_size * (input_size + 1):],
                                      (num_label, (hidden_size + 1))))
        self.params['W1'] = theta1
        self.params['W2'] = theta2
        self.params['b1'] = np.ones(hidden_size)
        self.params['b2'] = np.ones(num_label)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(x, theta1, theta2, b, b2):
    a1 = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, np.ones(x.shape[0]), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)
    return a1, z2, a2, z3, h


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def backprop(params, X, y, learning_rate, reg):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    a1, z2, a2, z3, h = forward_propagate(X, params['W1'], params['W2'],
                                             params['b1'], params['b2'])
    j = 0

    # for i in range(m):
    #     first_term = np.multiply(-y[i, :], np.log(h[i, :]))
    #     second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
    #     j += np.sum(first_term - second_term)
    # j = j / m

    j = -(y*np.log(h) - (1-y)*np.log(1-h)**2).mean()

    j += (float(learning_rate) / (2 * m)) *\
         (np.sum(np.power(params['W1'][:, 1:], 2)) +
          np.sum(np.power(params['W2'][:, 1:], 2))) * reg

    grads = {}
    dscore = h - y
    grads['W2'] = np.dot(a2.T, dscore)
    grads['b2'] = np.sum(dscore, axis=0)
    hid = np.dot(dscore, params["W2"])

    hid[hid <= 0] = 0
    grads['W1'] = np.dot(X.T, hid)
    grads['b1'] = np.sum(hid, axis=0)

    grads['W2'] = reg * params["W1"]
    grads['W1'] = reg * params["W2"]
    return j, grads


def train(x, y_onehot, learning_rate=1e-3,
          learning_rate_decay=0.95, reg=1e-5, num_iters=100, batch_size=200):
    print('train function ')
    num_train = x.shape[0]
    hidden_size = 25
    num_labels = 10

    mamadou = NeuralNet(hidden_size, x.shape[1],  num_labels)
    print("num_train", num_train)
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
        mamadou.params["W1"] += -learning_rate * mamadou.params["W1"]
        mamadou.params["W2"] += -learning_rate * mamadou.params["W2"]
        mamadou.params["b1"] += -learning_rate * mamadou.params["b1"]
        mamadou.params["b2"] += -learning_rate * mamadou.params["b2"]
        learning_rate *= learning_rate_decay
        print("iteration epoch", iteration, "/", iterations_per_epoch, "/n")

    return loss_history, mamadou


def predict(x, params):
    x_ = np.insert(x, 0, np.ones(x.shape[0]), axis=1)
    a1 = np.dot(x_, params["W1"].T) + params["b1"]
    a2 = np.insert(np.maximum(a1, 0), 0, np.ones(x.shape[0]), axis=1)
    score = np.dot(a2, params["W2"].T)
    return np.argmax(score, axis=1)


if __name__ == '__main__':
    import time
    start = time.time()
    data = loadmat('ex3/ex3data1.mat')
    X = data['X']
    y = data['y']

    Z = data['X'][4000:]
    z_out = data['y'][4000:]
    input_size = 400
    m = X.shape[0]

    encoder = OneHotEncoder(sparse=False)
    losses, neural = train(X, encoder.fit_transform(y))
    import random

    X = np.matrix(X)
    y = np.matrix(y)

    T = np.array([X[0],X[2990],X[3440],X[1230],X[10],X[870],X[4560],X[45]])
    yy = np.array([y[0],y[2990],y[3440],y[1230],y[10],y[870],y[4560],y[45]])

    print(X.shape, T.shape)
#    sample_indices = np.random.choice(np.arange(T.shape[0]), 2000)

  #  validation = T[sample_indices]
  #  hat = E[sample_indices]
 #   print("sample_indice===", sample_indices.shape, validation.shape)

    y_pred = predict(X[0], neural.params)
    y_pred1 = predict(X[2990], neural.params)
    y_pred2 = predict(X[3440], neural.params)
    y_pred3 = predict(X[1230], neural.params)
    y_pred4 = predict(X[10], neural.params)
    y_pred5 = predict(X[870], neural.params)
    y_pred6 = predict(X[4560], neural.params)
    y_pred7 = predict(X[45], neural.params)

    print('fin===', y_pred, y_pred1, y_pred3, y_pred4, y_pred5, y_pred6, y_pred7,
          "z====",  y[0], y[2990], y[3440], y[1230], y[10], y[870], y[4560], y[45])

#    y_pred = np.array(np.argmax(h, axis=1) + 1)
 #   correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]

  #  print(correct)
    #plt.plot(params)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()
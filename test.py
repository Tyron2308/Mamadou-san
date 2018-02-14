import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def forward_propagate(X, theta1, theta2):
    m = X.shape[0]
    a1 = np.insert(X, 0, values=np.ones(m), axis=1)
    z2 = a1 * theta1.T
    a2 = np.insert(sigmoid(z2), 0, values=np.ones(m), axis=1)
    z3 = a2 * theta2.T
    h = sigmoid(z3)

    return a1, z2, a2, z3, h


def cost(params, X, y, learning_rate, theta1, theta2):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    J2 = 0
    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
        #print("j==", J)
    J = J / m

    print(J, J2)
    params = params - learning_rate * J
    return J


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))


def backprop(params, X, y, learning_rate, theta1, theta2):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)

    print(params.shape)

    # theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],
    #                              (hidden_size, (input_size + 1))))
    # theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],
    #                               (num_labels, (hidden_size + 1))))

    a1, z2, a2, z3, h = forward_propagate(X, theta1, theta2)
    J = 0
    delta1 = np.zeros(theta1.shape)  # (25, 401)
    delta2 = np.zeros(theta2.shape)  # (10, 26)

    for i in range(m):
        first_term = np.multiply(-y[i, :], np.log(h[i, :]))
        second_term = np.multiply((1 - y[i, :]), np.log(1 - h[i, :]))
        J += np.sum(first_term - second_term)
    J = J / m

    J += (float(learning_rate) / (2 * m)) *\
         (np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2)))

    print(J)
    for t in range(m):
        a1t = a1[t, :]  # (1, 401)
        z2t = z2[t, :]  # (1, 25)
        a2t = a2[t, :]  # (1, 26)
        ht = h[t, :]  # (1, 10)
        yt = y[t, :]  # (1, 10)

        d3t = ht - yt  # (1, 10)

        z2t = np.insert(z2t, 0, values=np.ones(1))  # (1, 26)
        d2t = np.multiply((theta2.T * d3t.T).T, sigmoid_gradient(z2t))  # (1, 26)

        delta1 = delta1 + (d2t[:, 1:]).T * a1t
        delta2 = delta2 + d3t.T * a2t

    delta1 = delta1 / m
    delta2 = delta2 / m

    delta1[:, 1:] = delta1[:, 1:] + (theta1[:, 1:] * learning_rate) / m
    delta2[:, 1:] = delta2[:, 1:] + (theta2[:, 1:] * learning_rate) / m
    grad = np.concatenate((np.ravel(delta1), np.ravel(delta2)))
    params += grad
    return J, grad


if __name__ == '__main__':
    import time
    start = time.time()
    data = loadmat('ex3/ex3data1.mat')
    X = data['X']
    y = data['y']
    input_size = 400
    hidden_size = 25
    num_labels = 10
    learning_rate = 1

    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    params = (np.random.random(size=hidden_size * (input_size + 1) +
                                    num_labels * (hidden_size + 1)) - 0.5) * 0.25

    print(params.shape)
    # unravel the parameter array into parameter matrices for each layer

    encoder = OneHotEncoder(sparse=False)
    y_onehot = encoder.fit_transform(y)
    i = 0
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)],
                                  (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):],
                                  (num_labels, (hidden_size + 1))))

    cost(params, X, y, 0.01, theta1, theta2)
    #for i in range(400):
    j, params = backprop(params, X, y_onehot, 0.01, theta1, theta2)
       # print('j===', params)
    # while i < 400:
    #     j, params = backprop(params, 400, 2, 9, X, y_onehot, 0.01)
    #
    #     print('THETA', theta1.shape, theta2.shape, X.shape)
    #     j = cost(params, X, y_onehot, 0.001, theta1, theta2)
    #     i = i + 1

#    y_pred = np.array(np.argmax(h, axis=1) + 1)
 #   correct = [1 if a == b else 0 for (a, b) in zip(y_pred, y)]

  #  print(correct)
    plt.plot(j)
    plt.xlabel('iteration')
    plt.ylabel('training loss')
    plt.title('Training Loss history')
    plt.show()
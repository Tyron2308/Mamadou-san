import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from neural_net import *
from multiprocessing.pool import ThreadPool


def compute_training(X_train, y_label, lr, rs, iters):

    loss, n = train(X_train, y_label, lr, rs, iters)
    y_train_pred = predict(X_train, n.params)
    acc_train = np.mean(y_label == y_train_pred)
    return acc_train, loss, neural, (lr, rs)


def gridsearch_model(X_train, y_label):
    pool = ThreadPool(5)

    best_val = -1
    best_stats = None
    learning_rates = [1e-2, 1e-3]
    regularization_strengths = [0.4, 0.5, 0.6]
    results = {}
    iters = 1000

    current_thread = [] * 5
    for lr in learning_rates:
        for rs in regularization_strengths:
            current_thread.append(pool.apply_async(compute_training, (X_train, y_label, lr, rs, iters)))

    for idx, thread in enumerate(curr_thread):
        acc_train, loss, neural, tuples = thread.get()
        results[tuples] = acc_train

        if best_val < acc_train:
            best_stats = loss
            best_val = acc_train
            best_net = neural

    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e train accuracy: %f val accuracy: %f' % (lr, reg, train_accuracy)
    print 'best validation accuracy achieved during cross-validation: %f' % best_val
    return {
        "best_stat": best_stats,
        "best_net": best_net,
        "res": results
    }


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




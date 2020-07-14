import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D
from keras import Sequential
from functools import partial
from sklearn.model_selection import train_test_split

import ROC_PR


def get_model_SVM(kernel=0, degree=1, C=1, gamma=1):
    from sklearn.svm import SVC
    C = 10 ** (int(C))
    gamma = 10 ** (int(gamma))
    degree = int(degree)
    kernel = int(kernel)
    if kernel == 0:
        svm_model_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
    elif kernel == 1:
        svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train, y_train)
    else:
        svm_model_linear = SVC(kernel='rbf', C=C, gamma=gamma).fit(X_train, y_train)

    score1 = ROC_PR.ROC_ML(svm_model_linear, X_test, y_test, "SVM", 0)
    # accuracy = svm_model_linear.score(X_test, y_test)
    print(score1)
    return score1


def get_model_LR(C=1, penalty=1, solver=1):
    C = 1 ** (int(C))
    penalty = int(penalty)
    solver = int(solver)
    from sklearn.linear_model import LogisticRegression
    if penalty == 0:
        lr_model_linear = LogisticRegression(C=C, penalty='l1').fit(X_train, y_train)
    elif penalty == 1:
        if solver == 0:
            lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='newton-cg').fit(X_train, y_train)
        elif solver == 1:
            lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='sag').fit(X_train, y_train)
        else:
            lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='lbfgs').fit(X_train, y_train)
    elif penalty == 2:
        lr_model_linear = LogisticRegression(C=C, penalty='elasticnet', solver='saga').fit(X_train, y_train)
    else:
        lr_model_linear = LogisticRegression(C=C, penalty='none').fit(X_train, y_train)

    score1 = ROC_PR.ROC_ML(lr_model_linear, X_test, y_test, "LR", 0)
    # accuracy = svm_model_linear.score(X_test, y_test)
    print(score1)
    return score1


X_train, X_test, y_train, y_test = 0, 0, 0, 0


def BO_SVM(X, y, i):
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    global X_train
    X_train = X_train2
    global X_test
    X_test = X_test2
    global y_train
    y_train = y_train2
    global y_test
    y_test = y_test2

    fit_with_partial = partial(get_model_SVM)

    fit_with_partial(kernel=0, degree=1, C=1, gamma=1)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'C': (-10, 10), "degree": (0.9, 100), "kernel": (0.9, 3.1), 'gamma': (-5, 5)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=10, n_iter=10, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("resultttttttttttttt" + str(i))
    print(optimizer.max)

def BO_LR(X, y, i):
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    global X_train
    X_train = X_train2
    global X_test
    X_test = X_test2
    global y_train
    y_train = y_train2
    global y_test
    y_test = y_test2

    fit_with_partial = partial(get_model_LR)

    fit_with_partial(C=1, penalty=1, solver=1)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'C': (-10, 10), 'penalty': (0.9, 4.1), 'solver': (0.9, 3.1)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=5, n_iter=5, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("resultttttttttttttt" + str(i))
    print(optimizer.max)


if __name__ == '__main__':
    # | iter | target | C | degree | kernel |
    # | 1 | 0.8609 | -1.66 | 3.853 | 0.9001 |
    C = -1.66
    degree = 3.853
    kernel = 0.9001
    C = 10 ** (int(C))
    degree = int(degree)
    kernel = int(kernel)
    # if kernel == 0:
    #     svm_model_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
    # else:
    #     svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train, y_train)
    print(C)
    print(degree)
    print(kernel)

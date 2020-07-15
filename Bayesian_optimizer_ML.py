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
    all_scores = 0
    C = 10 ** (int(C))
    gamma = 10 ** (int(gamma))
    degree = int(degree)
    kernel = int(kernel)

    for i in range(0, len(labels)):
        dfCurrentDrug = labels[i]
        X = df_train.values.tolist()
        y = dfCurrentDrug.values.tolist()
        for i2 in range(len(y) - 1, -1, -1):
            if y[i2][0] != 0.0 and y[i2][0] != 1.0:
                del y[i2]
                del X[i2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
                                                            shuffle=True)

        if kernel == 0:
            svm_model_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
        elif kernel == 1:
            svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train, y_train)
        else:
            svm_model_linear = SVC(kernel='rbf', C=C, gamma=gamma).fit(X_train, y_train)

        try:
            score1 = ROC_PR.ROC_ML(svm_model_linear, X_test, y_test, "SVM", 0)
        except:
            score1 = svm_model_linear.score(X_test, y_test)
        print(i, flush=True)
        print(score1, flush=True)
        all_scores = all_scores + score1

    print(all_scores/len(labels), flush=True)
    return all_scores/len(labels)


def get_model_LR(C=1, penalty=1, solver=1, l1_ratio=1, max_iter=2):
    from sklearn.linear_model import LogisticRegression
    all_scores = 0
    C = 10 ** (int(C))
    penalty = int(penalty)
    solver = int(solver)
    l1_ratio = l1_ratio / 10
    max_iter = 10 ** max_iter
    print(max_iter)
    for i in range(0, len(labels)):
        dfCurrentDrug = labels[i]
        X = df_train.values.tolist()
        y = dfCurrentDrug.values.tolist()
        for i2 in range(len(y) - 1, -1, -1):
            if y[i2][0] != 0.0 and y[i2][0] != 1.0:
                del y[i2]
                del X[i2]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42,
                                                            shuffle=True)
        if penalty == 0:
            lr_model_linear = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=max_iter).fit(X_train, y_train)
        elif penalty == 1:
            if solver == 0:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='newton-cg', max_iter=max_iter).fit(X_train, y_train)
            elif solver == 1:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='sag', max_iter=max_iter).fit(X_train, y_train)
            else:
                lr_model_linear = LogisticRegression(C=C, penalty='l2', solver='lbfgs', max_iter=max_iter).fit(X_train, y_train)
        elif penalty == 2:
            lr_model_linear = LogisticRegression(C=C, penalty='elasticnet', solver='saga', max_iter=max_iter, l1_ratio=l1_ratio).fit(X_train, y_train)
        else:
            lr_model_linear = LogisticRegression(C=C, penalty='none', max_iter=max_iter).fit(X_train, y_train)

        score1 = ROC_PR.ROC_ML(lr_model_linear, X_test, y_test, "LR", 0)
        # accuracy = svm_model_linear.score(X_test, y_test)
        print(i, flush=True)
        print(score1, flush=True)
        all_scores = all_scores + score1

    print(all_scores / len(labels), flush=True)
    return all_scores / len(labels)


X_train, X_test, y_train, y_test = 0, 0, 0, 0
df_train, labels = 0, 0


def BO_SVM(X, y):
    global df_train
    df_train = X
    global labels
    labels = y
    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    # global X_train
    # X_train = X_train2
    # global X_test
    # X_test = X_test2
    # global y_train
    # y_train = y_train2
    # global y_test
    # y_test = y_test2

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
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), flush=True)

    print("resultttttttttttttt SVM" + str(i), flush=True)
    print(optimizer.max, flush=True)

def BO_LR(X, y):
    global df_train
    df_train = X
    global labels
    labels = y

    # X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    #
    # global X_train
    # X_train = X_train2
    # global X_test
    # X_test = X_test2
    # global y_train
    # y_train = y_train2
    # global y_test
    # y_test = y_test2

    fit_with_partial = partial(get_model_LR)

    fit_with_partial(C=1, penalty=1, solver=1, l1_ratio=1, max_iter=2)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'C': (-10, 10), 'penalty': (0.9, 3.1), 'solver': (0.9, 2.1), 'l1_ratio': (0, 10), 'max_iter': (1.9, 4.1)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res), flush=True)

    print("resultttttttttttttt LR" + str(i), flush=True)
    print(optimizer.max, flush=True)


if __name__ == '__main__':
    # | iter | target | C | degree | kernel |
    # | 1 | 0.8609 | -1.66 | 3.853 | 0.9001 |
    # C = -1.66
    # degree = 3.853
    # kernel = 0.9001
    # C = 10 ** (int(C))
    # degree = int(degree)
    # kernel = int(kernel)
    # if kernel == 0:
    #     svm_model_linear = SVC(kernel='linear', C=C).fit(X_train, y_train)
    # else:
    #     svm_model_linear = SVC(kernel='poly', C=C, degree=degree).fit(X_train, y_train)
    # resultttttttttttttt
    # LR29
    # {'target': 0.8636191441305866,
    #  'params': {'C': -7.953311423443483, 'l1_ratio': 4.140559878195683, 'max_iter': 3.427680347001039,
    #             'penalty': 1.811194392959186, 'solver': 0.9599441507353046}}

    # {'target': 0.8601012372054825,
    #  'params': {'C': 7.023950121643671, 'l1_ratio': 2.8797628407599776, 'max_iter': 4.1, 'penalty': 0.9, 'solver': 0.9}}
    C = 7.023950121643671
    l1_ratio = 2.8797628407599776
    max_iter = 4.1
    penalty = 0.9
    solver = 0.9
    max_iter = 10 ** max_iter
    C = 1 ** (int(C))
    penalty = int(penalty)
    solver = int(solver)
    l1_ratio = l1_ratio / 10
    print(C)
    print(l1_ratio)
    print(penalty)
    print(solver)
    print(max_iter)

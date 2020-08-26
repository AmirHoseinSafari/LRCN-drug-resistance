from keras import regularizers, Model, Input
from keras.layers import BatchNormalization, concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from loading_data import data_preprocess
import numpy as np
import keras.backend as K
from keras.layers import Dense, Dropout
from functools import partial

from evaluations import ROC_PR

NUM_CLASSES = 12
epochs = 60


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def get_model(dropout2_rate, l2_reg, dense_1_neurons,
              dense_2_neurons, dense_3_neurons, dense_4_neurons, dense_5_neurons, i1):
    # feat_num : The number of features in the data
    global feat_num
    input = Input(shape=(feat_num,))

    for i in range(0, i1):
        if i == 0:
            x = Dense(dense_1_neurons, kernel_regularizer=regularizers.l2(l2_reg))(input)
        elif i == 1:
            x = Dense(dense_2_neurons, kernel_regularizer=regularizers.l2(l2_reg))(x)
        elif i == 1:
            x = Dense(dense_3_neurons, kernel_regularizer=regularizers.l2(l2_reg))(x)
        elif i == 1:
            x = Dense(dense_4_neurons, kernel_regularizer=regularizers.l2(l2_reg))(x)
        else:
            x = Dense(dense_5_neurons, kernel_regularizer=regularizers.l2(l2_reg))(x)

        x = BatchNormalization()(x)
        x = Dropout(dropout2_rate)(x)

    wide_deep = concatenate([input, x])
    preds = Dense(12, activation='sigmoid', kernel_regularizer=regularizers.l2(l2_reg))(wide_deep)
    model = Model(inputs=input, outputs=preds)
    opt = Adam(lr=0.01)
    model.compile(optimizer=opt, loss=masked_loss_function, metrics=[])
    return model


def prepare_date(features, label):
    FrameSize = 200

    y = []
    for i in range(0, len(label)):
        label[i] = label[i].values.tolist()

    for j in range(0, len(label[0])):
        tmp = []
        for i in range(0, len(label)):
            if label[i][j][0] != 0.0 and label[i][j][0] != 1.0:
                tmp.extend([-1])
            else:
                tmp.extend(label[i][j])
        y.append(tmp)

    y = np.array(y)
    return features, y, FrameSize


def fit_with(dropout2_rate, l2_reg, dense_1_neurons_x128,
             dense_2_neurons_x128, dense_3_neurons_x128, dense_4_neurons_x128, dense_5_neurons_x128, i1):
    i1 = int(i1)
    import random
    if random.randint(0, 10) < 5:
        l2_reg = l2_reg * 0.1

    dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
    dense_2_neurons = max(int(dense_2_neurons_x128 * 64), 64)
    dense_3_neurons = max(int(dense_3_neurons_x128 * 64), 64)
    dense_4_neurons = max(int(dense_4_neurons_x128 * 64), 64)
    dense_5_neurons = max(int(dense_5_neurons_x128 * 64), 64)

    model = get_model(dropout2_rate, l2_reg, dense_1_neurons,
                      dense_2_neurons, dense_3_neurons, dense_4_neurons, dense_5_neurons, i1)

    return run_one_fold(model)


def run_one_fold(model):
    model.compile(
        loss=masked_loss_function,
        optimizer='Adam',
        metrics=[masked_accuracy]
    )
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=128,
        # shuffle=True,
        verbose=2,
        validation_data=(X_val, y_val)
    )

    score = ROC_PR.ROC_Score(model, X_val, y_val)
    score_test = ROC_PR.ROC_Score(model, X_test, y_test)
    score_for_each_drug = ROC_PR.ROC(model, X_test, y_test, ("wide-n-deep" + "BO_delete"), True)
    spec_recall = ROC_PR.PR(model, X_test, y_test)

    print('area under ROC curve for val:', score)
    print('area under ROC curve for test:', score_test)
    print(score_for_each_drug)
    print("recall at 95 spec: ", spec_recall)

    return score


X_train, X_test, X_val, y_train, y_test, y_val = 0, 0, 0, 0, 0, 0
feat_num = 0
X, y = 0, 0


def BO(X_train2, X_test2, X_val2, y_train2, y_test2, y_val2):
    global X_train
    X_train = X_train2
    global X_test
    X_test = X_test2
    global X_val
    X_val = X_val2
    global y_train
    y_train = y_train2
    global y_test
    y_test = y_test2
    global y_val
    y_val = y_val2
    global feat_num
    feat_num = X_train.shape[1]

    fit_with_partial = partial(fit_with)

    fit_with_partial(dropout2_rate=0.2, l2_reg=0.01, dense_1_neurons_x128=1, dense_2_neurons_x128=1,
                     dense_3_neurons_x128=1, dense_4_neurons_x128=1, dense_5_neurons_x128=1, i1=1)

    from bayes_opt import BayesianOptimization

    pbounds = {'dropout2_rate': (0.1, 0.5),
               'l2_reg': (0.1, 0.5),
               "dense_1_neurons_x128": (0.9, 8.1),
               "dense_2_neurons_x128": (0.9, 8.1),
               "dense_3_neurons_x128": (0.9, 8.1),
               "dense_4_neurons_x128": (0.9, 8.1),
               "dense_5_neurons_x128": (0.9, 8.1),
               "i1": (1.9, 5.1)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,
        random_state=1,
    )
    optimizer.maximize(init_points=10, n_iter=10, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("resultttttttttttttt")
    print(optimizer.max)

    import json

    with open('BO_result.txt', 'ab') as f:
        f.write(json.dumps(optimizer.max).encode())
        f.write("\n".encode())


def run_bayesian(df_train, labels):
    X, y, FrameSize = prepare_date(df_train, labels)

    for i in range(0, 10):
        print("fold: " + str(i))
        length = int(len(X)/10)
        if i == 0:
            X_train = X[length:]
            X_test = X[0:length]
            y_train = y[length:]
            y_test = y[0:length]
        elif i != 9:
            X_train = np.append(X[0:length*i], X[length*(i+1):], axis=0)
            X_test = X[length*i:length*(i+1)]
            y_train = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
            y_test = y[length * i:length * (i + 1)]
        else:
            X_train = X[0:length * i]
            X_test = X[length * i:]
            y_train = y[0:length * i]
            y_test = y[length * i:]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, shuffle=False)

        BO(X_train, X_test, X_val, y_train, y_test, y_val)



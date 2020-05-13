from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K
from sklearn.model_selection import StratifiedKFold

import Bayesian_optimizer
import ROC_PR
import plot
import data_preprocess
# from keras.utils.vis_utils import plot_model

import tensorflow as tf


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def weighted_binary_crossentropy_nan(y_true, y_pred):
    index = ~tf.is_nan(y_true)
    y_true = tf.boolean_mask(y_true, index)
    y_pred = tf.boolean_mask(y_pred, index)

    bce = K.binary_crossentropy(y_true, y_pred)

    bce = tf.where(tf.is_nan(bce), tf.zeros_like(bce), bce)
    bce = K.mean(bce, axis=-1)
    return bce


def model_256_128_64_2_StateFul(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping):
    batch_size = 128
    if len(X_train) // 128 != len(X_train) / 128:
        tmp = X_train[0]
        tmp2 = y_train[0]
        for j in range(0, len(tmp2)):
            tmp2[j] = 0
        for j in range (0, len(tmp)):
            for k in range(0, len(tmp[j])):
                if tmp[j][k] == 1:
                    tmp[j][k] = 0
        for j in range(0, ((((len(X_train) // 128) + 1) * 128) - len(X_train))):
            X_train = np.append(X_train, np.expand_dims(tmp, axis=0), axis=0)
            y_train = np.append(y_train, np.expand_dims(tmp2, axis=0), axis=0)

        for j in range(0, ((((len(X_test) // 128) + 1) * 128) - len(X_test))):
            X_test = np.append(X_test, np.expand_dims(tmp, axis=0), axis=0)
            y_test = np.append(y_test, np.expand_dims(tmp2, axis=0), axis=0)

    model = Sequential()
    model.add(LSTM(256, batch_input_shape=(batch_size, FrameSize, X[0].shape[1]), return_sequences=True, stateful=True))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, stateful=True))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='sigmoid'))

    model.compile(
        loss=masked_loss_function,
        optimizer='Adam',
        metrics=[masked_accuracy]
    )

    print(model.summary())

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/256_128_64_2_StateFul.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    plot.plot(history, "256_128_64_2_StateFul")

    # ROC_PR.ROC(model, X_test, y_test, "256_128_64_2_StateFul", True)


def model_256_128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping):
    model = Sequential()
    model.add(LSTM(256, input_shape=(FrameSize, X[0].shape[1]), return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='sigmoid'))

    model.compile(
        loss=masked_loss_function,
        optimizer='Adam',
        metrics=[masked_accuracy]
    )

    print(model.summary())

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=128,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/256_128_64_2.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    plot.plot(history, "256_128_64_2")
    ROC_PR.ROC(model, X_test, y_test, "256_128_64_2", True)


def model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name, dropout2_rate, dense_1, filterCNN, kernelCNN, LSTM1, LSTM2, recurrent_dropout):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    # model.add(TimeDistributed(Conv1D(filters=1, kernel_size=3, activation='relu', padding='same', input_shape=(FrameSize, X[0].shape[1], 1))))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=3)))
    # model.add(TimeDistributed(Flatten()))
    model.add(Dropout(dropout2_rate))
    # model.add(Conv1D(filters=5, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(filters=filterCNN, kernel_size=kernelCNN, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=3))
    # model.add(TimeDistributed(Flatten()))
    # model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(LSTM1, return_sequences=True, recurrent_dropout=recurrent_dropout))
    model.add(SpatialDropout1D(dropout2_rate))
    # model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(LSTM(LSTM2, return_sequences=False, recurrent_dropout=recurrent_dropout))
    model.add(Dropout(dropout2_rate))
    # model.add(Dense(64))
    model.add(Dense(dense_1))
    model.add(Dropout(dropout2_rate))
    model.add(Dense(12, activation='sigmoid'))

    model.compile(
        loss=masked_loss_function,
        optimizer='Adam',
        metrics=[masked_accuracy]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=128,
        # shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    # plot_model(model, to_file='model_plot.png', show_shapes=True)

    plot.plot(history, ("LRCN" + name))

    score = ROC_PR.ROC(model, X_test, y_test, ("LRCN" + name), True)
    return score


def prepareDate(features, label):
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

    X = features.values.tolist()

    for i in range(0, len(X)):
        if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
            for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
                X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)
    return X, y, FrameSize


def run_model_kfold(df_train, labels, epoch):

    X, y, FrameSize = prepareDate(df_train, labels)
    # kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    # X = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24],
    #      [25, 26]]
    # y = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [21, 22], [23, 24],
    #      [25, 26]]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    cvscores1 = []
    cvscores2 = []
    cvscores3 = []
    earlyStopping = EarlyStopping(monitor='val_masked_accuracy', mode='max', min_delta=0.1, verbose=1, patience=80)

    for i in range(0, 10):
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

        score1 = model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "1_" + str(i),
                                  0.3155266936013428, 240, 5, 5, 143, 216, 0.3)

        score2 = model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "2_" + str(i),
                                  0.1, 240, 5, 5, 143, 216, 0.3)

        score3 = model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "5_" + str(i),
                                  dropout2_rate=0.1783805364232113, dense_1=82, filterCNN=7, kernelCNN=5, LSTM1=140,
                                  LSTM2=509, recurrent_dropout=0.3)
        print("Area for 1")
        print(score1)
        print("Area for 2")
        print(score2)
        print("Area for 3")
        print(score3)
        cvscores1.append(score1)
        cvscores2.append(score2)
        cvscores3.append(score3)
    f = open('result/kfoldResult.txt', 'w')
    for ele in cvscores1:
        f.write(str(ele) + '\n')
    for ele in cvscores2:
        f.write(str(ele) + '\n')
    for ele in cvscores3:
        f.write(str(ele) + '\n')
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores2), np.std(cvscores2)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores3), np.std(cvscores3)))


def run_model(df_train, labels, epoch):
    X, y, FrameSize = prepareDate(df_train, labels)

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y[:, 7:9], shuffle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)

    earlyStopping = EarlyStopping(monitor='val_masked_accuracy', mode='max', min_delta=0.1, verbose=1, patience=80)

    # Bayesian_optimizer.BO(X_train, X_test, y_train, y_test)

    for i in range(0, 4):
        model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "1_" + str(i),
                                  0.3155266936013428, 240, 5, 5, 143, 216, 0.3)

        model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "2_" + str(i),
                                  0.1, 240, 5, 5, 143, 216, 0.3)

        # model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "3_" + str(i),
        #                           0.1, 256, 8, 6, 128, 256, 0.3)
        #
        # model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "4_" + str(i),
        #                           0.1, 256, 8, 6, 128, 256, 0.1)

        model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "5_" + str(i),
                                  dropout2_rate=0.1783805364232113, dense_1=82, filterCNN=7, kernelCNN=5, LSTM1=140,
                                  LSTM2=509, recurrent_dropout=0.3)

if __name__ == '__main__':
    df_train, labels = data_preprocess.process(6)
    run_model_kfold(df_train, labels, 10)



# def masked_multi_weighted_bce(alpha, y_pred):
#     y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
#     mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
#     num_not_missing = K.sum(mask, axis=-1)
#     alpha = K.abs(alpha)
#     bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
#     masked_bce = bce * mask
#     return K.sum(masked_bce, axis=-1) / num_not_missing
#
#
# def block_gradient_loss_function(self):
#     newY = []
#     newY_predicted = []
#     l = K.binary_crossentropy
#
#     def loss(y_true, y_pred):
#         y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#         y_true_ = K.cast(K.greater(y_true, 0.), K.floatx())
#         return l(y_true_, y_pred)
#     return loss
#     # print(K.binary_crossentropy(y, y_predicted))
#     # y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
#     # y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
#     # return K.binary_crossentropy(y_true_, y_pred)
#     # for i in range(0, len(y)):
#     #     if y != -1:
#     #         newY.append(y)
#     #         newY_predicted.append(y_predicted)
#     # return 0
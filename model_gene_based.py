from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D, GRU
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K

import Bayesian_optimizer
import ROC_PR
import plot
import data_preprocess
from keras.utils import to_categorical

import tensorflow as tf


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def model_lrcn_simple(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name, limited=False):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=5, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=3, padding='same'))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    if limited:
        model.add(Dense(7, activation='sigmoid'))
    else:
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
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    plot.plot(history, ("LRCN" + name))

    score = ROC_PR.ROC(model, X_test, y_test, ("LRCN" + name), True, limited=limited)
    return score


def model_lstm_simple(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name, limited=False):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    if limited:
        model.add(Dense(7, activation='sigmoid'))
    else:
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
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    plot.plot(history, ("LRCN" + name))

    score = ROC_PR.ROC(model, X_test, y_test, ("LRCN" + name), True, limited=limited)
    return score


def model_gru_simple(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name, limited=False):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(GRU(256, return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(GRU(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(128))
    model.add(Dropout(0.2))
    if limited:
        model.add(Dense(7, activation='sigmoid'))
    else:
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
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    plot.plot(history, ("LRCN" + name))

    score = ROC_PR.ROC(model, X_test, y_test, ("LRCN" + name), True, limited=limited)
    return score


def model_CNN_LSTM(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    model.add(Dropout(0.1))
    model.add(Conv1D(filters=8, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=3, padding='same'))

    model.add(LSTM(518, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.1))

    model.add(Dense(64))
    model.add(Dropout(0.1))

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
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='val_masked_accuracy', mode='max', save_best_only=True)]
    )

    plot.plot(history, ("LRCN" + name))

    score = ROC_PR.ROC(model, X_test, y_test, ("LRCN" + name), True)
    return score, ROC_PR.ROC_Score(model, X_train, y_train, limited=False)


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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, shuffle=True)
    X = np.append(X_train, X_test, axis=0)
    y = np.append(y_train, y_test, axis=0)

    cvscores1 = []
    cvscores_all = []
    earlyStopping = EarlyStopping(monitor='val_masked_accuracy', mode='max', min_delta=0.1, verbose=1, patience=60)

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

        score1, score_all = model_CNN_LSTM(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "24J_" + str(i))

        print("Area for 1")
        print(score1)
        print("________________________")
        print(score_all)
        cvscores1.append(score1)
        cvscores_all.append(score_all)
    f = open('result/kfoldResult.txt', 'w')
    for ele in cvscores1:
        f.write(str(ele) + '\n')
    for ele in cvscores_all:
        f.write(str(ele) + '\n')
    print(cvscores_all)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_all), np.std(cvscores_all)))



def run_model(df_train, labels, epoch, limited=False):
    X, y, FrameSize = prepareDate(df_train, labels)
    # X = to_categorical(X, dtype=np.int8)
    earlyStopping = EarlyStopping(monitor='val_masked_accuracy', mode='max', min_delta=0.1, verbose=1, patience=80)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=True)
    for i in range(0, 2):
        model_CNN_LSTM(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "gene_10_" + str(i))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
    for i in range(0, 2):
        model_CNN_LSTM(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "gene_20_" + str(i))

def run_bayesian(df_train, labels, limited=False, portion=0.1):
    X, y, FrameSize = prepareDate(df_train, labels)
    # X = to_categorical(X, dtype=np.int8)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=portion, random_state=1, shuffle=True)
    Bayesian_optimizer.BO(X_train, X_test, y_train, y_test, limited, portion=portion)


def run_all(df_train, labels, epoch):
    X, y, FrameSize = prepareDate(df_train, labels)
    # X = to_categorical(X, dtype=np.int8)
    earlyStopping = EarlyStopping(monitor='val_masked_accuracy', mode='max', min_delta=0.1, verbose=1, patience=80)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=True)
    res = []
    for i in range(0, 5):
        res.append(i)
        res.append(model_lrcn_simple(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "lrcn_gene_" + str(i)))
        res.append(model_lstm_simple(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "lstm_gene_" + str(i)))
        res.append(model_gru_simple(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "gru_gene_" + str(i)))
    print(res)

if __name__ == '__main__':
    df_train, labels = data_preprocess.process(6)
    run_model_kfold(df_train, labels, 10)



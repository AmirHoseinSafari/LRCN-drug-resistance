from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D, GRU, TimeDistributed, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K

from models import Bayesian_optimizer
from evaluations import plot, ROC_PR
from loading_data import data_preprocess


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


def model_CNN_LSTM_time(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name):
    print(X.shape)
    print(X_train.shape)
    print(X_test.shape)
    print(FrameSize)
    print(y_train.shape)
    print(y_test.shape)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
    # y_train = y_train.reshape(7060, 12, 1)
    # y_test = y_test.reshape(785, 12, 1)
    model = Sequential()
    model.add(Dropout(0.1))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=3, padding='same')))
    model.add(TimeDistributed(Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')))
    model.add(TimeDistributed(MaxPooling1D(pool_size=3, padding='same')))
    model.add(TimeDistributed(Flatten()))

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


def model_CNN_LSTM_shuffled_index(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    model.add(Dropout(0.20412603943141638))
    model.add(Conv1D(filters=7, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Dropout(0.20412603943141638))
    model.add(Conv1D(filters=4, kernel_size=5, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=5, padding='same'))

    model.add(LSTM(311, return_sequences=True, recurrent_dropout=0.3))
    model.add(Dropout(0.20412603943141638))
    model.add(LSTM(401, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.20412603943141638))

    model.add(Dense(228))
    model.add(Dropout(0.20412603943141638))
    model.add(Dense(347))
    model.add(Dropout(0.20412603943141638))
    model.add(Dense(154))
    model.add(Dropout(0.20412603943141638))
    model.add(Dense(404))
    model.add(Dropout(0.20412603943141638))

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


def model_CNN_LSTM_random_data(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name):
    print(X.shape)
    print(FrameSize)
    model = Sequential()

    model.add(Dropout(0.3311428861138142))
    model.add(Conv1D(filters=4, kernel_size=6, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Dropout(0.3311428861138142))
    model.add(Conv1D(filters=7, kernel_size=4, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Dropout(0.3311428861138142))
    model.add(Conv1D(filters=6, kernel_size=6, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))
    model.add(Dropout(0.3311428861138142))
    model.add(Conv1D(filters=4, kernel_size=4, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=4, padding='same'))

    model.add(LSTM(425, return_sequences=True, recurrent_dropout=0.3))
    model.add(Dropout(0.3311428861138142))
    model.add(LSTM(189, return_sequences=True, recurrent_dropout=0.3))
    model.add(Dropout(0.3311428861138142))
    model.add(LSTM(283, return_sequences=True, recurrent_dropout=0.3))
    model.add(Dropout(0.3311428861138142))
    model.add(LSTM(333, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.3311428861138142))

    model.add(Dense(331))
    model.add(Dropout(0.3311428861138142))
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


def model_shuffle_index_0(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, name, index):
    print(X.shape)
    print(FrameSize)

    model = Sequential()

    if index == 0:
        model.add(Dropout(0.3975303416300372))
        model.add(Conv1D(filters=4, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Dropout(0.3975303416300372))
        model.add(Conv1D(filters=6, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Dropout(0.3975303416300372))
        model.add(Conv1D(filters=7, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))

        model.add(LSTM(107, return_sequences=True, recurrent_dropout=0.3))
        model.add(Dropout(0.3975303416300372))
        model.add(LSTM(161, return_sequences=True, recurrent_dropout=0.3))
        model.add(Dropout(0.3975303416300372))
        model.add(LSTM(386, return_sequences=False, recurrent_dropout=0.3))
        model.add(Dropout(0.3975303416300372))

        model.add(Dense(90))
        model.add(Dropout(0.3975303416300372))
        model.add(Dense(503))
        model.add(Dropout(0.3975303416300372))
        model.add(Dense(319))
        model.add(Dropout(0.3975303416300372))
        model.add(Dense(151))
        model.add(Dropout(0.3975303416300372))

        model.add(Dense(12, activation='sigmoid'))
    elif index == 1:
        model.add(Dropout(0.1135579744586698))
        model.add(Conv1D(filters=4, kernel_size=5, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=4, padding='same'))

        model.add(LSTM(420, return_sequences=True, recurrent_dropout=0.3))
        model.add(Dropout(0.1135579744586698))
        model.add(LSTM(67, return_sequences=True, recurrent_dropout=0.3))
        model.add(Dropout(0.1135579744586698))
        model.add(LSTM(231, return_sequences=False, recurrent_dropout=0.3))
        model.add(Dropout(0.1135579744586698))

        model.add(Dense(112))
        model.add(Dropout(0.1135579744586698))
        model.add(Dense(92))
        model.add(Dropout(0.1135579744586698))

        model.add(Dense(12, activation='sigmoid'))
    elif index == 2:
        model.add(Dropout(0.3211287914743064))
        model.add(Conv1D(filters=7, kernel_size=3, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=3, padding='same'))
        model.add(Dropout(0.3211287914743064))
        model.add(Conv1D(filters=4, kernel_size=5, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=4, padding='same'))
        model.add(Dropout(0.3211287914743064))
        model.add(Conv1D(filters=5, kernel_size=4, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=4, padding='same'))

        model.add(LSTM(404, return_sequences=False, recurrent_dropout=0.3))
        model.add(Dropout(0.3211287914743064))

        model.add(Dense(69))
        model.add(Dropout(0.3211287914743064))
        model.add(Dense(70))
        model.add(Dropout(0.3211287914743064))
        model.add(Dense(171))
        model.add(Dropout(0.3211287914743064))
        model.add(Dense(453))
        model.add(Dropout(0.3211287914743064))

        model.add(Dense(12, activation='sigmoid'))
    elif index == 3:

        model.add(Dropout(0.3584182599371434))
        model.add(Conv1D(filters=8, kernel_size=5, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=5, padding='same'))
        model.add(Dropout(0.3584182599371434))
        model.add(Conv1D(filters=7, kernel_size=5, activation='relu', padding='same'))
        model.add(MaxPooling1D(pool_size=5, padding='same'))

        model.add(LSTM(438, return_sequences=True, recurrent_dropout=0.3))
        model.add(Dropout(0.3584182599371434))
        model.add(LSTM(500, return_sequences=False, recurrent_dropout=0.3))
        model.add(Dropout(0.3584182599371434))

        model.add(Dense(81))
        model.add(Dropout(0.3584182599371434))
        model.add(Dense(267))
        model.add(Dropout(0.3584182599371434))
        model.add(Dense(388))
        model.add(Dropout(0.3584182599371434))


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


def prepare_data(features, label):
    # TODO
    FrameSize = 1

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
        # if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
        #     for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
        #         X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)
    return X, y, FrameSize


def run_model_kfold(df_train, labels, epoch, index=-1):

    X, y, FrameSize = prepare_data(df_train, labels)
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

        score1, score_all = model_shuffle_index_0(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "24J_" + str(i), index)

        print("Area for 1")
        print(score1)
        print("________________________")
        print(score_all)
        cvscores1.append(score1)
        cvscores_all.append(score_all)
    f = open('result/kfoldResult_shuffle.txt', 'w')
    for ele in cvscores1:
        f.write(str(ele) + '\n')
    for ele in cvscores_all:
        f.write(str(ele) + '\n')
    print(cvscores1)
    print(cvscores_all)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores1), np.std(cvscores1)))
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores_all), np.std(cvscores_all)))


def run_model(df_train, labels, epoch, limited=False):
    X, y, FrameSize = prepare_data(df_train, labels)
    # X = to_categorical(X, dtype=np.int8)
    earlyStopping = EarlyStopping(monitor='val_masked_accuracy', mode='max', min_delta=0.1, verbose=1, patience=80)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, shuffle=True)
    for i in range(0, 2):
        model_CNN_LSTM_time(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "gene_10_" + str(i))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1, shuffle=True)
    for i in range(0, 2):
        model_CNN_LSTM(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping, "gene_20_" + str(i))


def run_bayesian(df_train, labels, limited=False, portion=0.1):
    X, y, FrameSize = prepare_data(df_train, labels)

    for i in range(0, 10):
        print("fold: " + str(i))
        length = int(len(X) / 10)
        if i == 0:
            X_train = X[length:]
            X_test = X[0:length]
            y_train = y[length:]
            y_test = y[0:length]
        elif i != 9:
            X_train = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
            X_test = X[length * i:length * (i + 1)]
            y_train = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
            y_test = y[length * i:length * (i + 1)]
        else:
            X_train = X[0:length * i]
            X_test = X[length * i:]
            y_train = y[0:length * i]
            y_test = y[length * i:]

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, shuffle=False)
        Bayesian_optimizer.BO(X_train, X_test, X_val, y_train, y_test, y_val, limited, portion)


def run_bayesian_single(df_train, labels, limited=False, portion=0.1):
    X, y, FrameSize = prepare_data(df_train, labels)
    X2 = X
    y2 = y
    for j in range(0, 12):
        print("drug: " + str(j))
        tmp = []
        tmp_x = []
        dele = []
        for k in range(0, len(y2)):
            if y2[k][j] == 1 or y2[k][j] == 0:
                tmp.append(y2[k][j])
                tmp_x.append(X2[k])
        y = tmp
        X = tmp_x
        for i in range(3, 4):
            print("fold: " + str(i))
            length = int(len(X) / 10)
            if i == 0:
                X_train = X[length:]
                X_test = X[0:length]
                y_train = y[length:]
                y_test = y[0:length]
            elif i != 9:
                X_train = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
                X_test = X[length * i:length * (i + 1)]
                y_train = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
                y_test = y[length * i:length * (i + 1)]
            else:
                X_train = X[0:length * i]
                X_test = X[length * i:]
                y_train = y[0:length * i]
                y_test = y[length * i:]

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1, shuffle=False)
            Bayesian_optimizer.BO(X_train, X_test, X_val, y_train, y_test, y_val, limited, portion)


def run_all(df_train, labels, epoch):
    X, y, FrameSize = prepare_data(df_train, labels)
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
    run_model(df_train, labels, 10)



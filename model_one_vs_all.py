from keras import Sequential
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D
import numpy as np
from sklearn.model_selection import train_test_split
import keras.backend as K

import ROC_PR
import plot
import data_preprocess

import tensorflow as tf


def weighted_binary_crossentropy_nan(y_true, y_pred):
    # y_pred is the probability of belonging to positive class as model returns
    # The below three lines remove None values
    index = ~tf.is_nan(y_true)
    y_true = tf.boolean_mask(y_true, index)
    y_pred = tf.boolean_mask(y_pred, index)

    bce = K.binary_crossentropy(y_true, y_pred)

    bce = tf.where(tf.is_nan(bce), tf.zeros_like(bce), bce)
    bce = K.mean(bce, axis=-1)
    # the K.cast(K.greater(y_true, 0.), K.floatx()) makes the labels of positive class, where are positive class weights, equal to 1 to make the formula correct

    #    #weighted_bce = - y_true * K.log(y_pred + 1e-6) - ((1.0 - K.cast(K.greater(y_true, 0.), K.floatx())) * K.log(1.0 - y_pred + 1e-6))
    #    weighted_bce = - tf.constant(globalVars.weight_matrix, dtype= np.float32) * K.log(y_pred + 1e-6) - (tf.constant(globalVars.weight_matrix, dtype= np.float32) *(1.0 - y_true) * K.log(1.0 - y_pred + 1e-6))
    #
    #    # it replaces nan values with zero
    #    weighted_bce = tf.where(tf.is_nan(weighted_bce), tf.zeros_like(weighted_bce), weighted_bce)
    #    weighted_bce = K.mean(weighted_bce, axis=-1)
    #    return K.in_train_phase(weighted_bce, bce)

    return bce
def masked_multi_weighted_bce(alpha, y_pred):
    import tensorflow as tf
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    # s = tf.Session()
    # print(y_pred.numpy())
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    num_not_missing = K.sum(mask, axis=-1)
    alpha = K.abs(alpha)
    bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
    masked_bce = bce * mask
    return K.sum(masked_bce, axis=-1) / num_not_missing


def block_gradient_loss_function(self):
    newY = []
    newY_predicted = []
    l = K.binary_crossentropy

    def loss(y_true, y_pred):
        y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
        y_true_ = K.cast(K.greater(y_true, 0.), K.floatx())
        return l(y_true_, y_pred)
    return loss
    # print(K.binary_crossentropy(y, y_predicted))
    # y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    # y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    # return K.binary_crossentropy(y_true_, y_pred)
    # for i in range(0, len(y)):
    #     if y != -1:
    #         newY.append(y)
    #         newY_predicted.append(y_predicted)
    # return 0


def model_256_128_64_2_StateFul(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping):
    batch_size = 128
    if len(X_train) // 128 != len(X_train) / 128:
        tmp = X_train[0]
        tmp2 = y_train[0]
        # print(tmp)
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
        loss=weighted_binary_crossentropy_nan,
        optimizer='Adam',
        metrics=['accuracy']
    )

    print(model.summary())

    # for i in range(0, 20):
    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=batch_size,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/256_128_64_2_StateFul.h5', monitor='val_accuracy', mode='max', save_best_only=True)]
    )
        # model.reset_states()
    # model.save_weights("result/256_128_64_2_StateFul.h5")

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
        loss=weighted_binary_crossentropy_nan,
        optimizer='Adam',
        metrics=['accuracy']
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
                   ModelCheckpoint('result/256_128_64_2.h5', monitor='val_accuracy', mode='max', save_best_only=True)]
    )

    # model.save_weights("result/256_128_64_2.h5")

    plot.plot(history, "256_128_64_2")
    print("here we call the function after training:")
    ROC_PR.ROC(model, X_test, y_test, "256_128_64_2", True)


def model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping):
    print(X.shape)
    print(FrameSize)
    model = Sequential()
    # model.add(TimeDistributed(Conv1D(filters=1, kernel_size=3, activation='relu', padding='same', input_shape=(FrameSize, X[0].shape[1], 1))))
    # model.add(TimeDistributed(MaxPooling1D(pool_size=3)))
    # model.add(TimeDistributed(Flatten()))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=5, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=3))
    # model.add(TimeDistributed(Flatten()))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(12, activation='sigmoid'))

    model.compile(
        loss=weighted_binary_crossentropy_nan,
        optimizer='Adam',
        metrics=['accuracy']
    )

    # print(model.summary())

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=128,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[earlyStopping,
                   ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='val_accuracy', mode='max', save_best_only=True)]
    )

    # model.save_weights("result/CNN256_LSTM128_64_2.h5")

    plot.plot(history, "CNN256_LSTM128_64_2")

    ROC_PR.ROC(model, X_test, y_test, "CNN256_LSTM128_64_2", True)


def run_model(df_train, labels, epoch):
    FrameSize = 200

    y = []
    for i in range(0, len(labels)):
        labels[i] = labels[i].values.tolist()

    for j in range(0, len(labels[0])):
        tmp = []
        for i in range(0, len(labels)):
            # if labels[i][j][0] != 0.0 and labels[i][j][0] != 1.0:
            #     tmp.extend([0])
            # else:
            tmp.extend(labels[i][j])
        y.append(tmp)

    X = df_train.values.tolist()

    for i in range(0, len(X)):
        if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
            for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
                X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    earlyStopping = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=1, verbose=1, patience=15)

    model_256_128_64_2_StateFul(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping)
    model_256_128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping)
    model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch, earlyStopping)


if __name__ == '__main__':
    df_train, labels = data_preprocess.process(6)
    run_model(df_train, labels, 10)

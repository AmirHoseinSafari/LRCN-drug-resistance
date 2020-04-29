from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, TimeDistributed, Conv1D, MaxPooling1D
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import ROC_PR
import plot
import data_preprocess
# from keras.utils.vis_utils import plot_model


# def modelSimple(FrameSize, X, X_train, X_test, y_train, y_test):
    # model2 = Sequential()
    # model2.add(LSTM(256, input_shape=(FrameSize, X[0].shape[1]), return_sequences=True))
    # model2.add(Dropout(0.2))
    # model2.add(LSTM(128, return_sequences=False))
    # model2.add(Dropout(0.2))
    # model2.add(Dense(64))
    # model2.add(Dropout(0.2))
    # model2.add(Dense(2, activation='softmax'))
    #
    # model2.compile(
    #     loss='categorical_crossentropy',
    #     optimizer='Adam',
    #     metrics=['accuracy']
    # )
    #
    # # print(model.summary())
    #
    # history = model2.fit(
    #     X,
    #     y,
    #     validation_split=0.1,
    #     epochs=20,
    #     batch_size=128
    # )
    #
    # plt.clf()
    # loss = history.history['loss']
    # val_loss = history.history['val_loss']
    # epochs = range(1, len(loss) + 1)
    # plt.plot(epochs, loss, 'g', label='Training loss')
    # plt.plot(epochs, val_loss, 'y', label='Validation loss')
    # plt.title('Training and validation loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.legend()
    # fig1 = plt.gcf()
    # plt.show()
    # plt.draw()
    # fig1.savefig('loss.png', dpi=100)
    #
    # plt.clf()
    # acc = history.history['accuracy']
    # val_acc = history.history['val_accuracy']
    # plt.plot(epochs, acc, 'g', label='Training acc')
    # plt.plot(epochs, val_acc, 'y', label='Validation acc')
    # plt.title('Training and validation accuracy')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.legend()
    # fig2 = plt.gcf()
    # plt.show()
    # plt.draw()
    # fig2.savefig('acc.png', dpi=100)

def model_256_128_64_2BS(FrameSize, X, X_train, X_test, y_train, y_test, epoch):
    model = Sequential()
    # model.add(Embedding(2, 50, input_length=None))
    # model.add(LSTM(256, return_sequences=True))

    model.add(LSTM(256, input_shape=(FrameSize, X[0].shape[1]), return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=128,
        verbose=2,
        shuffle=True,
        validation_data=(X_test, y_test),
        callbacks=[ModelCheckpoint('result/One_256_128_64_2BS.h5', monitor='val_accuracy', mode='max', save_best_only=True)]
    )

    # model.save_weights("result/One_256_128_64_2BS.h5")

    plot.plot(history, "One_256_128_64_2BS")

    ROC_PR.ROC(model, X_test, y_test, "One_256_128_64_2BS")

def model_256_128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch):
    model = Sequential()
    # model.add(Embedding(2, 50, input_length=None))
    # model.add(LSTM(256, return_sequences=True))

    model.add(LSTM(256, input_shape=(FrameSize, X[0].shape[1]), return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=128,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[ModelCheckpoint('result/One_256_128_64_2.h5', monitor='val_accuracy', mode='max', save_best_only=True)]

    )

    # model.save_weights("result/One_256_128_64_2.h5")
    # plot_model(model, to_file='model_plot1.png', show_shapes=True)

    plot.plot(history, "One_256_128_64_2")

    ROC_PR.ROC(model, X_test, y_test, "One_256_128_64_2")


def model_256_128_64_2_100Ep(FrameSize, X, X_train, X_test, y_train, y_test):
    model = Sequential()
    # model.add(Embedding(2, 50, input_length=None))
    # model.add(LSTM(256, return_sequences=True))

    model.add(LSTM(256, input_shape=(FrameSize, X[0].shape[1]), return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=100,
        batch_size=128,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test)
    )

    plot.plot(history, "One_256_128_64_2_100Ep")

    ROC_PR.ROC(model, X_test, y_test, "One_256_128_64_2_100Ep")


def model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch):
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=5, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=3))
    model.add(LSTM(256, return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(0.2))
    model.add(LSTM(128, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(0.2))
    model.add(Dense(64))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='sigmoid'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='Adam',
        metrics=['accuracy']
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epoch,
        batch_size=128,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test),
        callbacks=[ModelCheckpoint('result/CNN256_LSTM128_64_2.h5', monitor='accuracy', mode='max', save_best_only=True)]
    )

    # plot_model(model, to_file='model_plot.png', show_shapes=True)

    plot.plot(history, "One_CNN256_LSTM128_64_2")

    ROC_PR.ROC(model, X_test, y_test, "One_CNN256_LSTM128_64_2", False)


# def model_256_128_64_TD_2(FrameSize, X, X_train, X_test, y_train, y_test):
#     model = Sequential()
#     # model.add(Embedding(2, 50, input_length=None))
#     # model.add(LSTM(256, return_sequences=True))
#
#     model.add(LSTM(256, input_shape=(FrameSize, X[0].shape[1]), return_sequences=True, recurrent_dropout=0.3))
#     model.add(SpatialDropout1D(0.2))
#     # model.add(tf.keras.layers.Attention()([150, X[0].shape[1]]))
#     model.add(LSTM(128, return_sequences=True, recurrent_dropout=0.3))
#     model.add(Dropout(0.2))
#     # model.add(Dense(64))
#     # model.add(Dropout(0.2))
#     model.add(TimeDistributed(Dense(2, activation='softmax')))
#
#     model.compile(
#         loss='categorical_crossentropy',
#         optimizer='Adam',
#         metrics=['accuracy']
#     )
#
#     history = model.fit(
#         X,
#         y,
#         validation_split=0.1,
#         epochs=20,
#         batch_size=128
#     )
#
#     plt.clf()
#     loss = history.history['loss']
#     val_loss = history.history['val_loss']
#     epochs = range(1, len(loss) + 1)
#     plt.plot(epochs, loss, 'g', label='Training loss')
#     plt.plot(epochs, val_loss, 'y', label='Validation loss')
#     plt.title('Training and validation loss')
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.legend()
#     fig1 = plt.gcf()
#     plt.show()
#     plt.draw()
#     fig1.savefig('lossmodel_256_128_64_TD_2.png', dpi=100)
#
#     plt.clf()
#     acc = history.history['accuracy']
#     val_acc = history.history['val_accuracy']
#     plt.plot(epochs, acc, 'g', label='Training acc')
#     plt.plot(epochs, val_acc, 'y', label='Validation acc')
#     plt.title('Training and validation accuracy')
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.legend()
#     fig2 = plt.gcf()
#     plt.show()
#     plt.draw()
#     fig2.savefig('accmodel_256_128_64_TD_2.png', dpi=100)


def run_model(df_train, labels, epoch):
    FrameSize = 200

    dfStreptomycin = labels[0]

    print(dfStreptomycin.head(10))

    X = df_train.values.tolist()
    y = dfStreptomycin.values.tolist()

    # X = [[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [0.0, 0.0], [6.0, 0.0], [1.0, 0.0], [0.0, 0.0], [0.0, 0.0],
    # [0.0, 0.0]]

    for i in range(len(y) - 1, -1, -1):
        if y[i][0] != 0.0 and y[i][0] != 1.0:
            del y[i]
            del X[i]

    y = to_categorical(y)

    for i in range(0, len(X)):
        if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
            for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
                X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

    model_CNN256_LSTM128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch)
    # model_256_128_64_2BS(FrameSize, X, X_train, X_test, y_train, y_test, epoch)
    model_256_128_64_2(FrameSize, X, X_train, X_test, y_train, y_test, epoch)


if __name__ == '__main__':
    df_train, labels = data_preprocess.process(6)
    run_model(df_train, labels, 20)

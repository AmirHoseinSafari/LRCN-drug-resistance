import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D
from keras import Sequential
from functools import partial


NUM_CLASSES = 10


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def get_model(dropout2_rate=0.2, dense_1_neurons=64, filterCNN=5, kernelCNN=3, LSTM1=256, LSTM2=128):
    model = Sequential()
    model.add(Dropout(dropout2_rate))
    model.add(Conv1D(filters=filterCNN, kernel_size=kernelCNN, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=kernelCNN))
    model.add(LSTM(LSTM1, return_sequences=True, recurrent_dropout=0.3))
    model.add(SpatialDropout1D(dropout2_rate))
    model.add(LSTM(LSTM2, return_sequences=False, recurrent_dropout=0.3))
    model.add(Dropout(dropout2_rate))
    model.add(Dense(dense_1_neurons))
    model.add(Dropout(dropout2_rate))
    model.add(Dense(12, activation='sigmoid'))
    return model


def fit_with(dropout2_rate, dense_1_neurons_x128, filterCNN, kernelCNN, LSTM1, LSTM2):
    # Create the model using a specified hyperparameters.
    dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
    LSTM1 = max(int(LSTM1 * 256), 256)
    LSTM2 = max(int(LSTM2 * 128), 128)
    kernelCNN = max(int(kernelCNN), 3)
    filterCNN = max(int(filterCNN), 4)
    if kernelCNN >= filterCNN:
        kernelCNN = filterCNN - 1

    model = get_model(dropout2_rate, dense_1_neurons, filterCNN, kernelCNN, LSTM1, LSTM2)

    # Train the model for a specified number of epochs.
    model.compile(
        loss=masked_loss_function,
        optimizer='Adam',
        metrics=[masked_accuracy]
    )
    # TODO
    # Train the model with the train dataset.
    history = model.fit(
        X_train,
        y_train,
        epochs=50,
        batch_size=128,
        shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test)
    )

    # Evaluate the model with the eval dataset.
    score = model.evaluate(X_test, y_test, steps=10, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    # Return the accuracy.
    # print(history.history['val_masked_accuracy'])
    return score[1]


X_train, X_test, y_train, y_test = 0, 0, 0, 0


def BO(X_train2, X_test2, y_train2, y_test2):
    global X_train
    X_train = X_train2
    global X_test
    X_test = X_test2
    global y_train
    y_train = y_train2
    global y_test
    y_test = y_test2

    fit_with_partial = partial(fit_with)

    fit_with_partial(dropout2_rate=0.2, dense_1_neurons_x128=1, filterCNN=5, kernelCNN=3, LSTM1=1, LSTM2=1)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'dropout2_rate': (0.1, 0.5), "dense_1_neurons_x128": (0.9, 3.1), "filterCNN": (3.9, 8.1), "kernelCNN": (2.9, 6.1), "LSTM1": (0.9, 3.1), "LSTM2": (0.9, 3.1)}

    optimizer = BayesianOptimization(
        f=fit_with_partial,
        pbounds=pbounds,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=10, n_iter=10, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("resultttttttttttttt")
    print(optimizer.max)

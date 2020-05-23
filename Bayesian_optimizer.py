import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D
from keras import Sequential
from functools import partial
import ROC_PR


NUM_CLASSES = 10


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def get_model(dropout2_rate=0.2, dense_1_neurons=64,
              dense_2_neurons=64, dense_3_neurons=64, dense_4_neurons=64, dense_5_neurons=64,
              filterCNN1=5, kernelCNN1=3, poolCNN1=3,
              filterCNN2=5, kernelCNN2=3, poolCNN2=3,
              filterCNN3=5, kernelCNN3=3, poolCNN3=3,
              filterCNN4=5, kernelCNN4=3, poolCNN4=3,
              filterCNN5=5, kernelCNN5=3, poolCNN5=3,
              LSTM1=128, LSTM2=64, LSTM3=64, LSTM4=64, LSTM5=64, i1=1, i2=2, i3=1):

    model = Sequential()
    model.add(Dropout(dropout2_rate))
    for i in range(0, i1):
        if i == 0:
            model.add(Conv1D(filters=filterCNN1, kernel_size=kernelCNN1, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN1, padding='same'))
        elif i == 1:
            model.add(Conv1D(filters=filterCNN2, kernel_size=kernelCNN2, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN2, padding='same'))
        elif i == 2:
            model.add(Conv1D(filters=filterCNN3, kernel_size=kernelCNN3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN3, padding='same'))
        elif i == 3:
            model.add(Conv1D(filters=filterCNN4, kernel_size=kernelCNN4, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN4, padding='same'))
        elif i == 4:
            model.add(Conv1D(filters=filterCNN5, kernel_size=kernelCNN5, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN5, padding='same'))

    for i in range(0, i2):
        if i == i2 - 1:
            if i == 0:
                model.add(LSTM(LSTM1, return_sequences=False, recurrent_dropout=0.3))
                model.add(Dropout(dropout2_rate))
            elif i == 1:
                model.add(LSTM(LSTM2, return_sequences=False, recurrent_dropout=0.3))
                model.add(Dropout(dropout2_rate))
            elif i == 2:
                model.add(LSTM(LSTM3, return_sequences=False, recurrent_dropout=0.3))
                model.add(Dropout(dropout2_rate))
            elif i == 3:
                model.add(LSTM(LSTM4, return_sequences=False, recurrent_dropout=0.3))
                model.add(Dropout(dropout2_rate))
            elif i == 4:
                model.add(LSTM(LSTM5, return_sequences=False, recurrent_dropout=0.3))
                model.add(Dropout(dropout2_rate))
            break
        if i == 0:
            model.add(LSTM(LSTM1, return_sequences=True, recurrent_dropout=0.3))
            model.add(SpatialDropout1D(dropout2_rate))
        elif i == 1:
            model.add(LSTM(LSTM2, return_sequences=True, recurrent_dropout=0.3))
            model.add(SpatialDropout1D(dropout2_rate))
        elif i == 2:
            model.add(LSTM(LSTM3, return_sequences=True, recurrent_dropout=0.3))
            model.add(SpatialDropout1D(dropout2_rate))
        elif i == 3:
            model.add(LSTM(LSTM4, return_sequences=True, recurrent_dropout=0.3))
            model.add(SpatialDropout1D(dropout2_rate))
        elif i == 4:
            model.add(LSTM(LSTM5, return_sequences=False, recurrent_dropout=0.3))
            model.add(Dropout(dropout2_rate))
    for i in range(0, i3):
        if i == 0:
            model.add(Dense(dense_1_neurons))
            model.add(Dropout(dropout2_rate))
        if i == 1:
            model.add(Dense(dense_2_neurons))
            model.add(Dropout(dropout2_rate))
        if i == 2:
            model.add(Dense(dense_3_neurons))
            model.add(Dropout(dropout2_rate))
        if i == 3:
            model.add(Dense(dense_4_neurons))
            model.add(Dropout(dropout2_rate))
        if i == 4:
            model.add(Dense(dense_5_neurons))
            model.add(Dropout(dropout2_rate))
    model.add(Dense(12, activation='sigmoid'))
    return model


def fit_with(dropout2_rate, dense_1_neurons_x128,
              dense_2_neurons_x128, dense_3_neurons_x128, dense_4_neurons_x128, dense_5_neurons_x128,
              filterCNN1, kernelCNN1, poolCNN1,
              filterCNN2, kernelCNN2, poolCNN2,
              filterCNN3, kernelCNN3, poolCNN3,
              filterCNN4, kernelCNN4, poolCNN4,
              filterCNN5, kernelCNN5, poolCNN5,
              LSTM1, LSTM2, LSTM3, LSTM4, LSTM5, i1, i2, i3):
    # Create the model using a specified hyperparameters.
    i1 = int(i1)
    i2 = int(i2)
    i3 = int(i3)

    dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
    dense_2_neurons = max(int(dense_2_neurons_x128 * 64), 64)
    dense_3_neurons = max(int(dense_3_neurons_x128 * 64), 64)
    dense_4_neurons = max(int(dense_4_neurons_x128 * 64), 64)
    dense_5_neurons = max(int(dense_5_neurons_x128 * 64), 64)

    LSTM1 = max(int(LSTM1 * 64), 64)
    LSTM2 = max(int(LSTM2 * 64), 64)
    LSTM3 = max(int(LSTM3 * 64), 64)
    LSTM4 = max(int(LSTM4 * 64), 64)
    LSTM5 = max(int(LSTM5 * 64), 64)

    kernelCNN1 = max(int(kernelCNN1), 3)
    filterCNN1 = max(int(filterCNN1), 4)
    poolCNN1 = max(int(poolCNN1), 4)
    if poolCNN1 > kernelCNN1:
        poolCNN1 = kernelCNN1

    kernelCNN2 = max(int(kernelCNN2), 3)
    filterCNN2 = max(int(filterCNN2), 4)
    poolCNN2 = max(int(poolCNN2), 4)
    if poolCNN2 > kernelCNN2:
        poolCNN2 = kernelCNN2

    kernelCNN3 = max(int(kernelCNN3), 3)
    filterCNN3 = max(int(filterCNN3), 4)
    poolCNN3 = max(int(poolCNN3), 4)
    if poolCNN3 > kernelCNN3:
        poolCNN3 = kernelCNN3

    kernelCNN4 = max(int(kernelCNN4), 3)
    filterCNN4 = max(int(filterCNN4), 4)
    poolCNN4 = max(int(poolCNN4), 4)
    if poolCNN4 > kernelCNN4:
        poolCNN4 = kernelCNN4

    kernelCNN5 = max(int(kernelCNN5), 3)
    filterCNN5 = max(int(filterCNN5), 4)
    poolCNN5 = max(int(poolCNN5), 4)
    if poolCNN5 > kernelCNN5:
        poolCNN5 = kernelCNN5
    # if kernelCNN >= filterCNN:
    #     kernelCNN = filterCNN - 1

    model = get_model(dropout2_rate, dense_1_neurons,
              dense_2_neurons, dense_3_neurons, dense_4_neurons, dense_5_neurons,
              filterCNN1, kernelCNN1, poolCNN1,
              filterCNN2, kernelCNN2, poolCNN2,
              filterCNN3, kernelCNN3, poolCNN3,
              filterCNN4, kernelCNN4, poolCNN4,
              filterCNN5, kernelCNN5, poolCNN5,
              LSTM1, LSTM2, LSTM3, LSTM4, LSTM5, i1, i2, i3)

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
        epochs=1,
        batch_size=128,
        # shuffle=True,
        verbose=2,
        validation_data=(X_test, y_test)
    )

    # Evaluate the model with the eval dataset.
    # score = model.evaluate(X_test, y_test, steps=10, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # Return the accuracy.
    # print(history.history['val_masked_accuracy'])
    score = ROC_PR.ROC_Score(model, X_test, y_test)
    print('area under ROC curve:', score)
    return score


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

    fit_with_partial(dropout2_rate=0.2, dense_1_neurons_x128=1, dense_2_neurons_x128=1, dense_3_neurons_x128=1,
                     dense_4_neurons_x128=1, dense_5_neurons_x128=1,
                     filterCNN1=5, kernelCNN1=3, poolCNN1=3,
                     filterCNN2=5, kernelCNN2=3, poolCNN2=3,
                     filterCNN3=5, kernelCNN3=3, poolCNN3=3,
                     filterCNN4=5, kernelCNN4=3, poolCNN4=3,
                     filterCNN5=5, kernelCNN5=3, poolCNN5=3,
                     LSTM1=1, LSTM2=1, LSTM3=1, LSTM4=1, LSTM5=1, i1=1, i2=2, i3=1)

    from bayes_opt import BayesianOptimization

    # Bounded region of parameter space
    pbounds = {'dropout2_rate': (0.1, 0.5), "dense_1_neurons_x128": (0.9, 8.1),
               "dense_2_neurons_x128": (0.9, 8.1),
               "dense_3_neurons_x128": (0.9, 8.1),
               "dense_4_neurons_x128": (0.9, 8.1),
               "dense_5_neurons_x128": (0.9, 8.1),
               "filterCNN1": (3.9, 8.1), "kernelCNN1": (2.9, 6.1), "poolCNN1": (2.9, 6.1),
               "filterCNN2": (3.9, 8.1), "kernelCNN2": (2.9, 6.1), "poolCNN2": (2.9, 6.1),
               "filterCNN3": (3.9, 8.1), "kernelCNN3": (2.9, 6.1), "poolCNN3": (2.9, 6.1),
               "filterCNN4": (3.9, 8.1), "kernelCNN4": (2.9, 6.1), "poolCNN4": (2.9, 6.1),
               "filterCNN5": (3.9, 8.1), "kernelCNN5": (2.9, 6.1), "poolCNN5": (2.9, 6.1),
               "LSTM1": (0.9, 8.1), "LSTM2": (0.9, 8.1), "LSTM3": (0.9, 8.1), "LSTM4": (0.9, 8.1),"LSTM5": (0.9, 8.1),
               "i1": (1.9, 5.1), "i2": (1.9, 5.1), "i3": (1.9, 5.1),
               }

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


if __name__ == '__main__':
    ## first time run
    # LSTM1 = 2.2410735219192306
    # LSTM2 = 3.3880372347099432
    # dense_1_neurons_x128 = 3.756725814460823
    # dropout2_rate = 0.3155266936013428
    # filterCNN = 5.660616960493838
    # kernelCNN = 5.09270240126963
    # dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
    # LSTM1 = max(int(LSTM1 * 64), 64)
    # LSTM2 = max(int(LSTM2 * 64), 64)
    # kernelCNN = max(int(kernelCNN), 3)
    # filterCNN = max(int(filterCNN), 4)
    # print(dense_1_neurons)
    # print(LSTM1)
    # print(LSTM2)
    # print(kernelCNN)
    # print(filterCNN)

    # second
    # LSTM1 = 2.19096774380784
    # LSTM2 = 7.968090930682244
    # dense_1_neurons_x128 = 1.2814343371372734
    # dropout2_rate = 0.1783805364232113
    # filterCNN = 7.859897700
    # kernelCNN = 5.59569428296302
    # dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
    # LSTM1 = max(int(LSTM1 * 64), 64)
    # LSTM2 = max(int(LSTM2 * 64), 64)
    # kernelCNN = max(int(kernelCNN), 3)
    # filterCNN = max(int(filterCNN), 4)
    #
    # print(dense_1_neurons)
    # print(LSTM1)
    # print(LSTM2)
    # print(kernelCNN)
    # print(filterCNN)
    LSTM1 = 7.222626024691774
    LSTM2 = 1.6080972035979606
    LSTM3 = 3.931974900036375
    LSTM4 = 7.796804617083613
    LSTM5 = 4.738790051805723
    dense_1_neurons_x128 = 5.881515220443408
    dense_2_neurons_x128 = 3.171712543243653
    dense_3_neurons_x128 = 5.842806679307402
    dense_4_neurons_x128 = 6.909304837661085
    dense_5_neurons_x128 = 1.031675596878181
    dropout2_rate = 0.40005772597798706
    filterCNN1 = 8.053216573407276
    filterCNN2 = 7.0422957483953255
    filterCNN3 = 5.077864766670501
    filterCNN4 = 7.214973179496251
    filterCNN5 = 4.333549227626096
    i1 = 3.3332592837628963
    i2 = 4.807505609897905
    i3 = 2.839565274795774
    kernelCNN1 = 3.8208810834763156
    kernelCNN2 = 3.3160914307784886
    kernelCNN3 = 2.9619742651849505
    kernelCNN4 = 5.072273705407651
    kernelCNN5 = 3.5772099712001886
    poolCNN1 = 3.749749309991124
    poolCNN2 = 4.473034109697082
    poolCNN3 = 3.070760144374657
    poolCNN4= 4.737176337574441
    poolCNN5 = 3.3695314396985925
    i1 = int(i1)
    i2 = int(i2)
    i3 = int(i3)

    dense_1_neurons = max(int(dense_1_neurons_x128 * 64), 64)
    dense_2_neurons = max(int(dense_2_neurons_x128 * 64), 64)
    dense_3_neurons = max(int(dense_3_neurons_x128 * 64), 64)
    dense_4_neurons = max(int(dense_4_neurons_x128 * 64), 64)
    dense_5_neurons = max(int(dense_5_neurons_x128 * 64), 64)

    LSTM1 = max(int(LSTM1 * 64), 64)
    LSTM2 = max(int(LSTM2 * 64), 64)
    LSTM3 = max(int(LSTM3 * 64), 64)
    LSTM4 = max(int(LSTM4 * 64), 64)
    LSTM5 = max(int(LSTM5 * 64), 64)

    kernelCNN1 = max(int(kernelCNN1), 3)
    filterCNN1 = max(int(filterCNN1), 4)
    poolCNN1 = max(int(poolCNN1), 4)
    if poolCNN1 > kernelCNN1:
        poolCNN1 = kernelCNN1

    kernelCNN2 = max(int(kernelCNN2), 3)
    filterCNN2 = max(int(filterCNN2), 4)
    poolCNN2 = max(int(poolCNN2), 4)
    if poolCNN2 > kernelCNN2:
        poolCNN2 = kernelCNN2

    kernelCNN3 = max(int(kernelCNN3), 3)
    filterCNN3 = max(int(filterCNN3), 4)
    poolCNN3 = max(int(poolCNN3), 4)
    if poolCNN3 > kernelCNN3:
        poolCNN3 = kernelCNN3

    kernelCNN4 = max(int(kernelCNN4), 3)
    filterCNN4 = max(int(filterCNN4), 4)
    poolCNN4 = max(int(poolCNN4), 4)
    if poolCNN4 > kernelCNN4:
        poolCNN4 = kernelCNN4

    kernelCNN5 = max(int(kernelCNN5), 3)
    filterCNN5 = max(int(filterCNN5), 4)
    poolCNN5 = max(int(poolCNN5), 4)
    if poolCNN5 > kernelCNN5:
        poolCNN5 = kernelCNN5

    print(i2)
    print(LSTM1)
    print(LSTM2)
    print(LSTM3)
    print(LSTM4)

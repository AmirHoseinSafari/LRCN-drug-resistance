import numpy as np
import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D, Flatten
from keras import Sequential
from functools import partial

from keras.utils import plot_model

import ROC_PR


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

    if i2 == 0:
        model.add(Flatten())

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
    if limited :
        model.add(Dense(7, activation='sigmoid'))
    else:
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

    return run_one_fold(model)


def run_one_fold(model):
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
        epochs=epochs,
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
    score = ROC_PR.ROC_Score(model, X_test, y_test, limited=limited)
    print('area under ROC curve:', score)
    return score


def run_k_fold(model):
    global X_train, X_test, y_train, y_test
    global X, y, check
    if check == 0:
        check = 1
        X = np.append(X_train, X_test, axis=0)
        y = np.append(y_train, y_test, axis=0)
        X_train = 0
        X_test = 0
        y_train = 0
        y_test = 0

    cvscores = []
    scores_each_drug = []
    for i in range(0, 10):
        print("fold:" + str(i))
        length = int(len(X) / 10)
        if i == 0:
            X_train_tmp= X[length:]
            X_test_tmp = X[0:length]
            y_train_tmp = y[length:]
            y_test_tmp = y[0:length]
        elif i != 9:
            X_train_tmp = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
            X_test_tmp = X[length * i:length * (i + 1)]
            y_train_tmp = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
            y_test_tmp = y[length * i:length * (i + 1)]
        else:
            X_train_tmp = X[0:length * i]
            X_test_tmp = X[length * i:]
            y_train_tmp = y[0:length * i]
            y_test_tmp = y[length * i:]

        model.compile(
            loss=masked_loss_function,
            optimizer='Adam',
            metrics=[masked_accuracy]
        )

        # plot_model(model, to_file='model_plot.png', show_shapes=True)

        history = model.fit(
            X_train_tmp,
            y_train_tmp,
            epochs=epochs,
            batch_size=128,
            # shuffle=True,
            verbose=2,
            validation_data=(X_test_tmp, y_test_tmp)
        )

        score = ROC_PR.ROC_Score(model, X_train_tmp, y_train_tmp, limited=limited)
        print('area under ROC curve:', score)
        cvscores.append(score)
        scores_each_drug.append(ROC_PR.ROC(model, X_test_tmp, y_test_tmp, ("LRCN" + "BO_delete" + str(i)), True))
    print(np.mean(cvscores))
    print(scores_each_drug)
    return np.mean(cvscores)


X_train, X_test, y_train, y_test = 0, 0, 0, 0
limited = False
X, y = 0, 0
check = 0

def BO(X_train2, X_test2, y_train2, y_test2, limited2, portion):
    global X_train
    X_train = X_train2
    global X_test
    X_test = X_test2
    global y_train
    y_train = y_train2
    global y_test
    y_test = y_test2
    global limited
    limited = limited2

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
    pbounds_LSTM = {'dropout2_rate': (0.1, 0.5), "dense_1_neurons_x128": (0.9, 8.1),
               "dense_2_neurons_x128": (0.9, 8.1),
               "dense_3_neurons_x128": (0.9, 8.1),
               "dense_4_neurons_x128": (0.9, 8.1),
               "dense_5_neurons_x128": (0.9, 8.1),
               "filterCNN1": (3.9, 8.1), "kernelCNN1": (2.9, 6.1), "poolCNN1": (2.9, 6.1),
               "filterCNN2": (3.9, 8.1), "kernelCNN2": (2.9, 6.1), "poolCNN2": (2.9, 6.1),
               "filterCNN3": (3.9, 8.1), "kernelCNN3": (2.9, 6.1), "poolCNN3": (2.9, 6.1),
               "filterCNN4": (3.9, 8.1), "kernelCNN4": (2.9, 6.1), "poolCNN4": (2.9, 6.1),
               "filterCNN5": (3.9, 8.1), "kernelCNN5": (2.9, 6.1), "poolCNN5": (2.9, 6.1),
               "LSTM1": (0.9, 8.1), "LSTM2": (0.9, 8.1), "LSTM3": (0.9, 8.1), "LSTM4": (0.9, 8.1), "LSTM5": (0.9, 8.1),
               "i1": (0, 0), "i2": (1.9, 5.1), "i3": (1.9, 5.1),
               }

    pbounds_CNN = {'dropout2_rate': (0.1, 0.5), "dense_1_neurons_x128": (0.9, 8.1),
                    "dense_2_neurons_x128": (0.9, 8.1),
                    "dense_3_neurons_x128": (0.9, 8.1),
                    "dense_4_neurons_x128": (0.9, 8.1),
                    "dense_5_neurons_x128": (0.9, 8.1),
                    "filterCNN1": (3.9, 8.1), "kernelCNN1": (2.9, 6.1), "poolCNN1": (2.9, 6.1),
                    "filterCNN2": (3.9, 8.1), "kernelCNN2": (2.9, 6.1), "poolCNN2": (2.9, 6.1),
                    "filterCNN3": (3.9, 8.1), "kernelCNN3": (2.9, 6.1), "poolCNN3": (2.9, 6.1),
                    "filterCNN4": (3.9, 8.1), "kernelCNN4": (2.9, 6.1), "poolCNN4": (2.9, 6.1),
                    "filterCNN5": (3.9, 8.1), "kernelCNN5": (2.9, 6.1), "poolCNN5": (2.9, 6.1),
                    "LSTM1": (0.9, 8.1), "LSTM2": (0.9, 8.1), "LSTM3": (0.9, 8.1), "LSTM4": (0.9, 8.1),
                    "LSTM5": (0.9, 8.1),
                    "i1": (1.9, 5.1), "i2": (0, 0), "i3": (1.9, 5.1),
                    }

    pbounds_dense = {'dropout2_rate': (0.1, 0.5), "dense_1_neurons_x128": (0.9, 8.1),
                    "dense_2_neurons_x128": (0.9, 8.1),
                    "dense_3_neurons_x128": (0.9, 8.1),
                    "dense_4_neurons_x128": (0.9, 8.1),
                    "dense_5_neurons_x128": (0.9, 8.1),
                    "filterCNN1": (3.9, 8.1), "kernelCNN1": (2.9, 6.1), "poolCNN1": (2.9, 6.1),
                    "filterCNN2": (3.9, 8.1), "kernelCNN2": (2.9, 6.1), "poolCNN2": (2.9, 6.1),
                    "filterCNN3": (3.9, 8.1), "kernelCNN3": (2.9, 6.1), "poolCNN3": (2.9, 6.1),
                    "filterCNN4": (3.9, 8.1), "kernelCNN4": (2.9, 6.1), "poolCNN4": (2.9, 6.1),
                    "filterCNN5": (3.9, 8.1), "kernelCNN5": (2.9, 6.1), "poolCNN5": (2.9, 6.1),
                    "LSTM1": (0.9, 8.1), "LSTM2": (0.9, 8.1), "LSTM3": (0.9, 8.1), "LSTM4": (0.9, 8.1),
                    "LSTM5": (0.9, 8.1),
                    "i1": (1.9, 5.1), "i2": (1.9, 5.1), "i3": (0, 0),
                    }


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
               "LSTM1": (0.9, 8.1), "LSTM2": (0.9, 8.1), "LSTM3": (0.9, 8.1), "LSTM4": (0.9, 8.1), "LSTM5": (0.9, 8.1),
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

    print("resultttttttttttttt + random")
    print(optimizer.max)

    import json

    with open('BO_result.txt', 'ab') as f:
        if limited:
            f.write("True \n".encode())
        elif not limited:
            f.write("False \n".encode())
        if portion == 0.2:
            f.write("0.2 \n".encode())
        elif portion == 0.1:
            f.write("0.1 \n".encode())
        f.write(json.dumps(optimizer.max).encode())
        f.write("\n".encode())


if __name__ == '__main__':
    # LSTM1 = 6.222775762313878
    # LSTM2 = 1.6679769249718495
    # LSTM3 = 7.432809526425034
    # LSTM4 = 4.129349772672659
    # LSTM5 = 7.296327224664441
    # dense_1_neurons_x128 = 5.508703069163429
    # dense_2_neurons_x128 = 5.912649965701209
    # dense_3_neurons_x128 = 3.5709770005266606
    # dense_4_neurons_x128 = 5.344687894046625
    # dense_5_neurons_x128 = 7.126574894033522
    # dropout2_rate = 0.43369355853937297
    # filterCNN1 = 5.839768897056432
    # filterCNN2 = 7.203637104557576
    # filterCNN3 = 5.0526585360408784
    # filterCNN4 = 5.24220234023808
    # filterCNN5 = 6.558239117341969
    # i1 = 2.6017664815079016
    # i2 = 4.626566817095913
    # i3 = 2.018877147623796
    # kernelCNN1 = 4.016571188677377
    # kernelCNN2 = 4.581395969156278
    # kernelCNN3 = 5.944186720087927
    # kernelCNN4 = 3.642934120437684
    # kernelCNN5 = 4.001393933658716
    # poolCNN1 = 4.507134496841905
    # poolCNN2 = 4.220916148606445
    # poolCNN3 = 5.648968546118207
    # poolCNN4 = 4.111476205132963
    # poolCNN5 = 4.624578852167564


    # gene based model Jun 24
    LSTM1 = 8.1
    LSTM2 = 0.9
    LSTM3 = 0.9
    LSTM4 = 8.1
    LSTM5 = 0.9
    dense_1_neurons_x128 = 0.9
    dense_2_neurons_x128 = 8.1
    dense_3_neurons_x128 = 8.1
    dense_4_neurons_x128 = 0.9
    dense_5_neurons_x128 = 0.9
    dropout2_rate = 0.1
    filterCNN1 = 8.1
    filterCNN2 = 3.9
    filterCNN3 = 3.9
    filterCNN4 = 8.1
    filterCNN5 = 8.1
    i1 = 1.9
    i2 = 1.9
    i3 = 1.9
    kernelCNN1 = 2.9
    kernelCNN2 = 6.1
    kernelCNN3 = 6.1
    kernelCNN4 = 6.1
    kernelCNN5 = 2.9
    poolCNN1 = 6.1
    poolCNN2 = 2.9
    poolCNN3 = 6.1
    poolCNN4 = 2.9
    poolCNN5 = 6.1
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

    print(dense_1_neurons)
    print(filterCNN1)
    print(poolCNN1)
    print(LSTM4)



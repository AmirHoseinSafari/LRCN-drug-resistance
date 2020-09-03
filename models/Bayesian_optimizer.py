import numpy as np
import keras.backend as K
from keras.layers import SpatialDropout1D, LSTM, Dense, Dropout, MaxPooling1D, Conv1D, Flatten
from keras import Sequential
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
            # model.add(TimeDistributed(Conv1D(filters=filterCNN1, kernel_size=kernelCNN1, activation='relu', padding='same')))
            # model.add(TimeDistributed(MaxPooling1D(pool_size=poolCNN1, padding='same')))
            model.add(Conv1D(filters=filterCNN1, kernel_size=kernelCNN1, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN1, padding='same'))
        elif i == 1:
            # model.add(TimeDistributed(Conv1D(filters=filterCNN2, kernel_size=kernelCNN2, activation='relu', padding='same')))
            # model.add(TimeDistributed(MaxPooling1D(pool_size=poolCNN2, padding='same')))
            model.add(Conv1D(filters=filterCNN2, kernel_size=kernelCNN2, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN2, padding='same'))
        elif i == 2:
            model.add(Conv1D(filters=filterCNN3, kernel_size=kernelCNN3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN3, padding='same'))
        elif i == 3:
            # model.add(TimeDistributed(Conv1D(filters=filterCNN4, kernel_size=kernelCNN4, activation='relu', padding='same')))
            # model.add(TimeDistributed(MaxPooling1D(pool_size=poolCNN4, padding='same')))
            model.add(Conv1D(filters=filterCNN4, kernel_size=kernelCNN4, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN4, padding='same'))
        elif i == 4:
            # model.add(TimeDistributed(Conv1D(filters=filterCNN5, kernel_size=kernelCNN5, activation='relu', padding='same')))
            # model.add(TimeDistributed(MaxPooling1D(pool_size=poolCNN5, padding='same')))
            model.add(Conv1D(filters=filterCNN5, kernel_size=kernelCNN5, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=poolCNN5, padding='same'))


    # model.add(TimeDistributed(Flatten()))

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
    score = ROC_PR.ROC_Score(model, X_test, y_test) #TODO
    score_test = ROC_PR.ROC_Score(model, X_test, y_test)
    score_for_each_drug = ROC_PR.ROC(model, X_test, y_test, ("LRCN" + "BO_delete"), True)
    spec_recall, prec_recall = ROC_PR.PR(model, X_test, y_test)

    print('area under ROC curve for val:', score)
    print('area under ROC curve for test:', score_test)
    print(score_for_each_drug)
    print("recall at 95 spec: ", spec_recall)
    print("precision recall: ", prec_recall)
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
    if np.mean(cvscores) > 0.97:
        model.save()
    print(scores_each_drug)
    return np.mean(cvscores)


X_train, X_test, X_val, y_train, y_test, y_val = 0, 0, 0, 0, 0, 0
limited = False
X, y = 0, 0
check = 0


def BO(X_train2, X_test2, X_val2, y_train2, y_test2, y_val2, limited2, portion):
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
        pbounds=pbounds_LSTM,
        verbose=2,  # verbose = 1 prints only when a maximum is observed, verbose = 0 is silent
        random_state=1,
    )
    optimizer.maximize(init_points=15, n_iter=15, )

    for i, res in enumerate(optimizer.res):
        print("Iteration {}: \n\t{}".format(i, res))

    print("resultttttttttttttt + LSTM")
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

    # random index
    # LSTM1 = 4.866827177648857
    # LSTM2 = 6.266407902526815
    # LSTM3 = 5.718476832862929
    # LSTM4 = 2.8074208151722275
    # LSTM5 = 1.3776108078847793
    # dense_1_neurons_x128 = 3.564606224981565
    # dense_2_neurons_x128 = 5.433966050555264
    # dense_3_neurons_x128 = 2.413252871386845
    # dense_4_neurons_x128 = 6.31983998691946
    # dense_5_neurons_x128 = 1.3790626657496277
    # dropout2_rate = 0.20412603943141638
    # filterCNN1 = 7.27996916772205
    # filterCNN2 = 4.712423987017976
    # filterCNN3 = 6.585735699695748
    # filterCNN4 = 6.103615298319681
    # filterCNN5 = 7.784193475677272
    # i1 = 2.742549665558755
    # i2 = 2.111075490188876
    # i3 = 4.252211082523742
    # kernelCNN1 = 5.37096969453839
    # kernelCNN2 = 5.805010728011276
    # kernelCNN3 = 5.882310621429879
    # kernelCNN4 = 2.9446450335219105
    # kernelCNN5 = 3.6499586755885454
    # poolCNN1 = 4.873690742405304
    # poolCNN2 = 5.936852226200372
    # poolCNN3 = 5.940563581590655
    # poolCNN4 = 4.6812902022245
    # poolCNN5 = 5.829940319252078


    # random data
    # LSTM1 = 6.649945856294683
    # LSTM2 = 2.957175732485862
    # LSTM3 = 4.429825362863479
    # LSTM4 = 5.213594215050324
    # LSTM5 = 1.0118395839660173
    # dense_1_neurons_x128 = 5.173066139015788
    # dense_2_neurons_x128 = 4.022469712724099
    # dense_3_neurons_x128 = 6.7129958079708585
    # dense_4_neurons_x128 = 3.169762582286685
    # dense_5_neurons_x128 = 7.328798701381089
    # dropout2_rate = 0.3311428861138142
    # filterCNN1 = 4.672842846835353
    # filterCNN2 = 7.209302782067118
    # filterCNN3 = 6.470530943588722
    # filterCNN4 = 4.126418942709762
    # filterCNN5 = 5.6648134560047545
    # i1 = 4.073020277009375
    # i2 = 4.839525689528083
    # i3 = 1.9012864796523437
    # kernelCNN1 = 6.025629276899391
    # kernelCNN2 = 4.105057007186478
    # kernelCNN3 = 6.016107322746632
    # kernelCNN4 = 4.8350915231169695
    # kernelCNN5 = 5.552306585538217
    # poolCNN1 = 4.739076815065927
    # poolCNN2 = 4.909843834583521
    # poolCNN3 = 3.8138441014262203
    # poolCNN4 = 4.777866690099465
    # poolCNN5 = 5.300069643848511

    # resultttttttttttttt
    # {'target': 0.8751376639312831,
    #  'params': {'LSTM1': 5.318803550972627, 'LSTM2': 5.092327267789626, 'LSTM3': 5.074300778715073,
    #             'LSTM4': 5.154371285013191, 'LSTM5': 3.922872356885941, 'dense_1_neurons_x128': 7.227699918599786,
    #             'dense_2_neurons_x128': 4.095306036812141, 'dense_3_neurons_x128': 3.7199829472200494,
    #             'dense_4_neurons_x128': 3.977630850973221, 'dense_5_neurons_x128': 4.825030405910773,
    #             'dropout2_rate': 0.1410763101705484, 'filterCNN1': 6.562086491737755, 'filterCNN2': 5.438153788692958,
    #             'filterCNN3': 6.915774801695678, 'filterCNN4': 7.988122814241818, 'filterCNN5': 4.1966508672770075,
    #             'i1': 3.8420122980426488, 'i2': 2.6054389339552078, 'i3': 2.801688290718926,
    #             'kernelCNN1': 4.577781536806731, 'kernelCNN2': 4.7988255077194335, 'kernelCNN3': 5.158934408220259,
    #             'kernelCNN4': 2.950089012124746, 'kernelCNN5': 4.292786152234941, 'poolCNN1': 3.995376607628528,
    #             'poolCNN2': 5.230000391550719, 'poolCNN3': 5.397837232287488, 'poolCNN4': 4.796942756340975,
    #             'poolCNN5': 4.984760406804859}}

    # shuffle_0
    # LSTM1 = 1.673957729563869
    # LSTM2 = 2.525107237976554
    # LSTM3 = 6.033520658755272
    # LSTM4 = 4.929962270789826
    # LSTM5 = 0.9904030571456341
    # dense_1_neurons_x128 = 1.4182148137643047
    # dense_2_neurons_x128 = 7.864389576001958
    # dense_3_neurons_x128 = 4.990323325823583
    # dense_4_neurons_x128 = 2.3637112895591312
    # dense_5_neurons_x128 = 2.7167453609063283
    # dropout2_rate = 0.3975303416300372
    # filterCNN1 = 4.720803820659139
    # filterCNN2 = 6.341707494547682
    # filterCNN3 = 7.974083954170911
    # filterCNN4 = 7.456680966258148
    # filterCNN5 = 4.907360588419862
    # i1 = 3.4800630856599355
    # i2 = 3.8838582988204147
    # i3 = 4.552738878560572
    # kernelCNN1 = 3.4017324628674697
    # kernelCNN2 = 2.9594438469677105
    # kernelCNN3 = 3.1240708599015115
    # kernelCNN4 = 4.456304354998501
    # kernelCNN5 = 4.840254277290657
    # poolCNN1 = 4.72032459867674
    # poolCNN2 = 3.915559709830914
    # poolCNN3 = 6.063571694119836
    # poolCNN4 = 4.75518470158655
    # poolCNN5 = 4.116451752395362


    # shuffle 1
    # LSTM1 = 6.564698885407315
    # LSTM2 = 1.060982465032951
    # LSTM3 = 3.6209654209979787
    # LSTM4 = 7.124037697849187
    # LSTM5 = 2.6136971901330788
    # dense_1_neurons_x128 = 1.760132915954984
    # dense_2_neurons_x128 = 1.4455184532590837
    # dense_3_neurons_x128 = 7.089940718310026
    # dense_4_neurons_x128 = 4.2974734410395445
    # dense_5_neurons_x128 = 6.234513197480954
    # dropout2_rate = 0.1135579744586698
    # filterCNN1 = 4.703651493457163
    # filterCNN2 = 6.8204044527513545
    # filterCNN3 = 7.603973180352927
    # filterCNN4 = 8.005010651929128
    # filterCNN5 = 7.40119514686825
    # i1 = 1.9454584053060493
    # i2 = 3.822902581799232
    # i3 = 2.5096139051470683
    # kernelCNN1 = 5.595781973004557
    # kernelCNN2 = 4.084095044190034
    # kernelCNN3 = 5.881216939290972
    # kernelCNN4 = 3.0406574930663557
    # kernelCNN5 = 4.350748957769083
    # poolCNN1 = 3.2419983952038853
    # poolCNN2 = 4.836168587394216
    # poolCNN3 = 3.258980522254367
    # poolCNN4 = 3.3473004702257523
    # poolCNN5 = 3.121172880147331

    # shuffle 2
    # LSTM1 = 6.327908556920974
    # LSTM2 = 7.54577665593468
    # LSTM3 = 6.0229782621249965
    # LSTM4 = 1.7947509261995858
    # LSTM5 = 1.0431369636465282
    # dense_1_neurons_x128 = 1.0887191055195788
    # dense_2_neurons_x128 = 1.1038067137497212
    # dense_3_neurons_x128 = 2.6727196867419303
    # dense_4_neurons_x128 = 7.092201230516793
    # dense_5_neurons_x128 = 4.7795836632599
    # dropout2_rate = 0.3211287914743064
    # filterCNN1 = 7.436529747910344
    # filterCNN2 = 4.421527923503627
    # filterCNN3 = 5.072571451846786
    # filterCNN4 = 6.360188940124809
    # filterCNN5 = 7.972302142942633
    # i1 = 3.695296701618272
    # i2 = 1.9596713259934175
    # i3 = 4.462024552577972
    # kernelCNN1 = 3.645517676291265
    # kernelCNN2 = 5.482736625980094
    # kernelCNN3 = 4.141154061005349
    # kernelCNN4 = 5.663333934590171
    # kernelCNN5 = 5.290789256758991
    # poolCNN1 = 4.67996874876934
    # poolCNN2 = 3.336656722114192
    # poolCNN3 = 3.091736606439077
    # poolCNN4 = 3.2882990583703595
    # poolCNN5 = 3.0425660113432373

    # time dist
    # gpu1-2672
    # resultttttttttttttt + LSTM
    # {'target': 0.9378313573180093,
    #  'params': {'LSTM1': 5.51927670441363, 'LSTM2': 3.7080555418169725, 'LSTM3': 4.39913280309775,
    #             'LSTM4': 5.251035477023807, 'LSTM5': 4.8567450348616505, 'dense_1_neurons_x128': 7.5685062722864656,
    #             'dense_2_neurons_x128': 7.514880736561963, 'dense_3_neurons_x128': 3.7431044130495947,
    #             'dense_4_neurons_x128': 7.835490204773125, 'dense_5_neurons_x128': 2.1524808000993434,
    #             'dropout2_rate': 0.15053180777585495, 'filterCNN1': 4.4673324637962715, 'filterCNN2': 6.023781095842965,
    #             'filterCNN3': 3.9904041821516314, 'filterCNN4': 7.881474887114259, 'filterCNN5': 7.373884978917076,
    #             'i1': 1.9480607383747914, 'i2': 2.463828017840177, 'i3': 2.962603437973878,
    #             'kernelCNN1': 3.3191899033949337, 'kernelCNN2': 5.490370214805063, 'kernelCNN3': 4.003157288586539,
    #             'kernelCNN4': 5.908343943466774, 'kernelCNN5': 4.762445375830656, 'poolCNN1': 5.712262350117901,
    #             'poolCNN2': 5.60315022525511, 'poolCNN3': 5.797255419867677, 'poolCNN4': 4.371616850613782,
    #             'poolCNN5': 4.648309811265194}}
    # shuffle 3
    # LSTM1 = 6.857932101285899
    # LSTM2 = 7.8247565029282535
    # LSTM3 = 7.93855364364796
    # LSTM4 = 1.0370375040103659
    # LSTM5 = 5.222751913097705
    # dense_1_neurons_x128 = 1.269912297670297
    # dense_2_neurons_x128 = 4.180379847343162
    # dense_3_neurons_x128 = 6.0659390964767335
    # dense_4_neurons_x128 = 1.3230654491114158
    # dense_5_neurons_x128 = 2.852582606327128
    # dropout2_rate = 0.3584182599371434
    # filterCNN1 = 8.082383676023122
    # filterCNN2 = 7.43790273178718
    # filterCNN3 = 6.6151243767460155
    # filterCNN4 = 6.960156939837497
    # filterCNN5 = 4.7347663171361765
    # i1 = 2.0618254495008244
    # i2 = 2.7148722021526126
    # i3 = 3.2193678012456752
    # kernelCNN1 = 5.5634203327727
    # kernelCNN2 = 5.842728005722684
    # kernelCNN3 = 3.6940066126250337
    # kernelCNN4 = 5.545767440087675
    # kernelCNN5 = 5.548638745871845
    # poolCNN1 = 5.788653578553341
    # poolCNN2 = 5.9175930951933955
    # poolCNN3 = 4.538686028476428
    # poolCNN4 = 5.927841920999326
    # poolCNN5 = 5.0119463100374215

    # LSTM1 = 5.318803550972627
    # LSTM2 = 5.092327267789626
    # LSTM3 = 5.074300778715073
    # LSTM4 = 5.154371285013191
    # LSTM5 = 3.922872356885941
    # dense_1_neurons_x128 = 7.227699918599786
    # dense_2_neurons_x128 = 4.095306036812141
    # dense_3_neurons_x128 = 3.7199829472200494
    # dense_4_neurons_x128 = 3.977630850973221
    # dense_5_neurons_x128 = 4.825030405910773
    # dropout2_rate = 0.1410763101705484
    # filterCNN1 = 6.562086491737755
    # filterCNN2 = 5.438153788692958
    # filterCNN3 = 6.915774801695678
    # filterCNN4 = 7.988122814241818
    # filterCNN5 = 4.1966508672770075
    # i1 = 3.8420122980426488
    # i2 = 2.6054389339552078
    # i3 = 2.801688290718926
    # kernelCNN1 = 4.577781536806731
    # kernelCNN2 = 4.7988255077194335
    # kernelCNN3 = 5.158934408220259
    # kernelCNN4 = 2.950089012124746
    # kernelCNN5 = 4.292786152234941
    # poolCNN1 = 3.995376607628528
    # poolCNN2 = 5.230000391550719
    # poolCNN3 = 5.397837232287488
    # poolCNN4 = 4.796942756340975
    # poolCNN5 = 4.984760406804859

    # LSTM1 =  5.51927670441363
    #     # LSTM2 =  3.7080555418169725
    #     # LSTM3 =  4.39913280309775
    #     # LSTM4 =  5.251035477023807
    #     # LSTM5 =  4.8567450348616505
    #     # dense_1_neurons_x128 =  7.5685062722864656
    #     # dense_2_neurons_x128 =  7.514880736561963
    #     # dense_3_neurons_x128 =  3.7431044130495947
    #     # dense_4_neurons_x128 =  7.835490204773125
    #     # dense_5_neurons_x128 =  2.1524808000993434
    #     # dropout2_rate =  0.15053180777585495
    #     # filterCNN1 =  4.4673324637962715
    #     # filterCNN2 =  6.023781095842965
    #     # filterCNN3 =  3.9904041821516314
    #     # filterCNN4 =  7.881474887114259
    #     # filterCNN5 =  7.373884978917076
    #     # i1 =  1.9480607383747914
    #     # i2 =  2.463828017840177
    #     # i3 =  2.962603437973878
    #     # kernelCNN1 =  3.3191899033949337
    #     # kernelCNN2 =  5.490370214805063
    #     # kernelCNN3 =  4.003157288586539
    #     # kernelCNN4 =  5.908343943466774
    #     # kernelCNN5 =  4.762445375830656
    #     # poolCNN1 =  5.712262350117901
    #     # poolCNN2 =  5.60315022525511
    #     # poolCNN3 =  5.797255419867677
    #     # poolCNN4 =  4.371616850613782
    #     # poolCNN5 =  4.648309811265194

    # LSTM1 =  8.1
    # LSTM2 =  0.9
    # LSTM3 =  0.9
    # LSTM4 =  6.9983173772440965
    # LSTM5 =  0.9
    # dense_1_neurons_x128 =  8.1
    # dense_2_neurons_x128 =  8.1
    # dense_3_neurons_x128 =  0.9
    # dense_4_neurons_x128 =  0.9
    # dense_5_neurons_x128 =  0.9
    # dropout2_rate =  0.1
    # filterCNN1 =  3.9
    # filterCNN2 =  3.9
    # filterCNN3 =  8.1
    # filterCNN4 =  3.9
    # filterCNN5 =  8.1
    # i1 =  1.9
    # i2 =  1.9
    # i3 =  1.9
    # kernelCNN1 =  6.1
    # kernelCNN2 =  6.1
    # kernelCNN3 =  6.1
    # kernelCNN4 =  6.1
    # kernelCNN5 =  6.1
    # poolCNN1 =  6.1
    # poolCNN2 =  2.9
    # poolCNN3 =  2.9
    # poolCNN4 =  2.9
    # poolCNN5 =  6.1


    # pure LSTM
    LSTM1 = 5.559974417023026
    LSTM2 = 7.082064653671237
    LSTM3 = 5.372471089054764
    LSTM4 = 6.802611752726484
    LSTM5 = 1.7695998006449467
    dense_1_neurons_x128 = 5.6152943552398895
    dense_2_neurons_x128 = 3.4373195170534263
    dense_3_neurons_x128 = 3.59536032165431
    dense_4_neurons_x128 = 2.300509984026279
    dense_5_neurons_x128 = 3.8161670087956545
    dropout2_rate = 0.18725148040207906
    i1 = 0.0
    i2 = 2.417719456185797
    i3 = 4.1624599809776335

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

    print(i1)
    print(i2)
    print(i3)
    print("__")
    print(dropout2_rate)
    print("__")
    print(filterCNN1)
    print(kernelCNN1)
    print(poolCNN1)
    print("__")
    print(filterCNN2)
    print(kernelCNN2)
    print(poolCNN2)
    print("__")
    print(filterCNN3)
    print(kernelCNN3)
    print(poolCNN3)
    print("__")
    print(LSTM1)
    print(LSTM2)
    print(LSTM3)
    print(LSTM4)
    print("__")
    print(dense_1_neurons)
    print(dense_2_neurons)
    print(dense_3_neurons)
    print(dense_4_neurons)





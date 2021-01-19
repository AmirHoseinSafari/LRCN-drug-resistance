import pickle

import keras
import numpy as np
import pandas as pd
import keras.backend as K
from evaluations import ROC_PR

path = '../Data/'


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def load_data():
    test = pd.read_csv(path + 'bccdc_data.csv', header=None)
    test.set_index(test.columns[0], inplace=True, drop=True)

    label = pd.read_csv(path + 'bccdc_data_label.csv', header=None)

    return test, label


def prepare_date(features, label):
    FrameSize = 200

    y = label.values.tolist()
    # for i in range(0, len(label)):
    #     label[i] = label[i].values.tolist()


    X = features.values.tolist()

    for i in range(0, len(X)):
        if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
            X[i] = X[i][0:((len(X[i]) // FrameSize) * FrameSize)]
            # print((((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i])))
            # for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
            #     X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)
    return X, y, FrameSize


def prepare_date_wnd(features, label):
    FrameSize = 200

    y = label.values.tolist()

    y = np.array(y)
    features = np.array(features)
    #
    # for i in range(0, len(features)):
    #     for j in range(4187-3967):
    #         np.concatenate(features[i], [0])
    return features, y, FrameSize


def prepare_data_ml(features, label):
    FrameSize = 200


    y = label.values.tolist()

    y = np.array(y)
    features = np.array(features)

    return features, y, FrameSize


def lrcn_test(X, y):
    auc = []
    pr = []
    sr = []
    for i in range(0, 10):
        model = keras.models.load_model('../saved_models/LRCN/lrcn_' + str(i) + '.h5',
                                        custom_objects={'masked_loss_function': masked_loss_function,
                                                        'masked_accuracy': masked_accuracy})
        score_for_each_drug = ROC_PR.ROC(model, X, y, ("LRCN" + "BO_delete"), True, bccdc=True)
        spec_recall, prec_recall = ROC_PR.PR(model, X, y, bccdc=True)

        # print('AUC-ROC:', score_for_each_drug)
        # print("recall at 95 spec: ", spec_recall)
        # print("precision recall: ", prec_recall)
        auc.append(score_for_each_drug)
        pr.append(prec_recall)
        sr.append(spec_recall)
    print(auc)
    print(pr)
    print(sr)


def wnd_test(X, y):
    auc = []
    pr = []
    sr = []
    X_val2 = X.tolist()
    for i in range(0, len(X_val2)):
        X_val2[i] = X_val2[i][0:3967]
    X = np.array(X_val2)
    for i in range(1, 11):
        model = keras.models.load_model('../saved_models/WnD/WnD' + str(i) + '.h5',
                                        custom_objects={'masked_loss_function': masked_loss_function,
                                                        'masked_accuracy': masked_accuracy})
        score_for_each_drug = ROC_PR.ROC(model, X, y, ("wide-n-deep" + "BO_delete"), True, bccdc=True)
        spec_recall, prec_recall = ROC_PR.PR(model, X, y, bccdc=True)

        # print('AUC-ROC:', score_for_each_drug)
        # print("recall at 95 spec: ", spec_recall)
        # print("precision recall: ", prec_recall)
        auc.append(score_for_each_drug)
        pr.append(prec_recall)
        sr.append(spec_recall)
    print(auc)
    print(pr)
    print(sr)


def rf_test(X,y):
    auc = []
    pr = []
    sr = []
    drugs = [0, 1, 2, 6, 8]

    # for i in range(0, len(y[0])):
    #     X_val2 = X.tolist()
    #     y_val2 = y[:, i]
    #     y_val2 = y_val2.tolist()
    #
    #     for i2 in range(len(y_val2) - 1, -1, -1):
    #         if y_val2[i2] != 0.0 and y_val2[i2] != 1.0:
    #             del y_val2[i2]
    #             del X_val2[i2]

    for i in range(0, 8):
        a, p, s = [], [], []
        for j in range(0, len(drugs)):
            X_val2 = X.tolist()
            for i2 in range(0, len(X_val2)):
                X_val2[i2] = X_val2[i2][0:3967]
            y_val2 = y[:, j]
            y_val2 = y_val2.tolist()

            for i2 in range(len(y_val2) - 1, -1, -1):
                if y_val2[i2] != 0.0 and y_val2[i2] != 1.0:
                    del y_val2[i2]
                    del X_val2[i2]

            model = pickle.load(open('../saved_models/RF/rf'+str(drugs[j])+'_' + str(i) + '.sav', 'rb'))
            score_test, score_sr, score_pr = ROC_PR.ROC_ML(model, X_val2, y_val2, "RF", 0, rf=True)

            a.append(score_test)
            p.append(score_pr)
            s.append(score_sr)
        auc.append(a)
        pr.append(p)
        sr.append(a)
    print(auc)
    print(pr)
    print(sr)


if __name__ == '__main__':
    test, label = load_data()
    print(test.shape)
    test = test.loc[:, (test != 0).any(axis=0)]
    print(test.shape)
    X, y, FrameSize = prepare_date_wnd(test, label)
    wnd_test(X, y)


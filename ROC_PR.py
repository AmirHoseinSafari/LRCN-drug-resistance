from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt
import numpy as np

num_of_drugs = 12


def ROC(model, X_test, y_test, name, multi=False, limited=False):
    y_pred_keras_tmp = model.predict(X_test)
    y_pred_keras = []
    y_test_tmp = []
    scores = []
    global num_of_drugs
    if limited:
        num_of_drugs = 7

    if multi == False:
        for i in range(0, len(y_pred_keras_tmp)):
            y_pred_keras.append(y_pred_keras_tmp[i][1])
            y_test_tmp.append(y_test[i][1])
        ROC_maker(y_test_tmp, y_pred_keras, name)
    else:
        for i in range(0, num_of_drugs):  # len(y_test[0])):
            y_test_tmp = y_test[:, i]
            y_pred_keras = y_pred_keras_tmp[:, i]
            # bug? cahnge i2 to i
            i2 = 0
            while i2 < len(y_test_tmp):
                if y_test_tmp[i2] != 0 and y_test_tmp[i2] != 1:
                    y_test_tmp = np.delete(y_test_tmp, i2)
                    y_pred_keras = np.delete(y_pred_keras, i2)
                else:
                    i2 = i2 + 1
            try:
                if i != 0:
                    if i < num_of_drugs - 1:
                        scores.append(ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), False, False))
                    else:
                        scores.append(ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), False, True))
                else:
                    scores.append(ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), True, False))

            except():
                print("error on " + i + " " + y_test_tmp)
        y_test_tmp = []
        y_pred_keras = []
        for i in range(0, num_of_drugs):  # len(y_test[0])):
            y_test_tmp.extend(y_test[:, i])
            # print(y_test_tmp)
            y_pred_keras.extend(y_pred_keras_tmp[:, i])
        i = 0
        while i < len(y_test_tmp):
            if y_test_tmp[i] != 0 and y_test_tmp[i] != 1:
                y_test_tmp = np.delete(y_test_tmp, i)
                y_pred_keras = np.delete(y_pred_keras, i)
            else:
                i = i + 1
        ROC_maker(y_test_tmp, y_pred_keras, name + " _ All", True)
        # fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
        # auc_keras = auc(fpr_keras, tpr_keras)
        # print(auc_keras)
        return scores


def ROC_Score(model, X_test, y_test):
    y_pred_keras_tmp = model.predict(X_test)
    y_pred_keras = []
    y_test_tmp = []

    for i in range(0, num_of_drugs):
        y_test_tmp = y_test[:, i]
        y_pred_keras = y_pred_keras_tmp[:, i]
        i2 = 0
        while i2 < len(y_test_tmp):
            if y_test_tmp[i2] != 0 and y_test_tmp[i2] != 1:
                y_test_tmp = np.delete(y_test_tmp, i2)
                y_pred_keras = np.delete(y_pred_keras, i2)
            else:
                i2 = i2 + 1
    y_test_tmp = []
    y_pred_keras = []
    for i in range(0, num_of_drugs):
        y_test_tmp.extend(y_test[:, i])
        y_pred_keras.extend(y_pred_keras_tmp[:, i])
    i = 0
    while i < len(y_test_tmp):
        if y_test_tmp[i] != 0 and y_test_tmp[i] != 1:
            y_test_tmp = np.delete(y_test_tmp, i)
            y_pred_keras = np.delete(y_pred_keras, i)
        else:
            i = i + 1
    fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)
    # print(auc_keras)
    return auc_keras


def ROC_maker(y_test_tmp, y_pred_keras, name, clear=True, save=True):
    # print(y_test_tmp)
    fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    if clear:
        plt.clf()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='area = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve _ ' + name)
    plt.legend(loc='best')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    if save:
        fig1.savefig('result/ROC_' + name + '.png', dpi=100)
    return auc_keras
    # Zoom in view of the upper left corner.
    # plt.figure(2)
    # plt.xlim(0, 0.2)
    # plt.ylim(0.8, 1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve (zoomed in at top left)')
    # plt.legend(loc='best')
    # fig2 = plt.gcf()
    # plt.show()
    # plt.draw()
    # fig2.savefig('ROC_Zoom_' + name + '.png', dpi=100)


def ROC_ML(model, X_test, y_test, name, i):
    y_pred_keras_tmp = model.decision_function(X_test)
    fpr_keras, tpr_keras, _ = roc_curve(y_test, y_pred_keras_tmp)
    auc_keras = auc(fpr_keras, tpr_keras)

    if i == 0 and name == "SVM":
        plt.clf()
    if i == 6 and name == "SVM":
        plt.clf()
    plt.figure(i)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label=name + str(i) + ' = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve _ ' + name)
    plt.legend(loc='best')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('result/ROC_' + name + str(i) + '.png', dpi=100)
    return auc_keras
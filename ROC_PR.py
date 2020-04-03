from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from matplotlib import pyplot as plt

num_of_drugs = 12

def ROC(model, X_test, y_test, name, multi=False):
    print(y_test)
    # print("____________")
    # print(X_test)
    # print(len(X_test))
    y_pred_keras_tmp = model.predict(X_test)
    print(y_pred_keras_tmp)
    y_pred_keras = []
    y_test_tmp = []
    if multi == False:
        for i in range(0, len(y_pred_keras_tmp)):
            y_pred_keras.append(y_pred_keras_tmp[i][1])
            y_test_tmp.append(y_test[i][1])
        ROC_maker(y_test_tmp, y_pred_keras, name)
    else:
        for i in range(0, num_of_drugs):  # len(y_test[0])):
            y_test_tmp = y_test[:, i]
            # print(y_test_tmp)
            y_pred_keras = y_pred_keras_tmp[:, i]
            if i != 0:
                ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), False)
            else:
                ROC_maker(y_test_tmp, y_pred_keras, name + " _ " + str(i), True)
        y_test_tmp = []
        y_pred_keras = []
        for i in range(0, num_of_drugs):  # len(y_test[0])):
            y_test_tmp.extend(y_test[:, i])
            # print(y_test_tmp)
            y_pred_keras.extend(y_pred_keras_tmp[:, i])
        # print("_____")
        # print(y_test_tmp)
        # print(y_pred_keras)
        ROC_maker(y_test_tmp, y_pred_keras, name + " _ All" , True)


def ROC_maker(y_test_tmp, y_pred_keras, name, clear=True):
    fpr_keras, tpr_keras, _ = roc_curve(y_test_tmp, y_pred_keras)
    auc_keras = auc(fpr_keras, tpr_keras)

    if clear:
        plt.clf()
    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='Keras (area = {:.3f})'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve _ ' + name)
    plt.legend(loc='best')
    fig1 = plt.gcf()
    plt.show()
    plt.draw()
    fig1.savefig('result/ROC_' + name + '.png', dpi=100)
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

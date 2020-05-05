# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import ROC_PR
from sklearn.multiclass import OneVsRestClassifier

res = []

def svm(X, y, i):
    global res
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    ROC_PR.ROC_ML(svm_model_linear, X_test, y_test, "SVM", i)
    accuracy = svm_model_linear.score(X_test, y_test)
    print(accuracy)
    res.append(accuracy)

def lr(X, y, i):
    global res
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    from sklearn.linear_model import LogisticRegression
    lr_model_linear = LogisticRegression(C=1).fit(X_train, y_train)
    ROC_PR.ROC_ML(lr_model_linear, X_test, y_test, "LR", i)

    accuracy = lr_model_linear.score(X_test, y_test)
    print(accuracy)
    res.append(accuracy)

def model_run(df_train, labels):
    # dividing X, y into train and test data
    global res
    # TODO check before run
    for i in range(0, len(labels)):
        print(i)
        res.append(i)
        print(res)
        dfCurrentDrug = labels[i]
        X = df_train.values.tolist()
        y = dfCurrentDrug.values.tolist()
        for i2 in range(len(y) - 1, -1, -1):
            if y[i2][0] != 0.0 and y[i2][0] != 1.0:
                del y[i2]
                del X[i2]
        svm(X, y, i)
        lr(X, y, i)
    f = open('result/mlResult.txt', 'w')
    for ele in res:
        f.write(str(ele) + '\n')
    f.close()


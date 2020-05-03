# importing necessary libraries
from sklearn import datasets
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

res = []

def svm(X, y):
    global res
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training a linear SVM classifier
    from sklearn.svm import SVC
    svm_model_linear = SVC(kernel='linear', C=1).fit(X_train, y_train)
    svm_predictions = svm_model_linear.predict(X_test)

    accuracy = svm_model_linear.score(X_test, y_test)
    print(accuracy)
    res.append(accuracy)
    # # creating a confusion matrix
    # cm = confusion_matrix(y_test, svm_predictions)
    # print(cm)


def lr(X, y):
    global res
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    from sklearn.linear_model import LogisticRegression
    lr_model_linear = LogisticRegression(C=1).fit(X_train, y_train)

    accuracy = lr_model_linear.score(X_test, y_test)
    print(accuracy)
    res.append(accuracy)

def model_run(df_train, labels):
    # dividing X, y into train and test data
    global res
    for i in range(0, 2):
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
        svm(X, y)
        lr(X, y)
    f = open('mlResult.txt', 'w')
    for ele in res:
        f.write(str(ele) + '\n')
    f.close()


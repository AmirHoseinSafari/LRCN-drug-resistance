import csv
import multiprocessing
import re

from sklearn.model_selection import train_test_split
from functools import partial
from itertools import repeat
from multiprocessing import Pool, freeze_support
from feature_importance.base_approach import load_model
from models.model_gene_based import prepare_data
import numpy as np


offset = 1

def lime_importance(model, X, y, fold, instance_index=0):
    from lime import lime_tabular
    data_columns = []
    print("X shape")
    print(X.shape)
    for i in range(len(X[0])):
        data_columns.append(str(i))
    print(len(data_columns))
    explainer = lime_tabular.RecurrentTabularExplainer(X, training_labels=y, feature_names=data_columns)
    exp = explainer.explain_instance(X[instance_index], model.predict, num_features=150, labels=(1,))
    feuture_lists = exp.as_list()
    # exp.save_to_file("lime" + str(fold) + "_" + str(instance_index) + ".html")
    # print(feuture_lists[0][0])
    return feuture_lists


def features_processor(feature, res):
    fold_res = []
    for i in range(4001):
        fold_res.append(0)

    print(feature)

    for i in range(0, len(feature)):
        name = feature[i][0]
        numbers = re.findall(r'\d+', name)
        index = int(numbers[0]) * 200 + int(numbers[1])
        if feature[i][1] < 0:
            fold_res[index] = -1 * feature[i][1]
        else:
            fold_res[index] = feature[i][1]

    res.append(fold_res)
    return res


# def prepare_data(features, label):
#     FrameSize = 200
#
#     y = []
#     for i in range(0, len(label)):
#         label[i] = label[i].values.tolist()
#
#     for j in range(0, len(label[0])):
#         tmp = []
#         for i in range(0, len(label)):
#             if label[i][j][0] != 0.0 and label[i][j][0] != 1.0:
#                 tmp.extend([-1])
#             else:
#                 tmp.extend(label[i][j])
#         y.append(tmp)
#
#     X = features.values.tolist()
#
#     for i in range(0, len(X)):
#         if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
#             for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
#                 X[i].append(0)
#         X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))
#
#     X = np.array(X)
#     y = np.array(y)
#     return X, y, FrameSize


def main_function(df_train, labels):
    X, y, FrameSize = prepare_data(df_train, labels)
    feature_importance_score = []
    global offset

    for complexity in range(1, 9):
        feature_importance_score = []
        for i in range(0, 10):
            print("fold: " + str(i))
            length = int(len(X) / 10)
            if i == 0:
                X_train = X[length:]
                X_test = X[0:length]
                y_train = y[length:]
                y_test = y[0:length]
            elif i != 9:
                X_train = np.append(X[0:length * i], X[length * (i + 1):], axis=0)
                X_test = X[length * i:length * (i + 1)]
                y_train = np.append(y[0:length * i], y[length * (i + 1):], axis=0)
                y_test = y[length * i:length * (i + 1)]
            else:
                X_train = X[0:length * i]
                X_test = X[length * i:]
                y_train = y[0:length * i]
                y_test = y[length * i:]

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1,
                                                              shuffle=False)

            # i2 = []
            # for i22 in range(len(X_train) - 1):
            #     i2.append(i22)
            # model_current = load_model(i, complexity)
            # with multiprocessing.Pool(processes=250) as pool:
            #
            #     pool.map(partial(lime_importance, model=model_current, X=X_train, y=y_train,
            #                                 res=feature_importance_score, fold=i), i2)

            for i2 in range(int(len(X_train)/25 - 1)):
                features = lime_importance(model=load_model(i, complexity), X=X_train, y=y_train, fold=i, instance_index=25*i2+offset)
                print("len features")
                print(len(features))
                print("before:")
                print(len(feature_importance_score))
                features_processor(features, feature_importance_score)
                print("after:")
                print(len(feature_importance_score))

        with open(str(offset) + "_" + "feature_scores_lime_train_" + str(complexity) + ".csv", "w+") as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(feature_importance_score)
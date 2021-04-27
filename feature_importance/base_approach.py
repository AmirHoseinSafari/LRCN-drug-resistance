import csv

from sklearn.model_selection import train_test_split

from evaluations import ROC_PR
import keras.backend as K
from models.model_gene_based import prepare_data
from tensorflow import keras

import numpy as np


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


def decrease_score(model, score, X_test, y_test):
        X_test2 = np.array(X_test).astype(np.float)
        y_test2 = np.array(y_test).astype(np.float)
        new_score = ROC_PR.ROC_Score(model, X_test2, y_test2)
        return score - new_score


def run_ELI5(model, X_train, X_test, X_val, y_train, y_test, y_val):
    X_train2 = np.array(X_train).astype(np.float)
    X_test2 = np.array(X_test).astype(np.float)
    X_val2 = np.array(X_val).astype(np.float)

    y_train2 = np.array(y_train).astype(np.float)
    y_test2 = np.array(y_test).astype(np.float)
    y_val2 = np.array(y_val).astype(np.float)

    score = ROC_PR.ROC_Score(model, X_val2, y_val2)
    score_test = ROC_PR.ROC_Score(model, X_test2, y_test2)
    # score_for_each_drug = ROC_PR.ROC(model, X_test2, y_test2, ("LRCN" + "BO_delete"), True)
    spec_recall, prec_recall = ROC_PR.PR(model, X_test2, y_test2)

    print('area under ROC curve for val:', score)
    print('area under ROC curve for test:', score_test)
    print("recall at 95 spec: ", spec_recall)
    print("precision recall: ", prec_recall)

    def score(X_test, y_test):
        return ROC_PR.ROC_Score(model, X_test, y_test)

    from eli5.permutation_importance import get_score_importances

    feature_score = []

    for i in range(0, len(X_test2[0])):
        lst = []
        lst.append(i)
        base_score, score_decreases = get_score_importances(score, X_test2, y_test2, n_iter=1, columns_to_shuffle=lst)
        feature_importances = np.mean(score_decreases, axis=0)
        feature_score.append(feature_importances[0])
        print(i)

    print(feature_score)


def load_model(i, complexity):
    new_model = keras.models.load_model('saved_models/LRCN/LRCN' + str(complexity) + '_' + str(i) + '.h5',
                                        custom_objects={'masked_loss_function': masked_loss_function,
                                                        'masked_accuracy': masked_accuracy})

    # new_model = keras.models.load_model('asd.h5',
    #                                     custom_objects={'masked_loss_function': masked_loss_function,
    #                                                     'masked_accuracy': masked_accuracy})
    return new_model


def prepare_data_shuffle(features, label, iter_num, column_index):
    FrameSize = 200

    y = []
    # for i in range(0, len(label)):
    #     label[i] = label[i].values.tolist()

    for j in range(0, len(label[0])):
        tmp = []
        for i in range(0, len(label)):
            if label[i][j][0] != 0.0 and label[i][j][0] != 1.0:
                tmp.extend([-1])
            else:
                tmp.extend(label[i][j])
        y.append(tmp)

    # print(features[column_index])
    if iter_num == 0:
        features[column_index].values[:] = -1
    else:
        features[column_index] = np.random.permutation(features[column_index].values)
    # print(features[column_index])

    X = features.values.tolist()

    for i in range(0, len(X)):
        if len(X[i]) < ((len(X[i]) // FrameSize + 1) * FrameSize):
            for j in range(0, (((len(X[i]) // FrameSize + 1) * FrameSize) - len(X[i]))):
                X[i].append(0)
        X[i] = np.reshape(X[i], (FrameSize, len(X[i]) // FrameSize))

    X = np.array(X)
    y = np.array(y)
    return X, y, FrameSize


def original_score(df_train, labels):
    X, y, FrameSize = prepare_data(df_train, labels)

    scores = []

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
        model = load_model(i)

        X_test2 = np.array(X_test).astype(np.float)
        y_test2 = np.array(y_test).astype(np.float)

        scores.append(ROC_PR.ROC_Score(model, X_test2, y_test2))

    return scores


def run_feature_importance(df_train, labels):
    scores = original_score(df_train, labels)

    num_iter = 5

    feature_importance_score = []

    for i in range(0, 10):
        print("fold: " + str(i), flush=True)
        model = load_model(i)
        fold_scores = []
        for k in range(0, len(df_train)):
            print(k, flush=True)
            iter_scores = []
            for j in range(0, num_iter):
                labels_temp = labels.copy()
                df_train_temp = df_train.copy()
                X, y, FrameSize = prepare_data_shuffle(df_train_temp, labels_temp, j, column_index=k+1)
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

                iter_scores.append(decrease_score(model, scores[i], X_test, y_test))
            fold_scores.append(np.mean(iter_scores, axis=0))
        feature_importance_score.append(fold_scores)

    with open("feature_scores.csv", "w+") as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(feature_importance_score)


def kth_largest_element (k, array):
    index = []
    value = []
    for j in range(0, k):
        max = -1
        max_index = -1
        for i in range(0, len(array)):
            if array[i] > max:
                max = array[i]
                max_index = i
        array[max_index] = -1
        index.append(max_index)
        value.append(max)

    return value, index


def find_the_portion(imp_index, feature_index):
    intersect = []
    for i in range(0, len(imp_index)):
        for j in range(0, len(feature_index)):
            if feature_index[j] + 2 >= imp_index[i] >= feature_index[j] - 2:
                intersect.append(feature_index[j])
                break

    return intersect


def find_feature_importance(file_name='feature_importance/feature_scores_1_iter.csv', k=100):
    imp_index = [4, 5, 17, 52, 123, 206, 207, 294, 363, 364, 429, 510, 514, 598, 678, 679, 680, 689, 690, 691, 715, 716, 724, 725, 726, 730, 752, 1080, 1081, 1256, 1259, 1276, 1296, 1350, 1363, 1405, 1411, 1417, 1419, 1545, 1596, 1597, 1736, 1751, 1789, 1817, 1827, 1911, 1912, 1985, 2041, 2042, 2116, 2126, 2186, 2213, 2276, 2402, 2587, 2600, 2620, 2697, 2716, 2863, 2923, 2947, 2956, 2957, 2973, 2976, 3108, 3129, 3130, 3131, 3132, 3138, 3183, 3216, 3253, 3295, 3384, 3416, 3419, 3486, 3487, 3489, 3491, 3656, 3690, 3782, 3801, 3834, 3840, 3847, 4010, 4051, 4052, 4053, 4054, 4066, 4116, 4117, 4181]
    missed_indexes = [76, 137, 208, 513, 647, 883, 1242, 1310, 1360, 1398, 1485, 1852, 2023, 2088, 2100, 2192, 2204, 2309, 2317, 2325, 2400, 2437, 2564, 2657, 2840, 3513, 3514, 3611, 3651, 3807, 3808, 3904, 3973, 3984, 3985, 3986, 3987, 3988, 3989, 3990, 3991, 3992, 3993, 3994, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4005, 4006, 4007, 4008, 4009, 4010, 4011, 4012, 4013, 4015, 4016, 4017, 4018, 4019, 4020, 4021, 4022, 4023, 4024, 4026, 4027, 4028, 4029, 4030, 4031, 4032, 4033, 4034, 4035, 4036, 4037, 4038, 4039, 4040, 4041, 4042, 4043, 4044, 4045, 4046, 4047, 4048, 4049, 4050, 4051, 4052, 4053, 4054, 4055, 4056, 4057, 4058, 4059, 4060, 4061, 4062, 4063, 4064, 4065, 4066, 4067, 4068, 4069, 4071, 4072, 4073, 4074, 4075, 4076, 4077, 4078, 4079, 4080, 4081, 4082, 4083, 4084, 4085, 4086, 4087, 4088, 4089, 4090, 4091, 4092, 4093, 4094, 4095, 4096, 4097, 4098, 4099, 4100, 4101, 4102, 4103, 4104, 4105, 4106, 4107, 4108, 4109, 4110, 4111, 4112, 4113, 4115, 4116, 4117, 4118, 4119, 4120, 4121, 4122, 4123, 4124, 4125, 4126, 4127, 4128, 4129, 4130, 4131, 4132, 4133, 4134, 4135, 4136, 4137, 4138, 4139, 4141, 4142, 4143, 4144, 4145, 4146, 4147, 4148, 4149, 4150, 4151, 4152, 4153, 4154, 4155, 4156, 4157, 4158, 4159, 4160, 4161, 4164, 4165, 4169, 4171, 4177, 4178, 4179, 4180, 4181, 4182, 4183, 4184, 4185, 4186, 4187, 4188]

    ptr = 0
    for i in range(0, len(missed_indexes)):
        for j in range(0, len(imp_index)):
            if missed_indexes[i] < imp_index[j]:
                for k2 in range(j, len(imp_index)):
                    imp_index[k2] = imp_index[k2] - 1
                break

    # print(imp_index)

    file = open(file_name)
    fe = np.loadtxt(file, delimiter=",")
    fe = fe.transpose()

    mean_fe = []

    for i in range(0, len(fe)):
        mean = np.mean(fe[i])
        mean_fe.append(mean)

    value, index = kth_largest_element(k, mean_fe)
    # print(value)
    intersect = find_the_portion(imp_index, index)
    print(len(intersect))
    # print(intersect)
    # print(value)
    # print(index)
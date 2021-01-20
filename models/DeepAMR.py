#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:03:18 2019

@author: emma
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from sklearn.model_selection import train_test_split

"""
Created on Wed Oct 10 19:07:58 2018

@author: emma
"""

import pandas as pd
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.layers.noise import GaussianDropout
from collections import Counter
from sklearn import metrics
from sklearn.metrics import confusion_matrix, roc_curve
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from keras.engine.topology import Layer, InputSpec
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from keras import backend as K


num_num = 9


def masked_loss_function(y_true, y_pred):
    mask = K.cast(K.not_equal(y_true, -1), K.floatx())
    return K.binary_crossentropy(y_true * mask, y_pred * mask)


def masked_accuracy(y_true, y_pred):
    dtype = K.floatx()
    total = K.sum(K.cast(K.not_equal(y_true, -1), dtype))
    correct = K.sum(K.cast(K.equal(y_true, K.round(y_pred)), dtype))
    return correct / total


# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class eva_def:
    def Find_Optimal_Cutoff(self, target, predicted):
        """ Find the optimal probability cutoff point for a classification model related to event rate
        Parameters
        ----------
        target : Matrix with dependent or target data, where rows are observations

        predicted : Matrix with predicted data, where rows are observations

        Returns
        -------
        list type, with optimal cutoff value

        """
        # print(len(predicted))
        # print(predicted)
        # print(len(target))
        # print(target)
        for i in range(len(target)-1, -1, -1):
            if target[i] == -1:
                target = np.delete(target, i)
                predicted = np.delete(predicted, i, 0)

        # predicted_tmp = []
        # if len(predicted[0]) > 2:
        #     for i in range(len(predicted)):
        #         tmp = np.mean(predicted[i])
        #         a = []
        #         a.append(tmp)
        #         # predicted[i] = tmp
        #         predicted[i] = a
        #         # print(len(predicted[i]))
        #         # for j in range(len(predicted[i])-1, 0, -1):
        #         #     predicted[i] = np.delete(predicted[i], j)
        #     # predicted = predicted_tmp
        #
        # print(len(predicted))
        # print(predicted)
        # print(len(target))
        # print(target)

        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr))
        roc = pd.DataFrame({'tf': pd.Series(tpr - (1 - fpr), index=i), 'threshold': pd.Series(threshold, index=i)})
        roc_t = roc.ix[(roc.tf - 0).abs().argsort()[:1]]

        return list(roc_t['threshold'])

    def get_class_weights(self, y):
        counter = Counter(y)
        majority = max(counter.values())
        return {cls: float(majority / count) for cls, count in counter.items()}

        # calculate the performance (Accuracy, sensitivity, specificity, AUC)

    def performance_calculation(self, array1, array2, array3):
        from evaluations import ROC_PR
        # print(array1)
        # print(array2)
        for i in range(len(array1)-1, -1, -1):
            if array1[i] == -1:
                array1 = np.delete(array1, i)
                array2 = np.delete(array2, i, 0)
                array3 = np.delete(array3, i, 0)

        tn, fp, fn, tp = confusion_matrix(array1, array2).ravel()
        #         total=tn+fp+fn+tp
        #         acc= (tn+tp)/total
        sen = tp / (tp + fn)
        sps = tn / (tn + fp)

        fpr, tpr, thresholds = metrics.roc_curve(array1, array3)
        roc_auc = metrics.auc(fpr, tpr)

        precision = tp / (tp + fp)
        f1_score = 2 * (precision * sen) / (precision + sen)
        se95spe, pr = ROC_PR.SR_maker(array1, array3)
        return roc_auc, se95spe, pr, sen, sps, roc_auc, f1_score


class deepamr:
    def __init__(self, data, labels):
        self.X = data
        self.Y = labels
        self.n_splits = None
        self.test_size = None
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.batch_size = None
        self.autoencoder = None
        self.encoder = None
        self.ae_layer_dims = None
        self.deepamr_model = None
        self.class_threshold = None

    def makedivisible_to_all(self):
        self.x_train, self.y_train = self.makedivisible(self.x_train, self.y_train)
        self.x_test, self.y_test = self.makedivisible(self.x_test, self.y_test)
        self.x_val, self.y_val = self.makedivisible(self.x_val, self.y_val)

    def makedivisible(self, x, y):
        b_s = self.batch_size
        if x.shape[0] / b_s != 0:
            to_remove = np.size(x, axis=0) - int(np.floor(np.size(x, axis=0) / b_s) * b_s)
            x = x[:-to_remove]
            y = y[:-to_remove]
        return x, y

    def data_prep(self, N_splits=10, Test_size=0.1, Val_size=0.1, Batch_size=32):
        rand_sta = 333
        msss = MultilabelStratifiedShuffleSplit(n_splits=N_splits,
                                                test_size=Test_size, random_state=rand_sta)
        #        msss=MultilabelStratifiedKFold(n_splits=n_fold,random_state=rand_sta)
        train_list = []
        test_list = []
        for train_index, test_index in msss.split(self.X, self.Y):
            train_list.append(train_index)
            test_list.append(test_index)

        x_train_tmp = self.X.values[train_list[0]] #TODO as_matrix()
        y_train_tmp = self.Y.values[train_list[0]]

        x_test = self.X.values[test_list[0]]
        y_test = self.Y.values[test_list[0]]

        msss_cv = MultilabelStratifiedShuffleSplit(n_splits=N_splits,
                                                   test_size=Val_size, random_state=rand_sta)

        train_list = []
        val_list = []
        for train_index, val_index in msss_cv.split(x_train_tmp, y_train_tmp):
            train_list.append(train_index)
            val_list.append(val_index)

        x_train = x_train_tmp[train_list[0]]
        y_train = y_train_tmp[train_list[0]]

        x_val = x_train_tmp[val_list[0]]
        y_val = y_train_tmp[val_list[0]]

        print(type(y_train))
        for i in range(0, num_num):
            print("fold: " + str(i))
            length = int(len(self.X) / 10)
            if i == 0:
                X_train = self.X[length:]
                x_test = self.X[0:length]
                y_train = self.Y[length:]
                y_test = self.Y[0:length]
            elif i != 9:
                X_train = np.append(self.X[0:length * i], self.X[length * (i + 1):], axis=0)
                x_test = self.X[length * i:length * (i + 1)]
                y_train = np.append(self.Y[0:length * i], self.Y[length * (i + 1):], axis=0)
                y_test = self.Y[length * i:length * (i + 1)]
            else:
                X_train = self.X[0:length * i]
                x_test = self.X[length * i:]
                y_train = self.Y[0:length * i]
                y_test = self.Y[length * i:]

            x_train, x_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1,
                                                              shuffle=False)
        print(type(y_train))
        print(type(y_test))

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test.to_numpy()
        self.y_test = y_test.to_numpy()
        self.batch_size = Batch_size

        self.makedivisible_to_all()

    def AutoEncoder(self, dims=[739999, 500, 1000, 20], act='relu', init='uniform', drop_rate=0.3): #TODO should fix
        """
        Fully connected auto-encoder model, symmetric.
        Arguments:
            dims: list of number of units in each layer of encoder. dims[0] is input dim, dims[-1] is units in hidden layer.
                The decoder is symmetric with encoder. So number of layers of the auto-encoder is 2*len(dims)-1
            act: activation, not applied to Input, Hidden and Output layers
        return:
            (ae_model, encoder_model), Model of autoencoder and model of encoder
        """
        n_stacks = len(dims) - 1
        # input
        input_img = Input(shape=(dims[0],), name='input')
        x = input_img
        x = GaussianDropout(drop_rate)(x)
        # internal layers in encoder
        for i in range(n_stacks - 1):
            x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

        # hidden layer
        encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(
            x)  # hidden layer, features are extracted from here

        x = encoded
        # internal layers in decoder
        for i in range(n_stacks - 1, 0, -1):
            x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

        # output
        x = Dense(dims[0], kernel_initializer=init, activation='sigmoid', name='decoder_0')(x)
        decoded = x
        self.autoencoder = Model(inputs=input_img, outputs=decoded, name='AE')
        print(input_img)
        self.encoder = Model(inputs=input_img, outputs=encoded, name='encoder')
        self.ae_layer_dims = dims

    def pre_train(self, Epochs=100, Optimizer='SGD', Loss='binary_crossentropy', Callbacks=None):
        ae = self.autoencoder
        enc = self.encoder
        ae.compile(optimizer=Optimizer, loss=masked_loss_function)
        #
        ae.fit(self.x_train, self.x_train,
               batch_size=self.batch_size,
               epochs=Epochs,
               shuffle=False,
               validation_data=[self.x_val, self.x_val],
               callbacks=Callbacks)  # pretrin the
        ae.save_weights('./ae_best_weights.h5')  # save the best model
        ae.load_weights('./ae_best_weights.h5')  # load the best model?
        self.autoencoder = ae
        self.encoder = enc

    def build_model(self, act='sigmoid', init='uniform'):
        t11 = self.encoder.output
        t1 = Dense(1, kernel_initializer=init, activation=act, name='task1')(t11)
        # No.2
        #    st2 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t22 = self.encoder.output
        t2 = Dense(1, kernel_initializer=init, activation=act, name='task2')(t22)
        # No.3
        #    st3 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t33 = self.encoder.output
        t3 = Dense(1, kernel_initializer=init, activation=act, name='task3')(t33)
        # No.4
        #    st4 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t44 = self.encoder.output
        t4 = Dense(1, kernel_initializer=init, activation=act, name='task4')(t44)
        t55 = self.encoder.output
        t5 = Dense(1, kernel_initializer=init, activation=act, name='task5')(t55)
        t66 = self.encoder.output
        t6 = Dense(1, kernel_initializer=init, activation=act, name='task6')(t66)
        t77 = self.encoder.output
        t7 = Dense(1, kernel_initializer=init, activation=act, name='task7')(t77)
        t88 = self.encoder.output
        t8 = Dense(1, kernel_initializer=init, activation=act, name='task8')(t88)
        t99 = self.encoder.output
        t9 = Dense(1, kernel_initializer=init, activation=act, name='task9')(t99)
        t1010 = self.encoder.output
        t10 = Dense(1, kernel_initializer=init, activation=act, name='task10')(t1010)
        t1111 = self.encoder.output
        t101 = Dense(1, kernel_initializer=init, activation=act, name='task11')(t1111)
        t1212 = self.encoder.output
        t12 = Dense(1, kernel_initializer=init, activation=act, name='task12')(t1212)
        # Compile model
        self.deepamr_model = Model(inputs=self.encoder.input,
                                   output=[t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t101, t12, self.autoencoder.output])

    def train(self, Epochs=100, Loss=['binary_crossentropy', 'binary_crossentropy',
                                      'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
              Loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], Optimizer='Nadam', Callbacks=None):
        self.deepamr_model.compile(loss=masked_loss_function,
                                   loss_weights=Loss_weights,
                                   optimizer=Optimizer, metrics=[masked_accuracy])

        self.deepamr_model.fit(self.x_train, [self.y_train[:, 0], self.y_train[:, 1],
                                              self.y_train[:, 2], self.y_train[:, 3],
                                              self.y_train[:, 4], self.y_train[:, 5],
                                              self.y_train[:, 6], self.y_train[:, 7],
                                              self.y_train[:, 8], self.y_train[:, 9],
                                              self.y_train[:, 10], self.y_train[:, 11], self.x_train],
                               validation_data=(self.x_val, [self.y_val[:, 0], self.y_val[:, 1],
                                                             self.y_val[:, 2], self.y_val[:, 3],
                                                             self.y_val[:, 4], self.y_val[:, 5],
                                                             self.y_val[:, 6], self.y_val[:, 7],
                                                             self.y_val[:, 8], self.y_val[:, 9],
                                                             self.y_val[:, 10], self.y_val[:, 11],
                                                             self.x_val]),
                               shuffle=False,
                               epochs=Epochs,
                               batch_size=self.batch_size,
                               callbacks=Callbacks)
        self.deepamr_model.save_weights('./best_weights.h5')
        self.deepamr_model.load_weights('./best_weights.h5')

        #        # optimise threshold: use train and validation data together to determine the best threshold
        x_train_tmp = np.concatenate((self.x_train, self.x_val), axis=0)
        y_train_tmp = np.concatenate((self.y_train, self.y_val), axis=0)

        train_pred = self.deepamr_model.predict(x_train_tmp, batch_size=self.batch_size)
        th = []
        for i in range(y_train_tmp.shape[1]):
            th.append(eva_def().Find_Optimal_Cutoff(y_train_tmp[:, i], train_pred[i]))

        self.class_threshold = th

    def predict(self):
        y_pred_prob_tmp = self.deepamr_model.predict(self.x_test, batch_size=self.batch_size)
        # self.deepamr_model.save('../saved_models/DPAMR/deepamr' + str(num_num-1) + '.h5')
        y_pred_prob = self.deepamr_model.predict(self.x_test, batch_size=self.batch_size)
        th = self.class_threshold
        for i in range(len(th)):
            y_pred_prob_tmp[i] = np.where(y_pred_prob_tmp[i] > th[i], 1, 0)
            perf_mat = eva_def().performance_calculation(self.y_test[:, i], y_pred_prob_tmp[0], y_pred_prob[0])
            # print("1")
            print('Performance to label %s is:' % i, perf_mat)


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))  # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# ----------------------------------------------------------------------
class deepamr_cluster(deepamr):
    def __init__(self, data, labels, cluster_labels):
        print('deepamr_cluster')
        self.deepamr_cluster = None
        self.L = cluster_labels
        self.n_cluster = 8
        self.l_train = None
        self.l_val = None
        self.l_test = None
        super(deepamr_cluster, self).__init__(data, labels)

    def makedivisible_to_all(self):
        self.x_train, self.y_train, self.l_train = self.makedivisible(self.x_train, self.y_train, self.l_train)
        self.x_test, self.y_test, self.l_test = self.makedivisible(self.x_test, self.y_test, self.l_test)
        self.x_val, self.y_val, self.l_val = self.makedivisible(self.x_val, self.y_val, self.l_val)

    def makedivisible(self, x, y, l):
        b_s = self.batch_size
        if x.shape[0] / b_s != 0:
            to_remove = np.size(x, axis=0) - int(np.floor(np.size(x, axis=0) / b_s) * b_s)
            x = x[:-to_remove]
            y = y[:-to_remove]
            l = l[:-to_remove]
        return x, y, l

    def data_prep(self, N_splits=10, Test_size=0.3, Val_size=0.2, Batch_size=32):
        rand_sta = 333
        msss = MultilabelStratifiedShuffleSplit(n_splits=N_splits,
                                                test_size=Test_size, random_state=rand_sta)
        #        msss=MultilabelStratifiedKFold(n_splits=n_fold,random_state=rand_sta)
        train_list = []
        test_list = []
        for train_index, test_index in msss.split(self.X, self.Y):
            train_list.append(train_index)
            test_list.append(test_index)

        x_train_tmp = self.X.as_matrix()[train_list[0]]
        y_train_tmp = self.Y.as_matrix()[train_list[0]]
        l_train_tmp = self.L.as_matrix()[train_list[0]]

        x_test = self.X.as_matrix()[test_list[0]]
        y_test = self.Y.as_matrix()[test_list[0]]
        l_test = self.L.as_matrix()[test_list[0]]

        msss_cv = MultilabelStratifiedShuffleSplit(n_splits=N_splits,
                                                   test_size=Val_size, random_state=rand_sta)

        train_list = []
        val_list = []
        for train_index, val_index in msss_cv.split(x_train_tmp, y_train_tmp):
            train_list.append(train_index)
            val_list.append(val_index)

        x_train = x_train_tmp[train_list[0]]
        y_train = y_train_tmp[train_list[0]]
        l_train = l_train_tmp[train_list[0]]
        x_val = x_train_tmp[val_list[0]]
        y_val = y_train_tmp[val_list[0]]
        l_val = l_train_tmp[val_list[0]]

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.l_train = l_train
        self.l_test = l_test
        self.l_val = l_val
        self.batch_size = Batch_size

        self.makedivisible_to_all()

    def AutoEncoder(self, dims=[5823, 500, 1000, 20], act='relu', init='uniform', drop_rate=0.3):
        super(deepamr_cluster, self).AutoEncoder(dims, act, init, drop_rate)

    def pre_train(self, Epochs=5, Optimizer='SGD', Loss='binary_crossentropy', Callbacks=None):
        super(deepamr_cluster, self).pre_train(Epochs=Epochs, Optimizer=Optimizer, Loss=Loss, Callbacks=Callbacks)

    def build_model(self, act='sigmoid', init='uniform', ):
        t11 = self.encoder.output
        t1 = Dense(1, kernel_initializer=init, activation=act, name='task1')(t11)
        # No.2
        #    st2 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t22 = self.encoder.output
        t2 = Dense(1, kernel_initializer=init, activation=act, name='task2')(t22)
        # No.3
        #    st3 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t33 = self.encoder.output
        t3 = Dense(1, kernel_initializer=init, activation=act, name='task3')(t33)
        # No.4
        #    st4 = Dense(cls_hidden_layer[1], init='normal')(cls2)
        t44 = self.encoder.output
        t4 = Dense(1, kernel_initializer=init, activation=act, name='task4')(t44)
        # cluster layer
        #        y=pd.factorize(lin_train_tmp)[0]
        #        n_clusters=8
        clustering_layer = ClusteringLayer(self.n_cluster, name='clustering')(
            self.encoder.output)  # attach output of bottleneck to cluster layer

        # Compile model
        self.deepamr_cluster = Model(inputs=self.encoder.input,
                                     output=[t1, t2, t3, t4, clustering_layer])

    #        model = Model(inputs=ae.input,
    #                      outputs=[clustering_layer,ae.output])                     # construct model with two output: decoded output + cluster output
    def target_distribution(self, q=None):
        weight = q ** 2 / q.sum(0)
        return (weight.T / weight.sum(1)).T

    def train(self, Epochs=100, Loss=['binary_crossentropy', 'binary_crossentropy',
                                      'binary_crossentropy', 'binary_crossentropy', 'categorical_crossentropy'],
              Loss_weights=[1, 1, 1, 1, 0.1], Optimizer='Nadam'):
        x_train_tmp = np.concatenate((self.x_train, self.x_val), axis=0)
        y_train_tmp = np.concatenate((self.y_train, self.y_val), axis=0)
        kmeans = KMeans(n_clusters=self.n_cluster, n_init=20, random_state=27)  # kmeans baseline model
        y_pred = kmeans.fit_predict(self.encoder.predict(x_train_tmp))  # prediction using kmeans
        self.deepamr_cluster.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

        y_pred_last = np.copy(y_pred)
        self.deepamr_cluster.compile(loss=masked_loss_function,
                                     loss_weights=Loss_weights, optimizer=Optimizer)

        loss = 0
        index = 0
        maxiter = 8000
        update_interval = 140
        index_array = np.arange(self.x_train.shape[0])
        tol = 0.001  # tolerance theshold to stop training
        # start train
        import metrics #TODO install
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:  # update p every update_interval
                _, _, _, _, q = self.deepamr_cluster.predict(self.x_train, verbose=0)  # predict using recompiled model
                p = self.target_distribution(q)  # update the auxiliary target distribution p

                # evaluate the clustering performance
                l_pred = q.argmax(1)  # replace probs by those obtained
                if self.l_train is not None:
                    acc = np.round(metrics.acc(self.l_train[:, 0], l_pred), 5)
                    nmi = np.round(metrics.nmi(self.l_train[:, 0], l_pred), 5)
                    ari = np.round(metrics.ari(self.l_train[:, 0], l_pred), 5)
                    loss = np.round(loss, 5)
                    print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

                # check stop criterion: if two predictions remain the same then stop
                delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
                y_pred_last = np.copy(y_pred)
                print('delta_label', delta_label)
                if ite > 0 and delta_label < tol:
                    print('delta_label ', delta_label, '< tol ', tol)
                    print('Reached tolerance threshold. Stopping training.')
                    break

            idx = index_array[index * self.batch_size: min((index + 1) * self.batch_size,
                                                           x_train_tmp.shape[0])]  # find index of batch to train
            loss = self.deepamr_cluster.train_on_batch(x=x_train_tmp[idx],
                                                       y=[y_train_tmp[idx, 0], y_train_tmp[idx, 1], y_train_tmp[idx, 2],
                                                          y_train_tmp[idx, 3], p[idx]])  # retrain on batch
            index = index + 1 if (index + 1) * self.batch_size <= x_train_tmp.shape[0] else 0

        x_train_tmp = np.concatenate((self.x_train, self.x_val), axis=0)
        y_train_tmp = np.concatenate((self.y_train, self.y_val), axis=0)

        train_pred = self.deepamr_cluster.predict(x_train_tmp, batch_size=self.batch_size)
        th = []
        for i in range(y_train_tmp.shape[1]):
            th.append(eva_def().Find_Optimal_Cutoff(y_train_tmp[:, i], train_pred[i]))

        self.class_threshold = th

        self.deepamr_cluster.save_weights('./b_DEC_model_final.h5')
        self.deepamr_cluster.load_weights('./b_DEC_model_final.h5')

    def predict(self):
        # final Eval.
        y_pred_prob_tmp = self.deepamr_cluster.predict(self.x_test, batch_size=self.batch_size)
        y_pred_prob = self.deepamr_cluster.predict(self.x_test, batch_size=self.batch_size)
        th = self.class_threshold
        for i in range(len(th)):
            y_pred_prob_tmp[i] = np.where(y_pred_prob_tmp[i] > th[i], 1, 0)
            perf_mat = eva_def().performance_calculation(self.y_test[:, i], y_pred_prob_tmp[0], y_pred_prob[0])
            print("2")
            print('Performance to label %s is:' % i, perf_mat)

            # evaluate the clustering performance
        y_pred = y_pred_prob[-1].argmax(1)

        import seaborn as sns
        import sklearn.metrics
        #    # plot confusion matrix
        sns.set(font_scale=3)
        confusion_mat = sklearn.metrics.confusion_matrix(self.l_test, y_pred)
        #
        plt.figure(figsize=(16, 14))
        sns.heatmap(confusion_mat, annot=True, fmt="d", annot_kws={"size": 20});
        plt.title("Confusion matrix", fontsize=30)
        plt.ylabel('True label', fontsize=25)
        plt.xlabel('Clustering label', fontsize=25)
        plt.savefig('confusion.eps', format='eps', dpi=1000)
        plt.savefig('confusion.png', format='png', dpi=1000)
        plt.show()



# if __name__=='__main__':
#     print('DeepAMR')
#     import numpy as np
#     import tensorflow as tf
#     import random as rn
#     import os
#     os.environ['PYTHONHASHSEED'] = '0'
#
#     from keras import backend as K
#
#     np.random.seed(27)
#     rn.seed(27)
#     tf.set_random_seed(27)
#
#     sess = tf.Session(graph=tf.get_default_graph())
#     K.set_session(sess)
#     main_deepamr()
#     K.clear_session()



# results:
# fold 0:
# Performance to label 0 is: (0.8801388488978468, 0.6299516908212561, 0.8759455444301599, 0.7681159420289855, 0.8278145695364238, 0.8801388488978468, 0.7607655502392345)
# Performance to label 1 is: (0.9180984426644434, 0.6734955185659411, 0.8898982259892602, 0.7429577464788732, 0.9302325581395349, 0.9180984426644434, 0.7992424242424242)
# Performance to label 2 is: (0.8307983890073442, 0.23217247097844115, 0.4561131018904826, 0.7014925373134329, 0.8444444444444444, 0.8307983890073442, 0.5766871165644172)
# Performance to label 3 is: (0.6936581642389161, 0.012658227848101266, 0.38465305281757634, 0.7848101265822784, 0.542713567839196, 0.6936581642389161, 0.5344827586206896)
# Performance to label 4 is: (0.7083333333333334, 0.36363636363636365, 0.3372166534403573, 0.45454545454545453, 0.7638888888888888, 0.7083333333333334, 0.30303030303030304)
# Performance to label 5 is: (0.8019793459552496, 0.35857142857142854, 0.6681422486283526, 0.8142857142857143, 0.5903614457831325, 0.8019793459552496, 0.5846153846153846)
# Performance to label 6 is: (0.8667300648069879, 0.15660749506903354, 0.8723457655187179, 0.6420118343195266, 0.9357142857142857, 0.8667300648069879, 0.7457044673539519)
# Performance to label 7 is: (0.6535812672176309, 0.07954545454545454, 0.42865892075581535, 0.7727272727272727, 0.42424242424242425, 0.6535812672176309, 0.5037037037037037)
# Performance to label 8 is: (0.8807614262247561, 0.555490311215502, 0.6726077435310325, 0.8244274809160306, 0.7869198312236287, 0.8807614262247561, 0.6352941176470588)
# Performance to label 9 is: (0.44047619047619047, 0, 0.07186020183047574, 0.0, 0.5714285714285714, 0.44047619047619047, nan)
# Performance to label 10 is: (0.6827956989247312, 0.02150537634408602, 0.43363336490926285, 0.8709677419354839, 0.49206349206349204, 0.6827956989247312, 0.6)
# Performance to label 11 is: (0.7077261161230627, 0.022727272727272728, 0.46697895214493707, 0.8333333333333334, 0.5419847328244275, 0.7077261161230627, 0.6077348066298343)
#
# fold 1:
# Performance to label 0 is: (0.8577013620312763, 0.5482456140350878, 0.8526101066299268, 0.7850877192982456, 0.7955271565495208, 0.8577013620312763, 0.7600849256900212)
# Performance to label 1 is: (0.9078615859820933, 0.598159509202454, 0.8855237115944542, 0.7269938650306749, 0.8979118329466357, 0.9078615859820933, 0.7808896210873145)
# Performance to label 2 is: (0.8435694962345545, 0.33156028368794327, 0.6114584048353118, 0.7659574468085106, 0.8144329896907216, 0.8435694962345545, 0.6545454545454545)
# Performance to label 3 is: (0.6501567398119122, 0.022988505747126436, 0.3434951322616347, 0.8045977011494253, 0.4681818181818182, 0.6501567398119122, 0.510948905109489)
# Performance to label 4 is: (0.8151260504201681, 0.29411764705882354, 0.5286661062117399, 0.8235294117647058, 0.7142857142857143, 0.8151260504201681, 0.509090909090909)
# Performance to label 5 is: (0.73421926910299, 0.22023809523809523, 0.5402385519633548, 0.7976190476190477, 0.5581395348837209, 0.73421926910299, 0.5903083700440529)
# Performance to label 6 is: (0.87994373885918, 0.6336898395721925, 0.8957968674469847, 0.6631016042780749, 0.9140625, 0.87994373885918, 0.7572519083969466)
# Performance to label 7 is: (0.658641975308642, 0.016666666666666666, 0.43565252176278346, 0.8, 0.4722222222222222, 0.658641975308642, 0.5818181818181818)
# Performance to label 8 is: (0.8676925162689805, 0.4475961538461538, 0.7085771818499332, 0.825, 0.7700650759219089, 0.8676925162689805, 0.6633165829145728)
# Performance to label 9 is: (0.44736842105263164, 0, 0.07924869037772264, 0.25, 0.631578947368421, 0.44736842105263164, 0.10526315789473685)
# Performance to label 10 is: (0.6304573804573805, 0.06153846153846154, 0.35825132462056397, 0.8615384615384616, 0.41216216216216217, 0.6304573804573805, 0.5384615384615384)
# Performance to label 11 is: (0.6636100386100386, 0.07142857142857142, 0.4014489736649104, 0.8571428571428571, 0.4797297297297297, 0.6636100386100386, 0.5797101449275363)
#
# fold 2:
# Performance to label 0 is: (0.8750301651325543, 0.5808080808080808, 0.8628596725180622, 0.803030303030303, 0.8122866894197952, 0.8750301651325543, 0.7718446601941747)
# Performance to label 1 is: (0.9296294645510192, 0.7367957746478874, 0.9018592414344555, 0.7922535211267606, 0.9135021097046413, 0.9296294645510192, 0.8181818181818182)
# Performance to label 2 is: (0.8483771929824562, 0.35526315789473684, 0.581632820399052, 0.7894736842105263, 0.8466666666666667, 0.8483771929824562, 0.6593406593406593)
# Performance to label 3 is: (0.6360227846275868, 0.00966183574879227, 0.2987575649974345, 0.8115942028985508, 0.47761194029850745, 0.6360227846275868, 0.4869565217391304)
# Performance to label 4 is: (0.8209876543209877, 0.2222222222222222, 0.4754189680812755, 0.8333333333333334, 0.7037037037037037, 0.8209876543209877, 0.5263157894736842)
# Performance to label 5 is: (0.8468555900621118, 0.359375, 0.7031611050384518, 0.921875, 0.5838509316770186, 0.8468555900621118, 0.6210526315789473)
# Performance to label 6 is: (0.8776785714285713, 0.4136904761904762, 0.8785370137399289, 0.6904761904761905, 0.919047619047619, 0.8776785714285713, 0.7707641196013288)
# Performance to label 7 is: (0.660233918128655, 0.14473684210526316, 0.4023698540061219, 0.7894736842105263, 0.4444444444444444, 0.660233918128655, 0.5084745762711864)
# Performance to label 8 is: (0.8861566484517304, 0.48360655737704916, 0.7078202102325432, 0.8524590163934426, 0.7606837606837606, 0.8861566484517304, 0.6153846153846153)
# Performance to label 9 is: (0.723404255319149, 0, 0.3184373863285707, 0.8333333333333334, 0.5957446808510638, 0.723404255319149, 0.33333333333333337)
# Performance to label 10 is: (0.7116898849182314, 0.09615384615384616, 0.4193399846984569, 0.9038461538461539, 0.4645669291338583, 0.7116898849182314, 0.5628742514970061)
# Performance to label 11 is: (0.7115332428765264, 0.09090909090909091, 0.42539433997724296, 0.8545454545454545, 0.5, 0.7115332428765264, 0.5562130177514794)
#
# fold 3:
# Performance to label 0 is: (0.8803293594665589, 0.5707213578500707, 0.8655791053525389, 0.7524752475247525, 0.8469387755102041, 0.8803293594665589, 0.761904761904762)
# Performance to label 1 is: (0.9343155469678953, 0.7396551724137931, 0.8953697775357377, 0.7379310344827587, 0.9547413793103449, 0.9343155469678953, 0.8152380952380952)
# Performance to label 2 is: (0.8867532467532467, 0.4, 0.5953124679146662, 0.7866666666666666, 0.8636363636363636, 0.8867532467532467, 0.6704545454545454)
# Performance to label 3 is: (0.6355334163553341, 0.015151515151515152, 0.2731647444575385, 0.7727272727272727, 0.5205479452054794, 0.6355334163553341, 0.45945945945945943)
# Performance to label 4 is: (0.7705696202531646, 0.0625, 0.3584300941820125, 0.6875, 0.7848101265822784, 0.7705696202531646, 0.5)
# Performance to label 5 is: (0.7685381805826892, 0.18032786885245902, 0.4459333863413718, 0.8852459016393442, 0.6113989637305699, 0.7685381805826892, 0.568421052631579)
# Performance to label 6 is: (0.8871237815652968, 0.5921450151057401, 0.8797923694461623, 0.6253776435045317, 0.9339622641509434, 0.8871237815652968, 0.7314487632508834)
# Performance to label 7 is: (0.5733965672990063, 0.024390243902439025, 0.281836298704978, 0.7804878048780488, 0.4537037037037037, 0.5733965672990063, 0.4848484848484849)
# Performance to label 8 is: (0.8933940855258492, 0.45762711864406785, 0.6837792291681101, 0.8220338983050848, 0.8075313807531381, 0.8933940855258492, 0.6319218241042345)
# Performance to label 9 is: (0.673076923076923, 0, 0.07367677710814965, 0.3333333333333333, 0.7115384615384616, 0.673076923076923, 0.10526315789473684)
# Performance to label 10 is: (0.7174475423972406, 0.14285714285714285, 0.40176147960566866, 0.8979591836734694, 0.5140845070422535, 0.7174475423972406, 0.5432098765432098)
# Performance to label 11 is: (0.7444080741953082, 0.15384615384615385, 0.4336024820416164, 0.9615384615384616, 0.5886524822695035, 0.7444080741953082, 0.625)
#
# fold 4:
# Performance to label 0 is: (0.8851735566021282, 0.632051282051282, 0.874998925136844, 0.8205128205128205, 0.826530612244898, 0.8851735566021282, 0.7881773399014778)
# Performance to label 1 is: (0.9320031038687508, 0.7140893470790378, 0.9110935611511538, 0.7800687285223368, 0.9290322580645162, 0.9320031038687508, 0.823956442831216)
# Performance to label 2 is: (0.887573385518591, 0.41142857142857137, 0.6299505884805321, 0.8142857142857143, 0.8595890410958904, 0.887573385518591, 0.6785714285714286)
# Performance to label 3 is: (0.6586931818181818, 0.0014204545454545455, 0.3680572182914597, 0.8068181818181818, 0.49, 0.6586931818181818, 0.5440613026819924)
# Performance to label 4 is: (0.7877984084880636, 0.025641025641025644, 0.26299907106853415, 0.6153846153846154, 0.7241379310344828, 0.7877984084880636, 0.35555555555555557)
# Performance to label 5 is: (0.7870161290322581, 0.20625, 0.5993838350888296, 0.7875, 0.6129032258064516, 0.7870161290322581, 0.6206896551724137)
# Performance to label 6 is: (0.8836433167757959, 0.6630329457364342, 0.8915499455505748, 0.6831395348837209, 0.9393203883495146, 0.8836433167757959, 0.7781456953642384)
# Performance to label 7 is: (0.5968013468013468, 0.03333333333333333, 0.40857088101613653, 0.75, 0.40404040404040403, 0.5968013468013468, 0.5487804878048781)
# Performance to label 8 is: (0.8908747951389963, 0.601027397260274, 0.771716095610163, 0.8493150684931506, 0.8036117381489842, 0.8908747951389963, 0.6946778711484594)
# Performance to label 9 is: (0.9743589743589743, 0, 0.25, 1.0, 0.6666666666666666, 0.9743589743589743, 0.13333333333333333)
# Performance to label 10 is: (0.6428571428571429, 0, 0.32595315121780355, 0.8301886792452831, 0.4357142857142857, 0.6428571428571429, 0.5)
# Performance to label 11 is: (0.6234936488980567, 0.003278688524590164, 0.32802629894366686, 0.7540983606557377, 0.46357615894039733, 0.6234936488980567, 0.48936170212765956)
#
# fold 5:
# Performance to label 0 is: (0.8852338943324978, 0.607843137254902, 0.8832564517167552, 0.751131221719457, 0.8345323741007195, 0.8852338943324978, 0.7667436489607391)
# Performance to label 1 is: (0.9242907419378007, 0.6539215686274511, 0.9011570011315966, 0.7124183006535948, 0.9296703296703297, 0.9242907419378007, 0.7841726618705036)
# Performance to label 2 is: (0.8906789413118527, 0.5316455696202531, 0.6953521187579574, 0.759493670886076, 0.8601398601398601, 0.8906789413118527, 0.6703910614525139)
# Performance to label 3 is: (0.6792441439265878, 0.02032520325203252, 0.3711916697794703, 0.7560975609756098, 0.5297029702970297, 0.6792441439265878, 0.5188284518828452)
# Performance to label 4 is: (0.6444444444444445, 0.1, 0.1535400619153082, 0.5, 0.7037037037037037, 0.6444444444444445, 0.25641025641025644)
# Performance to label 5 is: (0.8076923076923077, 0.30303030303030304, 0.5672876719069185, 0.8181818181818182, 0.6208791208791209, 0.8076923076923077, 0.5714285714285715)
# Performance to label 6 is: (0.8979741019214704, 0.4161793372319688, 0.8942430199269068, 0.6608187134502924, 0.9404761904761905, 0.8979741019214704, 0.7622259696458684)
# Performance to label 7 is: (0.6610576923076924, 0.020833333333333332, 0.3959700865731379, 0.7916666666666666, 0.49038461538461536, 0.6610576923076924, 0.5467625899280576)
# Performance to label 8 is: (0.895231738969225, 0.4296774193548387, 0.7352917927732844, 0.8193548387096774, 0.8137931034482758, 0.895231738969225, 0.6997245179063362)
# Performance to label 9 is: (0.35135135135135137, 0, 0.07181458168590611, 0.0, 0.5945945945945946, 0.35135135135135137, nan)
# Performance to label 10 is: (0.7106481481481481, 0.046296296296296294, 0.38019525765701123, 0.8333333333333334, 0.5277777777777778, 0.7106481481481481, 0.5389221556886228)
# Performance to label 11 is: (0.7405071119356833, 0.05454545454545454, 0.4136822998980574, 0.8545454545454545, 0.54421768707483, 0.7405071119356833, 0.5562130177514794)
#
# fold 6:
# Performance to label 0 is: (0.8999295278365045, 0.724031007751938, 0.9014374181807489, 0.8046511627906977, 0.8552188552188552, 0.8999295278365045, 0.802784222737819)
# Performance to label 1 is: (0.9366946267930951, 0.7311197916666667, 0.91599128747925, 0.7777777777777778, 0.9321663019693655, 0.9366946267930951, 0.8250460405156539)
# Performance to label 2 is: (0.8830167305681421, 0.5153508771929824, 0.6536021913384907, 0.8026315789473685, 0.8774834437086093, 0.8830167305681421, 0.7011494252873564)
# Performance to label 3 is: (0.7250685220324689, 0.043010752688172046, 0.44333297351899625, 0.7956989247311828, 0.5784313725490197, 0.7250685220324689, 0.5849802371541502)
# Performance to label 4 is: (0.851123595505618, 0.125, 0.23908413512428295, 0.75, 0.7303370786516854, 0.851123595505618, 0.31578947368421056)
# Performance to label 5 is: (0.8232697343808455, 0.3786008230452675, 0.7079835922319744, 0.8395061728395061, 0.6606060606060606, 0.8232697343808455, 0.6634146341463414)
# Performance to label 6 is: (0.886338680926916, 0.6481481481481481, 0.8896798102889187, 0.6909090909090909, 0.9317647058823529, 0.886338680926916, 0.776831345826235)
# Performance to label 7 is: (0.6383816684568564, 0.043859649122807015, 0.42490017831843774, 0.8070175438596491, 0.47959183673469385, 0.6383816684568564, 0.5974025974025974)
# Performance to label 8 is: (0.8900064042850488, 0.48552631578947364, 0.719787400421305, 0.8421052631578947, 0.8185840707964602, 0.8900064042850488, 0.7071823204419889)
# Performance to label 9 is: (0.3904761904761904, 0, 0.11198717426010414, 0.3333333333333333, 0.5142857142857142, 0.3904761904761904, 0.16)
# Performance to label 10 is: (0.7484819431128156, 0.047619047619047616, 0.4445718733725944, 0.9206349206349206, 0.5302013422818792, 0.7484819431128156, 0.6073298429319371)
# Performance to label 11 is: (0.7564384767774599, 0.0847457627118644, 0.44618398468534726, 0.8983050847457628, 0.538961038961039, 0.7564384767774599, 0.5792349726775956)
#
# fold 7:
# Performance to label 0 is: (0.9008971988219986, 0.6218637992831542, 0.8693848064327324, 0.7956989247311828, 0.8152866242038217, 0.9008971988219986, 0.7551020408163266)
# Performance to label 1 is: (0.9191506737443854, 0.7096774193548387, 0.8760584110435817, 0.7455197132616488, 0.9219409282700421, 0.9191506737443854, 0.7938931297709925)
# Performance to label 2 is: (0.884370801612181, 0.41818181818181804, 0.6123534247095007, 0.8142857142857143, 0.8244514106583072, 0.884370801612181, 0.6229508196721312)
# Performance to label 3 is: (0.6485100305135363, 0, 0.30666575477464153, 0.8082191780821918, 0.5023696682464455, 0.6485100305135363, 0.4978902953586498)
# Performance to label 4 is: (0.8161157024793387, 0.21212121212121213, 0.2901270356388289, 0.9090909090909091, 0.8068181818181818, 0.8161157024793387, 0.5263157894736842)
# Performance to label 5 is: (0.811794973283198, 0.3387096774193548, 0.5759204994145697, 0.8548387096774194, 0.6196319018404908, 0.811794973283198, 0.5988700564971751)
# Performance to label 6 is: (0.8858669565462913, 0.6578549848942598, 0.8895069036100471, 0.6676737160120846, 0.9392523364485982, 0.8858669565462913, 0.7647058823529411)
# Performance to label 7 is: (0.6284395198522623, 0.02631578947368421, 0.42759111131602157, 0.7894736842105263, 0.4842105263157895, 0.6284395198522623, 0.5960264900662251)
# Performance to label 8 is: (0.9018220585384763, 0.5522388059701491, 0.7068534534509721, 0.8432835820895522, 0.8051948051948052, 0.9018220585384763, 0.6706231454005934)
# Performance to label 9 is: (0.7941176470588236, 0, 0.19894133644133644, 1.0, 0.6764705882352942, 0.7941176470588236, 0.4210526315789474)
# Performance to label 10 is: (0.6639884088514226, 0.06730769230769232, 0.33932461617095006, 0.8846153846153846, 0.4452054794520548, 0.6639884088514226, 0.5139664804469274)
# Performance to label 11 is: (0.7183359830418654, 0.06862745098039216, 0.3727990550813462, 0.8823529411764706, 0.5405405405405406, 0.7183359830418654, 0.5487804878048781)
#
# fold 8:
# Performance to label 0 is: (0.8854015104274926, 0.6217320261437909, 0.8760453640798878, 0.803921568627451, 0.8056537102473498, 0.8854015104274926, 0.7754137115839244)
# Performance to label 1 is: (0.9399548872180452, 0.7365546218487394, 0.9077644071424741, 0.8, 0.9326315789473684, 0.9399548872180452, 0.8358208955223881)
# Performance to label 2 is: (0.902983234714004, 0.32783882783882784, 0.6513555039519794, 0.8589743589743589, 0.8621794871794872, 0.902983234714004, 0.7127659574468086)
# Performance to label 3 is: (0.6305418719211823, 0.02857142857142857, 0.28948478492159124, 0.9, 0.45320197044334976, 0.6305418719211823, 0.5163934426229507)
# Performance to label 4 is: (0.7407797681770285, 0.07692307692307693, 0.2517836747701935, 0.8461538461538461, 0.5616438356164384, 0.7407797681770285, 0.3928571428571428)
# Performance to label 5 is: (0.7297215496368039, 0.11864406779661017, 0.3997573809963642, 0.864406779661017, 0.5059523809523809, 0.7297215496368039, 0.5284974093264249)
# Performance to label 6 is: (0.9008572919869298, 0.6915932364848774, 0.890903018324075, 0.7120743034055728, 0.9399538106235565, 0.9008572919869298, 0.7944732297063902)
# Performance to label 7 is: (0.6832120582120582, 0.08108108108108109, 0.3586639094520663, 0.918918918918919, 0.41346153846153844, 0.6832120582120582, 0.5151515151515151)
# Performance to label 8 is: (0.8823829570098226, 0.3908582089552239, 0.6529063859996831, 0.8656716417910447, 0.7905982905982906, 0.8823829570098226, 0.6666666666666666)
# Performance to label 9 is: (0.622093023255814, 0, 0.11068376068376069, 0.75, 0.5813953488372093, 0.622093023255814, 0.24)
# Performance to label 10 is: (0.6445393196529403, 0.04716981132075472, 0.3392675450637152, 0.9245283018867925, 0.41605839416058393, 0.6445393196529403, 0.5384615384615384)
# Performance to label 11 is: (0.6625216888374783, 0.057692307692307696, 0.35013998958541925, 0.9423076923076923, 0.42857142857142855, 0.6625216888374783, 0.5536723163841807)
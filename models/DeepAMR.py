#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 19:03:18 2019

@author: emma
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        print(len(predicted))
        print(predicted)
        print(len(target))
        print(target)
        for i in range(len(target)-1, -1, -1):
            if target[i] == -1:
                target = np.delete(target, i)
                predicted = np.delete(predicted, i, 0)

        predicted_tmp = []
        if len(predicted[0]) > 2:
            for i in range(len(predicted)):
                tmp = np.mean(predicted[i])
                a = []
                a.append(tmp)
                # predicted[i] = tmp
                predicted[i] = a
                # print(len(predicted[i]))
                # for j in range(len(predicted[i])-1, 0, -1):
                #     predicted[i] = np.delete(predicted[i], j)
            # predicted = predicted_tmp

        print(len(predicted))
        print(predicted)
        print(len(target))
        print(target)

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
        tn, fp, fn, tp = confusion_matrix(array1, array2).ravel()
        #         total=tn+fp+fn+tp
        #         acc= (tn+tp)/total
        sen = tp / (tp + fn)
        sps = tn / (tn + fp)

        fpr, tpr, thresholds = metrics.roc_curve(array1, array3)
        roc_auc = metrics.auc(fpr, tpr)

        precision = tp / (tp + fp)
        f1_score = 2 * (precision * sen) / (precision + sen)
        return sen, sps, roc_auc, f1_score


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

        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_test = x_test
        self.y_test = y_test
        self.batch_size = Batch_size

        self.makedivisible_to_all()

    def AutoEncoder(self, dims=[120, 500, 1000, 20], act='relu', init='uniform', drop_rate=0.3): #TODO should fix
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
        # Compile model
        self.deepamr_model = Model(inputs=self.encoder.input,
                                   output=[t1, t2, t3, t4, self.autoencoder.output])

    def train(self, Epochs=100, Loss=['binary_crossentropy', 'binary_crossentropy',
                                      'binary_crossentropy', 'binary_crossentropy', 'binary_crossentropy'],
              Loss_weights=[1, 1, 1, 1, 1], Optimizer='Nadam', Callbacks=None):
        self.deepamr_model.compile(loss=masked_loss_function,
                                   loss_weights=Loss_weights,
                                   optimizer=Optimizer, metrics=[masked_accuracy])

        self.deepamr_model.fit(self.x_train, [self.y_train[:, 0], self.y_train[:, 1],
                                              self.y_train[:, 2], self.y_train[:, 3], self.x_train],
                               validation_data=(self.x_val, [self.y_val[:, 0], self.y_val[:, 1],
                                                             self.y_val[:, 2], self.y_val[:, 3], self.x_val]),
                               shuffle=False,
                               epochs=Epochs,
                               batch_size=self.batch_size,
                               callbacks=Callbacks)
        self.deepamr_model.save_weights('./best_weights.h5')
        self.deepamr_model.load_weights('./best_weights.h5')

        #        # optimise threshold: use train and validation data together to determine the best threshold
        x_train_tmp = np.concatenate((self.x_train, self.x_val), axis=0)
        y_train_tmp = np.concatenate((self.y_train, self.y_val), axis=0)

        print(x_train_tmp)
        train_pred = self.deepamr_model.predict(x_train_tmp, batch_size=self.batch_size)
        print(train_pred)
        th = []
        print("y_train_tmp.shape[1]: {}", y_train_tmp.shape[1])
        for i in range(y_train_tmp.shape[1]):
            th.append(eva_def().Find_Optimal_Cutoff(y_train_tmp[:, i], train_pred[i]))

        self.class_threshold = th

    def predict(self):
        y_pred_prob_tmp = self.deepamr_model.predict(self.x_test, batch_size=self.batch_size)
        y_pred_prob = self.deepamr_model.predict(self.x_test, batch_size=self.batch_size)
        th = self.class_threshold
        for i in range(len(th)):
            y_pred_prob_tmp[i] = np.where(y_pred_prob_tmp[i] > th[i], 1, 0)
            perf_mat = eva_def().performance_calculation(self.y_test[:, i], y_pred_prob_tmp[0], y_pred_prob[0])
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




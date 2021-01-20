#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 12:10:38 2019

@author: emma
"""

# computing an auxiliary target distribution

from tensorflow.keras.callbacks import *
from tensorflow.keras import backend as K
import numpy as np


class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitude of the cycle can be scaled on a per-iteration or
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each
        cycle iteration.
    For more detail, please see paper.

    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```

    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1 / (2. ** (x - 1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step_size))
        x = np.abs(self.clr_iterations / self.step_size - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())

    def on_batch_end(self, epoch, logs=None):

        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        K.set_value(self.model.optimizer.lr, self.clr())


import pandas as pd
from keras.callbacks import EarlyStopping, ModelCheckpoint
# from clr_callback import CyclicLR
from models.DeepAMR import deepamr, deepamr_cluster


def data_processor(df_train, df_label):
    df_train = df_train.merge(df_label, left_index=True, right_index=True)

    print('train set: {0}'.format(df_train.shape))
    labels = []

    labels_list = []

    dfStreptomycin = df_train[['streptomycin']]
    labels.append(dfStreptomycin)
    labels_list.append(df_train['streptomycin'])

    dfRifampicin = df_train[['rifampicin']]
    labels.append(dfRifampicin)
    labels_list.append(df_train['rifampicin'])

    dfPyrazinamide = df_train[['pyrazinamide']]
    labels.append(dfPyrazinamide)
    labels_list.append(df_train['pyrazinamide'])

    dfOfloxacin = df_train[['ofloxacin']]
    labels.append(dfOfloxacin)
    labels_list.append(df_train['ofloxacin'])


    dfMoxifloxacin = df_train[['moxifloxacin']]
    labels.append(dfMoxifloxacin)
    labels_list.append(df_train['moxifloxacin'])

    dfKanamycin = df_train[['kanamycin']]
    labels.append(dfKanamycin)
    labels_list.append(df_train['kanamycin'])

    dfIsoniazid = df_train[['isoniazid']]
    labels.append(dfIsoniazid)
    labels_list.append(df_train['isoniazid'])


    dfEthionamide = df_train[['ethionamide']]
    labels.append(dfEthionamide)
    labels_list.append(df_train['ethionamide'])

    dfEthambutol = df_train[['ethambutol']]
    labels.append(dfEthambutol)
    labels_list.append(df_train['ethambutol'])


    dfCiprofloxacin = df_train[['ciprofloxacin']]
    labels.append(dfCiprofloxacin)
    labels_list.append(df_train['ciprofloxacin'])

    dfCapreomycin = df_train[['capreomycin']]
    labels.append(dfCapreomycin)
    labels_list.append(df_train['capreomycin'])

    dfAmikacin = df_train[['amikacin']]
    labels.append(dfAmikacin)
    labels_list.append(df_train['amikacin'])

    df_train = df_train.drop(['streptomycin'], axis=1)
    df_train = df_train.drop(['rifampicin'], axis=1)
    df_train = df_train.drop(['pyrazinamide'], axis=1)
    df_train = df_train.drop(['ofloxacin'], axis=1)

    df_train = df_train.drop(['moxifloxacin'], axis=1)
    df_train = df_train.drop(['kanamycin'], axis=1)
    df_train = df_train.drop(['isoniazid'], axis=1)

    df_train = df_train.drop(['ethionamide'], axis=1)
    df_train = df_train.drop(['ethambutol'], axis=1)

    df_train = df_train.drop(['ciprofloxacin'], axis=1)
    df_train = df_train.drop(['capreomycin'], axis=1)
    df_train = df_train.drop(['amikacin'], axis=1)

    return df_train, pd.concat(labels, axis=1).replace(np.nan, -1)



from loading_data import load_data


def process(numOfFiles, nrow=0):
    # ../../../../ project / compbio - lab / Drug - resistance - TB /
    df_train = load_data.load_data(list(range(38, numOfFiles)), '../Data/', nrow)

    print('train set: {0}'.format(df_train.shape))

    return df_train


def main_deepamr():
    # load data-------------------
    print('load data...')
    X1 = process(2)
    # X1 = pd.read_csv('../Data/gene_data.csv', index_col=0)
    # Y1 = pd.read_csv('../Data/AllLabels.csv', index_col=0)

    # X, Y = data_processor(X1, Y1)
    # print(X)

    # X1 = pd.read_csv('../Data/gene_data_19.csv', index_col=0)
    Y1 = pd.read_csv('../Data/AllLabels.csv', index_col=0)

    X, Y = data_processor(X1, Y1)
    # print(X)
    # Y = Y.replace(np.nan, -1)
    M = deepamr(X, Y)
    # -------------------------------------------------------
    print('preparing data in train, validation and test...')
    M.data_prep()
    # -----------------------------------------------------------------
    print('construct deep autoencoder...')
    M.AutoEncoder()
    # ----------------------------------------------------------
    print('pre_training...')
    best_weights_filepath = './ae_best_weights.hdf5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='auto')
    esp = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    clr = CyclicLR(base_lr=0.001, max_lr=0.9,
                   step_size=100., mode='triangular2')
    call_backs = [clr, esp, saveBestModel]
    M.pre_train(Epochs=5, Callbacks=call_backs)
    # -------------------------------------------------------------
    print('contruct deepamr model...')
    M.build_model()
    print(M.deepamr_model.summary())
    # -----------------------------------------------------------
    print('train deeparm...')
    best_weights_filepath = './best_weights.hdf5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='auto')
    esp = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    clr = CyclicLR(base_lr=0.0001, max_lr=0.003,
                   step_size=100., mode='triangular2')
    call_backs = [clr, esp, saveBestModel]
    M.train(Epochs=5, Callbacks=call_backs)
    # ----------------------------------------------------------
    print('testing....')
    M.predict()


def main_deepamr_cluster():
    print('loading data...')
    X = pd.read_csv('Data.csv', index_col=0)
    Y = pd.read_csv('Labels.csv', index_col=0)
    L = pd.read_csv('Cluster_label.csv', index_col=0)
    M = deepamr_cluster(X, Y, L)
    # --------------------------------------------
    # -------------------------------------------------------
    print('preparing data in train, validation and test...')
    M.data_prep()
    # -----------------------------------------------------------------
    print('construct deep autoencoder...')
    M.AutoEncoder()
    # ----------------------------------------------------------
    print('pre_training...')
    best_weights_filepath = './ae_best_weights.hdf5'
    saveBestModel = ModelCheckpoint(best_weights_filepath, monitor='val_loss', verbose=1, save_best_only=True,
                                    mode='auto')
    esp = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    clr = CyclicLR(base_lr=0.001, max_lr=0.9,
                   step_size=100., mode='triangular2')
    call_backs = [clr, esp, saveBestModel]
    M.pre_train(Epochs=1, Callbacks=call_backs)
    # -------------------------------------------------------------
    print('contruct deepamr model...')
    M.build_model()
    print(M.deepamr_cluster.summary())
    # -----------------------------------------------------------
    print('train deeparm_cluster...')
    M.train(Epochs=1)
    # ----------------------------------------------------------
    print('testing....')
    M.predict()


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    import random as rn
    import os

    os.environ['PYTHONHASHSEED'] = '0'

    from keras import backend as K

    np.random.seed(27)
    rn.seed(27)
    # tf.set_random_seed(27)
    print('--------------------------------------------')
    print('DeepAMR Demo')
    # sess = tf.Session(graph=tf.get_default_graph())
    # K.set_session(sess)
    main_deepamr()
    K.clear_session()

    # print('--------------------------------------------')
    # print('DeepAMR_Cluster Demo')
    # sess = tf.Session(graph=tf.get_default_graph())
    # K.set_session(sess)
    # main_deepamr_cluster()
    # K.clear_session()
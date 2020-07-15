#!/usr/bin/env python3
import numpy as np
import pandas as pd

"""
Created on Tue Mar 10 18:23:59 2020

@author: nafiseh
"""

num = 20


def load_data(lst, path, nrow=0):
    lst = sorted(np.unique(lst))
    dtypes = dict(zip(list(range(0, 20000)), [np.int8] * 20000))

    if lst[0] == 1:
        dt = pd.read_csv(path + 'file-' + str(lst[0]) + '.csv', header=None, nrows=num)
        dt.set_index(dt.columns[0], inplace=True, drop=True)
        dt = dt.astype(np.int8)
    else:
        dt = pd.read_csv(path + 'file-' + str(lst[0]) + '.csv', header=None, dtype=dtypes, nrows=num)
        isolates = pd.read_csv(path + 'Isolates.csv', header=None)[0].values.tolist()
        dt.index = isolates

    for i in lst[1:]:
        tmp = pd.read_csv(path + 'file-' + str(i) + '.csv', header=None, dtype=dtypes, nrows=num)
        tmp.index = dt.index
        dt = pd.concat([dt, tmp], axis='columns', ignore_index=True)

    dt.columns = list(range(0, len(dt.columns)))

    print(dt.shape)
    dt = dt.loc[:, dt.sum() >= nrow]
    print("After dropping")
    print(dt.shape)

    return (dt)


def load_label(path):
    dt = pd.read_csv(path + 'AllLabels' + '.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)
    return dt


def load_data_gene(lst, path):
    lst = sorted(np.unique(lst))

    if lst[0] == 0:
        dt = pd.read_pickle(path + 'Isolate_Gene_Seqs_' + str(lst[0]) + '_200.pkl')
        # print(dt.iloc[0])
        dt.set_index(dt.columns[0])
        # dt = pd.DataFrame(
        #     np.row_stack([dt.columns, dt.values]),
        #     columns=dt.iloc[0]
        # )



    for i in lst[1:]:
        tmp = pd.read_pickle(path + 'Isolate_Gene_Seqs_' + str(lst[i]) + '_200.pkl')
        tmp.index = dt.index
        dt = pd.concat([dt, tmp], axis='columns', ignore_index=True)

    dt.columns = list(range(0, len(dt.columns)))

    print(dt.shape)
    print(dt.head())
    return (dt)


def load_data_gene_dataset(path):
    dt = pd.read_csv(path + 'gene_data.csv', header=None)
    dt.set_index(dt.columns[0], inplace=True, drop=True)
    return dt


def load_data_shuffle_dataset(path, index_file):
    dt = pd.read_csv(path + "shuffled_index_" + str(index_file) + ".csv", header=None)
    dt.set_index(dt.columns[0], inplace=True, drop=True)
    return dt


def load_data_random_dataset(path):
    dt = pd.read_csv(path + 'random_data.csv', header=None)
    dt.set_index(dt.columns[0], inplace=True, drop=True)
    return dt


if __name__ == '__main__':
    # LoadData(list(range(1, 2)), 'Data/', 0)
    load_data_gene(list(range(0, 1)), 'Data/')
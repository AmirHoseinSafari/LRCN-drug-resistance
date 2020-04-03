#!/usr/bin/env python3
import numpy as np
import pandas as pd

"""
Created on Tue Mar 10 18:23:59 2020

@author: nafiseh
"""

num = 40
def LoadData(lst, path):
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

    return (dt)


def LoadLabel(path):
    dt = pd.read_csv(path + 'AllLabels' + '.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)
    return dt

import pandas as pd

import data_preprocess


def LoadLabel(path):
    dt = pd.read_csv(path + 'AllLabels' + '.csv')
    dt.set_index(dt.columns[0], inplace=True, drop=True)
    return dt


df_train, dt = data_preprocess.process(2, 0)

# dt = LoadLabel('Data/')

# print(dt)

dfPyrazinamide = dt[2]
# print(dfPyrazinamide)

arr = dfPyrazinamide.values.tolist()
zero = 0
one = 0
nan = 0
for i in range(0, len(arr)):
    if arr[i][0] == 1:
        one = one + 1
    elif arr[i][0] == 0:
        zero = zero + 1
    else:
        nan = nan + 1
print("pyrazinamide")
print(one)
print(zero)
print(nan)


# dfPyrazinamide = dt[['moxifloxacin']]
# # print(dfPyrazinamide)
#
# arr = dfPyrazinamide.values.tolist()
# zero = 0
# one = 0
# nan = 0
# for i in range(0, len(arr)):
#     if arr[i][0] == 1:
#         one = one + 1
#     elif arr[i][0] == 0:
#         zero = zero + 1
#     else:
#         nan = nan + 1
# print("moxifloxacin")
# print(one)
# print(zero)
# print(nan)
#
# dfPyrazinamide = dt[['streptomycin']]
# # print(dfPyrazinamide)
#
# arr = dfPyrazinamide.values.tolist()
# zero = 0
# one = 0
# nan = 0
# for i in range(0, len(arr)):
#     if arr[i][0] == 1:
#         one = one + 1
#     elif arr[i][0] == 0:
#         zero = zero + 1
#     else:
#         nan = nan + 1
# print("streptomycin")
# print(one)
# print(zero)
# print(nan)



dfPyrazinamide = dt[9]

arr = dfPyrazinamide.values.tolist()
zero = 0
one = 0
nan = 0
for i in range(0, len(arr)):
    if arr[i][0] == 1:
        one = one + 1
    elif arr[i][0] == 0:
        zero = zero + 1
    else:
        nan = nan + 1
print("ciprofloxacin")
print(one)
print(zero)
print(nan)


dfPyrazinamide = dt[7]

arr = dfPyrazinamide.values.tolist()
zero = 0
one = 0
nan = 0
for i in range(0, len(arr)):
    if arr[i][0] == 1:
        one = one + 1
    elif arr[i][0] == 0:
        zero = zero + 1
    else:
        nan = nan + 1
print("ethionamide")
print(one)
print(zero)
print(nan)
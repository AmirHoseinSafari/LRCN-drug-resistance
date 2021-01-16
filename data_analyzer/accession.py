import pandas as pd

l = []
new = []
f = open("/Users/amir/PycharmProjects/Lab/LSTM-DR/LSTM-drug-resistance/data_analyzer/list.txt", "r")
for x in f:
    tmp = x.split(',')
    for i in range(0, len(tmp)):
        l.append(tmp[i].rstrip("\n"))

print(len(l))


f = open("/Users/amir/PycharmProjects/Lab/LSTM-DR/LSTM-drug-resistance/data_analyzer/list copy.txt", "r")
for x in f:
    tmp = x.split('/')
    print(tmp[0])
    l.append(tmp[0].rstrip("\n"))

print(len(l))

dt = pd.read_csv('/Users/amir/PycharmProjects/Lab/LSTM-DR/LSTM-drug-resistance/Data/shuffled_index.csv', header=None)
dt.set_index(dt.columns[0], inplace=True, drop=True)

isolates = dt.index.values
isolates = list(isolates)

print(isolates)
e = 0
ee = 0
for i in range(0 , len(isolates)):
    exixt = 0
    for j in range(0, len(l)):
        if isolates[i] == l[j]:
            e = e + 1
            exixt = 1
            break
    if exixt == 0:
        ee = ee + 1
        print(isolates[i])

print(e)
print(ee)
import random
import data_loader
import csv

data = []

df_train = data_loader.process(2)

isolates = df_train.index.values
isolates = list(isolates)


for i2 in range(0, len(isolates)):
    tmp = []
    tmp.append(isolates[i2])
    tmp.extend([random.randrange(-12, 7, 1) for i in range(3979)])
    for i3 in range(1, len(tmp)):
        if tmp[i3] < 0:
            tmp[i3] = 0
    data.append(tmp)



with open("random_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)


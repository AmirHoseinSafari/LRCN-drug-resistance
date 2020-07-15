import random
import data_loader
import csv

import data_preprocess


def random_dataset():
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


def shuffle_data():
    df_train, labels = data_preprocess.process(38, gene_dataset=True)
    import random
    print(df_train.head())
    for i in range(0, 4):
        cols = df_train.columns.tolist()
        random.shuffle(cols)
        df_train = df_train[cols]
        print(df_train.head())
        df_train.to_csv("shuffled_index_" + str(i) +".csv")



if __name__ == '__main__':
    shuffle_data()


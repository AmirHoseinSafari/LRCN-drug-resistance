import LoadData
import pandas as pd


def process(numOfFiles, nrow=0, gene=False):
    if not gene:
        df_train = LoadData.LoadData(list(range(1, numOfFiles)), 'Data/', nrow)
    else:
        df_train = LoadData.load_data_gene(list(range(0, numOfFiles)), 'Data/')

    # df_train = df_train[df_train.columns[df_train.sum() > 5]]

    df_label = LoadData.LoadLabel('Data/')

    print('label set: {0}'.format(df_label.shape))

    df_train = df_train.merge(df_label, left_index=True, right_index=True)

    print('train set: {0}'.format(df_train.shape))
    labels = []

    dfStreptomycin = df_train[['streptomycin']]
    labels.append(dfStreptomycin)

    dfRifampicin = df_train[['rifampicin']]
    labels.append(dfRifampicin)

    dfPyrazinamide = df_train[['pyrazinamide']]
    labels.append(dfPyrazinamide)

    dfOfloxacin = df_train[['ofloxacin']]
    labels.append(dfOfloxacin)

    dfMoxifloxacin = df_train[['moxifloxacin']]
    labels.append(dfMoxifloxacin)

    dfKanamycin = df_train[['kanamycin']]
    labels.append(dfKanamycin)

    dfIsoniazid = df_train[['isoniazid']]
    labels.append(dfIsoniazid)

    dfEthionamide = df_train[['ethionamide']]
    labels.append(dfEthionamide)

    dfEthambutol = df_train[['ethambutol']]
    labels.append(dfEthambutol)

    dfCiprofloxacin = df_train[['ciprofloxacin']]
    labels.append(dfCiprofloxacin)

    dfCapreomycin = df_train[['capreomycin']]
    labels.append(dfCapreomycin)

    dfAmikacin = df_train[['amikacin']]
    labels.append(dfAmikacin)

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
    # print(dfStreptomycin.head(10))
    if gene:
        df_train = one_hot_gene(df_train)

    return df_train, labels


def one_hot_gene(df_train):
    # df_train = df_train.iloc[0:50]
    tmp = df_train.values.tolist()
    for i in range(0, len(tmp)):
        if i % 100 == 0:
            print(i)
        for j in range(0, len(tmp[0])):
            one = []
            for k in range(0, len(tmp[i][j])):
                if tmp[i][j][k] == 'A':
                    one.append(True)
                    one.append(False)
                    one.append(False)
                    one.append(False)
                elif tmp[i][j][k] == 'C':
                    one.append(False)
                    one.append(True)
                    one.append(False)
                    one.append(False)
                elif tmp[i][j][k] == 'T':
                    one.append(False)
                    one.append(False)
                    one.append(True)
                    one.append(False)
                elif tmp[i][j][k] == 'G':
                    one.append(False)
                    one.append(False)
                    one.append(False)
                    one.append(True)
                else:
                    print("Error!")
            tmp[i][j] = one
    # print(pd.DataFrame(tmp))
    return pd.DataFrame(tmp)
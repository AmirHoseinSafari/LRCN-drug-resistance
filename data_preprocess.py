import load_data
import pandas as pd


def process(num_of_files, nrow=0, gene=False, limited=False, gene_dataset=False, shuffle_index=False, random_data=False,
            shuffle_operon=False, index_file=0):
    # ../../../../ project / compbio - lab / Drug - resistance - TB /
    if gene:
        df_train = load_data.load_data_gene(list(range(0, num_of_files)), 'Data/')
    elif gene_dataset:
        df_train = load_data.load_data_gene_dataset('Data/')
    elif shuffle_index:
        df_train = load_data.load_data_shuffle_dataset('Data/', index_file)
    elif random_data:
        df_train = load_data.load_data_random_dataset('Data/')
    elif shuffle_operon:
        df_train = load_data.load_data_operon_dataset('Data/')
    else:
        df_train = load_data.load_data(list(range(1, num_of_files)), 'Data/', nrow)

    # df_train = df_train[df_train.columns[df_train.sum() > 5]]

    df_label = load_data.load_label('Data/')

    if limited:
        df_label = df_label.drop(['ciprofloxacin'], axis=1)
        df_label = df_label.drop(['capreomycin'], axis=1)
        df_label = df_label.drop(['amikacin'], axis=1)
        df_label = df_label.drop(['ethionamide'], axis=1)
        df_label = df_label.drop(['moxifloxacin'], axis=1)


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

    if not limited:
        dfMoxifloxacin = df_train[['moxifloxacin']]
        labels.append(dfMoxifloxacin)

    dfKanamycin = df_train[['kanamycin']]
    labels.append(dfKanamycin)

    dfIsoniazid = df_train[['isoniazid']]
    labels.append(dfIsoniazid)

    if not limited:
        dfEthionamide = df_train[['ethionamide']]
        labels.append(dfEthionamide)

    dfEthambutol = df_train[['ethambutol']]
    labels.append(dfEthambutol)

    if not limited:
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
    if not limited:
        df_train = df_train.drop(['moxifloxacin'], axis=1)
    df_train = df_train.drop(['kanamycin'], axis=1)
    df_train = df_train.drop(['isoniazid'], axis=1)
    if not limited:
        df_train = df_train.drop(['ethionamide'], axis=1)
    df_train = df_train.drop(['ethambutol'], axis=1)
    if not limited:
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
from loading_data import load_data
import pandas as pd
from scipy.stats import pearsonr

def common_elements(l):
    import matplotlib.pyplot as plt
    import seaborn as sb
    df_m = []
    print(len(l[0]))
    for i in range(0, len(l)):
        for i2 in range(0, len(l[i])):
            if l[i][i2] != 0 and l[i][i2] != 1:
                l[i][i2] = -1
    for i in range(0, len(l)):
        r = []
        for j in range(0, len(l)):
            one = 0
            corr, _ = pearsonr(l[i], l[j])
            for i2 in range(0, len(l[i])):
                if l[i][i2] == l[j][i2] == 0:
                    one = one + 1
            r.append(corr)
            # print(corr)
        df_m.append(r)
        print("____________")

    fig, ax = plt.subplots(figsize=(12, 6))
    print(df_m)
    x_axis_labels = ["Streptomycin",	"Rifampicin",	"Pyrazinamide",	"Ofloxacin",	"Moxifloxacin",	"Kanamycin",	"Isoniazid",	"Ethionamide",	"Ethambutol",	"Ciprofloxacin",	"Capreomycin",	"amikacin"]
    sb.heatmap(df_m, xticklabels=x_axis_labels, yticklabels=x_axis_labels)

    plt.savefig('heatmap.png',  bbox_inches='tight')
    plt.show()
    return 0


def process(num_of_files, nrow=0, gene=False, limited=False, gene_dataset=False, shuffle_index=False, random_data=False,
            shuffle_operon=False, shuffle_operon_locally=False, shuffle_operon_group=False, index_file=0):
    # ../../../../project/compbio-lab/Drug-resistance-TB/
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
    elif shuffle_operon_locally:
        df_train = load_data.load_data_operon_locally_dataset('Data/')
    elif shuffle_operon_group:
        df_train = load_data.load_data_operon_group_dataset('Data/')
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


    if not limited:
        dfMoxifloxacin = df_train[['moxifloxacin']]
        labels.append(dfMoxifloxacin)
        labels_list.append(df_train['moxifloxacin'])

    dfKanamycin = df_train[['kanamycin']]
    labels.append(dfKanamycin)
    labels_list.append(df_train['kanamycin'])


    dfIsoniazid = df_train[['isoniazid']]
    labels.append(dfIsoniazid)
    labels_list.append(df_train['isoniazid'])


    if not limited:
        dfEthionamide = df_train[['ethionamide']]
        labels.append(dfEthionamide)
        labels_list.append(df_train['ethionamide'])

    dfEthambutol = df_train[['ethambutol']]
    labels.append(dfEthambutol)
    labels_list.append(df_train['ethambutol'])


    if not limited:
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


    # common_elements(labels_list)

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
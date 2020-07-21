from loading_data import load_data


def process(numOfFiles, nrow=0):
    # ../../../../ project / compbio - lab / Drug - resistance - TB /
    df_train = load_data.load_data(list(range(1, numOfFiles)), 'Data/', nrow)

    print('train set: {0}'.format(df_train.shape))

    return df_train



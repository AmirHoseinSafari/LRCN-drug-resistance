import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import load_data
import pandas as pd


def process(numOfFiles, nrow=0):
    # ../../../../ project / compbio - lab / Drug - resistance - TB /
    df_train = load_data.load_data(list(range(1, numOfFiles)), '../Data/', nrow)

    print('train set: {0}'.format(df_train.shape))

    return df_train



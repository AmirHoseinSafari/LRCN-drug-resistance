import models
from loading_data import data_preprocess
from models import ML_methods, model_one_vs_all

epochs = 200

# drop columns below 5 SNP
# df_train, labels = data_preprocess.process(6, 5)
# model.run_model(df_train, labels, epochs)
# model_one_vs_all.run_model(df_train, labels, epochs)

# without dropping


# df_train, labels = data_preprocess.process(1, gene=True)
# def run(i):
    # if i == 0:
    #     model_one_vs_all.run_bayesian(df_train, labels, epochs, limited=False, portion=0.1)
    # elif i == 1:
    #     model_one_vs_all.run_bayesian(df_train, labels, epochs, limited=False, portion=0.2)
    # elif i == 2:
    #     model_one_vs_all.run_bayesian(df_train2, labels2, epochs, limited=True, portion=0.1)
    # else:
    #     model_one_vs_all.run_bayesian(df_train2, labels2, epochs, limited=True, portion=0.2)

# def BO():
#     df_train, labels = data_preprocess.process(3, limited=False)
#     df_train2, labels2 = data_preprocess.process(3, limited=True)
#
#     pool = multiprocessing.Pool(processes=4)
#     pool.map (run, (i for i in range(0, 4)))

def train():
    df_train, labels = data_preprocess.process(2)
    print(type(df_train))
    models.model_one_vs_all.run_model(df_train, labels, epochs, limited=True)
    ML_methods.model_run(df_train, labels)


if __name__ == '__main__':
    train()
import data_preprocess
import model_gene_based
import ML_methods


epochs = 200


def train():
    df_train, labels = data_preprocess.process(38, gene_dataset=True)
    ML_methods.model_run(df_train, labels)
    # model_gene_based.run_model(df_train, labels, epochs)
    # model_gene_based.run_bayesian(df_train, labels)
    # model_gene_based.run_all(df_train, labels, epochs)
    # model_gene_based.run_model_kfold(df_train,labels,epochs)


def train_shuffle():
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=0)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=0)
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=1)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=1)
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=2)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=2)
    df_train, labels = data_preprocess.process(38, shuffle_index=True, index_file=3)
    model_gene_based.run_model_kfold(df_train, labels, epochs, index=3)


if __name__ == '__main__':
    train()
    # train_shuffle()

import data_preprocess
import model_gene_based
import ML_methods


epochs = 200

def train():
    df_train, labels = data_preprocess.process(38, shuffle_index=True)
    # ML_methods.model_run(df_train, labels)
    # model_gene_based.run_model(df_train, labels, epochs)
    model_gene_based.run_model_kfold(df_train, labels, epochs)
    # model_gene_based.run_all(df_train, labels, epochs)
    # model_gene_based.run_model_kfold(df_train,labels,epochs)
    df_train, labels = data_preprocess.process(38, random_data=True)
    model_gene_based.run_model_kfold_tmp(df_train, labels, epochs)


if __name__ == '__main__':
    train()
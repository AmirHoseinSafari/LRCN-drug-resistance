import data_preprocess
import model_gene_based
import ML_methods


epochs = 200

def train():
    df_train, labels = data_preprocess.process(38, gene_dataset=True, shuffle_index=False)
    # ML_methods.model_run(df_train, labels)
    # model_gene_based.run_model(df_train, labels, epochs)
    model_gene_based.run_bayesian(df_train, labels)
    # model_gene_based.run_all(df_train, labels, epochs)
    # model_gene_based.run_model_kfold(df_train,labels,epochs)

if __name__ == '__main__':
    train()
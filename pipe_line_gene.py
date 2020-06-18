import data_preprocess
import model_gene_based


epochs = 200

def train():
    df_train, labels = data_preprocess.process(38, gene_dataset=True)
    model_gene_based.run_model(df_train, labels, epochs)


if __name__ == '__main__':
    train()
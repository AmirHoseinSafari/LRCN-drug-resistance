import data_preprocess
import model
import model_one_vs_all
import ML_methods

epochs = 2

# drop columns below 5 SNP
# df_train, labels = data_preprocess.process(6, 5)
# model.run_model(df_train, labels, epochs)
# model_one_vs_all.run_model(df_train, labels, epochs)

# without dropping


# df_train, labels = data_preprocess.process(1, gene=True)
df_train, labels = data_preprocess.process(3, limited=True)
model_one_vs_all.run_model(df_train, labels, epochs, limited=True)


# df_train, labels = data_preprocess.process(6)
# ML_methods.model_run(df_train, labels)



import data_preprocess
import model
import model_one_vs_all

epochs = 200

# drop columns below 5 SNP
# df_train, labels = data_preprocess.process(6, 5)
# model.run_model(df_train, labels, epochs)
# model_one_vs_all.run_model(df_train, labels, epochs)

# without dropping
df_train, labels = data_preprocess.process(6)
# model.run_model(df_train, labels, epochs)
model_one_vs_all.run_model(df_train, labels, epochs)



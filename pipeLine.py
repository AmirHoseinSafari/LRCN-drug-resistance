import data_preprocess
import model
import model_one_vs_all

epochs = 200

df_train, labels = data_preprocess.process(6)
model.run_model(df_train, labels, epochs)
model_one_vs_all.run_model(df_train, labels, epochs)

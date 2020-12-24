## LRCN drug resistance

In this repo we implemented the methods which we used at "Predicting drug resistance in M. tuberculosis using a Long-term Recurrent Convolutional Networks architecture" paper

The main method is Long-term Recurrent Convolutional Network (LRCN).

We used the datasets at [M.tuberculosis-dataset-for-drug-resistant](https://github.com/AmirHoseinSafari/M.tuberculosis-dataset-for-drug-resistant) repository for this code.

### pipe_line_gene.py:

This is the main class for running the models on the gene-based dataset, it simply loads the data using "data_preprocess.process()" and then run the proper model on it.

### pipe_line.py:

This is the main class for running the models on the SNP-based dataset, it simply loads the data using "data_preprocess.process()" and then run the proper model on it.

### create_dataset/gene_dataset_creator.py:

This code creates the gene dataset, using the SNP-based dataset. 

### create_dataset/operons_finder.py

This code finds the operon indexes in the gene-based dataset.

### create_dataset/shuffle_operon_dataset_creator.py:

We shuffled the features in this code as we explained in the paper.

### loading_data package:

Codes here are for loading the proper dataset. You need to call the "data_preprocess.process()" function for this purpose. 
parameters are as follow:

---
num_of_files: this is for the SNP-based data, you can specify the number of files that you want to load.
nrow: num of rows which are loaded.
limited: default is False, if you set it True, five of the drugs would be drop (ciprofloxacin, capreomycin, amikacin, ethionamide, moxifloxacin)
gene_dataset: default is False, if you set it True then the gene-based data would be loaded. 
---

You don't need to modify or call the other methods.

### models/model_gene_based.py:

you need to run the "run_bayesian()" function with proper data as input. it will run the proper functions in other classes and will print the output.

### models/Bayesian_optimizer.py:
This file has the main implementation of the model. Which is the implementation of LRCN with K-fold and the Bayesian optimization.


---

## Citation
If you found the content of this repository useful, please cite us:

https://www.biorxiv.org/content/10.1101/2020.11.07.372136v1?rss=1

---

## How to use LRCN

In order to run LRCN, you first need to have your dataset properly transformed to the LRCN expected format; that is a `.csv` file which includes a table in which rows correspond to the samples (isolates) and the coloumns contain features. The features could be in SNP binary form or in gene aggregated format (please refer to our [paper](https://www.biorxiv.org/content/10.1101/2020.11.07.372136v1?rss=1) and our dataset and its creation repos [here](https://github.com/AmirHoseinSafari/M.tuberculosis-dataset-for-drug-resistant) and [here](https://github.com/AmirHoseinSafari/Genotype-collector-and-SNP-dataset-creator) for more details on how to generate these data).
For this tutorial, we are going to use a mini-dataset which is a subset of the dataset we used in [our paper](https://www.biorxiv.org/content/10.1101/2020.11.07.372136v1?rss=1).

This mini-dataset contains mutation burden for each gene for the multiple isolates as shown here.

For demonstration purpose, we will be using a randomized label set for 12 drug compounds and use it as the prediction target.

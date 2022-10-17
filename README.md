# Adaptive-m6A

### Sample code for research "Identification of species-specific RNA N6-methyladinosine modification sites from RNA sequences"

### train.py: 

contains the code for training from skratch. In this file, the dataset from file './data/mm10/mm10_positive_training.fa' and './data/mm10/mm10_negative_training.fa' were used for the training. 

The generated model would be saved under the folder './Result/Adapt_Train_mm10/fold_X', which is the default saving path. 'X' refers to the cross-validation fold number.

### test.py: 

loading the pre-trained model (for instance, generated from previous training step) and use other dataset for independent testing. Here the dataset from './data/mm10/mm10_positive_testing.fa' and './data/mm10/mm10_negative_testing.fa' were used for the independent testing process.

### residue2idx.pkl 

look-up table for embedding layer process.


### data: 

in the data.zip file, some demo data were provided. These data could be directly used for demonstration. You can also use your desired dataset.

### checkpoint.pth.tar: 

pre-trained model for demonstration.



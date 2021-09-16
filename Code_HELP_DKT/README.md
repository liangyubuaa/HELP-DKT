# HELP-DKT Model

## Introduction

The folder consists of four different modules.

`HELP_DKT_Model.py` is the core file which contains HELP-DKT model architecture.

`RunModel.py` is the program to run our model. It first reads train/test dataset and pre-process data before training. Then, we run the model for 500 epochs and output test result.

`getDataA/B/C.py` are three similar Pyhton files to divide original data into train and test dataset according to the requirements of Task A/B/C.

`processData.py` modifies the forms of data for training model.

`main.py` is the main program. You can easily run Task A/B/C by running `main.py` only once, because the `main.py` will automatically run our model three times.

## Dataset
The original Python code files are compressed in the path:
[../Data/Original_Codes.zip](../Data/Original_Codes.zip)

### Processing data

In `./data` folder:
- `ModelInput` contains the training/testing dataset of model (get from [Program_Vector_Embeddings.CSV](../Data/Program_Vector_Embeddings.CSV))
- `ModelOutput` contains the auc, loss and student's ability levels results of model.  

## Usage

This repository is the `PyTorch` implementation for HELP-DKT.

The hyperParameters are in the `RunModel.py` file.


Run:

```
HELP_DKT/Code_HELP_DKT/main.py
```

the file will automatically run the model three times for Task A/B/C separately.

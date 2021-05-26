# Kernel Method Project


Project from the MVA course Kernel methods for machine learning.

## Introduction 
This [Kaggle](https://www.kaggle.com/c/machine-learning-with-kernel-methods-2021) competition is a sequence classification task: 
predicting whether a DNA sequence region is binding site to a specific transcription factor.

Transcription factors (TFs) are regulatory proteins that bind specific sequence motifs in the genome to activate or repress transcription of target genes.
Genome-wide protein-DNA binding maps can be profiled using some experimental techniques and thus all genomics can be classified into two classes for a TF of interest: bound or unbound.
In this challenge, we will work with three datasets corresponding to three different TFs.

## Leaderboard


## Run the  submission
Just run this command in a terminal:

```bash
python start.py
```

## Kernels method available:
### Kernels for numeric data:
- Linear kernel
- Polynomial kernel
- Rbf kernel

### Kernels for string data:
- Spectrum kernel
- Mismatch kernel


## Methods availables:

- Logistic regression (KLR)
- Support vector machine (KSVM)

### Experiments:
 

Train for 3 sets with different parameters
```bash
python -m src.experiments.exp_combine_set

```
Train for each set
```bash
python -m src.experiments.exp_seperate_set
```
Train all data at the same time
```bash
python -m src.experiments.exp_baseline
```

## Resources

| Path | Description
| :--- | :----------
| [Kernel]() | Main folder.
| &boxvr;&nbsp; [data]() | data folder.
| &boxvr;&nbsp; [fit]() | Folder to store some Gram matrixes for train, validation and test.
| &boxvr;&nbsp; [results]() | Store the results of the classification.
| &boxvr;&nbsp; [src]() | the main source codes.
| &boxv;&nbsp; &boxvr;&nbsp; [lib]() | library.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [kernels]() | kernels for numeric and string data.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [methods]() | KLR and KSVM.
| &boxv;&nbsp; &boxvr;&nbsp; &boxvr;&nbsp; [tools]()   | data_processing and utils.
| &boxv;&nbsp; &boxvr;&nbsp; [experiments]() | some experiments test cases.
| &boxv;&nbsp; &boxvr;&nbsp; [config.py]() | some configurations.
| &boxvr;&nbsp; [start.py]() | run the test case.

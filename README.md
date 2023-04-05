# Ge3Net

Experiments were run on three genotypes - humans, dogs and ancient and for geography, unsupervised space constructed from pca and umap. 
1. Build labels by running the script buildLabels.py
2. For training, run trainer.py
3. For inference only with a pre-trained model, run inference.py

## Learnt Representations from Ge3Net
![Learnt Representations from Ge3Net](./images/LearntRepresentations.svg)

<!-- ## Project structure


```console
$ tree
.
├── README.md
├── data                  # <-- Directory with raw and intermediate data
│   ├── data.xml          # <-- Initial XML StackOverflow dataset (raw data)
│   ├── data.xml.dvc      # <-- .dvc file - a placeholder/pointer to raw data
│   ├── features          # <-- Extracted feature matrices
│   │   ├── test.pkl
│   │   └── train.pkl
│   └── prepared          # <-- Processed dataset (split and TSV formatted)
│       ├── test.tsv
│       └── train.tsv
├── evaluation
│   ├── importance.png    # <-- Feature importance plot
│   └── plots             # <-- Data points for ROC, PRC, confusion matrix
│       ├── confusion_matrix.json
│       ├── precision_recall.json
│       └── roc.json
├── dvc.lock
├── dvc.yaml              # <-- DVC pipeline file
├── model.pkl             # <-- Trained model file
├── params.yaml           # <-- Parameters file
├── evaluation.json       # <-- Binary classifier final metrics (e.g. AUC)
└── src                   # <-- Source code to run the pipeline stages
    ├── evaluate.py
    ├── featurization.py
    ├── prepare.py
    ├── requirements.txt  # <-- Python dependencies needed in the project
    └── train.py
``` -->
## Acknowledgements
Here we reference publicly available third party code implementations that are used/modified in our code base
BOCD implementation available at <https://github.com/gwgundersen/bocd> based on the original Bayesian Changepoint Detection paper <https://arxiv.org/abs/0710.3742>
Pyadmix module implementation from <https://github.com/AI-sandbox/gnomix>

## Citation
If you find Ge3Net useful for your research, please consider citing our paper/software:
```
@article{Rastogi_Ge3Net_Inferring_Continuous_2021,
author = {Rastogi, Richa and Kumar, Arvind S. and Hilmarsson, Helgi and Bustamante, Carlos D. and Montserrat, Daniel Mas and Ioannidis, Alexander G.},
doi = {10.5281/zenodo.7800263},
title = {{Ge3Net: Inferring Continuous Population Structure Coordinates Along the Genome }},
year = {2021}
}
```

## Feedback
Please send feedback/issues related to this repository or the paper to [here](rr568@cornell.edu)


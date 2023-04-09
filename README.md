# Ge3Net
Our work targets the disparities in genomic medicine that are emerging between European-descent populations and all other populations. 

Personalized genomic predictions are revolutionizing medical diagnosis and treatment. These predictions rely on associations between health outcomes (disease severity, drug response, cancer risk) and correlated neighboring positions along the genome. However, these local genomic correlations differ widely amongst worldwide populations, necessitating that genetic research include all human populations. For admixed populations further computational challenges arise, because individuals of diverse combined ancestries inherit genomic segments from multiple ancestral populations. To extend population-specific associations to such individuals, their multiple ancestries must be identified along their genome (local ancestry inference, LAI). Here we introduce Ge3Net, Genomic Geographic Geometric Network, the first LAI method to identify ancestral origin of \emph{each segment} of an individual's genome as a \emph{continuous coordinate}, rather than an ethnic category, using a transformer based framework, yielding higher resolution \emph{local} ancestry inference, and eliminating a need for ethnic labels.

By annotating ancestry along the genome accurately, and with simple to use coordinates, we hope to enable genetic researchers to incorporate ancestry-specific genetic effects into their future models with ease. This could help to extend the benefits of such research and models to more diverse cohorts. Ge3Net is particularly targeted at improving genetic modeling applied to admixed individuals. Such individuals inherit genomic segments from diverse populations that have very different genetic correlation (linkage). This ancestry-specific structure must be identified for each segment of the genome to apply appropriate ancestry-specific risk models.

## Paper
The paper can be accessed from [Ge3Net.pdf](Ge3Net.pdf). A short version of this work was presented at Neurips Learning Meaningful Representations of Life, 2020


## Learnt Representations from Ge3Net
![Learnt Representations from Ge3Net](./images/LearntRepresentations.svg)

## Demo
Below is an example of geographic predictions (x and y axes) from Ge3Net with the ground-truth ancestral origin for each piece of an admixed individual's chromosome 22 shown extending along the z axis (Yoruba segment orange, Spanish segment blue, and Vietnamese segment green) and the predicted ancestral location for each piece of the chromosome shown alongside in pink.

![Demo](./images/Ge3Net_Demo.gif)

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

## Repo
Experiments were run on three genotypes - humans, dogs and ancient and for geography, unsupervised space constructed from pca and umap. 
1. Build labels by running the script buildLabels.py
2. For training, run trainer.py
3. For inference only with a pre-trained model, run inference.py

Ge3Net mdefault model is ``` src\models\Model_H.py```

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


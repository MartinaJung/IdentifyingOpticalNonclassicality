[![codecov](https://codecov.io/gh/MartinaJung/IdentifyingOpticalNonclassicality/graph/badge.svg?token=KVAJBASKAX)](https://codecov.io/gh/MartinaJung/IdentifyingOpticalNonclassicality) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18647273.svg)](https://doi.org/10.5281/zenodo.18647273)

# Learning to detect optical nonclassicality

This repository contains code for the setup and training of the algebraic classifier, a machine learning–based model developed to identify optical nonclassicality from data. Importantly, the model is interpretable in the sense that the learned nonclassicality criterion can be extracted after training. The analytical formula then represents an indicator for whether a state is nonclassical.


<img src="./img/VisualAbstract.png" alt="Visual abstract" style="height: 250px;"/>

# Installation

First, close the git repository and navigate to the folder with
```
gt clone https://github.com/MartinaJung/IdentifyingOpticalNonclassicality.git
cd IdentifyingOpticalNonclassicality
```
Then, create create an environment with
```
conda env create -f ml_nonclassicality.yaml 
```
and you are ready to use the library!


# Example usage
Let's imagine, you already have a dataset consisting of $M$ measurement samples per state. That is for each state you have samples of the form 
$$\left\lbrace x_\nu^\alpha\right\rbrace_{\nu=1,...,d_x}^{\alpha=1,...,M}$$
with $d_x$ being the number of modes and $M$ the number of measurement samples per state. 
Let's assume, you have $3$ modes and $1000$ samples per state and saved the dataset with the name "ds1_test" in your *dataset* subfolder. In order to setup and train the algebraic classifier, edit the respective hyperparameters in the file *hyperparams.py*.

Then, activate your environment and run the code
```
conda activate ml_nonclassicality
python3 main_AlCla.py 'ds1_test' 0.0
```
This will train the algebraic classifier on your dataset without using a regularization that drives the model's bias towards classical states.

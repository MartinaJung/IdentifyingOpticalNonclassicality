[![codecov](https://codecov.io/gh/MartinaJung/IdentifyingOpticalNonclassicality/graph/badge.svg?token=KVAJBASKAX)](https://codecov.io/gh/MartinaJung/IdentifyingOpticalNonclassicality) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18647273.svg)](https://doi.org/10.5281/zenodo.18647273)

# Learning to detect optical nonclassicality

This repository contains code for the setup and training of the algebraic classifier, a machine learning–based model developed to identify optical nonclassicality from data. Importantly, the model is interpretable in the sense that the learned nonclassicality criterion can be extracted after training. The analytical formula then represents an indicator for whether a state is nonclassical.


<img src="./img/VisualAbstract.png" alt="Visual abstract" style="height: 250px;"/>

# Usage

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

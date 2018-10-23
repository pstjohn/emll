# Ensemble Modeling with Linear-Logarithmic Kinetics (emll)

This repository hosts code for the in-progress manuscript [*Bayesian inference of metabolic kinetics from genome-scale multiomics data* by Peter C. St. John, Jonathan Strutz, Linda J. Broadbelt, Keith E.J. Tyo, and Yannick J. Bomble](https://doi.org/10.1101/450163).

General code for solving for the steady-state metabolite and flux values as a function of elasticity parameters, enzyme expression, and external metabolite concentrations is found in `emll/linlog_model.py`. Theano code to perform the regularized linear regression (and integrate this operation into pymc3 models) is found in `emll/theano_utils.py`.

The `notebooks` directory contains the main code used to generate figures in the manuscript. `wu2004.ipynb` contains a simple model of an *in vitro* pathway, used to compare NUTS and ADVI inference methods. `contador.ipynb` compares the given methodology to an earlier application of metabolic ensemble modeling. `hackett.ipynb` demonstrates how the method can scale to near genome-scale models and omics datasets.

A duplicate of the python enviroment I used to perform the calculations should be creatable using anaconda
```
$ conda env create -f environment.yml
$ source activate idp_new
```
It uses the intelpython distribution for some faster blas routines, at least on the processors I developed this method on.

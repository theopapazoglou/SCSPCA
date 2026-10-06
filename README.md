# Sparse Covariance Supervised Principal Component Analysis (SCSPCA)
This repository contains the code accompanying the paper "Theodosios Papazoglou and Guosheng Yin. Sparse Covariance Supervised Principal Component Analysis. Transactions on Machine Learning Research (TMLR), 2026. OpenReview".

It provides the implementation of SCS-PCA, implementations of all competing methods, and scripts that reproduce the simulation studies reported in the paper.

# Repository Structure

- dependencies.py : Contains all third-party imports used by the code. Every other module loads them with 'from dependencies import *'.
- scspca.py : SCS-PCA, ManPG solver (manpg_orth_sparse), semi-smooth Newton subrproblem and the wrapper scspca.
- competing_methods.py : Contains implementation of HSIC-SPCA, Bair's supervised PCA, CSPCA, SPLS, SPCA and SSPCA, along with their helper functions.
- simulations_iid.py : Simulation study with independent predictors, including paired Wilcoxon signed-rank tests.
- simulations_cor.py : Simulation study with correlated predictors, including paired Wilcoxon signed-rank tests.

# Reproducing the simulation studies
Run from the repository directory: 

python simulations_iid.py $\\$
python simulations_cor.py

Each script prints, for every method and number of components $q$, the test MSE (mean and s.e.), the number of selected variables, and support recovery (TP,FP,FN,precision,recall,F1), followed by paired Wilcoxon singed-rank and paired $t$-tests of SCS-PCA against each competitor. The scripts run by default the linear simulation (Simulation 1). To run the non-linear simulation (Simulation 2), uncomment the response generation in the 'make_dataset' function for either script. 

All random quantities are seeded: dataset $i$ is generated with seed $123 + i$. Within each dataset, the data split uses 'random_state=1888' and NumPy's global generator is seeded once with '1924'.

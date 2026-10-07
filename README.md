# Sparse Covariance Supervised Principal Component Analysis (SCSPCA)

This repository contains the code accompanying the paper 

> Theodosios Papazoglou and Guosheng Yin. **Sparse Covariance Supervised Principal Component Analysis.** *Transactions on Machine Learning Research (TMLR)*, 2026. [OpenReview].

It provides the implementation of SCS-PCA, implementations of all competing methods, and scripts that reproduce the empirical studies reported in the paper.

## Method

Given a standardised predictor matrix $X \in \mathbb{R}^{n\times p}$ and response $Y$, SCS-PCA estimates $q$ sparse, orthonormal loading vectors by solving

$$
\min_{W\in\mathrm{St}(p,q)}\; -\operatorname{tr}\!\left(W^\top C W\right)+\eta\|W\|_1, \qquad C=X^\top YY^\top X+\kappa X^\top X,
$$
where $\mathrm{St}(p,q)=\{W: W^\top W=I_q\}$ is the Stiefel manifold. The problem is solved with the manifold proximal gradient method (ManPG; Chen et al., 2020), using a regularised semi-smooth Newton method for the proximal subproblem.

---


## Repository Structure

| File | Contents |
|---|---|
| `dependencies.py` | Contains all third-party imports used by the code. Every other module loads them with 'from dependencies import *'. |
| `scspca.py` | SCS-PCA: ManPG solver (`manpg_orth_sparse`), semi-smooth Newton subproblem, and the wrapper `scspca`. |
| `competing_methods.py` | Contains implementation of HSIC-SPCA, Bair's supervised PCA, CSPCA, SPLS, SPCA, SSPCA, and SDSPCA along with their helper functions. |
| `simulations_iid.py` | Simulation study with independent predictors, including paired Wilcoxon signed-rank tests. |
| `simulations_cor.py` | Simulation study with correlated predictors, including paired Wilcoxon signed-rank tests. |
| `real_analysis_colonca.py` | Real analysis implemented on Alon's et al. (1999) colon cancer classification dataset. |
| `requirements.txt` | Python package requirements. |
 
---

## Installation

The code requires Python ≥ 3.9.

```bash
git clone https://github.com/theopapazoglou/SCSPCA.git
cd SCSPCA
pip install -r requirements.txt
```
All modules must be in the same directory, since they import each other.

---

## Example implementation -- Quick start

```python
import numpy as np
from sklearn.preprocessing import StandardScaler
from scspca import scspca

rng = np.random.default_rng(0)
X = rng.normal(size=(100,500))
Y = (3 * X[:,0] - 2 * X[:,1] + rng.normal(scale=0.1, size=100)).reshape(-1,1)

X = StandardScaler().fit_transform(X)
Y = StandardScaler().fit_transform(Y)

W = scspca(X,Y,n_components=2, eta=3000, kappa=0.1)[0] # p x q sparse loading matrix
Z = X @ W # projected data
selected = np.where(np.any(np.abs(W) > 1e-7, axis=1))[0] # selected variables
```

`eta` is the sparsity penalty (a scalar or a length-`p` vector) and `kappa` weights the unsupervised term $X^\top X$. `scspca` returns the tuple produced by `manpg_orth_sparse`; its first element is the loading matrix.

---


## Simulation studies
Run from the repository directory: 

```bash
python simulations_iid.py 
python simulations_cor.py
```

Each script prints, for every method and number of components $q$, the test MSE (mean and s.e.), the number of selected variables, and support recovery (TP,FP,FN,precision,recall,F1), followed by paired Wilcoxon singed-rank and paired $t$-tests of SCS-PCA against each competitor. The scripts run by default the linear simulation (Simulation 1). To run the non-linear simulation (Simulation 2), uncomment the response generation in the 'make_dataset' function for either script. 

The functions can also be called from Python or Jupyter:

```python
import numpy as np
from simulations_iid import run_simulation, paired_significance, summary_table
 
np.random.seed(1924)
sim = run_simulation(n_datasets=100, n=100, p=500, n_components_list=[2, 3, 4], n_jobs=50)
summary_table(sim)
paired_significance(sim, reference='SCSPCA', metric='MSE')
```

## Real-data analysis

```bash
python real_analysis_colonca.py
```

Runs the colon cancer classification analysis (Alon et al., 1999). The dataset is extracted from R's Bioconductor (https://bioconductor.org/packages/release/data/experiment/html/colonCA.html). Its first column is the class label and the remaining columns are gene expression levels. 

The analysis uses 10 random splits (training 60%, validation 20%, test 20%) and $q=2$ components. $X$ is standardised on the training set. Each method's projected training data are used to fit a logistic regression and tuning parameters are chosen by validation log-loss. For the binary response, the supervised methods use the delta kernel.

## Citation

If you use this code, please cite: 

```bibtex
@article{papazoglou2026scspca,
  title   = {Sparse Covariance Supervised Principal Component Analysis},
  author  = {Papazoglou, Theodosios and Yin, Guosheng},
  journal = {Transactions on Machine Learning Research},
  year    = {2026},
  url     = {(https://openreview.net/forum?id=c2XDHSfZt2)}
}
```

---
 

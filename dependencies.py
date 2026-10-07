"""
Third-party dependencies shared by all modules in this repository
(independent and correlated simulation settings).

Every other module imports from here with `from dependencies import *`,
so this is the single place to check (or change) what the code relies on.
"""
import time
import argparse
from functools import lru_cache

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from scipy import linalg, stats
from scipy.linalg import eigh, lu_factor, lu_solve, pinv, qr, svd, toeplitz
from scipy.optimize import root_scalar
from scipy.sparse.linalg import LinearOperator, cg
from scipy.spatial.distance import cdist

from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, precision_score, accuracy_score,
                             roc_auc_score, log_loss)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



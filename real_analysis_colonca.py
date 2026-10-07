"""
Colon cancer classification (Alon et al., 1999).
"""
from dependencies import *
from scspca import manpg_orth_sparse
from competing_methods import (delta_kernel, barshan_pca, cspca_nystrom, bair_spca,
                               spca, sspca, SDSPCA)
from simulations_iid import count_nonzero_vars

METHODS_CLF = ['PCA', 'LDA', 'HSIC', 'Bair', 'CSPCA', 'SCSPCA', 'SDSPCA', 'SPCA', 'SSPCA']
METRICS = ['precision', 'error', 'accuracy', 'non_zero_vars', 'auc']

GRIDS = {
    'CSPCA':        [0.01, 0.1, 1, 10, 100],
    'SCSPCA':       [1800.1900, 2000, 2100, 2500, 2800, 3000, 3500, 4000, 4100],
    'SDSPCA_alpha': [1, 10, 100, 1000, 10000, 1000000, 10000000],
    'SDSPCA_beta':  [1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16,
                     1e-15, 1e-14, 1e-13, 1e-12, 1e-11],
    'SPCA':         [0.01, 0.1, 0.5, 1],              # same penalty for every component
    'SSPCA':        [32, 24, 16, 12, 8, 4, 2],        # c = sqrt(p) / value
}



def supervised_matrix(X, Y, kappa, epsilon=1e-4):
    """C = X^T K X + kappa X^T X, K = delta kernel + epsilon I."""
    K = delta_kernel(Y, epsilon)
    return X.T @ K @ X + kappa * X.T @ X


def scspca_clf(X, Y, q, eta, kappa=1.0, epsilon=1e-4, maxiter=1000):
    p = X.shape[1]
    C = supervised_matrix(X, Y, kappa, epsilon)
    return manpg_orth_sparse(C, q, p, eta * np.ones(p), maxiter=maxiter)[0]



def _logit(Z_train, Y_train):
    return LogisticRegression(solver='lbfgs', max_iter=1000).fit(Z_train, Y_train)


def _val_loss(Z_train, Y_train, Z_val, Y_val):
    return log_loss(Y_val, _logit(Z_train, Y_train).predict_proba(Z_val)[:, 1])


def _test_metrics(Y_test, Y_pred, Y_proba, nz):
    acc = accuracy_score(Y_test, Y_pred)
    return {'precision': precision_score(Y_test, Y_pred, zero_division=0),
            'error': 1 - acc, 'accuracy': acc, 'non_zero_vars': nz,
            'auc': roc_auc_score(Y_test, Y_proba)}


def _eval_projection(Z_train, Y_train, Z_test, Y_test, nz):
    clf = _logit(Z_train, Y_train)
    return _test_metrics(Y_test, clf.predict(Z_test), clf.predict_proba(Z_test)[:, 1], nz)


def _tune(loss_fn, grid):
    """Grid value with the smallest validation log-loss (first one on ties)."""
    best_val, best = grid[0], float('inf')
    for v in grid:
        loss = loss_fn(v)
        if loss < best:
            best, best_val = loss, v
    return best_val



def evaluate_split_clf(X_train, Y_train, X_val, Y_val, X_test, Y_test, q, grids, threshold=0.1):
    p = X_train.shape[1]
    out = {}

    # PCA
    pca = PCA(n_components=q)
    Z_train = pca.fit_transform(X_train)
    out['PCA'] = _eval_projection(Z_train, Y_train, pca.transform(X_test), Y_test, p)

    # LDA
    n_comp_lda = min(q, len(np.unique(Y_train)) - 1, p)
    lda = LinearDiscriminantAnalysis(n_components=n_comp_lda).fit(X_train, Y_train)
    out['LDA'] = _test_metrics(Y_test, lda.predict(X_test), lda.predict_proba(X_test)[:, 1], p)

    # HSIC-SPCA 
    W = barshan_pca(X_train, Y_train, delta_kernel(Y_train), q)['W']
    out['HSIC'] = _eval_projection(X_train @ W, Y_train, X_test @ W, Y_test, count_nonzero_vars(W))

    # Bair
    b = bair_spca(X_train, Y_train, q, threshold)
    sel = b['selected']
    out['Bair'] = _eval_projection(b['Z'], Y_train, b['pca'].transform(X_test[:, sel]), Y_test,
                                   int(sel.sum()))

    # CSPCA 
    def cspca_W(v, m):
        return cspca_nystrom(X_train, Y_train, q, p=p, m=m, lambda_=v)['W']
    lam = _tune(lambda v: _val_loss(X_train @ (W := cspca_W(v, 55)), Y_train, X_val @ W, Y_val),
                grids['CSPCA'])
    W = cspca_W(lam, 145)
    out['CSPCA'] = _eval_projection(X_train @ W, Y_train, X_test @ W, Y_test, count_nonzero_vars(W))

    # SCS-PCA
    eta = _tune(lambda v: _val_loss(X_train @ (W := scspca_clf(X_train, Y_train, q, v)), Y_train,
                                    X_val @ W, Y_val), grids['SCSPCA'])
    W = scspca_clf(X_train, Y_train, q, eta)
    out['SCSPCA'] = _eval_projection(X_train @ W, Y_train, X_test @ W, Y_test, count_nonzero_vars(W))

    # SDSPCA
    best_ab, best = None, float('inf')
    for alpha in grids['SDSPCA_alpha']:
        for beta in grids['SDSPCA_beta']:
            sd = SDSPCA(n_components=q, alpha=alpha, beta=beta)
            loss = _val_loss(sd.fit_transform(X_train, Y_train), Y_train, sd.transform(X_val), Y_val)
            if loss < best:
                best, best_ab = loss, (alpha, beta)
    sd = SDSPCA(n_components=q, alpha=best_ab[0], beta=best_ab[1])
    Q_train = sd.fit_transform(X_train, Y_train)
    out['SDSPCA'] = _eval_projection(Q_train, Y_train, sd.transform(X_test), Y_test,
                                     count_nonzero_vars(sd.Q_))

    # SPCA
    def spca_W(v):
        return spca(X_train, K=q, para=[v] * q, type_="predictor",
                    sparse="penalty", lambda_=1e-4)['loadings']
    v = _tune(lambda v: _val_loss(X_train @ (W := spca_W(v)), Y_train, X_val @ W, Y_val),
              grids['SPCA'])
    W = spca_W(v)
    out['SPCA'] = _eval_projection(X_train @ W, Y_train, X_test @ W, Y_test, count_nonzero_vars(W))

    # SSPCA 
    def sspca_fit(c, X_eval):
        return sspca(X_train, Y_train, K=q, c=c, X_test=X_eval, kernel_type='delta', max_iter=1000)
    c_grid = [np.sqrt(p) / d for d in grids['SSPCA']]
    c = _tune(lambda c: _val_loss((r := sspca_fit(c, X_val))['Z'], Y_train, r['Z_test'], Y_val), c_grid)
    r = sspca_fit(c, X_test)
    out['SSPCA'] = _eval_projection(r['Z'], Y_train, r['Z_test'], Y_test, count_nonzero_vars(r['V']))

    return out



def _run_split(X, Y, n_components, split_idx, seed, grids, threshold):
    current_seed = seed + split_idx
    np.random.seed(current_seed)

    X_train_val, X_test, Y_train_val, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=current_seed)
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train_val, Y_train_val, test_size=0.25, random_state=current_seed)

    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    return evaluate_split_clf(X_train, Y_train, X_val, Y_val, X_test, Y_test,
                              n_components, grids, threshold)


def run_real_data_classification(data, n_splits=10, n_components_list=[2], threshold=0.1,
                                 seed=1924, grids=GRIDS, n_jobs=10):
    Y = data[:, 0]
    X = data[:, 1:]

    tasks = [(i, n_components, split_idx)
             for i, n_components in enumerate(n_components_list)
             for split_idx in range(n_splits)]
    outs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_run_split)(X, Y, n_components, split_idx, seed, grids, threshold)
        for _, n_components, split_idx in tasks)

    results = {m: {k: [[] for _ in n_components_list] for k in METRICS} for m in METHODS_CLF}
    for (i, _, _), out in zip(tasks, outs):
        for m in METHODS_CLF:
            for k in METRICS:
                results[m][k][i].append(out[m][k])

    rows = []
    for i, n_components in enumerate(n_components_list):
        for m in METHODS_CLF:
            row = {'method': m, 'q': n_components}
            for k in METRICS:
                vals = np.asarray(results[m][k][i], dtype=float)
                row[f'{k}_mean'] = vals.mean()
                row[f'{k}_se'] = vals.std(ddof=1) / np.sqrt(len(vals))
            rows.append(row)
    return pd.DataFrame(rows), results



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SCS-PCA real-data analysis: colon cancer classification")
    ap.add_argument("--data", type=str, default="colonCA_combined.csv")
    ap.add_argument("--n_splits", type=int, default=10)
    ap.add_argument("--n_components", type=int, nargs="+", default=[2])
    ap.add_argument("--n_jobs", type=int, default=10)
    ap.add_argument("--out", type=str, default=None,
                    help="optional prefix for CSV output (<out>_summary.csv)")
    args = ap.parse_args()

    data = pd.read_csv(args.data).to_numpy()
    print("Dataset shape:", data.shape)

    summary, raw = run_real_data_classification(data, n_splits=args.n_splits,
                                                n_components_list=args.n_components,
                                                seed=1924, n_jobs=args.n_jobs)
    pd.set_option('display.float_format', '{:.4f}'.format)
    print(summary.to_string(index=False))

    if args.out:
        summary.to_csv(f"{args.out}_summary.csv", index=False)
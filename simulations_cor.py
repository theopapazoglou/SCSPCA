"""
Correlated simulation studies:
(1) Y = 3 X1 - 2 X2 - 5 X3 + 4 X4 + eps,  eps ~ N(0, 0.1^2).
(2) Y = exp(X1) + 4sin(X2) - 3 X3 + X4 + eps, eps ~ N(0, 0.1^2).
The code runs by default simulation 1. The response generation for simulation 2 is commented out on
the make_dataset function. 
"""
from dependencies import *
from scspca import scspca
from competing_methods import hsic_spca, rbf_kernel, bair_spca, cspca, spls, spca, sspca
from simulations_iid import METHODS, _mask, count_nonzero_vars, paired_significance, summary_table



def run_all_methods_analysis(data, n_splits=1, n_components_list=[2, 3, 4],
                             threshold=0.1, seed=1888, return_supports=False):
    np.random.seed(seed)
    Y = data[:, 0:1]
    X = data[:, 1:]
    p = X.shape[1]

    cols = {'n_components': n_components_list}
    for m in METHODS:
        cols[f'mse_{m}'] = np.zeros(len(n_components_list))
        cols[f'non_zero_vars_{m}'] = np.zeros(len(n_components_list))
    results = pd.DataFrame(cols)
    supports = {m: [np.zeros(p, dtype=bool) for _ in n_components_list] for m in METHODS}

    eta_sparse_grid = [3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100]
    eta_spls_grid = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    para_spca_grid = [[0.01]*4, [0.05]*4, [0.1]*4, [1]*4]
    c_sspca_grid = [1, 1.05, np.sqrt(p)/20, np.sqrt(p)/16, np.sqrt(p)/12, np.sqrt(p)/8, np.sqrt(p)/4, np.sqrt(p)/2]
    kappa = 0.1

    for i, n_components in enumerate(n_components_list):
        metrics = {key: [] for key in results.columns if key != 'n_components'}
        for _ in range(n_splits):
            X_tv, X_test, Y_tv, Y_test = train_test_split(X, Y, test_size=0.2, random_state=seed)
            X_train, X_val, Y_train, Y_val = train_test_split(X_tv, Y_tv, test_size=0.25, random_state=seed)
            sX = StandardScaler().fit(X_train)
            X_train, X_val, X_test = sX.transform(X_train), sX.transform(X_val), sX.transform(X_test)
            sY = StandardScaler().fit(Y_train)
            Y_train, Y_val, Y_test = sY.transform(Y_train), sY.transform(Y_val), sY.transform(Y_test)

            # PCA
            pca = PCA(n_components=n_components)
            lr = LinearRegression().fit(pca.fit_transform(X_train), Y_train)
            metrics['mse_PCA'].append(mean_squared_error(Y_test, lr.predict(pca.transform(X_test))))
            metrics['non_zero_vars_PCA'].append(p)
            supports['PCA'][i] = np.ones(p, dtype=bool)

            # PLS
            pls = PLSRegression(n_components=n_components).fit(X_train, Y_train)
            metrics['mse_PLS'].append(mean_squared_error(Y_test, pls.predict(X_test)))
            metrics['non_zero_vars_PLS'].append(p)
            supports['PLS'][i] = np.ones(p, dtype=bool)

            # HSIC-SPCA
            Kmat = rbf_kernel(Y_train, 0.1)
            W_hsic = hsic_spca(X_train, Y_train, Kmat, n_components)['W']
            lr_h = LinearRegression().fit(X_train @ W_hsic, Y_train)
            metrics['mse_HSIC'].append(mean_squared_error(Y_test, lr_h.predict(X_test @ W_hsic)))
            metrics['non_zero_vars_HSIC'].append(count_nonzero_vars(W_hsic))
            supports['HSIC'][i] = _mask(W_hsic)

            # Bair
            b = bair_spca(X_train, Y_train, n_components, threshold)
            sel = b['selected']
            lr_b = LinearRegression().fit(b['Z'], Y_train)
            metrics['mse_Bair'].append(mean_squared_error(Y_test, lr_b.predict(b['pca'].transform(X_test[:, sel]))))
            metrics['non_zero_vars_Bair'].append(int(sel.sum()))
            supports['Bair'][i] = sel.astype(bool)

            # CSPCA 
            best_lambda, best = 0.01, float('inf')
            for lambda_ in [0.01, 0.1, 1, 10, 100]:
                W = cspca(X_train, Y_train, n_components, lambda_)['W']
                lr_c = LinearRegression().fit(X_train @ W, Y_train)
                mse_ = mean_squared_error(Y_val, lr_c.predict(X_val @ W))
                if mse_ < best:
                    best, best_lambda = mse_, lambda_
            W_cspca = cspca(X_train, Y_train, n_components, best_lambda)['W']
            lr_c = LinearRegression().fit(X_train @ W_cspca, Y_train)
            metrics['mse_CSPCA'].append(mean_squared_error(Y_test, lr_c.predict(X_test @ W_cspca)))
            metrics['non_zero_vars_CSPCA'].append(count_nonzero_vars(W_cspca))
            supports['CSPCA'][i] = _mask(W_cspca)

            # SCS-PCA 
            best_eta, best = eta_sparse_grid[0], float('inf')
            for eta_s in eta_sparse_grid:
                W = scspca(X_train, Y_train, n_components, eta_s, kappa=kappa)[0]
                reg = LinearRegression().fit(X_train @ W, Y_train)
                mse_ = mean_squared_error(Y_val, reg.predict(X_val @ W))
                if mse_ < best:
                    best, best_eta = mse_, eta_s
            W_scspca = scspca(X_train, Y_train, n_components, best_eta, kappa=kappa)[0]
            reg = LinearRegression().fit(X_train @ W_scspca, Y_train)
            metrics['mse_SCSPCA'].append(mean_squared_error(Y_test, reg.predict(X_test @ W_scspca)))
            metrics['non_zero_vars_SCSPCA'].append(count_nonzero_vars(W_scspca))
            supports['SCSPCA'][i] = _mask(W_scspca)

            # SPLS 
            best_eta_spls, best = eta_spls_grid[0], float('inf')
            for eta_s in eta_spls_grid:
                bh = spls(X_train, Y_train, K=n_components, eta=eta_s, kappa=0.2,
                          scale_x=False, scale_y=False)['betahat']
                mse_ = mean_squared_error(Y_val, X_val @ bh)
                if mse_ < best:
                    best, best_eta_spls = mse_, eta_s
            bh = spls(X_train, Y_train, K=n_components, eta=best_eta_spls, kappa=0.2,
                      scale_x=False, scale_y=False)['betahat']
            metrics['mse_SPLS'].append(mean_squared_error(Y_test, X_test @ bh))
            metrics['non_zero_vars_SPLS'].append(count_nonzero_vars(bh))
            supports['SPLS'][i] = _mask(bh)

            # SPCA 
            best_para, best = para_spca_grid[0], float('inf')
            for para in para_spca_grid:
                W = spca(X_train, K=n_components, para=para, type_="predictor",
                         sparse="penalty", lambda_=1e-4)['loadings']
                lr_s = LinearRegression().fit(X_train @ W, Y_train)
                mse_ = mean_squared_error(Y_val, lr_s.predict(X_val @ W))
                if mse_ < best:
                    best, best_para = mse_, para
            W_spca = spca(X_train, K=n_components, para=best_para, type_="predictor",
                          sparse="penalty", lambda_=1e-4)['loadings']
            lr_s = LinearRegression().fit(X_train @ W_spca, Y_train)
            metrics['mse_SPCA'].append(mean_squared_error(Y_test, lr_s.predict(X_test @ W_spca)))
            metrics['non_zero_vars_SPCA'].append(count_nonzero_vars(W_spca))
            supports['SPCA'][i] = _mask(W_spca)

            # SSPCA
            best_c, best = c_sspca_grid[0], float('inf')
            for c_s in c_sspca_grid:
                r = sspca(X_train, Y_train, K=n_components, c=c_s, X_test=X_val,
                          kernel_type='rbf', sigma=0.1)
                reg = LinearRegression().fit(r['Z'], Y_train)
                mse_ = mean_squared_error(Y_val, reg.predict(r['Z_test']))
                if mse_ < best:
                    best, best_c = mse_, c_s
            r = sspca(X_train, Y_train, K=n_components, c=best_c, X_test=X_test,
                      kernel_type='rbf', sigma=0.1)
            reg = LinearRegression().fit(r['Z'], Y_train)
            metrics['mse_SSPCA'].append(mean_squared_error(Y_test, reg.predict(r['Z_test'])))
            metrics['non_zero_vars_SSPCA'].append(count_nonzero_vars(r['V']))
            supports['SSPCA'][i] = _mask(r['V'])

        for key in metrics:
            results.loc[i, key] = np.mean(metrics[key])

    if return_supports:
        return results, supports
    return results



def make_dataset(i, n, p, rho=0.7):
    np.random.seed(123 + i)
    cov = toeplitz(rho ** np.arange(p))
    X = np.random.multivariate_normal(np.zeros(p), cov, size=n)
    X1, X2, X3, X4 = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
    error = np.random.normal(0, 0.1, n)
    Y = 3 * X1 - 2 * X2 - 5 * X3 + 4 * X4 + error
    #Y = np.exp(X1) + 4 * np.sin(X2) - 3 * X3 + X4 + error
    return np.column_stack((Y[:, np.newaxis], X))


def _process_one_dataset(i, n, p, n_splits, n_components_list, rho):
    print(f"[{time.strftime('%H:%M:%S')}] Dataset {i+1} started", flush=True)
    data = make_dataset(i, n, p, rho)
    res, sup = run_all_methods_analysis(data, n_splits=n_splits,
                                        n_components_list=n_components_list,
                                        return_supports=True)
    print(f"[{time.strftime('%H:%M:%S')}] Dataset {i+1} done", flush=True)
    return i, res, sup


def run_simulation(n_datasets=100, n=100, p=500, n_splits=1, n_components_list=[2, 3, 4],
                   true_support=None, verbose=True, n_jobs=5, rho=0.7):
    if true_support is None:
        true_support = [0, 1, 2, 3]
    true_mask = np.zeros(p, dtype=bool)
    true_mask[list(true_support)] = True
    n_true = int(true_mask.sum())
    nc = len(n_components_list)

    results = {m: {'MSE': np.zeros((n_datasets, nc)),
                   'NonZeroVars': np.zeros((n_datasets, nc))} for m in METHODS}
    all_supports = {m: np.zeros((n_datasets, nc, p), dtype=bool) for m in METHODS}

    outputs = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_process_one_dataset)(i, n, p, n_splits, n_components_list, rho)
        for i in range(n_datasets)
    )

    for i, res, sup in outputs:
        for m in METHODS:
            results[m]['MSE'][i] = res[f'mse_{m}']
            results[m]['NonZeroVars'][i] = res[f'non_zero_vars_{m}']
            for j in range(nc):
                all_supports[m][i, j] = sup[m][j]

    def stat(mat):
        return {'avg': np.nanmean(mat, axis=0), 'se': np.nanstd(mat, axis=0) / np.sqrt(n_datasets)}

    perf = {m: {'MSE': stat(results[m]['MSE']), 'NonZeroVars': stat(results[m]['NonZeroVars'])}
            for m in METHODS}

    recovery = {}
    for m in METHODS:
        prec = np.full((n_datasets, nc), np.nan)
        rec = np.full((n_datasets, nc), np.nan)
        f1 = np.full((n_datasets, nc), np.nan)
        tp_a = np.zeros((n_datasets, nc))
        fp_a = np.zeros((n_datasets, nc))
        fn_a = np.zeros((n_datasets, nc))
        for i in range(n_datasets):
            for j in range(nc):
                sel = all_supports[m][i, j]
                tp = int(np.sum(sel & true_mask))
                fp = int(np.sum(sel & ~true_mask))
                fn = n_true - tp
                tp_a[i, j], fp_a[i, j], fn_a[i, j] = tp, fp, fn
                prec[i, j] = tp / (tp + fp) if (tp + fp) > 0 else np.nan
                rec[i, j] = tp / (tp + fn) if (tp + fn) > 0 else np.nan
                if not np.isnan(prec[i, j]) and not np.isnan(rec[i, j]) and (prec[i, j] + rec[i, j]) > 0:
                    f1[i, j] = 2 * prec[i, j] * rec[i, j] / (prec[i, j] + rec[i, j])
                else:
                    f1[i, j] = 0.0
        recovery[m] = {'precision': stat(prec), 'recall': stat(rec), 'f1': stat(f1),
                       'TP': stat(tp_a), 'FP': stat(fp_a), 'FN': stat(fn_a)}

    if verbose:
        print(f"\nCorrelated simulation (rho={rho}; Y = exp(X1) + 4sin(X2) - 3X3 + X4 + eps); "
              f"true support = {list(true_support)}")
        for m in METHODS:
            print(f"  {m}:")
            for j, ncomp in enumerate(n_components_list):
                print(f"    q={ncomp}: MSE={perf[m]['MSE']['avg'][j]:.4f}+/-{perf[m]['MSE']['se'][j]:.4f}"
                      f" | #nz={perf[m]['NonZeroVars']['avg'][j]:.1f}"
                      f" | TP={recovery[m]['TP']['avg'][j]:.2f} FP={recovery[m]['FP']['avg'][j]:.2f}"
                      f" FN={recovery[m]['FN']['avg'][j]:.2f}"
                      f" | P={recovery[m]['precision']['avg'][j]:.2f}"
                      f" R={recovery[m]['recall']['avg'][j]:.2f} F1={recovery[m]['f1']['avg'][j]:.2f}")

    return {'stats': perf, 'raw_results': results, 'supports': all_supports,
            'recovery': recovery, 'true_support': list(true_support),
            'n_components_list': n_components_list}



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="SCS-PCA correlated simulation study")
    ap.add_argument("--n_datasets", type=int, default=100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=500)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--n_components", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--n_jobs", type=int, default=100)
    ap.add_argument("--out", type=str, default=None,
                    help="optional prefix for CSV output (<out>_summary.csv, <out>_wilcoxon.csv)")
    args = ap.parse_args()

    np.random.seed(1924)
    sim = run_simulation(n_datasets=args.n_datasets, n=args.n, p=args.p, rho=args.rho,
                         n_components_list=args.n_components, n_jobs=args.n_jobs)

    print("\n=== Paired significance vs SCS-PCA (test MSE) ===")
    sig = paired_significance(sim, reference='SCSPCA', metric='MSE')
    print(sig.to_string(index=False))

    if args.out:
        summary_table(sim).to_csv(f"{args.out}_summary.csv", index=False)
        sig.to_csv(f"{args.out}_wilcoxon.csv", index=False)
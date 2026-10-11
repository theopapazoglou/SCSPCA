from dependencies import *
from scspca import manpg_orth_sparse
from competing_methods import spls, spca, sspca
from simulations_iid import count_nonzero_vars
from simulations_cor import make_dataset

PARETO_METHODS = ('SCSPCA', 'SPCA', 'SPLS', 'SSPCA')


def pareto_grids(p):
    return {
        'SCSPCA': list(range(2000, 4001, 100)),
        'SPCA':   [0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
        'SPLS':   [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.85, 0.9],
        'SSPCA':  [1, 1.05, np.sqrt(p) / 20, np.sqrt(p) / 16, np.sqrt(p) / 12, np.sqrt(p) / 8,
                   np.sqrt(p) / 4, np.sqrt(p) / 2, np.sqrt(p)],
    }


def split_scale(data, seed=1888):
    Y, X = data[:, 0:1], data[:, 1:]
    Xtv, Xte, Ytv, Yte = train_test_split(X, Y, test_size=0.2, random_state=seed)
    Xtr, _, Ytr, _ = train_test_split(Xtv, Ytv, test_size=0.25, random_state=seed)
    sX = StandardScaler().fit(Xtr)
    sY = StandardScaler().fit(Ytr)
    return sX.transform(Xtr), sY.transform(Ytr), sX.transform(Xte), sY.transform(Yte)


def _eval_one_task(mname, val, ds_idx, Xtr, Ytr, Xte, Yte, n_components, kappa=0.1):
    np.random.seed(1888 + ds_idx)
    p = Xtr.shape[1]
    try:
        if mname == 'SCSPCA':
            C = Xtr.T @ Ytr @ Ytr.T @ Xtr + kappa * Xtr.T @ Xtr
            W = manpg_orth_sparse(C, n_components, p, val * np.ones(p))[0]
            reg = LinearRegression().fit(Xtr @ W, Ytr)
            return count_nonzero_vars(W), mean_squared_error(Yte, reg.predict(Xte @ W))
        if mname == 'SPCA':
            W = spca(Xtr, K=n_components, para=[val] * n_components, type_="predictor",
                     sparse="penalty", lambda_=1e-4)['loadings']
            reg = LinearRegression().fit(Xtr @ W, Ytr)
            return count_nonzero_vars(W), mean_squared_error(Yte, reg.predict(Xte @ W))
        if mname == 'SPLS':
            bh = spls(Xtr, Ytr, K=n_components, eta=val, kappa=0.2,
                      scale_x=False, scale_y=False)['betahat']
            return count_nonzero_vars(bh), mean_squared_error(Yte, Xte @ bh)
        if mname == 'SSPCA':
            r = sspca(Xtr, Ytr, K=n_components, c=val, X_test=Xte, kernel_type='rbf', sigma=0.1)
            reg = LinearRegression().fit(r['Z'], Ytr)
            return count_nonzero_vars(r['V']), mean_squared_error(Yte, reg.predict(r['Z_test']))
    except Exception as e:
        print(f"  [warn] {mname} val={val} dataset={ds_idx} failed: {e}", flush=True)
        return None, None


def pareto_front(n_datasets=100, n=100, p=500, n_components=2, rho=0.7,
                 methods=PARETO_METHODS, verbose=True, n_jobs=50):
    grids = pareto_grids(p)
    datasets = [split_scale(make_dataset(i, n, p, rho)) for i in range(n_datasets)]

    out = {}
    for mname in methods:
        tasks = [(val, i, ds) for val in grids[mname] for i, ds in enumerate(datasets)]
        print(f"[{time.strftime('%H:%M:%S')}] Pareto {mname}: {len(tasks)} evaluations", flush=True)
        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(_eval_one_task)(mname, val, i, *ds, n_components) for val, i, ds in tasks)

        curve = []
        for k, val in enumerate(grids[mname]):
            block = results[k * n_datasets:(k + 1) * n_datasets]
            nzs = [nz for nz, mse in block if nz is not None]
            mses = [mse for nz, mse in block if nz is not None]
            if mses:
                curve.append({'penalty': float(val),
                              'nonzeros_avg': float(np.mean(nzs)),
                              'mse_avg': float(np.mean(mses)),
                              'mse_se': float(np.std(mses) / np.sqrt(len(mses)))})
        out[mname] = pd.DataFrame(curve)
        if verbose:
            print(f"\nPareto front -- {mname} (q={n_components})")
            print(out[mname].to_string(index=False))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Sparsity-vs-MSE Pareto fronts (correlated design)")
    ap.add_argument("--n_datasets", type=int, default=100)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--p", type=int, default=500)
    ap.add_argument("--rho", type=float, default=0.7)
    ap.add_argument("--n_components", type=int, default=2)
    ap.add_argument("--n_jobs", type=int, default=50)
    ap.add_argument("--out", type=str, default=None,
                    help="optional prefix for CSV output (<out>_<method>.csv)")
    args = ap.parse_args()

    fronts = pareto_front(n_datasets=args.n_datasets, n=args.n, p=args.p, rho=args.rho,
                          n_components=args.n_components, n_jobs=args.n_jobs)
    if args.out:
        for m, df in fronts.items():
            df.to_csv(f"{args.out}_{m}.csv", index=False)
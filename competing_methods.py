"""
Implements all supervised and sparse supervised methods used in the experimental section.
"""
from dependencies import *


# HSIC-SPCA
def rbf_kernel(Y, sigma):
    dist_matrix = cdist(Y, Y, 'euclidean')
    return np.exp(-dist_matrix**2 / (2 * sigma**2))


def hsic_spca(X, Y, K, q):
    C = X.T @ K @ X
    eigenvalues, eigenvectors = linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    return {'W': eigenvectors[:, :q], 'eigenvalues': eigenvalues[order][:q]}


# Bair supervised PCA
def bair_spca(X, Y, n_components, threshold=0.1):
    fs = np.abs(np.corrcoef(X.T, Y.T)[:-1, -1])
    sel = fs >= threshold
    if sel.sum() == 0:
        sel[np.argmax(fs)] = True
    pca = PCA(n_components=min(n_components, int(sel.sum())))
    Z = pca.fit_transform(X[:, sel])
    return {'selected': sel, 'pca': pca, 'Z': Z}


# CSPCA
def cspca(X, Y, q, lambda_):
    C = (X.T @ Y) @ (Y.T @ X) + lambda_ * (X.T @ X)
    eigenvalues, eigenvectors = linalg.eigh(C)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]
    return {'W': eigenvectors[:, :q], 'eigenvalues': eigenvalues[order][:q]}


# SPLS
def ust(b, eta):
    b_ust = np.zeros_like(b)
    if eta < 1:
        valb = np.abs(b) - eta * np.max(np.abs(b))
        mask = valb >= 0
        b_ust[mask] = valb[mask] * np.sign(b)[mask]
    return b_ust


def spls_dv(Z, eta, kappa, eps, maxstep):
    p, q = Z.shape
    Znorm1 = np.median(np.abs(Z))
    Z = Z / Znorm1 if Znorm1 > 0 else Z
    if q == 1:
        c = ust(Z.ravel(), eta)
    else:
        M = Z @ Z.T
        dis = 10
        i = 1
        if kappa == 0.5:
            c = np.ones(p) * 10
            c_old = c.copy()
            while dis > eps and i <= maxstep:
                mcsvd = linalg.svd(M @ c, full_matrices=False)
                a = mcsvd[0] @ mcsvd[2].T
                c = ust(M @ a, eta)
                dis = np.max(np.abs(c - c_old))
                c_old = c.copy()
                i += 1
        elif 0 < kappa < 0.5:
            kappa2 = (1 - kappa) / (1 - 2 * kappa)
            c = np.ones(p) * 10
            c_old = c.copy()

            def h(lambda_val, M, c, kappa2):
                alpha = linalg.solve(M + lambda_val * np.eye(p), M @ c)
                return alpha.T @ alpha - 1 / kappa2**2

            while dis > eps and i <= maxstep:
                while h(eps, M, c, kappa2) * h(1e30, M, c, kappa2) > 0:
                    if h(eps, M, c, kappa2) <= 1e5:
                        M *= 2
                        c *= 2
                lambdas = root_scalar(lambda x: h(x, M, c, kappa2),
                                      bracket=[eps, 1e30], method="brentq").root
                a = kappa2 * linalg.solve(M + lambdas * np.eye(p), M @ c)
                c = ust(M @ a, eta)
                dis = np.max(np.abs(c - c_old))
                c_old = c.copy()
                i += 1
    return c


def correctp(x, y, eta, K, kappa, select, fit):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n, p = x.shape
    eta = np.atleast_1d(eta)
    if np.min(eta) < 0 or np.max(eta) >= 1:
        if np.max(eta) == 1:
            raise ValueError("eta should be strictly less than 1!")
        raise ValueError("eta should be between 0 and 1!")
    K = np.atleast_1d(K)
    if np.max(K) > p:
        raise ValueError("K cannot exceed the number of predictors!")
    if np.max(K) >= n:
        raise ValueError("K cannot exceed the sample size!")
    if np.min(K) <= 0 or not np.all(K == K.astype(int)):
        raise ValueError("K should be a positive integer!")
    if kappa > 0.5 or kappa < 0:
        print("kappa should be between 0 and 0.5! kappa=0.5 is used.")
        kappa = 0.5
    if select not in ["pls2", "simpls"]:
        select = "pls2"
    if fit not in ["simpls", "kernelpls", "widekernelpls", "oscorespls"]:
        fit = "simpls"
    return {"K": K[0], "eta": eta[0], "kappa": kappa, "select": select, "fit": fit}


def spls(x, y, K, eta, kappa=0.5, select="pls2", fit="simpls", scale_x=True, scale_y=False,
         eps=1e-4, maxstep=100, trace=False):
    x = np.asarray(x)
    y = np.asarray(y)
    if x.ndim == 1:
        x = x[:, np.newaxis]
    if y.ndim == 1:
        y = y[:, np.newaxis]
    n, p = x.shape
    q = y.shape[1]
    params = correctp(x, y, eta, K, kappa, select, fit)
    K = params["K"]
    eta = params["eta"]
    kappa = params["kappa"]
    select = params["select"]
    fit = params["fit"]
    mu = np.mean(y, axis=0)
    y = y - mu
    meanx = np.mean(x, axis=0)
    x = x - meanx
    if scale_x:
        normx = np.sqrt(np.sum(x**2, axis=0) / (n - 1))
        if np.any(normx < np.finfo(float).eps):
            raise ValueError("Some columns of X have zero variance")
        x = x / normx
    else:
        normx = np.ones(p)
    if scale_y:
        normy = np.sqrt(np.sum(y**2, axis=0) / (n - 1))
        if np.any(normy < np.finfo(float).eps):
            raise ValueError("Some columns of Y have zero variance")
        y = y / normy
    else:
        normy = np.ones(q)
    betahat = np.zeros((p, q))
    betamat = []
    x1 = x.copy()
    y1 = y.copy()
    new2As = []
    A = np.array([], dtype=int)
    pj = np.zeros((p, 1))
    for k in range(K):
        Z = x1.T @ y1
        what = spls_dv(Z, eta, kappa, eps, maxstep)
        A = np.unique(np.where((np.abs(what) > 0) | (np.abs(betahat[:, 0]) > 0))[0])
        new2A = np.where((np.abs(what) > 0) & (np.abs(betahat[:, 0]) == 0))[0]
        xA = x[:, A]
        plsfit = PLSRegression(n_components=min(k + 1, len(A)), scale=False)
        plsfit.fit(xA, y)
        betahat = np.zeros((p, q))
        coef = plsfit.coef_
        if q == 1:
            if coef.ndim == 1:
                coef = coef[:, np.newaxis]
            elif coef.shape[0] == 1:
                coef = coef.T
        betahat[A, :] = coef
        betamat.append(betahat.copy())
        pj = plsfit.x_rotations_
        if pj.ndim == 1:
            pj = pj[:, np.newaxis]
        if select == "pls2":
            y1 = y - x @ betahat
        elif select == "simpls":
            pw = pj @ linalg.solve(pj.T @ pj, pj.T)
            x1 = x.copy()
            x1[:, A] = x[:, A] - x[:, A] @ pw
        new2As.append(new2A)
    return {"betahat": betahat, "A": A, "betamat": betamat, "new2As": new2As,
            "mu": mu, "meanx": meanx, "normx": normx, "normy": normy, "eta": eta,
            "K": K, "kappa": kappa, "select": select, "fit": fit, "projection": pj}


# SPCA 
def rootmatrix(x):
    eigvals, eigvecs = np.linalg.eigh(x)
    eigvals = (eigvals + np.abs(eigvals)) / 2
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def convcheck(beta1, beta2):
    a = np.max(np.abs(beta1 + beta2), axis=0)
    b = np.max(np.abs(beta1 - beta2), axis=0)
    return np.max(np.minimum(a, b))


def updateRR(xnew, R=None, xold=None, lambda_=0, eps=1e-16):
    xtx = (np.sum(xnew**2) + lambda_) / (1 + lambda_)
    norm_xnew = np.sqrt(xtx)
    if R is None:
        return np.array([[norm_xnew]]), 1
    Xtx = np.dot(xnew.T, xold) / (1 + lambda_)
    r = linalg.solve_triangular(R, Xtx, lower=False, trans='T')
    rpp = norm_xnew**2 - np.sum(r**2)
    if rpp <= eps:
        rpp = eps
        rank = R.shape[0]
    else:
        rpp = np.sqrt(rpp)
        rank = R.shape[0] + 1
    R_new = np.zeros((R.shape[0] + 1, R.shape[1] + 1))
    R_new[:-1, :-1] = R
    R_new[:-1, -1] = r
    R_new[-1, -1] = rpp
    return R_new, rank


def solvebeta(x, y, paras, max_steps=None, sparse="penalty", eps=1e-16):
    lambda_ = paras[0]
    penalty_param = paras[1]
    n, m = x.shape
    im = np.arange(m)
    if lambda_ > 0:
        maxvars = m
    else:
        maxvars = min(m, n - 1)
        if m == n:
            maxvars = m
    if max_steps is None:
        max_steps = 50 * maxvars
    d1 = np.sqrt(lambda_)
    d2 = 1 / np.sqrt(1 + lambda_)
    Cvec = np.dot(y.T, x) * d2
    residuals = np.concatenate([y, np.zeros(m)])
    penalty = np.array([np.max(np.abs(Cvec))])
    if sparse == "penalty" and penalty[0] * 2 / d2 <= penalty_param:
        return np.zeros(m)
    beta = np.zeros(m)
    first_in = np.zeros(m, dtype=int)
    active = []
    ignores = []
    actions = [None] * max_steps
    Sign = []
    R = None
    k = 0
    while k < max_steps and len(active) < maxvars - len(ignores):
        action = []
        k += 1
        inactive = im if k == 1 else np.setdiff1d(im, active + ignores)
        C = Cvec[inactive]
        Cmax = np.max(np.abs(C))
        new = np.abs(C) == Cmax
        C = C[~new]
        new = inactive[new]
        for inew in new:
            R, rank = updateRR(x[:, inew], R, x[:, active] if active else None, lambda_, eps)
            if rank == len(active):
                R = R[:len(active), :len(active)] if len(active) > 0 else np.array([[]])
                ignores.append(inew)
                action.append(-inew)
            else:
                if first_in[inew] == 0:
                    first_in[inew] = k
                active.append(inew)
                Sign.append(np.sign(Cvec[inew]))
                action.append(inew)
        Gi1 = linalg.solve_triangular(R, linalg.solve_triangular(R, Sign, lower=False, trans='T'), lower=False)
        A = 1 / np.sqrt(np.sum(Gi1 * Sign))
        w = A * Gi1
        u1 = np.dot(x[:, active], w) * d2
        u2 = np.zeros(m)
        u2[active] = d1 * d2 * w
        u = np.concatenate([u1, u2])
        if len(active) == maxvars - len(ignores):
            gamhat = Cmax / A
        else:
            a = (np.dot(u1, x[:, np.setdiff1d(im, active + ignores)]) + d1 * u2[np.setdiff1d(im, active + ignores)]) * d2
            gam = np.concatenate([(Cmax - C) / (A - a), (Cmax + C) / (A + a)])
            gamhat = np.min(gam[gam > eps]) if len(gam) > 0 else Cmax / A
        b1 = beta[active]
        z1 = -b1 / w
        zmin = np.min(z1[z1 > eps]) if np.any(z1 > eps) else gamhat
        drops = False
        if zmin < gamhat:
            gamhat = zmin
            drops = z1 == zmin
        beta2 = beta.copy()
        beta[active] += gamhat * w
        residuals -= gamhat * u
        Cvec = (np.dot(residuals[:n], x) + d1 * residuals[n:]) * d2
        penalty = np.append(penalty, penalty[-1] - np.abs(gamhat * A))
        if sparse == "penalty" and penalty[-1] * 2 / d2 <= penalty_param:
            s1 = penalty[-1] * 2 / d2
            s2 = penalty[-2] * 2 / d2 if len(penalty) > 1 else s1
            beta = ((s2 - penalty_param) / (s2 - s1)) * beta + ((penalty_param - s1) / (s2 - s1)) * beta2
            beta *= d2
            break
        if isinstance(drops, np.ndarray) and drops.any() or drops is True:
            dropid = np.where(drops)[0] if isinstance(drops, np.ndarray) else []
            for _id in reversed(dropid):
                R = R[:-1, :-1] if R.shape[0] > 1 else np.array([[]])
            dropid = np.array(active)[drops] if isinstance(drops, np.ndarray) else []
            beta[dropid] = 0
            active = [a for i, a in enumerate(active) if not drops[i]] if isinstance(drops, np.ndarray) else active
            Sign = [s for i, s in enumerate(Sign) if not drops[i]] if isinstance(drops, np.ndarray) else Sign
        if sparse == "varnum" and len(active) >= penalty_param:
            break
        actions[k-1] = action
    return beta


def spca(x, K, para, type_="predictor", sparse="penalty", use_corr=False, lambda_=1e-6,
         max_iter=200, trace=False, eps_conv=1e-3):
    x = np.asarray(x)
    n, p = x.shape if type_ == "predictor" else (None, x.shape[0])
    if type_ == "predictor":
        mean_x = np.mean(x, axis=0)
        scale_x = np.std(x, axis=0) if use_corr else np.ones(p)
        if use_corr:
            scale_x[scale_x == 0] = 1.0
        x = (x - mean_x) / scale_x
    elif type_ == "Gram":
        x = rootmatrix(x)
    U, s, Vt = linalg.svd(x, full_matrices=False)
    v = Vt.T
    total_variance = np.sum(s**2)
    alpha = v[:, :K]
    beta = alpha.copy()
    for i in range(K):
        y = np.dot(x, alpha[:, i])
        beta[:, i] = solvebeta(x, y, paras=[lambda_, para[i]], sparse=sparse)
    xtx = np.dot(x.T, x)
    temp = beta.copy()
    norm_temp = np.sqrt(np.sum(temp**2, axis=0))
    norm_temp[norm_temp == 0] = 1
    temp = temp / norm_temp
    k = 0
    diff = 1
    while k < max_iter and diff > eps_conv:
        k += 1
        alpha = np.dot(xtx, beta)
        U, _, Vt = linalg.svd(alpha, full_matrices=False)
        alpha = np.dot(U, Vt)
        for i in range(K):
            y = np.dot(x, alpha[:, i])
            beta[:, i] = solvebeta(x, y, paras=[lambda_, para[i]], sparse=sparse)
        norm_beta = np.sqrt(np.sum(beta**2, axis=0))
        norm_beta[norm_beta == 0] = 1
        beta2 = beta / norm_beta
        diff = convcheck(beta2, temp)
        temp = beta2
    norm_beta = np.sqrt(np.sum(beta**2, axis=0))
    norm_beta[norm_beta == 0] = 1
    beta = beta / norm_beta
    return {"type": type_, "K": K, "loadings": beta, "var_all": total_variance,
            "para": para, "lambda": lambda_}


# SSPCA 
def soft_threshold(a, tau):
    return np.sign(a) * np.maximum(np.abs(a) - tau, 0)


def find_tau(a, c, max_iter=5000, tol=1e-4):
    v_temp = soft_threshold(a, 0)
    if np.linalg.norm(v_temp) > 0:
        v_temp = v_temp / np.linalg.norm(v_temp)
        if np.sum(np.abs(v_temp)) <= c:
            return 0
    tau_min, tau_max = 0, np.max(np.abs(a))
    best_tau, best_norm_diff = 0, float('inf')
    for _ in range(max_iter):
        tau = (tau_min + tau_max) / 2
        v_temp = soft_threshold(a, tau)
        if np.linalg.norm(v_temp) > 0:
            v_temp = v_temp / np.linalg.norm(v_temp)
            l1_norm = np.sum(np.abs(v_temp))
            norm_diff = abs(l1_norm - c)
            if norm_diff < best_norm_diff:
                best_tau, best_norm_diff = tau, norm_diff
            if norm_diff < tol:
                return tau
            if l1_norm > c:
                tau_min = tau
            else:
                tau_max = tau
        else:
            tau_min = tau
    return best_tau


def compute_kernel(Y, kernel_type='linear', sigma=1.0, epsilon=1e-6):
    n = Y.shape[0]
    Y = Y.reshape(-1, 1)
    if kernel_type == 'linear':
        L = Y @ Y.T
    elif kernel_type == 'rbf':
        pairwise_dists = np.sum(Y**2, axis=1).reshape(-1, 1) + np.sum(Y**2, axis=1) - 2 * (Y @ Y.T)
        L = np.exp(-pairwise_dists / (2 * sigma**2))
    elif kernel_type == 'delta':
        L = (Y == Y.T).astype(float)
    else:
        raise ValueError(f"Unsupported kernel type: {kernel_type}")
    return L + epsilon * np.eye(n)


def sspca(X, Y, K, c, X_test=None, Y_test=None, kernel_type='linear',
          sigma=1.0, max_iter=15000, tol=1e-6, verbose=False, standardize=True):
    n, p = X.shape
    Y = Y.reshape(-1) if Y.ndim == 1 else Y[:, 0]
    if standardize:
        X = StandardScaler().fit_transform(X)
        Y = StandardScaler().fit_transform(Y.reshape(-1, 1)).reshape(-1)
    L = compute_kernel(Y, kernel_type, sigma)
    U_L, S_L, Vt_L = np.linalg.svd(L, hermitian=True)
    S_L = np.clip(S_L, 0, None)
    Delta = U_L @ np.diag(np.sqrt(S_L))
    e = np.ones((n, 1))
    H = np.eye(n) - (1/n) * (e @ e.T)
    Psi = Delta.T @ H @ X
    V = np.zeros((p, K))
    U = np.zeros((n, K))
    Lambda = np.zeros(K)
    Psi_k = Psi.copy()
    for k in range(K):
        v_k = np.random.randn(p)
        v_k = v_k / np.linalg.norm(v_k)
        for iter_idx in range(max_iter):
            v_old = v_k.copy()
            u_k = Psi_k @ v_k
            if np.linalg.norm(u_k) < 1e-10:
                u_k = np.random.randn(n)
            if k > 0:
                u_k = u_k - U[:, :k] @ (U[:, :k].T @ u_k)
                if np.linalg.norm(u_k) < 1e-10:
                    u_k = np.random.randn(n)
                u_k = u_k / np.linalg.norm(u_k)
            a = Psi_k.T @ u_k
            tau = find_tau(a, c)
            v_k = soft_threshold(a, tau)
            if np.linalg.norm(v_k) < 1e-10:
                v_k = np.random.randn(p)
            v_k = v_k / np.linalg.norm(v_k)
            if np.linalg.norm(v_k - v_old) < tol * np.linalg.norm(v_old):
                break
        Lambda[k] = u_k.T @ Psi_k @ v_k
        U[:, k] = u_k
        V[:, k] = v_k
        Psi_k = Psi_k - Lambda[k] * np.outer(u_k, v_k)
    result = {'Z': X @ V, 'V': V}
    if X_test is not None:
        result['Z_test'] = X_test @ V
    return result

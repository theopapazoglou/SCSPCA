"""
Implements Sparse Covariance Supervised Principal Component Analysis (SCS-PCA)
using manifold proximal gradient descent (ManPG).
"""
from dependencies import *


def proximal_l1(b, lambda_, q):
    Z = np.sign(b) * np.maximum(np.abs(b) - lambda_, 0)
    act_set = (Z != 0).astype(int)
    inact_set = 1 - act_set
    return Z, act_set, inact_set


_dup_cache = {}
def duplication_matrix(q):
    if q in _dup_cache:
        return _dup_cache[q]
    n = q * (q + 1) // 2
    Dn = np.zeros((q * q, n))
    idx = 0
    for i in range(q):
        for j in range(i, q):
            Dn[i * q + j, idx] = 1
            if i != j:
                Dn[j * q + i, idx] = 1
            idx += 1
    pDn = np.linalg.pinv(Dn)
    _dup_cache[q] = (Dn, pDn)
    return Dn, pDn


def linop(Blkd, x, q, t, reg):
    V = np.zeros_like(x)
    for i in range(q):
        V[:, i] = Blkd[i] @ x[:, i]
    return 2 * t * (V + V.T) + reg * x


def semi_newton_matrix(n, q, W, t, C, mut, inner_tol, inner_max_iter, Lam0, Dn, pDn):
    WtW = np.eye(q)
    Wt = W.T
    stop_flag = 0
    Lam = Lam0.copy()
    W_Lam_prod = C + 2 * t * (W @ Lam)
    Z, Act_set, Inact_set = proximal_l1(W_Lam_prod, mut, q)
    ZW = Z.T @ W
    R_Lam = ZW + ZW.T - 2 * np.eye(q)
    RE = pDn @ R_Lam.flatten()
    r_l = np.linalg.norm(R_Lam, 'fro')
    lambda_ = 0.2
    j = 0
    while r_l**2 > inner_tol:
        reg = lambda_ * max(min(r_l, 0.1), 1e-11)
        nnzZ = np.count_nonzero(Z)
        if q < 15:
            if nnzZ > q * (q + 1) // 2:
                g = np.zeros((q * q, q * q))
                for i in range(q):
                    g[i*q:(i+1)*q, i*q:(i+1)*q] = Wt @ (Act_set[:, i, None] * W)
                G = 4 * t * (pDn @ (g @ Dn))
                lu, piv = lu_factor(G + reg * np.eye(q * (q + 1) // 2))
                new_d = -lu_solve((lu, piv), RE)
            else:
                Wstack = np.zeros((nnzZ, q * q))
                dim = 0
                for i in range(q):
                    row = np.where(Act_set[:, i])[0]
                    Wstack[dim:dim+len(row), i*q:(i+1)*q] = W[row, :]
                    dim += len(row)
                V = Wstack @ Dn
                U = 4 * t * (pDn @ Wstack.T)
                lu, piv = lu_factor(np.eye(nnzZ) + (1/reg) * (V @ U))
                new_d = -(1/reg * RE - (1/reg**2) * U @ lu_solve((lu, piv), V @ RE))
            new_d = Dn @ new_d
            new_d = new_d.reshape(q, q)
        else:
            Blkd = [None] * q
            for i in range(q):
                ind = Act_set[:, i].astype(bool)
                if np.sum(ind) < n / 2:
                    W_ind = W[ind, :]
                    Blkd[i] = W_ind.T @ W_ind
                else:
                    ind = Inact_set[:, i].astype(bool)
                    W_ind = Wt[:, ind]
                    Blkd[i] = WtW - W_ind @ W_ind.T
            op = LinearOperator(
                (q * q, q * q), dtype=float,
                matvec=lambda v: linop(Blkd, v.reshape(q, q), q, t, reg).ravel())
            new_d, _ = cg(op, -R_Lam.ravel(), rtol=min(1e-4, 1e-3 * r_l))
            new_d = new_d.reshape(q, q)
        t_new = 1.0
        W_d_prod = 2 * t * (W @ new_d)
        W_Lam_new_prod = W_Lam_prod + t_new * W_d_prod
        Z, Act_set, Inact_set = proximal_l1(W_Lam_new_prod, mut, q)
        ZW = Z.T @ W
        R_Lam_new = ZW + ZW.T - 2 * np.eye(q)
        r_l_new = np.linalg.norm(R_Lam_new, 'fro')
        while r_l_new**2 >= r_l**2 * (1 - 0.001 * t_new) and t_new > 1e-3:
            t_new *= 0.5
            W_Lam_new_prod = W_Lam_prod + t_new * W_d_prod
            Z, Act_set, Inact_set = proximal_l1(W_Lam_new_prod, mut, q)
            ZW = Z.T @ W
            R_Lam_new = ZW + ZW.T - 2 * np.eye(q)
            r_l_new = np.linalg.norm(R_Lam_new, 'fro')
        Lam += t_new * new_d
        r_l = r_l_new
        R_Lam = R_Lam_new
        RE = pDn @ R_Lam.flatten()
        W_Lam_prod = W_Lam_new_prod
        if j > inner_max_iter:
            stop_flag = 1
            break
        j += 1
    return Z, j, Lam, r_l, stop_flag




def orthonormalize_stable(PY, epsilon=1e-8):
    M = PY.T @ PY + epsilon * np.eye(PY.shape[1])
    U, Sigma, Vt = svd(M, full_matrices=False)
    Sigma_inv_sqrt = np.diag([1 / np.sqrt(s) if s > epsilon else 0.0 for s in Sigma])
    T = U @ Sigma_inv_sqrt @ Vt
    return PY @ T, T



def manpg_orth_sparse(C, q, n, eta, maxiter=10000, tol=1e-5, inner_iter=50, phi_init=None):
    start_time = time.time()
    eta = eta.reshape(-1, 1)
    # Only the top-q eigenpairs are needed (initialisation and Lipschitz constant).
    eigenvalues, eigenvectors = eigh(C, subset_by_index=[n - q, n - 1])
    if phi_init is None:
        phi_init = eigenvectors
    Dn, pDn = duplication_matrix(q)
    L = 2 * abs(max(eigenvalues))
    t = 2 / L
    t_min = 1e-4

    W = phi_init.copy()
    AW = C @ W
    F = [-np.sum(W * AW) + np.sum(eta * np.abs(W))]
    num_inner = np.zeros(maxiter)
    num_linesearch = 0
    num_inexact = 0
    alpha = 1.0
    for iter_ in range(1, maxiter):
        ngx = 2 * AW
        neg_pgx = ngx
        if alpha < t_min or num_inexact > 10:
            inner_tol = max(5e-16, min(1e-14, 1e-5 * tol * t**2))
        else:
            inner_tol = max(1e-13, min(1e-11, 1e-3 * tol * t**2))
        Lam_init = np.zeros((q, q)) if iter_ == 1 else Lam
        PY, num_inner[iter_], Lam, _, in_flag = semi_newton_matrix(
            n, q, W, t, W + t * neg_pgx, eta * t, inner_tol, inner_iter, Lam_init, Dn, pDn
        )
        if in_flag:
            num_inexact += 1
        alpha = 1.0
        D = PY - W
        CD = C @ D

        Z, T = orthonormalize_stable(PY)
        AZ = (AW + alpha * CD) @ T
        f_trial = -np.sum(Z * AZ)
        F_trial = f_trial + np.sum(eta * np.abs(Z))
        normDsquared = np.linalg.norm(D, 'fro')**2
        if normDsquared / t**2 < tol:
            break
        while F_trial >= F[-1] - 0.5 / t * alpha * normDsquared:
            alpha *= 0.5
            num_linesearch += 1
            if alpha < t_min:
                num_inexact += 1
                break
            PY = W + alpha * D
            Z, T = orthonormalize_stable(PY)
            AZ = (AW + alpha * CD) @ T
            f_trial = -np.sum(Z * AZ)
            F_trial = f_trial + np.sum(eta * np.abs(Z))
        W = Z
        AW = AZ
        F.append(F_trial)
    W_manpg = W
    time_manpg = time.time() - start_time
    mean_ssn = np.sum(num_inner) / iter_
    sparsity = np.count_nonzero(W_manpg == 0) / (n * q)
    F_manpg = F[-1]
    flag_succ = 1 if iter_ < maxiter - 1 else 0
    return W_manpg, F_manpg, sparsity, time_manpg, iter_, flag_succ, num_linesearch, mean_ssn



def scspca_matrix(X, Y, kappa=0.1):
    return X.T @ Y @ Y.T @ X + kappa * X.T @ X


def scspca(X, Y, n_components, eta, kappa=0.1, **manpg_kwargs):
    p = X.shape[1]
    C = scspca_matrix(X, Y, kappa)
    eta_vec = eta * np.ones(p) if np.isscalar(eta) else np.asarray(eta, dtype=float)
    return manpg_orth_sparse(C, n_components, p, eta_vec, **manpg_kwargs)
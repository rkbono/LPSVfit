"""
lpsvfit — Legendre Polynomial Secular Variation fitting
"""
from dataclasses import dataclass
from math import comb, pi

import numpy as np
from numpy.polynomial import legendre
from scipy.optimize import minimize
from scipy.special import assoc_legendre_p_all, assoc_legendre_p, gammaln

__version__ = "0.5.0"

K_OUT = 4


def premix_c(kmax):
    """Model G to LPSV polynomial coefficients"""
    c = np.zeros(kmax + 1)
    c[0] = pi**2 / 4 - 2
    for n in range(1, kmax // 2 + 1):
        c[2*n] = (4*n + 1) * 16.0**n / (2 * n**2 * (2*n + 1)**2 * comb(2*n, n)**2)
    return c


def alpha_to_modelg(alpha):
    """LPSV to Model G by projection"""
    k = np.arange(2, len(alpha))
    c = C_PREMIX[k]
    b2 = np.sum(alpha[2:] * c / (2*k + 1)) / np.sum(c**2 / (2*k + 1))
    a2 = alpha[0] - C_PREMIX[0] * b2
    return np.degrees(np.sqrt(max(a2, 0.0))), np.sqrt(max(b2, 0.0))


C_PREMIX = premix_c(12)


def degrees(K, even=False):
    """selected Legendre degrees"""
    if np.isscalar(K):
        k = np.arange(int(K) + 1)
        return k[k % 2 == 0] if even else k
    return np.unique(np.asarray(list(K), dtype=int))


def _x(lat):
    return np.sin(np.radians(np.asarray(lat, dtype=float)))


def _A(model, lat, k=None, a1=True):
    """Design matrix at the given latitudes."""
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    if model == "lpsv":
        return legendre.legvander(_x(lat), int(k.max()))[:, k]
    P = legendre.legvander(_x(lat), 4)
    cols = [np.ones(lat.size), np.radians(lat)**2]
    if model == "mg+":
        if a1:
            cols.append(P[:, 1])
    elif model != "mg":
        raise ValueError(f"unknown model {model!r}")
    return np.column_stack(cols)


def _T(model, k=None, a1=True):
    """model params to alpha_k transform"""
    if model == "lpsv":
        T = np.zeros((K_OUT + 1, k.size))
        for j, kk in enumerate(k):
            if kk <= K_OUT:
                T[kk, j] = 1.0
        return T
    p = 2 + (model == "mg+" and bool(a1))
    T = np.zeros((K_OUT + 1, p))
    T[0, 0] = 1.0
    T[:, 1] = C_PREMIX[:K_OUT + 1]
    if model == "mg+" and a1:
        T[1, 2] = 1.0
    return T


def _labels(model, a1=True):
    if model == "mg":
        return np.array(["a2", "b2"], dtype=object)
    return np.array(["a2", "b2"] + (["a1"] if a1 else []), dtype=object)


MU_FLOOR = 1e-12


def _bounds(model, k=None, S_max=180.0, a1=True):
    """G theta >= h keeping MU_FLOOR <= S_hat^2 <= radians(S_max)^2 on dense lat grid 
        (Model G family also a2 >= 0, b2 >= 0)."""
    Ag = _A(model, np.linspace(-90.0, 90.0, 181), k, a1)
    rows = [Ag, -Ag]
    rhs = [np.full(181, MU_FLOOR), np.full(181, -np.radians(S_max)**2)]
    if model in ("mg", "mg+"):
        G0 = np.zeros((2, Ag.shape[1]))
        G0[0, 0] = G0[1, 1] = 1.0
        rows, rhs = [G0] + rows, [np.zeros(2)] + rhs
    return np.vstack(rows), np.concatenate(rhs)


def _degS(y):
    return np.degrees(np.sqrt(np.clip(y, 0.0, None)))


def predict(alpha_k, lat):
    """Dispersion S from LPSV at given lat"""
    return _degS(legendre.legval(_x(lat), alpha_k))


def sigma2_hat(S, n_sites=None):
    """estimated single-site variance: (weighted mean S^2)^2, rad^4."""
    n = None if n_sites is None else np.broadcast_to(
        np.asarray(n_sites, dtype=float), np.shape(S))
    return np.average(np.radians(S)**2, weights=n)**2


def _w(lat, sigma2, n_sites):
    """Pooled-pole weights w_i = (n_i - n_i/N_tot) / sigma2; equal weights if n is None."""
    N = np.atleast_1d(lat).size
    if n_sites is None: return np.full(N, 1.0 / sigma2)
    n = np.broadcast_to(np.asarray(n_sites, dtype=float), (N,))
    return n * (1.0 - 1.0 / n.sum()) / sigma2


def n_eff(lat, n_sites, reference="pooled"):
    """Effective VGP site count: 'known' n_i | 'local' n_i - 1 | 'pooled' n_i (1 - 1/N_tot)."""
    if n_sites is None:
        raise ValueError("n_eff requires n_sites")
    N = np.atleast_1d(lat).size
    n = np.broadcast_to(np.asarray(n_sites, dtype=float), (N,))
    if np.any(n <= 0):
        raise ValueError("n_sites must be positive")
    if reference == "known":
        return n.copy()
    if reference == "local":
        if np.any(n <= 1):
            raise ValueError("reference='local' requires n_sites > 1")
        return n - 1.0
    if reference == "pooled":
        return n * (1.0 - 1.0 / n.sum())
    raise ValueError(f"unknown reference {reference!r}")


def _ohm(k):
    return np.where(k == 0, 0.0, 4 * pi * (k + 1) * (2*k + 1) * (2*k + 3))


def _Lam(model, k, L, w, a1=True):
    """Penalty matrix

    lpsv : Lam = diag(L wbar ohm_k) on the active degrees.
    mg   : unpenalized.
    mg+  : just the a1 (degree-1 LPSV) term, following lpsv
    """
    if model == "lpsv":
        return np.diag(L * np.mean(w) * _ohm(k))
    if model == "mg+":
        kk = np.arange(K_OUT + 1)
        T = _T(model, None, a1).copy()
        T[:, :2] = 0.0
        return T.T @ np.diag(L * np.mean(w) * _ohm(kk)) @ T
    return np.zeros((2, 2))



def _gls(A, w, y, Lam, G=None, h=None):
    """model fit (theta) minimizing |sqrt(w)(A theta - y)|^2 + theta' Lam theta, with bounds G theta >= h."""
    B, z = np.sqrt(w)[:, None] * A, np.sqrt(w) * y
    ew, V = np.linalg.eigh(Lam)
    Bp = np.sqrt(np.clip(ew, 0.0, None))[:, None] * V.T
    c = np.linalg.lstsq(np.vstack([B, Bp]),
                        np.concatenate([z, np.zeros(A.shape[1])]), rcond=None)[0]
    if G is None or np.min(G @ c - h) >= -1e-9: return c
    obj = lambda c: float((B @ c - z) @ (B @ c - z) + c @ Lam @ c)
    jac = lambda c: 2.0 * (B.T @ (B @ c - z) + Lam @ c)
    return minimize(obj, c, jac=jac, method="SLSQP",
                    constraints=[{"type": "ineq", "fun": lambda c: G @ c - h,
                                  "jac": lambda c: G}],
                    options={"maxiter": 200, "ftol": 1e-12}).x


def _sd(Cov):
    return np.sqrt(np.clip(np.diag(Cov), 0.0, None))


def _diagnose(A, w, Lam):
    """returns diagnostics of sample distribution (design matrix)
    M = A'WA; 
    R = (M+Lam)^-1 M; 
    Cov = (M+Lam)^-1 M (M+Lam)^-1; 
    kappa = cond M.
    """
    B = np.sqrt(w)[:, None] * A
    M = B.T @ B
    P = np.linalg.pinv(M + Lam, hermitian=True)
    R, Cov = P @ M, P @ M @ P
    s = np.linalg.svd(B, compute_uv=False)
    kappa = float((s[0] / s[-1])**2) if s[-1] > 0 else np.inf
    return M, R, 0.5 * (Cov + Cov.T), kappa



@dataclass
class Fit:
    model: str
    lat: np.ndarray
    S: np.ndarray
    theta: np.ndarray
    k: np.ndarray
    Cov: np.ndarray
    alpha_k: np.ndarray
    Cov_k: np.ndarray
    S_hat: np.ndarray
    sd: np.ndarray
    sd_k: np.ndarray
    rms: float
    chi2: float
    sigma2: float
    edof: float
    kappa: float
    converged: bool = True

    def predict(self, lat, z=None):
        """Fitted dispersion S (deg) at given latitudes.
        z : scalar for z-sigma confidence bands (e.g. 1.96).
        """
        klist = self.k if self.model == "lpsv" else None
        a1 = self.model != "mg+" or "a1" in self.k
        A = _A(self.model, lat, klist, a1)
        y = A @ self.theta
        if z is None:
            return _degS(y)
        half = z * np.sqrt(np.einsum("ij,jk,ik->i", A, self.Cov, A))
        return _degS(y), _degS(y - half), _degS(y + half)


def fit(model, lat, S, K=4, even=False, L=5e-5, n_sites=None, sigma2=None,
        S_max=180.0, a1=True, ridge=True):
    """Fit selected zonal dispersion model to VGP dispersion data

    model : 'mg' | 'mg+' | 'lpsv'
    lat, S : (N,) deg
    K, even : LPSV degrees (max degree or set)
    L : ohmic penalty strength (lpsv and mg+; mg is unpenalized)
    n_sites : (N,), scalar, or None -- sites per locality (None: equal weights)
    sigma2 : rad^4 single-site variance; None -> sigma2_hat(S, n_sites)
    a1 : mg+ only -- include the P1 asymmetry column
    ridge : mg+ only -- False fits unpenalized (e.g. noise-free true curves)
    """
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    y = np.radians(S)**2
    k = degrees(K, even) if model == "lpsv" else None
    A, T = _A(model, lat, k, a1), _T(model, k, a1)
    G, h = _bounds(model, k, S_max, a1)
    if sigma2 is None:
        sigma2 = sigma2_hat(S, n_sites)
    w = _w(lat, sigma2, n_sites)
    Lam = _Lam(model, k, 0.0 if (model == "mg+" and not ridge) else L, w, a1)

    th = _gls(A, w, y, Lam, G, h)
    _, R, Cov, kappa = _diagnose(A, w, Lam)
    S_hat = _degS(A @ th)
    Cov_k = T @ Cov @ T.T

    return Fit(model=model, lat=lat, S=S, theta=th,
               k=k if model == "lpsv" else _labels(model, a1),
               Cov=Cov, alpha_k=T @ th, Cov_k=Cov_k, S_hat=S_hat,
               rms=float(np.sqrt(np.mean((S - S_hat)**2))),
               chi2=float((y - A @ th) @ (w * (y - A @ th))),
               sd=_sd(Cov), sd_k=_sd(Cov_k),
               sigma2=float(sigma2), edof=float(np.trace(R)), kappa=kappa)


def to_modelg(f):
    """Model G(+) fit -> (a, b) with fit standard deviations.
    returns : dict(a, b, sd_a, sd_b[, a1, sd_a1])
    """
    if f.model not in ("mg", "mg+"):
        raise ValueError(f"to_modelg expects a Model G(+) fit, got {f.model!r}")
    a2, b2 = np.clip(f.theta[:2], 0.0, None)
    out = dict(a=float(np.degrees(np.sqrt(a2))), b=float(np.sqrt(b2)),
               sd_a=float(np.degrees(f.sd[0] / (2 * np.sqrt(a2)))) if a2 > 0 else np.inf,
               sd_b=float(f.sd[1] / (2 * np.sqrt(b2))) if b2 > 0 else np.inf)
    if f.model == "mg+":
        j = list(f.k).index("a1") if "a1" in list(f.k) else None
        if j is not None:
            out["a1"] = float(f.theta[j])
            out["sd_a1"] = float(f.sd[j])
    return out


def make_fit(model, theta=None, alpha_k=None, lat=None, K=None, even=False,
             L=5e-5, n_sites=None, sigma2=None, a1=True, ridge=True):
    """Instantiate a Fit directly from parameters

    Provide either:
    theta : model parameters (mg: (a2, b2); mg+: (a2, b2, a1);
              lpsv: alpha on the active degrees)
    alpha_k : Legendre coefficients (rad^2).
    """
    if (theta is None) == (alpha_k is None):
        raise ValueError("give exactly one of theta, alpha_k")
    if model == "lpsv":
        p = (alpha_k if theta is None else theta)
        p = np.atleast_1d(np.asarray(p, dtype=float))
        k = degrees(p.size - 1 if K is None else K, even)
        theta = p[k] if theta is None else p
        if k.size != theta.size:
            raise ValueError("theta length does not match degrees(K, even)")
    elif model in ("mg", "mg+"):
        k = None
        if theta is None:
            T = _T(model, None, a1)
            aa = np.zeros(K_OUT + 1)
            av = np.atleast_1d(np.asarray(alpha_k, dtype=float))[:K_OUT + 1]
            aa[:av.size] = av
            D = np.diag(1.0 / (2 * np.arange(K_OUT + 1) + 1))
            theta = np.linalg.solve(T.T @ D @ T, T.T @ D @ aa)
        else:
            theta = np.atleast_1d(np.asarray(theta, dtype=float))
    else:
        raise ValueError(f"unknown model {model!r}")
    T = _T(model, k, a1)
    labels = k if model == "lpsv" else _labels(model, a1)
    if theta.size != T.shape[1]:
        raise ValueError("theta length does not match the model parameters")

    if lat is None:
        return Fit(model=model, lat=None, S=None, theta=theta, k=labels,
                   Cov=None, alpha_k=T @ theta, Cov_k=None, S_hat=None,
                   sd=None, sd_k=None, rms=0.0, chi2=0.0,
                   sigma2=np.nan if sigma2 is None else float(sigma2),
                   edof=np.nan, kappa=np.nan)

    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    A = _A(model, lat, k, a1)
    S_hat = _degS(A @ theta)
    if sigma2 is None:
        sigma2 = sigma2_hat(S_hat, n_sites)
    w = _w(lat, sigma2, n_sites)
    Lam = _Lam(model, k, 0.0 if (model == "mg+" and not ridge) else L, w, a1)
    _, R, Cov, kappa = _diagnose(A, w, Lam)
    Cov_k = T @ Cov @ T.T
    return Fit(model=model, lat=lat, S=S_hat, theta=theta, k=labels,
               Cov=Cov, alpha_k=T @ theta, Cov_k=Cov_k, S_hat=S_hat,
               sd=_sd(Cov), sd_k=_sd(Cov_k), rms=0.0, chi2=0.0,
               sigma2=float(sigma2), edof=float(np.trace(R)), kappa=kappa)



_PARAM_NAMES = {
    "mg":  np.array(["a", "b"], dtype=object),
    "mg+": np.array(["a", "b", "d"], dtype=object),
}


@dataclass
class BootResult:
    model: str
    param_names: np.ndarray
    params: np.ndarray
    alpha_k: np.ndarray
    lat: np.ndarray
    S_pred: np.ndarray

    @property
    def B(self):
        return self.params.shape[0]

    def summary(self, ci=95):
        lo, hi = (100 - ci) / 2, (100 + ci) / 2
        return dict(
            param_median=np.median(self.params, axis=0),
            param_ci=np.percentile(self.params, [lo, hi], axis=0).T,
            S_median=np.median(self.S_pred, axis=0),
            S_ci=np.percentile(self.S_pred, [lo, hi], axis=0).T,
        )


def _readout(fb):
    if fb.model in ("mg", "mg+"):
        g = to_modelg(fb)
        row = [g["a"], g["b"]]
        if fb.model == "mg+" and "a1" in g:
            row.append(g["a1"])
        return row
    return fb.alpha_k.tolist()


def bootstrap(model, lat, S, B=1000, seed=None, lat_out=None, **kw):
    lat = np.atleast_1d(np.asarray(lat, dtype=float))
    S = np.atleast_1d(np.asarray(S, dtype=float))
    N = lat.size
    n = kw.pop("n_sites", None)
    n = None if n is None else np.broadcast_to(np.asarray(n, dtype=float), (N,))
    kw.setdefault("sigma2", sigma2_hat(S, n))
    if lat_out is None:
        lat_out = np.linspace(-90, 90, 37)
    lat_out = np.atleast_1d(np.asarray(lat_out, dtype=float))
    rng = np.random.default_rng(seed)

    alpha_draws = np.empty((B, K_OUT + 1))
    param_draws = []
    S_draws = np.empty((B, lat_out.size))

    for b in range(B):
        i = rng.integers(0, N, N)
        fb = fit(model, lat[i], S[i],
                 n_sites=None if n is None else n[i], **kw)
        alpha_draws[b] = fb.alpha_k
        param_draws.append(_readout(fb))
        S_draws[b] = fb.predict(lat_out)

    names = _PARAM_NAMES.get(model,
                np.array([f"a{k}" for k in range(K_OUT + 1)], dtype=object))

    return BootResult(model=model, param_names=names,
                      params=np.array(param_draws), alpha_k=alpha_draws,
                      lat=lat_out, S_pred=S_draws)



def schmidt(lmax, z):
    """Schmidt semi-normalized P_l^m(z) and d/dtheta, each (l, m, node)."""
    z = np.atleast_1d(np.asarray(z, dtype=float))
    a = np.asarray(assoc_legendre_p_all(lmax, lmax, z, diff_n=1))
    l = np.arange(lmax + 1)[:, None]
    m = np.arange(lmax + 1)[None, :]
    norm = np.where(m <= l,
                    (-1.0)**m * np.sqrt(np.where(m == 0, 1.0, 2.0)
                                        * np.exp(gammaln(l - m + 1)
                                                 - gammaln(l + m + 1))),
                    0.0)[:, :, None]
    P = norm * a[0, :, :lmax + 1, :]
    dP = -norm * a[1, :, :lmax + 1, :] * np.sqrt(np.clip(1.0 - z**2, 0.0, None))
    return P, dP


def projection_weights(lmax=5, kmax=None, tol=1e-10):
    """Integration constants of alpha_k = sum W[k,l,l',m] C[l,l',m] (Table S3)."""
    if kmax is None:
        kmax = 2 * (lmax + 1)
    z, wq = legendre.leggauss((kmax + 2 * lmax) // 2 + 6)
    P, dP = schmidt(max(lmax, kmax), z)
    s2 = np.clip(1.0 - z**2, 1e-300, None)
    W = np.zeros((kmax + 1, lmax + 1, lmax + 1, lmax + 1))
    for l in range(1, lmax + 1):
        for lp in range(1, lmax + 1):
            for m in range(min(l, lp) + 1):
                t1 = z * dP[l, m] + 0.5 * (l + 1) * np.sqrt(s2) * P[l, m]
                t2 = z * dP[lp, m] + 0.5 * (lp + 1) * np.sqrt(s2) * P[lp, m]
                phi = (t1 * t2 + m**2 * P[l, m] * P[lp, m] / s2) \
                    * (2.0 if m == 0 else 1.0) / 2.0
                for kk in range(kmax + 1):
                    W[kk, l, lp, m] = (2 * kk + 1) / 2.0 * np.sum(wq * phi * P[kk, 0])
    W[np.abs(W) < tol] = 0.0
    return W


def C_from_gauss(g, h, lmax=5, normalize=True):
    """Second moments C[l,l',m] of Gauss coefficient series g, h [t, l, m]."""
    g = np.atleast_3d(np.asarray(g, dtype=float).T).T
    h = np.atleast_3d(np.asarray(h, dtype=float).T).T
    if normalize:
        g10 = g[:, 1, 0][:, None, None]
        g, h = g / g10, h / g10
    t = g.shape[0]
    C = np.zeros((lmax + 1, lmax + 1, lmax + 1))
    for m in range(lmax + 1):
        G, H = g[:, :lmax + 1, m], h[:, :lmax + 1, m]
        C[:, :, m] = (G.T @ G + H.T @ H) / t
        C[:m, :, m] = C[:, :m, m] = 0.0
    C[0, :, :] = C[:, 0, :] = 0.0
    C[1, :, 0] = C[:, 1, 0] = 0.0
    return C


def alpha_from_C(C, W):
    """alpha_k from second-moment tensor"""
    return np.tensordot(W, C, axes=3)



def ggp2lpsv(alpha, beta, kmax=12, lmax=5, g10=18, ca=0.547):
    """alpha_k from Giant Gaussian Process parameters."""
    def _schmidt_lm(l, m, z):
        z = np.atleast_1d(np.asarray(z, dtype=float))
        a = np.asarray(assoc_legendre_p(l, m, z, diff_n=1))
        norm = (-1.0)**m * np.sqrt((1.0 if m == 0 else 2.0)
                                   * np.exp(gammaln(l - m + 1) - gammaln(l + m + 1)))
        P = norm * a[0]
        dP = -norm * a[1] * np.sqrt(np.clip(1.0 - z**2, 0.0, None))
        return P, dP

    def _F(c, s, l, m):
        P, dP = _schmidt_lm(l, m, c)
        t1 = c * dP + 0.5 * (l + 1) * s * P
        t2 = m * P / s
        return t1**2 + t2**2

    def _G(l):
        return ca**(2 * l) / ((l + 1) * (2 * l + 1))

    cx, wq = legendre.leggauss((kmax + 2 * lmax) // 2 + 6)
    sx = np.sqrt(np.clip(1.0 - cx**2, 1e-300, None))
    a2 = (alpha / g10)**2

    alpha_k = {}
    for k in range(kmax + 1):
        Pk, _ = _schmidt_lm(k, 0, cx)
        tot = 0.0
        for l in range(1, lmax + 1):
            for m in range(l + 1):
                if l == 1 and m == 0: continue
                b2 = beta**2 if (l - m) % 2 else 1.0
                tot += (2 * k + 1) / 2 * np.sum(wq * Pk * _F(cx, sx, l, m)) * _G(l) * a2 * b2
        alpha_k[k] = tot
    return alpha_k

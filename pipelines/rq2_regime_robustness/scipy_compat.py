#!/usr/bin/env python3
# =============================================================================
# scipy_compat.py
# Pure-NumPy fallbacks for scipy / statsmodels.
# Activated automatically if scipy fails to import (Windows DLL error).
#
# Usage in every stage file:
#     from scipy_compat import stats, sm, multipletests
# =============================================================================

import numpy as np

_SCIPY_OK = False
try:
    from scipy import stats as _real_stats
    from statsmodels.stats.multitest import multipletests as _real_mt
    import statsmodels.api as _real_sm
    _SCIPY_OK = True
except Exception as _e:
    import warnings
    warnings.warn(
        f"scipy/statsmodels unavailable ({_e}). "
        "Using pure-NumPy fallbacks — install via: "
        "conda install scipy statsmodels",
        RuntimeWarning, stacklevel=2,
    )


# =============================================================================
# Pure-NumPy stats fallbacks
# =============================================================================

def _erf(x):
    """Abramowitz & Stegun erf approximation (max error 1.5e-7)."""
    t = 1.0 / (1.0 + 0.3275911 * np.abs(x))
    p = t * (0.254829592
             + t * (-0.284496736
                    + t * (1.421413741
                           + t * (-1.453152027
                                  + t * 1.061405429))))
    r = 1.0 - p * np.exp(-(x ** 2))
    return np.where(x >= 0, r, -r)


def _norm_cdf(x):
    return 0.5 * (1.0 + _erf(np.asarray(x, float) / np.sqrt(2.0)))


def _norm_sf(x):
    return 1.0 - _norm_cdf(x)


def _chi2_cdf(x, df):
    """Wilson-Hilferty normal approximation for chi2 CDF."""
    x   = np.asarray(x, float)
    z   = ((x / df) ** (1.0 / 3) - (1.0 - 2.0 / (9 * df))) \
          / np.sqrt(2.0 / (9 * df))
    return float(_norm_cdf(z))


def _t_sf(t, df):
    """Survival function of t-distribution via normal approx."""
    if df > 200:
        return float(_norm_sf(abs(t)))
    return float(_norm_sf(abs(t)) * 1.05)   # conservative adjustment


def _spearmanr(x, y):
    """Spearman rank correlation (pure numpy)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan")
    rx = np.argsort(np.argsort(x)).astype(float) + 1.0
    ry = np.argsort(np.argsort(y)).astype(float) + 1.0
    d2  = ((rx - ry) ** 2).sum()
    rho = 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))
    rho = float(np.clip(rho, -1.0, 1.0))
    if abs(rho) >= 1.0:
        return rho, 0.0
    t   = rho * np.sqrt((n - 2) / (1 - rho ** 2))
    p   = 2.0 * _t_sf(abs(t), df=n - 2)
    return rho, float(p)


class _StatsFallback:
    """Minimal scipy.stats-compatible namespace."""

    @staticmethod
    def spearmanr(x, y):
        return _spearmanr(x, y)

    class norm:
        @staticmethod
        def cdf(x):
            return _norm_cdf(x)

        @staticmethod
        def sf(x):
            return _norm_sf(x)

    class chi2:
        @staticmethod
        def sf(x, df):
            return 1.0 - _chi2_cdf(x, df)

        @staticmethod
        def cdf(x, df):
            return _chi2_cdf(x, df)


# =============================================================================
# Pure-NumPy OLS with optional Newey-West HAC SE
# =============================================================================

def _nw_cov(X, e, lags):
    n, k = X.shape
    S = X.T @ np.diag(e ** 2) @ X
    for l in range(1, lags + 1):
        w  = 1.0 - l / (lags + 1)
        Sl = (X[l:].T * (e[l:] * e[:-l])) @ X[:-l]
        S += w * (Sl + Sl.T)
    XtXi = np.linalg.pinv(X.T @ X)
    return XtXi @ S @ XtXi


class _OLSResult:
    def __init__(self, params, tvalues, pvalues, bse, rsquared, nobs):
        self.params   = params
        self.tvalues  = tvalues
        self.pvalues  = pvalues
        self.bse      = bse
        self.rsquared = rsquared
        self.nobs     = nobs


class _OLS:
    def __init__(self, y, X):
        self._y = np.asarray(y, float)
        self._X = np.asarray(X, float)

    def fit(self, cov_type="nonrobust", cov_kwds=None):
        y, X = self._y, self._X
        n, k = X.shape
        try:
            XtXi = np.linalg.inv(X.T @ X)
        except np.linalg.LinAlgError:
            XtXi = np.linalg.pinv(X.T @ X)
        beta  = XtXi @ X.T @ y
        yhat  = X @ beta
        resid = y - yhat
        sse   = (resid ** 2).sum()
        sst   = ((y - y.mean()) ** 2).sum()
        r2    = 1.0 - sse / sst if sst > 0 else 0.0

        if cov_type == "HAC":
            lags = (cov_kwds or {}).get("maxlags", 5)
            V    = _nw_cov(X, resid, lags)
        else:
            s2 = sse / max(n - k, 1)
            V  = s2 * XtXi

        se    = np.sqrt(np.maximum(np.diag(V), 0))
        t_vec = np.where(se > 0, beta / se, 0.0)
        p_vec = np.array([2.0 * _t_sf(abs(t), df=max(n - k, 1))
                          for t in t_vec])

        # Map to column names
        try:
            import pandas as pd
            if hasattr(self._X, "columns"):
                cols = list(self._X.columns)
            else:
                cols = [f"x{i}" for i in range(k)]
        except Exception:
            cols = [f"x{i}" for i in range(k)]

        params   = dict(zip(cols, beta))
        tvalues  = dict(zip(cols, t_vec))
        pvalues  = dict(zip(cols, p_vec))
        bse      = dict(zip(cols, se))
        return _OLSResult(params, tvalues, pvalues, bse, r2, n)


class _SM:
    OLS = _OLS

    @staticmethod
    def add_constant(X, has_constant="raise"):
        try:
            import pandas as pd
            if isinstance(X, pd.Series):
                df = X.to_frame()
            elif isinstance(X, pd.DataFrame):
                df = X.copy()
            else:
                df = pd.DataFrame(X)
            if "const" not in df.columns:
                df.insert(0, "const", 1.0)
            return df
        except Exception:
            arr = np.atleast_2d(np.asarray(X, float))
            if arr.shape[0] == 1:
                arr = arr.T
            return np.hstack([np.ones((len(arr), 1)), arr])


# =============================================================================
# Benjamini-Hochberg FDR (pure Python)
# =============================================================================

def _multipletests(pvals, alpha=0.05, method="fdr_bh"):
    pvals = np.asarray(pvals, float)
    n     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.arange(1, n + 1)

    if method in ("fdr_bh", "fdr_by"):
        thresh = ranks / n * alpha
        corr   = np.minimum(1.0, pvals[order] * n / ranks)
        for i in range(n - 2, -1, -1):
            corr[i] = min(corr[i], corr[i + 1])
        pvals_c          = np.empty(n)
        pvals_c[order]   = corr
    else:  # bonferroni
        pvals_c = np.minimum(1.0, pvals * n)

    reject = pvals_c <= alpha
    return reject, pvals_c, alpha / n, alpha / n


# =============================================================================
# Public interface — always import from here
# =============================================================================

if _SCIPY_OK:
    stats         = _real_stats
    multipletests = _real_mt
    sm            = _real_sm
else:
    stats         = _StatsFallback()
    multipletests = _multipletests
    sm            = _SM()

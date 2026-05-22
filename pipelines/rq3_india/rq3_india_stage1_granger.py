#!/usr/bin/env python3
# =============================================================================
# rq3_india_stage1_granger.py
# Stage 1: Granger Causality — Indian Cross-Asset Network
#
# Core India hypotheses tested:
#   H_FX:     USD/INR Granger-causes IT sector (INFY.NS) with 1-3 day lag
#   H_GLOBAL: SPY overnight Granger-causes ^NSEI (global coupling)
#   H_OIL:    CL=F Granger-causes ONGC.NS / ^NSEI (energy transmission)
#   H_SECTOR: ^NSEBANK Granger-causes ^NSEI (banking sector leads broad market)
#
# Method: identical to US stage 1 — AIC lag selection, F-test, BH correction.
# The same statistical machinery; only the universe and data differ.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from rq3_india_config import (
    DATA_DIR, OUTPUT_DIR, GRANGER_MAX_LAG,
    GRANGER_ALPHA, MIN_CAUSAL_PAIRS,
)

try:
    from scipy_compat import stats, multipletests
except ImportError:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests


# ── loaders ──────────────────────────────────────────────────────────────────
def load_returns() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "rq3_india_log_returns.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found. Run stage0 first.")
    return pd.read_parquet(path)


# ── Granger F-test (identical logic to US stage1) ────────────────────────────
def _ols(Y, X):
    """OLS: return beta, RSS."""
    Xc  = np.hstack([np.ones((len(X), 1)), X])
    b   = np.linalg.lstsq(Xc, Y, rcond=None)[0]
    rss = float(((Y - Xc @ b) ** 2).sum())
    return b, rss


def _aic(rss, n, k):
    return n * np.log(rss / n) + 2 * k


def _build_lags(y, x, p):
    T  = len(y)
    Y  = y[p:]
    Xr = np.column_stack([y[p - k: T - k] for k in range(1, p + 1)])
    if x is None:
        return Y, Xr
    Xu = np.column_stack([x[p - k: T - k] for k in range(1, p + 1)])
    return Y, np.hstack([Xr, Xu])


def granger_f(y: np.ndarray, x: np.ndarray,
               max_lag: int = GRANGER_MAX_LAG) -> dict:
    """H0: x does NOT Granger-cause y."""
    empty = dict(F=np.nan, p_val=1.0, lag=0,
                 reject_raw=False, AIC_r=np.nan, AIC_u=np.nan)
    mask  = ~(np.isnan(y) | np.isnan(x))
    y, x  = y[mask].astype(float), x[mask].astype(float)
    if len(y) < max_lag * 4:
        return empty

    # AIC-select lag on restricted model
    best_aic, best_p = np.inf, 1
    for p in range(1, max_lag + 1):
        Y_r, X_r = _build_lags(y, None, p)
        _, rss_r  = _ols(Y_r, X_r)
        a = _aic(rss_r, len(Y_r), p + 1)
        if a < best_aic:
            best_aic, best_p = a, p
    p = best_p

    Y_r, X_r = _build_lags(y, None, p);  _, rss_r = _ols(Y_r, X_r)
    Y_u, X_u = _build_lags(y, x,    p);  _, rss_u = _ols(Y_u, X_u)
    n, k = len(Y_r), p
    if rss_u <= 0 or rss_r <= rss_u:
        return {**empty, "lag": p}

    F     = ((rss_r - rss_u) / k) / (rss_u / max(n - 2 * k - 1, 1))
    # chi-squared approximation for p-value
    from scipy.stats import f as f_dist
    try:
        p_val = float(f_dist.sf(F, k, n - 2 * k - 1))
    except Exception:
        p_val = float(np.exp(-F / 2))

    _, aic_u = _ols(Y_u, X_u)

    return dict(F=round(F, 4), p_val=round(p_val, 6), lag=p,
                reject_raw=(p_val < GRANGER_ALPHA),
                AIC_r=round(best_aic, 2),
                AIC_u=round(_aic(rss_u, len(Y_u), 2 * p + 1), 2))


def run_granger(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Test all N*(N-1) directed pairs."""
    tickers = [c for c in log_ret.columns if log_ret[c].notna().sum() > 500]
    pairs, rows = [(c, t) for c in tickers for t in tickers if c != t], []
    log.info(f"[GRANGER] Testing {len(pairs)} directed pairs "
             f"({len(tickers)} tickers) ...")

    for cause, target in pairs:
        y = log_ret[target].values
        x = log_ret[cause].values
        res = granger_f(y, x)
        rows.append(dict(cause=cause, target=target, **res))

    df = pd.DataFrame(rows)
    # BH FDR correction
    reject_bh, _, _, _ = multipletests(
        df["p_val"].fillna(1).values, alpha=GRANGER_ALPHA, method="fdr_bh"
    )
    df["reject_BH"] = reject_bh
    return df


def summarise_india_hypotheses(df: pd.DataFrame):
    """Print results for the four key India-specific hypotheses."""
    hypotheses = [
        ("USD/INR → INFY.NS  (FX → IT sector)",
         "USDINR=X", "INFY.NS"),
        ("USD/INR → TCS.NS   (FX → IT sector)",
         "USDINR=X", "TCS.NS"),
        ("SPY → ^NSEI        (Global overnight coupling)",
         "SPY", "^NSEI"),
        ("CL=F → ^NSEI       (Oil → broad market)",
         "CL=F", "^NSEI"),
        ("^NSEBANK → ^NSEI   (Banking leads broad)",
         "^NSEBANK", "^NSEI"),
        ("^NSEI → INFY.NS    (Market → IT sector)",
         "^NSEI", "INFY.NS"),
        ("^NSEI → SBIN.NS    (Market → PSU bank)",
         "^NSEI", "SBIN.NS"),
        ("GC=F → ^NSEI       (Gold → equity)",
         "GC=F", "^NSEI"),
    ]
    print("\n  KEY INDIA HYPOTHESES:")
    print("  " + "-" * 62)
    for desc, cause, target in hypotheses:
        row = df[(df["cause"] == cause) & (df["target"] == target)]
        if row.empty:
            print(f"  {desc:<45} — not tested")
            continue
        r   = row.iloc[0]
        sig = "REJECT H0 ✓" if r["reject_BH"] else "fail"
        print(f"  {desc:<45} F={r['F']:>7.3f}  "
              f"p={r['p_val']:.4f}  {sig}")


def plot_heatmap(df: pd.DataFrame, save_path: str):
    """Granger F-stat heatmap: cause → target."""
    tickers = sorted(set(df["cause"].tolist() + df["target"].tolist()))
    mat     = pd.DataFrame(0.0, index=tickers, columns=tickers)
    for _, row in df.iterrows():
        if row["reject_BH"]:
            mat.loc[row["target"], row["cause"]] = float(row["F"])

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat.values, cmap="YlOrRd", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="Granger F-stat (BH-significant only)")
    ax.set_xticks(range(len(tickers)))
    ax.set_yticks(range(len(tickers)))
    ax.set_xticklabels(tickers, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tickers, fontsize=8)
    ax.set_xlabel("CAUSE →", fontsize=10)
    ax.set_ylabel("← TARGET", fontsize=10)
    ax.set_title("India Cross-Asset Granger Causality\n"
                 "(BH-significant pairs only; row=target, col=cause)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[PLOT] Saved → {save_path}")


def main():
    log.info("=" * 65)
    log.info("  RQ3 INDIA STAGE 1 — Granger Causality Network")
    log.info("=" * 65)

    log_ret = load_returns()
    df      = run_granger(log_ret)

    # Save
    out_csv = os.path.join(OUTPUT_DIR, "rq3_india_granger_results.csv")
    df.to_csv(out_csv, index=False)

    n_bh    = int(df["reject_BH"].sum())
    n_pairs = len(df)
    h1_ok   = n_bh >= MIN_CAUSAL_PAIRS

    # Summary
    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  RQ3 INDIA STAGE 1 — GRANGER RESULTS")
    print(sep)
    print(f"  Pairs tested          : {n_pairs}")
    print(f"  BH-significant (q=.05): {n_bh}  ({n_bh/n_pairs:.1%})")
    print(f"  H1 condition (>={MIN_CAUSAL_PAIRS}):   "
          f"{'CONFIRMED ✓' if h1_ok else 'NOT confirmed'}")
    print(sep)

    summarise_india_hypotheses(df)

    # Top 15 causal pairs
    top = (df[df["reject_BH"]]
           .sort_values("F", ascending=False)
           .head(15))
    if not top.empty:
        print("\n  TOP 15 GRANGER PAIRS (BH-significant):")
        print("  " + "-" * 58)
        for _, r in top.iterrows():
            print(f"  {r['cause']:<18} → {r['target']:<18} "
                  f"F={r['F']:>8.3f}  p={r['p_val']:.5f}  lag={r['lag']}")

    print(sep)

    with open(os.path.join(OUTPUT_DIR, "rq3_india_stage1_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"Pairs tested: {n_pairs}\n"
                f"BH-significant: {n_bh}\n"
                f"H1: {'CONFIRMED' if h1_ok else 'NOT confirmed'}\n")

    plot_heatmap(df, os.path.join(OUTPUT_DIR, "rq3_india_granger_heatmap.png"))

    log.info("\n✓ Stage 1 complete. Run rq3_india_stage2_dcc.py next.\n")
    return df


if __name__ == "__main__":
    main()

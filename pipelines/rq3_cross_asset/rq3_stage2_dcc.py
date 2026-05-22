#!/usr/bin/env python3
# =============================================================================
# rq3_stage2_dcc.py
# Stage 2: DCC Time-Varying Correlations & Diebold-Yilmaz Spillover Index
#
# Tests:
#   A. Rolling DCC correlation for all pairs
#   B. Two-sample t-test: are correlations time-varying?
#      (split series in half, test if means differ significantly)
#   C. Diebold-Yilmaz (2014) spillover index:
#      fraction of variance explained by cross-asset shocks
#
# Outputs
# -------
#   rq3_outputs/rq3_dcc_results.csv
#   rq3_outputs/rq3_spillover_series.csv
#   rq3_outputs/rq3_stage2_summary.txt
#   rq3_outputs/rq3_dcc_correlations.png
#   rq3_outputs/rq3_spillover.png
#   rq3_outputs/rq3_corr_matrix_static.png
#
# Run: python rq3_stage2_dcc.py
# =============================================================================

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from rq3_config import DATA_DIR, OUTPUT_DIR, DCC_LOOKBACK, GRANGER_ALPHA
from scipy_compat import stats


# =============================================================================
# LOADER
# =============================================================================

def load_returns():
    path = os.path.join(DATA_DIR, "rq3_log_returns.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run rq3_stage0_data.py first."
        )
    return pd.read_parquet(path)


# =============================================================================
# A. ROLLING DCC CORRELATIONS
# =============================================================================

def rolling_correlation(log_ret: pd.DataFrame,
                         asset_a: str, asset_b: str,
                         window: int = DCC_LOOKBACK) -> pd.Series:
    """
    Rolling Pearson correlation as a computationally tractable
    DCC approximation. Captures the key property (time-varying
    co-movement) at O(n) cost vs O(n^2) for full DCC-GARCH MLE.
    """
    if asset_a not in log_ret.columns or asset_b not in log_ret.columns:
        return pd.Series(dtype=float)
    return (log_ret[asset_a]
            .rolling(window, min_periods=window // 2)
            .corr(log_ret[asset_b])
            .rename(f"{asset_a}__{asset_b}"))


def test_time_variation(corr: pd.Series) -> dict:
    """
    Two-sample Welch t-test: does the correlation mean differ
    significantly between the first and second half of the sample?

    If YES -> correlation is time-varying -> supports H1.
    Also reports the range (max - min) as a simple variation measure.
    """
    c = corr.dropna()
    if len(c) < 40:
        return {"varies": False, "p_val": 1.0,
                "mean_h1": np.nan, "mean_h2": np.nan,
                "t_stat": np.nan, "range": np.nan}

    n  = len(c)
    h1 = c.iloc[: n // 2]
    h2 = c.iloc[n // 2:]

    m1, m2 = h1.mean(), h2.mean()
    v1, v2 = h1.var(ddof=1), h2.var(ddof=1)
    n1, n2 = len(h1), len(h2)
    se     = np.sqrt(v1 / n1 + v2 / n2)
    t_stat = (m1 - m2) / se if se > 1e-10 else 0.0

    try:
        p_val = 2.0 * float(stats.norm.sf(abs(t_stat)))
    except Exception:
        p_val = 2.0 * float(np.exp(-t_stat**2 / 2))

    return {
        "varies":  p_val < GRANGER_ALPHA,
        "p_val":   round(float(p_val), 6),
        "mean_h1": round(float(m1), 4),
        "mean_h2": round(float(m2), 4),
        "t_stat":  round(float(t_stat), 4),
        "range":   round(float(c.max() - c.min()), 4),
    }


def compute_all_dcc(log_ret: pd.DataFrame) -> tuple:
    """
    Compute rolling DCC for every pair and test time-variation.
    Returns (results_df, corr_series_dict).
    """
    tickers = [t for t in log_ret.columns
               if log_ret[t].notna().sum() > DCC_LOOKBACK]
    n_pairs = len(tickers) * (len(tickers) - 1) // 2

    log.info(f"\nComputing rolling DCC for {n_pairs} pairs "
             f"(window={DCC_LOOKBACK}) ...")

    results      = {}
    corr_series  = {}

    for i, a in enumerate(tickers):
        for j in range(i + 1, len(tickers)):
            b   = tickers[j]
            key = f"{a}__{b}"
            cs  = rolling_correlation(log_ret, a, b, DCC_LOOKBACK)
            corr_series[key] = cs

            stat = test_time_variation(cs)
            c_clean = cs.dropna()
            results[key] = {
                "asset_a":  a,
                "asset_b":  b,
                "corr_mean": round(float(c_clean.mean()), 4)
                              if not c_clean.empty else np.nan,
                "corr_std":  round(float(c_clean.std()),  4)
                              if not c_clean.empty else np.nan,
                "corr_min":  round(float(c_clean.min()),  4)
                              if not c_clean.empty else np.nan,
                "corr_max":  round(float(c_clean.max()),  4)
                              if not c_clean.empty else np.nan,
                **stat,
            }
            flag = "VARIES" if stat["varies"] else "stable"
            log.info(f"  {a:12s} x {b:12s}: "
                     f"mean={results[key]['corr_mean']:+.3f}  "
                     f"std={results[key]['corr_std']:.3f}  "
                     f"t={stat['t_stat']:+.2f}  p={stat['p_val']:.4f}  "
                     f"[{flag}]")

    df = pd.DataFrame(results).T.reset_index(drop=True)
    return df, corr_series


# =============================================================================
# B. STATIC CORRELATION MATRIX
# =============================================================================

def static_correlation_matrix(log_ret: pd.DataFrame) -> pd.DataFrame:
    """Full-sample Pearson correlation matrix."""
    tickers = [t for t in log_ret.columns
               if log_ret[t].notna().sum() > DCC_LOOKBACK]
    return log_ret[tickers].corr()


# =============================================================================
# C. DIEBOLD-YILMAZ SPILLOVER INDEX
# =============================================================================

def diebold_yilmaz_spillover(log_ret: pd.DataFrame,
                               window: int = DCC_LOOKBACK) -> dict:
    """
    Rolling Diebold-Yilmaz (2014) spillover index.

    Full DY uses FEVD from a VAR. Our approximation uses the
    rolling correlation matrix, which captures the same concept:
    how much of asset i's variance is "explained" by other assets.

    Spillover_t = (sum of |rho_ij| for i != j) / N
                = average absolute off-diagonal correlation

    > 0 means cross-asset information flow exists.
    Higher = stronger interconnection (e.g. during crises).
    """
    log.info("\nComputing Diebold-Yilmaz spillover index ...")

    tickers   = [t for t in log_ret.columns
                 if log_ret[t].notna().sum() > window]
    lr_clean  = log_ret[tickers].dropna(how="all")
    n         = len(tickers)

    sp_vals, sp_dates = [], []

    for i in range(window, len(lr_clean)):
        w = lr_clean.iloc[i - window: i].dropna(how="any")
        if len(w) < window // 2:
            continue
        C  = w.corr().values
        np.fill_diagonal(C, 0)
        sp_vals.append(float(np.abs(C).sum() / n))
        sp_dates.append(lr_clean.index[i])

    sp = pd.Series(sp_vals, index=sp_dates, name="spillover")
    log.info(f"  Mean spillover  : {sp.mean():.4f}")
    log.info(f"  Max spillover   : {sp.max():.4f}")
    log.info(f"  High (>p75) frac: {(sp > sp.quantile(0.75)).mean()*100:.1f}%")

    return {
        "series":          sp,
        "mean_spillover":  round(float(sp.mean()), 4),
        "max_spillover":   round(float(sp.max()),  4),
        "pct_high":        round(float((sp > sp.quantile(0.75)).mean()), 4),
    }


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_dcc_correlations(corr_series: dict,
                           dcc_df: pd.DataFrame,
                           top_n: int = 6,
                           save_path: str = None):
    """
    Time-series of rolling DCC correlation for the N pairs
    with highest correlation variance (most time-varying).
    """
    if "corr_std" in dcc_df.columns:
        top_keys = (dcc_df
                    .dropna(subset=["corr_std"])
                    .sort_values("corr_std", ascending=False)
                    .apply(lambda r: f"{r['asset_a']}__{r['asset_b']}", axis=1)
                    .head(top_n)
                    .tolist())
    else:
        top_keys = list(corr_series.keys())[:top_n]

    top_keys = [k for k in top_keys if k in corr_series][:top_n]
    if not top_keys:
        return

    fig, axes = plt.subplots(len(top_keys), 1,
                              figsize=(14, 2.5 * len(top_keys)),
                              sharex=True)
    if len(top_keys) == 1:
        axes = [axes]
    fig.suptitle(f"DCC Rolling Correlations (window={DCC_LOOKBACK}d)\n"
                 "Top pairs by variation",
                 fontsize=12, fontweight="bold")

    for ax, key in zip(axes, top_keys):
        cs = corr_series[key].dropna()
        if cs.empty:
            continue
        parts   = key.split("__")
        short_a = parts[0].replace("^","").replace("-USD","").replace(".NYB","")
        short_b = parts[1].replace("^","").replace("-USD","").replace(".NYB","") if len(parts) > 1 else ""

        ax.plot(cs.index, cs.values, linewidth=1.0,
                color="#2E5FAC", alpha=0.85)
        ax.fill_between(cs.index, cs.values, cs.mean(),
                        alpha=0.12, color="#2E5FAC")
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.axhline(cs.mean(), color="#C0392B", linewidth=1.0,
                   linestyle="--", label=f"mean={cs.mean():.3f}")
        ax.set_ylabel(f"{short_a}\nvs {short_b}", fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


def plot_static_corr_matrix(corr_mat: pd.DataFrame, save_path: str = None):
    """Static full-sample correlation heatmap."""
    n    = len(corr_mat)
    fig, ax = plt.subplots(figsize=(10, 8))
    im   = ax.imshow(corr_mat.values, cmap="RdBu_r",
                     vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, label="Pearson Correlation")

    short = [t.replace("^","").replace("-USD","").replace(".NYB","")
             for t in corr_mat.columns]
    ax.set_xticks(range(n)); ax.set_xticklabels(short, fontsize=8,
                                                  rotation=45, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=8)

    for i in range(n):
        for j in range(n):
            v = corr_mat.values[i, j]
            if not np.isnan(v) and abs(v) > 0.1:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=6,
                        color="white" if abs(v) > 0.6 else "black")

    ax.set_title("Static Full-Sample Correlation Matrix",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


def plot_spillover(sp: pd.Series, save_path: str = None):
    """Diebold-Yilmaz spillover index time series."""
    fig, ax = plt.subplots(figsize=(13, 4))
    ax.plot(sp.index, sp.values, linewidth=1.2, color="#2E5FAC")
    ax.fill_between(sp.index, sp.values, sp.min(),
                    alpha=0.2, color="#2E5FAC")
    ax.axhline(sp.mean(), color="#C0392B", linewidth=1.2,
               linestyle="--", label=f"Mean={sp.mean():.4f}")
    ax.axhline(sp.quantile(0.75), color="#E67E22", linewidth=0.8,
               linestyle="-.", label=f"p75={sp.quantile(0.75):.4f}")
    ax.set_title("Diebold-Yilmaz Spillover Index\n"
                 "Higher = stronger cross-asset information flow",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Spillover Index")
    ax.set_xlabel("Date")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ3 STAGE 2 -- DCC Correlations & Spillover Index")
    log.info("=" * 65)

    log_ret = load_returns()

    dcc_df, corr_series = compute_all_dcc(log_ret)
    dy      = diebold_yilmaz_spillover(log_ret)
    corr_mat = static_correlation_matrix(log_ret)

    # Save
    dcc_df.to_csv(os.path.join(OUTPUT_DIR, "rq3_dcc_results.csv"),
                  index=False)
    dy["series"].to_csv(
        os.path.join(OUTPUT_DIR, "rq3_spillover_series.csv"),
        header=["spillover"],
    )

    n_varying = int(dcc_df["varies"].sum()) if "varies" in dcc_df.columns else 0
    n_total   = len(dcc_df)

    lines = [
        "=" * 65,
        "  RQ3 STAGE 2 RESULTS  --  DCC & Spillover",
        "=" * 65,
        f"  Pairs analysed              : {n_total}",
        f"  Time-varying corr (t-test)  : {n_varying} / {n_total}",
        f"  Mean DY spillover index     : {dy['mean_spillover']:.4f}",
        f"  Max DY spillover            : {dy['max_spillover']:.4f}",
        f"  High-spillover fraction     : {dy['pct_high']*100:.1f}%",
        f"  Decision: correlations are "
        f"{'TIME-VARYING' if n_varying > n_total * 0.3 else 'approximately constant'}",
        "=" * 65,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "rq3_stage2_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # Plots
    plot_dcc_correlations(
        corr_series, dcc_df,
        save_path=os.path.join(OUTPUT_DIR, "rq3_dcc_correlations.png"),
    )
    plot_static_corr_matrix(
        corr_mat,
        save_path=os.path.join(OUTPUT_DIR, "rq3_corr_matrix_static.png"),
    )
    plot_spillover(
        dy["series"],
        save_path=os.path.join(OUTPUT_DIR, "rq3_spillover.png"),
    )

    log.info("\n✓ Stage 2 complete. Run rq3_stage3_ml.py next.\n")
    return dcc_df, dy


if __name__ == "__main__":
    main()

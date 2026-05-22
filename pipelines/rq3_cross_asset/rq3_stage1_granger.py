#!/usr/bin/env python3
# =============================================================================
# rq3_stage1_granger.py
# Stage 1: Granger Causality Network
#
# Tests all N*(N-1) directed asset pairs for Granger causality.
# For each pair (cause -> target) fits:
#
#   Restricted:    y_t = a0 + sum_k a_k*y_{t-k} + e_t
#   Unrestricted:  y_t = a0 + sum_k a_k*y_{t-k} + sum_k b_k*x_{t-k} + e_t
#
#   F = [(RSS_R - RSS_U)/p] / [RSS_U/(n-2p-1)]
#
# AIC selects optimal lag p for each pair.
# BH FDR correction applied across all tests.
#
# H0 partial reject: >= MIN_CAUSAL_PAIRS BH-significant pairs.
#
# Outputs
# -------
#   rq3_outputs/rq3_granger_results.csv
#   rq3_outputs/rq3_stage1_summary.txt
#   rq3_outputs/rq3_granger_heatmap.png
#   rq3_outputs/rq3_granger_network.png
#
# Run: python rq3_stage1_granger.py
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

from rq3_config import (
    DATA_DIR, OUTPUT_DIR,
    GRANGER_MAX_LAG, GRANGER_ALPHA, MIN_CAUSAL_PAIRS,
)
from scipy_compat import stats, multipletests


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
# HELPERS
# =============================================================================

def _chi2_sf(x, df):
    """Chi-squared survival function (Wilson-Hilferty approximation)."""
    x  = float(x)
    z  = ((x / df) ** (1.0 / 3) - (1.0 - 2.0 / (9 * df))) \
         / np.sqrt(2.0 / (9 * df))
    try:
        return float(stats.norm.sf(z))
    except Exception:
        return float(np.exp(-x / 2))


def _f_pval(F, df1, df2):
    """Approximate p-value for F(df1, df2) via chi-squared tail."""
    return _chi2_sf(max(F, 0.0) * df1, df1)


def _build_lag_matrix(y, x, p):
    """
    Build lagged design matrices for VAR Granger test.
    y: target series (1-D numpy array)
    x: cause series or None
    p: number of lags
    Returns (Y, X_matrix) both starting at index p.
    """
    T = len(y)
    Y = y[p:]
    own_lags = np.column_stack([y[p - k: T - k] for k in range(1, p + 1)])
    if x is None:
        return Y, own_lags
    cross_lags = np.column_stack([x[p - k: T - k] for k in range(1, p + 1)])
    return Y, np.hstack([own_lags, cross_lags])


def _add_const(X):
    """Prepend a column of ones."""
    return np.hstack([np.ones((len(X), 1)), X])


# =============================================================================
# GRANGER F-TEST
# =============================================================================

def granger_f_test(y: np.ndarray, x: np.ndarray,
                    max_lag: int = GRANGER_MAX_LAG) -> dict:
    """
    Bivariate Granger causality F-test: does x Granger-cause y?

    1. Select optimal lag p by AIC on the AR-only (restricted) model.
    2. Compute F-statistic comparing restricted vs unrestricted models.
    3. Return F, p-value, optimal lag, and rejection flag.
    """
    n_raw = min(len(y), len(x))
    y     = np.array(y[-n_raw:], float)
    x     = np.array(x[-n_raw:], float)

    # Remove any NaN rows pairwise
    mask = ~(np.isnan(y) | np.isnan(x))
    y, x = y[mask], x[mask]
    n    = len(y)

    NaN_result = {"F": np.nan, "p_val": 1.0, "lag": 1,
                  "reject": False, "aic_lag": 1}

    if n < max_lag * 4 + 20:
        return NaN_result

    # ── AIC lag selection on AR(p) restricted model ───────────────
    best_aic, best_p = np.inf, 1
    for p in range(1, max_lag + 1):
        Y_p, Xr_p = _build_lag_matrix(y, None, p)
        if len(Y_p) < p + 10:
            continue
        Xrc = _add_const(Xr_p)
        try:
            b, res, _, _ = np.linalg.lstsq(Xrc, Y_p, rcond=None)
            e    = Y_p - Xrc @ b
            rss  = float(e @ e)
            aic  = len(Y_p) * np.log(max(rss / len(Y_p), 1e-15)) + 2 * (p + 1)
            if aic < best_aic:
                best_aic, best_p = aic, p
        except Exception:
            continue

    p = best_p

    # ── Fit restricted and unrestricted models ─────────────────────
    Y, Xr  = _build_lag_matrix(y, None, p)
    _, Xur = _build_lag_matrix(y, x,    p)

    if len(Y) < 2 * p + 10:
        return NaN_result

    Xrc  = _add_const(Xr)
    Xurc = _add_const(Xur)

    try:
        br,  _,  _, _ = np.linalg.lstsq(Xrc,  Y, rcond=None)
        bur, _,  _, _ = np.linalg.lstsq(Xurc, Y, rcond=None)

        e_r  = Y - Xrc  @ br
        e_ur = Y - Xurc @ bur

        rss_r  = float(e_r  @ e_r)
        rss_ur = float(e_ur @ e_ur)

        n_obs = len(Y)
        df1   = p                       # number of added x-lags
        df2   = n_obs - 2 * p - 1       # residual df of unrestricted

        if df2 <= 0 or rss_ur <= 1e-15:
            return NaN_result

        F     = max(((rss_r - rss_ur) / df1) / (rss_ur / df2), 0.0)
        p_val = _f_pval(F, df1, df2)

        return {
            "F":       round(float(F),     4),
            "p_val":   round(float(p_val), 6),
            "lag":     p,
            "aic_lag": best_p,
            "reject":  p_val < GRANGER_ALPHA,
        }
    except Exception:
        return NaN_result


# =============================================================================
# RUN ALL PAIRS
# =============================================================================

def run_all_granger(log_ret: pd.DataFrame) -> pd.DataFrame:
    """
    Iterate over all N*(N-1) directed pairs.
    Apply BH FDR correction across all p-values.
    """
    tickers  = [t for t in log_ret.columns
                if log_ret[t].notna().sum() > 200]
    n_pairs  = len(tickers) * (len(tickers) - 1)

    log.info(f"\nRunning {n_pairs} Granger tests "
             f"({len(tickers)} assets, max_lag={GRANGER_MAX_LAG}) ...")

    rows = []
    done = 0
    for target in tickers:
        y = log_ret[target].values
        for cause in tickers:
            if cause == target:
                continue
            x = log_ret[cause].reindex(log_ret[target].index).values
            r = granger_f_test(y, x, GRANGER_MAX_LAG)
            rows.append({
                "cause":  cause,
                "target": target,
                "F":      r["F"],
                "p_val":  r["p_val"],
                "lag":    r["lag"],
                "reject_raw": r["reject"],
            })
            done += 1
            if done % 20 == 0:
                log.info(f"  ... {done}/{n_pairs} done")

    df = pd.DataFrame(rows)

    # BH FDR correction
    valid = df["p_val"].notna()
    if valid.sum() > 0:
        rej_bh, p_bh, _, _ = multipletests(
            df.loc[valid, "p_val"], alpha=GRANGER_ALPHA, method="fdr_bh"
        )
        df.loc[valid, "reject_BH"] = rej_bh
        df.loc[valid, "p_BH"]      = np.round(p_bh, 6)
    else:
        df["reject_BH"] = False
        df["p_BH"]      = 1.0

    df["reject_BH"] = df["reject_BH"].fillna(False)

    n_raw = int(df["reject_raw"].sum())
    n_bh  = int(df["reject_BH"].sum())
    log.info(f"\n  Significant (raw alpha={GRANGER_ALPHA}): {n_raw}")
    log.info(f"  Significant (BH corrected)            : {n_bh}")
    log.info(f"  H0 partial: "
             f"{'REJECT (>=' + str(MIN_CAUSAL_PAIRS) + ' pairs)' if n_bh >= MIN_CAUSAL_PAIRS else 'fail to reject'}")

    return df


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_granger_heatmap(gc_df: pd.DataFrame, save_path=None):
    """
    N x N heatmap of -log10(p-value) for directed Granger tests.
    Rows = target, Cols = cause.
    Stars mark BH-significant pairs.
    """
    tickers = sorted(set(gc_df["target"].tolist() + gc_df["cause"].tolist()))
    n       = len(tickers)
    idx     = {t: i for i, t in enumerate(tickers)}

    mat_p   = np.ones((n, n))
    mat_sig = np.zeros((n, n), dtype=bool)

    for _, row in gc_df.iterrows():
        i = idx.get(row["target"])
        j = idx.get(row["cause"])
        if i is not None and j is not None and not np.isnan(row["p_val"]):
            mat_p[i, j]   = max(float(row["p_val"]), 1e-10)
            mat_sig[i, j] = bool(row.get("reject_BH", False))

    log_p = -np.log10(mat_p)
    np.fill_diagonal(log_p, 0)

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(log_p, cmap="YlOrRd", aspect="auto", vmin=0, vmax=4)
    plt.colorbar(im, ax=ax, label="-log10(p-value)")

    short = [t.replace("^", "").replace("-USD", "")
              .replace(".NYB", "") for t in tickers]

    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(plt.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=True, color="#CCCCCC", zorder=2,
                ))
            elif mat_sig[i, j]:
                ax.text(j, i, "*", ha="center", va="center",
                        fontsize=14, color="black",
                        fontweight="bold", zorder=3)

    ax.set_xticks(range(n))
    ax.set_xticklabels(short, fontsize=8, rotation=45, ha="right")
    ax.set_yticks(range(n))
    ax.set_yticklabels(short, fontsize=8)
    ax.set_xlabel("Cause", fontsize=10)
    ax.set_ylabel("Target", fontsize=10)
    ax.set_title(
        "Granger Causality Heatmap  (-log10 p-value)\n"
        "* = BH-significant  |  Row = target, Col = cause",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


def plot_granger_network(gc_df: pd.DataFrame, save_path=None):
    """
    Directed network graph of BH-significant Granger pairs.
    Arrow thickness is proportional to F-statistic.
    """
    col = "reject_BH" if "reject_BH" in gc_df.columns else "reject_raw"
    sig = gc_df[gc_df[col] == True].copy()

    if sig.empty:
        log.warning("  No significant pairs for network plot.")
        return

    tickers = sorted(set(sig["cause"].tolist() + sig["target"].tolist()))
    n       = len(tickers)
    angles  = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pos     = {t: (np.cos(a), np.sin(a)) for t, a in zip(tickers, angles)}

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect("equal")

    F_max = sig["F"].max() if sig["F"].notna().any() else 1.0

    for _, row in sig.iterrows():
        if row["cause"] not in pos or row["target"] not in pos:
            continue
        x0, y0 = pos[row["cause"]]
        x1, y1 = pos[row["target"]]
        f_norm  = float(row["F"]) / F_max if not np.isnan(row["F"]) else 0.3
        lw      = 0.5 + 3.0 * f_norm
        ax.annotate(
            "",
            xy=(x1, y1), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color="#2E5FAC",
                lw=lw, connectionstyle="arc3,rad=0.1",
            ),
        )

    for ticker, (x, y) in pos.items():
        short = (ticker.replace("^", "")
                       .replace("-USD", "")
                       .replace(".NYB", ""))
        ax.scatter(x, y, s=500, color="#1B3A6B", zorder=5)
        ax.text(x * 1.18, y * 1.18, short,
                ha="center", va="center",
                fontsize=9, fontweight="bold", color="#1B3A6B")

    ax.set_xlim(-1.6, 1.6)
    ax.set_ylim(-1.6, 1.6)
    ax.axis("off")
    ax.set_title(
        "Granger Causality Network (BH-significant pairs)\n"
        "Arrow width proportional to F-statistic",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


def plot_top_pairs_timeseries(gc_df: pd.DataFrame,
                               log_ret: pd.DataFrame,
                               save_path=None):
    """
    Rolling 63-day cross-correlation for top-5 causal pairs.
    Shows the dynamic nature of the lead-lag relationships.
    """
    col  = "reject_BH" if "reject_BH" in gc_df.columns else "reject_raw"
    top5 = (gc_df[gc_df[col] == True]
            .dropna(subset=["F"])
            .sort_values("F", ascending=False)
            .head(5))

    if top5.empty:
        return

    fig, axes = plt.subplots(len(top5), 1,
                              figsize=(13, 2.5 * len(top5)),
                              sharex=True)
    if len(top5) == 1:
        axes = [axes]
    fig.suptitle("Top Granger Pairs -- Rolling 63-day Cross-Correlation",
                 fontsize=12, fontweight="bold")

    for ax, (_, row) in zip(axes, top5.iterrows()):
        cause  = row["cause"]
        target = row["target"]
        if cause not in log_ret.columns or target not in log_ret.columns:
            continue
        roll_corr = (log_ret[cause]
                     .rolling(63)
                     .corr(log_ret[target])
                     .dropna())
        ax.plot(roll_corr.index, roll_corr.values,
                color="#2E5FAC", linewidth=1.0, alpha=0.8)
        ax.axhline(0, color="black", linewidth=0.5, linestyle=":")
        ax.axhline(roll_corr.mean(), color="#C0392B",
                   linewidth=1.0, linestyle="--",
                   label=f"mean={roll_corr.mean():.3f}")
        short_c = cause.replace("^","").replace("-USD","").replace(".NYB","")
        short_t = target.replace("^","").replace("-USD","").replace(".NYB","")
        ax.set_ylabel(f"{short_c}->{short_t}\nF={row['F']:.1f}",
                      fontsize=8)
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Date", fontsize=9)
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
    log.info("  RQ3 STAGE 1 -- Granger Causality Network")
    log.info("=" * 65)

    log_ret = load_returns()
    gc_df   = run_all_granger(log_ret)

    # Save
    out_csv = os.path.join(OUTPUT_DIR, "rq3_granger_results.csv")
    gc_df.to_csv(out_csv, index=False)
    log.info(f"\nResults saved -> {out_csv}")

    # Summary
    n_bh = int(gc_df.get("reject_BH", gc_df["reject_raw"]).sum())
    lines = [
        "=" * 65,
        "  RQ3 STAGE 1 RESULTS  --  Granger Causality",
        "=" * 65,
        f"  Total directed pairs   : {len(gc_df)}",
        f"  Significant (raw)      : {int(gc_df['reject_raw'].sum())}",
        f"  Significant (BH)       : {n_bh}",
        f"  Required for H1        : >= {MIN_CAUSAL_PAIRS}",
        f"  H0 partial decision    : "
        f"{'REJECT H0' if n_bh >= MIN_CAUSAL_PAIRS else 'fail to reject'}",
        "=" * 65,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "rq3_stage1_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # Top pairs
    top = (gc_df[gc_df["reject_raw"] == True]
           .dropna(subset=["F"])
           .sort_values("F", ascending=False)
           .head(10))
    if not top.empty:
        log.info(f"\nTop Granger-causal pairs:")
        log.info(f"\n{top[['cause','target','F','p_val','lag']].to_string(index=False)}")

    # Plots
    plot_granger_heatmap(
        gc_df,
        save_path=os.path.join(OUTPUT_DIR, "rq3_granger_heatmap.png"),
    )
    plot_granger_network(
        gc_df,
        save_path=os.path.join(OUTPUT_DIR, "rq3_granger_network.png"),
    )
    plot_top_pairs_timeseries(
        gc_df, log_ret,
        save_path=os.path.join(OUTPUT_DIR,
                               "rq3_top_pairs_correlation.png"),
    )

    log.info("\n✓ Stage 1 complete. Run rq3_stage2_dcc.py next.\n")
    return gc_df


if __name__ == "__main__":
    main()

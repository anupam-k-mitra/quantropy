#!/usr/bin/env python3
# =============================================================================
# rq2_stage2_anova.py
# Stage 2: Statistical tests — are detected regimes genuinely different?
#
# Tests
# -----
# 1. One-way ANOVA          — parametric mean comparison (+ eta-squared)
# 2. Kruskal-Wallis         — non-parametric (no normality assumption)
# 3. Levene's test          — variance homogeneity (sigma part of H0)
# 4. Welch pairwise t-tests — Bonferroni-corrected regime pairs
# 5. Transition matrix      — regime persistence / duration check
#
# Outputs
# -------
#   rq2_outputs/rq2_stage2_summary.txt
#   rq2_outputs/rq2_stage2_summary.csv
#   rq2_outputs/rq2_pairwise_tests.csv
#   rq2_outputs/rq2_transition_matrix.csv
#   rq2_outputs/rq2_return_distributions.png
#   rq2_outputs/rq2_transition_heatmap.png
#
# Run: python rq2_stage2_anova.py
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

from regime_config import DATA_DIR, OUTPUT_DIR, ALPHA
from scipy_compat import stats


# =============================================================================
# LOADER
# =============================================================================

def load_data():
    labels_path = os.path.join(DATA_DIR, "rq2_regime_labels.parquet")
    ret_path    = os.path.join(DATA_DIR, "rq2_log_returns.parquet")
    for p in [labels_path, ret_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run rq2_stage1_detection.py first."
            )
    return (pd.read_parquet(labels_path),
            pd.read_parquet(ret_path))


def _get_groups(log_ret, labels, ticker, min_obs=5):
    """Align returns and labels; return list of per-regime arrays."""
    if ticker not in log_ret.columns:
        ticker = log_ret.columns[0]
    r = log_ret[ticker].dropna()
    # Align on common index — labels covers feature subset, r covers full dates
    common = r.index.intersection(labels.index)
    r      = r.loc[common]
    lbl    = labels.loc[common]
    df     = pd.DataFrame({"ret": r, "regime": lbl}).dropna()
    return df, [
        df.loc[df["regime"] == reg, "ret"].values
        for reg in sorted(df["regime"].unique())
        if (df["regime"] == reg).sum() >= min_obs
    ]


# =============================================================================
# HELPER: chi-squared CDF (for F / H p-values without scipy)
# =============================================================================

def _chi2_sf(x, df):
    """Survival function of chi-squared via Wilson-Hilferty."""
    z = ((x / df) ** (1.0 / 3) - (1.0 - 2.0 / (9 * df))) \
        / np.sqrt(2.0 / (9 * df))
    try:
        return float(stats.norm.sf(z))
    except Exception:
        return float(np.exp(-x / 2))   # rough fallback


def _f_pval(F, df1, df2):
    """Approximate p-value for F(df1, df2) statistic."""
    # Use chi2 approximation: F*df1 ~ chi2(df1) for large df2
    return _chi2_sf(F * df1, df1)


# =============================================================================
# 1.  ONE-WAY ANOVA
# =============================================================================

def anova_test(log_ret, labels, ticker="SPY"):
    """
    One-way ANOVA: H0: mu_1 = mu_2 = mu_3 = mu_4.

    Manual computation for full transparency.
    Effect size: eta-squared (eta^2 > 0.06 = medium, > 0.14 = large).
    """
    df, groups = _get_groups(log_ret, labels, ticker)
    if len(groups) < 2:
        log.warning("  ANOVA: fewer than 2 groups. Skipping.")
        return {}

    grand_mean = df["ret"].mean()
    k          = len(groups)
    n_total    = sum(len(g) for g in groups)

    ss_btw = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_wth = sum(((g - g.mean()) ** 2).sum() for g in groups)
    ss_tot = ss_btw + ss_wth

    df1 = k - 1
    df2 = n_total - k
    ms_btw = ss_btw / df1
    ms_wth = ss_wth / max(df2, 1)
    F      = ms_btw / ms_wth if ms_wth > 0 else 0.0
    p      = _f_pval(F, df1, df2)
    eta2   = ss_btw / ss_tot if ss_tot > 0 else 0.0

    effect = ("large"  if eta2 > 0.14 else
              "medium" if eta2 > 0.06 else "small")

    log.info(f"\n── One-way ANOVA ({ticker}) ─────────────────────────────────")
    log.info(f"  F({df1}, {df2}) = {F:.4f}   p = {p:.6f}   "
             f"eta^2 = {eta2:.4f} ({effect})")
    log.info(f"  H0: {'REJECT' if p < ALPHA else 'fail to reject'}")

    # Per-regime means for reporting
    for i, (reg, grp) in enumerate(
            zip(sorted(df["regime"].unique()), groups)):
        log.info(f"    R{reg}: mean={grp.mean()*252*100:.2f}% ann.  "
                 f"n={len(grp)}")

    return {
        "F_stat": round(F, 4), "p_val": round(p, 6),
        "eta_sq": round(eta2, 4), "df_between": df1, "df_within": df2,
        "reject_H0": p < ALPHA,
    }


# =============================================================================
# 2.  KRUSKAL-WALLIS
# =============================================================================

def kruskal_wallis_test(log_ret, labels, ticker="SPY"):
    """
    Kruskal-Wallis H-test: non-parametric ANOVA.
    Does not assume normality — essential for fat-tailed return series.
    H ~ chi2(k-1) under H0.
    """
    df, groups = _get_groups(log_ret, labels, ticker)
    if len(groups) < 2:
        return {}

    all_vals = np.concatenate(groups)
    N        = len(all_vals)
    ranks    = np.argsort(np.argsort(all_vals)).astype(float) + 1.0

    H   = 0.0
    pos = 0
    for g in groups:
        ng  = len(g)
        rg  = ranks[pos: pos + ng]
        H  += ng * (rg.mean() - (N + 1) / 2) ** 2
        pos += ng
    H = 12.0 / (N * (N + 1)) * H

    df_kw = len(groups) - 1
    p     = _chi2_sf(H, df_kw)

    log.info(f"\n── Kruskal-Wallis ({ticker}) ──────────────────────────────────")
    log.info(f"  H = {H:.4f}   df = {df_kw}   p = {p:.6f}")
    log.info(f"  H0: {'REJECT' if p < ALPHA else 'fail to reject'}")

    return {
        "H_stat": round(H, 4), "p_val": round(p, 6),
        "df": df_kw, "reject_H0": p < ALPHA,
    }


# =============================================================================
# 3.  LEVENE'S TEST (variance homogeneity)
# =============================================================================

def levene_test(log_ret, labels, ticker="SPY"):
    """
    Levene (median-centred): tests H0: sigma_1 = sigma_2 = ... = sigma_k.
    A significant result means volatility differs across regimes —
    this alone justifies regime-adaptive risk management.
    """
    df, groups = _get_groups(log_ret, labels, ticker)
    if len(groups) < 2:
        return {}

    k       = len(groups)
    n_total = sum(len(g) for g in groups)

    z_groups    = [np.abs(g - np.median(g)) for g in groups]
    grand_z     = np.concatenate(z_groups).mean()
    ss_btw = sum(len(z) * (z.mean() - grand_z) ** 2 for z in z_groups)
    ss_wth = sum(((z - z.mean()) ** 2).sum() for z in z_groups)

    df1 = k - 1
    df2 = n_total - k
    ms_btw = ss_btw / df1
    ms_wth = ss_wth / max(df2, 1)
    W   = ms_btw / ms_wth if ms_wth > 0 else 0.0
    p   = _f_pval(W, df1, df2)

    log.info(f"\n── Levene Test ({ticker}) ─────────────────────────────────────")
    log.info(f"  W = {W:.4f}   p = {p:.6f}")
    log.info(f"  H0 (equal variances): {'REJECT' if p < ALPHA else 'fail to reject'}")

    return {
        "W_stat": round(W, 4), "p_val": round(p, 6),
        "reject_H0": p < ALPHA,
    }


# =============================================================================
# 4.  PAIRWISE WELCH T-TESTS
# =============================================================================

def pairwise_tests(log_ret, labels, ticker="SPY"):
    """
    Welch's t-test for each pair of regimes.
    Bonferroni-corrected threshold: alpha / C(k,2).
    """
    df, groups = _get_groups(log_ret, labels, ticker)
    regs       = sorted(df["regime"].unique())
    k          = len(regs)
    bonf       = ALPHA / max((k * (k - 1) // 2), 1)

    rows = []
    for i in range(k):
        for j in range(i + 1, k):
            g1 = df.loc[df["regime"] == regs[i], "ret"].values
            g2 = df.loc[df["regime"] == regs[j], "ret"].values

            n1, n2 = len(g1), len(g2)
            m1, m2 = g1.mean(), g2.mean()
            v1, v2 = g1.var(ddof=1), g2.var(ddof=1)
            se     = np.sqrt(v1 / n1 + v2 / n2)
            t      = (m1 - m2) / se if se > 0 else 0.0

            # Welch-Satterthwaite df
            num = (v1 / n1 + v2 / n2) ** 2
            den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
            dof = num / den if den > 0 else min(n1, n2)

            # p-value via normal approximation (conservative for large n)
            try:
                p = 2.0 * float(stats.norm.sf(abs(t)))
            except Exception:
                p = 2.0 * np.exp(-t ** 2 / 2)

            rows.append({
                "pair":        f"R{regs[i]} vs R{regs[j]}",
                "mean_diff_ann": round((m1 - m2) * 252, 4),
                "t_stat":      round(t, 4),
                "p_val":       round(p, 6),
                "reject_bonf": p < bonf,
                "n1": n1, "n2": n2,
            })

    result = pd.DataFrame(rows)
    log.info(f"\n── Pairwise Tests (Bonferroni alpha={bonf:.4f}) ─────────────")
    log.info(f"\n{result[['pair','mean_diff_ann','t_stat','p_val','reject_bonf']].to_string(index=False)}")
    return result


# =============================================================================
# 5.  TRANSITION MATRIX
# =============================================================================

def transition_matrix(labels):
    """
    Empirical Markov transition matrix P(regime t+1 | regime t).
    High diagonal = persistent regimes = tradeable signals.
    """
    regs   = sorted(labels.unique())
    counts = pd.DataFrame(0, index=regs, columns=regs)
    for t in range(len(labels) - 1):
        counts.loc[labels.iloc[t], labels.iloc[t + 1]] += 1
    trans = counts.div(counts.sum(axis=1).replace(0, 1), axis=0)

    avg_dur = {}
    for r in regs:
        p_stay      = trans.loc[r, r]
        avg_dur[r]  = round(1.0 / (1.0 - p_stay), 1) if p_stay < 1.0 else float("inf")

    log.info(f"\n── Transition Matrix ─────────────────────────────────────────")
    log.info(f"\n{trans.round(3).to_string()}")
    log.info(f"\nAvg regime duration (days): {avg_dur}")
    return trans, avg_dur


# =============================================================================
# VISUALISATION
# =============================================================================

COLOURS = ["#2E5FAC", "#C0392B", "#27AE60", "#E67E22"]
NAMES   = ["Bull", "Bear", "Sideways", "Crisis"]


def plot_distributions(log_ret, labels, ticker="SPY", save_path=None):
    """Per-regime return distribution with normal overlay and stats."""
    if ticker not in log_ret.columns:
        ticker = log_ret.columns[0]
    r      = log_ret[ticker].dropna()
    common = r.index.intersection(labels.index)
    r      = r.loc[common]
    labels = labels.loc[common]
    df     = pd.DataFrame({"ret": r, "regime": labels}).dropna()
    regs = sorted(df["regime"].unique())

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    fig.suptitle(f"Return Distributions by Regime ({ticker})",
                 fontsize=14, fontweight="bold")

    for i, reg in enumerate(regs[:4]):
        ax   = axes[i]
        data = df.loc[df["regime"] == reg, "ret"].values * 252
        n, mu, sigma = len(data), data.mean(), data.std()

        ax.hist(data, bins=50, density=True, alpha=0.6,
                color=COLOURS[i % 4], edgecolor="white",
                label=f"n={n}")

        x = np.linspace(data.min(), data.max(), 200)
        if sigma > 0:
            ax.plot(x,
                    np.exp(-0.5 * ((x - mu) / sigma) ** 2)
                    / (sigma * np.sqrt(2 * np.pi)),
                    color="black", linewidth=1.5, linestyle="--",
                    label="Normal fit")

        ax.axvline(0,  color="grey",          linewidth=0.8, linestyle=":")
        ax.axvline(mu, color=COLOURS[i % 4], linewidth=1.5, linestyle="-",
                   label=f"mu={mu*100:.1f}%")

        name = NAMES[i] if i < len(NAMES) else f"R{reg}"
        ax.set_title(f"Regime {reg}: {name}", fontsize=11,
                     fontweight="bold", color=COLOURS[i % 4])
        ax.set_xlabel("Annualised Return")
        ax.set_ylabel("Density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        skew = pd.Series(data).skew()
        kurt = pd.Series(data).kurt()
        ax.text(0.98, 0.97,
                f"sigma={sigma*100:.1f}%\nSkew={skew:.2f}\nKurt={kurt:.2f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="white", alpha=0.7))

    for i in range(len(regs), 4):
        axes[i].set_visible(False)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved → {save_path}")
    plt.close()


def plot_transition_heatmap(trans, save_path=None):
    """Heatmap of Markov transition probabilities."""
    k   = len(trans)
    fig, ax = plt.subplots(figsize=(7, 5))
    im  = ax.imshow(trans.values, cmap="Blues", vmin=0, vmax=1,
                    aspect="auto")
    plt.colorbar(im, ax=ax, label="Transition Probability")

    for i in range(k):
        for j in range(k):
            v = trans.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=11, fontweight="bold",
                    color="white" if v > 0.5 else "black")

    tick_labels = [f"R{r}\n{NAMES[r] if r < len(NAMES) else ''}"
                   for r in range(k)]
    ax.set_xticks(range(k)); ax.set_xticklabels(tick_labels, fontsize=9)
    ax.set_yticks(range(k)); ax.set_yticklabels(tick_labels, fontsize=9)
    ax.set_title("Regime Transition Matrix (diagonal = persistence)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved → {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ2 STAGE 2 — Statistical Tests (ANOVA / KW / Levene)")
    log.info("=" * 65)

    labels_df, log_ret = load_data()
    consensus          = labels_df["consensus"]
    ticker = "SPY" if "SPY" in log_ret.columns else log_ret.columns[0]

    # ── Tests ─────────────────────────────────────────────────────
    anova_res = anova_test(log_ret, consensus, ticker)
    kw_res    = kruskal_wallis_test(log_ret, consensus, ticker)
    lev_res   = levene_test(log_ret, consensus, ticker)
    pair_df   = pairwise_tests(log_ret, consensus, ticker)
    trans, avg_dur = transition_matrix(consensus)

    # ── Save ──────────────────────────────────────────────────────
    pair_df.to_csv(
        os.path.join(OUTPUT_DIR, "rq2_pairwise_tests.csv"), index=False
    )
    trans.to_csv(
        os.path.join(OUTPUT_DIR, "rq2_transition_matrix.csv")
    )

    summary = {
        "anova_F":       anova_res.get("F_stat", np.nan),
        "anova_p":       anova_res.get("p_val",  np.nan),
        "anova_eta2":    anova_res.get("eta_sq", np.nan),
        "anova_reject":  anova_res.get("reject_H0", False),
        "kw_H":          kw_res.get("H_stat",   np.nan),
        "kw_p":          kw_res.get("p_val",     np.nan),
        "kw_reject":     kw_res.get("reject_H0", False),
        "levene_W":      lev_res.get("W_stat",   np.nan),
        "levene_p":      lev_res.get("p_val",     np.nan),
        "levene_reject": lev_res.get("reject_H0", False),
        "pairs_reject":  int(pair_df["reject_bonf"].sum())
                         if not pair_df.empty else 0,
    }
    pd.Series(summary).to_csv(
        os.path.join(OUTPUT_DIR, "rq2_stage2_summary.csv"),
        header=False,
    )

    # Text report
    n_pairs = summary["pairs_reject"]
    total_p = len(pair_df) if not pair_df.empty else 0
    lines   = [
        "=" * 65,
        "  RQ2 STAGE 2 RESULTS  —  Statistical Tests",
        "=" * 65,
        "",
        f"  ANOVA:           F={summary['anova_F']:.4f}  "
        f"p={summary['anova_p']:.6f}  "
        f"eta2={summary['anova_eta2']:.4f}  "
        f"{'REJECT H0' if summary['anova_reject'] else 'fail'}",
        f"  Kruskal-Wallis:  H={summary['kw_H']:.4f}  "
        f"p={summary['kw_p']:.6f}  "
        f"{'REJECT H0' if summary['kw_reject'] else 'fail'}",
        f"  Levene (var):    W={summary['levene_W']:.4f}  "
        f"p={summary['levene_p']:.6f}  "
        f"{'REJECT H0' if summary['levene_reject'] else 'fail'}",
        f"  Pairwise (Bonf): {n_pairs}/{total_p} pairs reject H0",
        "=" * 65,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "rq2_stage2_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # ── Plots ─────────────────────────────────────────────────────
    plot_distributions(
        log_ret, consensus, ticker,
        save_path=os.path.join(OUTPUT_DIR,
                               "rq2_return_distributions.png"),
    )
    plot_transition_heatmap(
        trans,
        save_path=os.path.join(OUTPUT_DIR,
                               "rq2_transition_heatmap.png"),
    )

    log.info("\n✓ Stage 2 complete. Run rq2_stage3_adaptive.py next.\n")
    return summary


if __name__ == "__main__":
    main()

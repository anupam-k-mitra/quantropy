#!/usr/bin/env python3
# =============================================================================
# rq2_india_stage2_anova.py — Statistical Tests Across Regimes
#
# Tests: ANOVA/KW on returns, vol, INR, gold, crude across 3 regimes.
# Also: regime-specific structural break analysis (Chow test at India dates).
# H0: performance is identical across regimes.
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

from rq2_india_config import (
    DATA_DIR, OUTPUT_DIR, ALPHA, INDIA_BREAKS,
    REGIME_NAMES_3, RISK_FREE_ANNUAL,
)
try:
    from scipy_compat import stats
except ImportError:
    from scipy import stats


def load_data():
    labels  = pd.read_parquet(
        os.path.join(DATA_DIR, "rq2_india_regime_labels.parquet"))
    feat    = pd.read_parquet(
        os.path.join(DATA_DIR, "rq2_india_features.parquet"))
    log_ret = pd.read_parquet(
        os.path.join(DATA_DIR, "rq2_india_log_returns.parquet"))
    return labels, feat, log_ret


def regime_performance(feat, labels, log_ret, label_col="hmm_3state"):
    """Compute per-regime statistics for all key variables."""
    lbl   = labels[label_col].dropna().astype(int)
    common= feat.index.intersection(lbl.index)
    feat_c= feat.loc[common]
    lbl_c = lbl.loc[common]

    rf_daily = RISK_FREE_ANNUAL / 252
    metrics  = ["nsei_ret_1d", "vol_primary", "usdinr_5d",
                "gold_21d", "crude_21d", "spy_21d",
                "nsei_dd_252", "turbulence"]
    results  = []

    for s in sorted(lbl_c.unique()):
        mask  = lbl_c == s
        sub   = feat_c[mask]
        r_sub = sub["nsei_ret_1d"]
        row   = {
            "regime":      s,
            "label":       REGIME_NAMES_3.get(s, str(s)),
            "n_days":      int(mask.sum()),
            "pct_days":    round(mask.mean() * 100, 1),
            "ann_ret":     round(float(r_sub.mean()) * 252, 4),
            "ann_vol":     round(float(r_sub.std()) * np.sqrt(252), 4),
            "sharpe":      round(
                (float(r_sub.mean()) - rf_daily)
                / max(float(r_sub.std()), 1e-9) * np.sqrt(252), 3),
            "max_dd":      round(float(sub["nsei_dd_252"].min()), 4)
                           if "nsei_dd_252" in sub else np.nan,
            "hit_rate":    round(float((r_sub > 0).mean()), 3),
        }
        for m in metrics:
            if m in sub.columns:
                row[f"mean_{m}"] = round(float(sub[m].mean()), 6)
        results.append(row)

    return pd.DataFrame(results)


def anova_kruskal(feat, labels, label_col="hmm_3state"):
    """
    ANOVA (parametric) and Kruskal-Wallis (non-parametric) tests.
    H0: variable distribution is identical across regimes.
    """
    lbl   = labels[label_col].dropna().astype(int)
    common= feat.index.intersection(lbl.index)
    lbl_c = lbl.loc[common]

    test_vars = ["nsei_ret_1d", "vol_primary", "usdinr_5d",
                 "gold_21d", "crude_21d", "spy_21d",
                 "nsei_vol_21d", "nsei_dd_252"]
    rows = []

    for var in test_vars:
        if var not in feat.columns:
            continue
        groups = [feat.loc[common][lbl_c == s][var].dropna().values
                  for s in sorted(lbl_c.unique())]
        groups = [g for g in groups if len(g) >= 5]
        if len(groups) < 2:
            continue

        # ANOVA
        try:
            F, p_anova = stats.f_oneway(*groups)
        except Exception:
            F, p_anova = np.nan, 1.0

        # Kruskal-Wallis
        try:
            H, p_kw = stats.kruskal(*groups)
        except Exception:
            H, p_kw = np.nan, 1.0

        rows.append({
            "variable":  var,
            "F_stat":    round(F, 4)  if not np.isnan(F) else np.nan,
            "p_anova":   round(p_anova, 6),
            "H_stat":    round(H, 4)  if not np.isnan(H) else np.nan,
            "p_kruskal": round(p_kw, 6),
            "reject_H0": (p_anova < ALPHA) or (p_kw < ALPHA),
        })

    return pd.DataFrame(rows)


def chow_test_india_breaks(feat):
    """
    Chow test at each India structural break date.
    Tests whether mean Nifty return differs significantly before/after.
    """
    if "nsei_ret_1d" not in feat.columns:
        return pd.DataFrame()

    rows = []
    r    = feat["nsei_ret_1d"].dropna()
    for dt_str, desc in INDIA_BREAKS.items():
        dt = pd.Timestamp(dt_str)
        if dt not in r.index and dt > r.index.max():
            continue
        # Find nearest date in index
        idx_pos = r.index.searchsorted(dt)
        if idx_pos < 30 or idx_pos > len(r) - 30:
            continue
        before = r.iloc[max(0, idx_pos - 126): idx_pos].values
        after  = r.iloc[idx_pos: min(len(r), idx_pos + 126)].values
        if len(before) < 20 or len(after) < 20:
            continue
        t, p = stats.ttest_ind(before, after)
        rows.append({
            "date":       dt_str,
            "event":      desc[:40],
            "ret_before": round(float(before.mean()) * 252, 4),
            "ret_after":  round(float(after.mean())  * 252, 4),
            "vol_before": round(float(before.std())  * np.sqrt(252), 4),
            "vol_after":  round(float(after.std())   * np.sqrt(252), 4),
            "t_stat":     round(t, 4),
            "p_val":      round(p, 6),
            "significant":p < ALPHA,
        })
    return pd.DataFrame(rows)


def plot_regime_distributions(feat, labels, save_path, label_col="hmm_3state"):
    """Box plots of key variables across regimes."""
    lbl    = labels[label_col].dropna().astype(int)
    common = feat.index.intersection(lbl.index)
    feat_c = feat.loc[common]
    lbl_c  = lbl.loc[common]

    plot_vars = [("nsei_ret_1d",  "Nifty Daily Return"),
                 ("vol_primary",  "Volatility (India VIX or RVol)"),
                 ("usdinr_5d",    "USD/INR 5d Return"),
                 ("gold_21d",     "Gold 21d Return"),
                 ("crude_21d",    "Crude 21d Return")]
    plot_vars = [(v, l) for v, l in plot_vars if v in feat_c.columns]

    fig, axes = plt.subplots(1, len(plot_vars),
                             figsize=(3.5 * len(plot_vars), 5))
    if len(plot_vars) == 1:
        axes = [axes]

    colours = ["#1D9E75", "#E24B4A", "#6B21A8"]
    names   = REGIME_NAMES_3

    for ax, (var, label) in zip(axes, plot_vars):
        data   = [feat_c[lbl_c == s][var].dropna().values
                  for s in sorted(lbl_c.unique())]
        labels_= [names.get(s, str(s)) for s in sorted(lbl_c.unique())]
        bp     = ax.boxplot(data, patch_artist=True, notch=True,
                            medianprops={"color": "white", "linewidth": 2})
        for patch, col in zip(bp["boxes"], colours[:len(data)]):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        ax.set_xticklabels(labels_, fontsize=10)
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("India RQ2 — Variable Distributions by Regime",
                 fontweight="bold", fontsize=12, y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[PLOT] Saved → {save_path}")


def main():
    log.info("=" * 65)
    log.info("  RQ2 INDIA STAGE 2 — ANOVA & Structural Break Tests")
    log.info("=" * 65)

    labels, feat, log_ret = load_data()

    # Per-regime performance
    perf = regime_performance(feat, labels, log_ret)
    perf.to_csv(os.path.join(OUTPUT_DIR, "rq2_india_regime_perf.csv"),
                index=False)

    # ANOVA / Kruskal-Wallis
    anova_df = anova_kruskal(feat, labels)
    anova_df.to_csv(os.path.join(OUTPUT_DIR, "rq2_india_anova.csv"),
                    index=False)

    # Chow test at India structural breaks
    chow_df = chow_test_india_breaks(feat)
    if not chow_df.empty:
        chow_df.to_csv(os.path.join(OUTPUT_DIR, "rq2_india_chow.csv"),
                       index=False)

    # Print summary
    print(f"\n{'='*65}")
    print(f"  RQ2 INDIA STAGE 2 — REGIME PERFORMANCE")
    print(f"{'='*65}")
    print(f"\n{perf[['regime','label','n_days','pct_days','ann_ret','ann_vol','sharpe']].to_string(index=False)}")

    print(f"\n{'='*65}")
    print(f"  ANOVA / KRUSKAL-WALLIS (H0: identical across regimes)")
    print(f"{'='*65}")
    print(f"\n{anova_df.to_string(index=False)}")

    n_reject = int(anova_df["reject_H0"].sum()) if not anova_df.empty else 0
    print(f"\n  Reject H0: {n_reject}/{len(anova_df)} variables")
    print(f"  H1 (ANOVA condition): "
          f"{'CONFIRMED ✓' if n_reject > 0 else 'NOT confirmed'}")

    if not chow_df.empty:
        print(f"\n{'='*65}")
        print(f"  CHOW TEST AT INDIA STRUCTURAL BREAKS")
        print(f"{'='*65}")
        for _, r in chow_df.iterrows():
            sig = "✓ significant" if r["significant"] else "not significant"
            print(f"  {r['date']} {r['event'][:30]:<32} "
                  f"t={r['t_stat']:>6.3f}  p={r['p_val']:.4f}  {sig}")

    # Plot
    plot_regime_distributions(
        feat, labels,
        save_path=os.path.join(OUTPUT_DIR, "rq2_india_distributions.png")
    )

    log.info("\n✓ Stage 2 complete. Run rq2_india_stage3_adaptive.py next.\n")
    return perf, anova_df, chow_df


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# =============================================================================
# rq2_stage1_detection.py
# Stage 1: Regime Detection — three independent methods + consensus.
#
# Methods
# -------
# A. Hidden Markov Model (HMM)      — respects temporal ordering
# B. Gaussian Mixture Model (GMM)   — density-based, BIC-optimal k
# C. Bai-Perron structural breaks   — statistical changepoint detection
# D. Consensus                      — majority vote across A/B/C
#
# Outputs
# -------
#   data/rq2_regime_labels.parquet  — HMM / GMM / BP / consensus labels
#   data/rq2_gmm_probs.parquet      — soft GMM probabilities per regime
#   rq2_outputs/rq2_regime_stats.csv
#   rq2_outputs/rq2_regime_timeline.png
#
# Run: python rq2_stage1_detection.py
# =============================================================================

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from regime_config import (
    DATA_DIR, OUTPUT_DIR, N_REGIMES,
    HMM_N_INIT, HMM_N_ITER, HMM_COVARIANCE_TYPE,
    GMM_N_INIT, BP_MIN_SIZE, BP_MAX_BREAKS, ALPHA,
)
from scipy_compat import stats


# =============================================================================
# LOADER
# =============================================================================

def load_data():
    feat_path = os.path.join(DATA_DIR, "rq2_regime_features.parquet")
    ret_path  = os.path.join(DATA_DIR, "rq2_log_returns.parquet")
    for p in [feat_path, ret_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run rq2_stage0_data.py first."
            )
    return (pd.read_parquet(feat_path),
            pd.read_parquet(ret_path))


# =============================================================================
# HELPERS
# =============================================================================

def _relabel_by_vol(labels, log_ret, primary="SPY"):
    """
    Canonical relabelling: sort regimes by ascending annualised volatility.
    0 = lowest vol (Bull), 3 = highest vol (Crisis).
    This ensures consistent ordering across methods and runs.
    """
    if primary not in log_ret.columns:
        primary = log_ret.columns[0]
    # CRITICAL: reindex returns to match label index (labels is a subset
    # of full dates after dropna in feature engineering)
    r = log_ret[primary].reindex(labels.index)

    vols = {}
    for reg in labels.unique():
        mask   = (labels == reg)
        subset = r.loc[mask].dropna()
        vols[reg] = subset.std() * np.sqrt(252) if len(subset) > 2 else 0.0

    # Sort original labels by volatility ascending
    sorted_regs = sorted(vols, key=vols.get)
    mapping     = {old_r: new_r for new_r, old_r in enumerate(sorted_regs)}
    return labels.map(mapping)


def _merge_to_n(labels, log_ret, n, primary="SPY"):
    """
    Merge excess Bai-Perron segments into exactly n regimes
    by grouping segments of similar volatility.
    """
    unique = sorted(labels.unique())
    if len(unique) <= n:
        return labels

    if primary not in log_ret.columns:
        primary = log_ret.columns[0]
    # Align returns to label index
    r = log_ret[primary].reindex(labels.index)

    seg_vols = {seg: r.loc[labels == seg].dropna().std() for seg in unique}
    thresholds = np.quantile(list(seg_vols.values()),
                              np.linspace(0, 1, n + 1))
    new = labels.copy()
    for seg, vol in seg_vols.items():
        bucket = min(int(np.searchsorted(thresholds[1:], vol)), n - 1)
        new[labels == seg] = bucket
    return new


def _count_runs(mask):
    """Count number of consecutive True runs in a boolean Series."""
    if mask.empty:
        return 0
    transitions = mask.astype(int).diff().fillna(mask.astype(int).iloc[0])
    return int((transitions == 1).sum()
               + (mask.astype(int).iloc[0] == 1))


# =============================================================================
# METHOD A: HMM
# =============================================================================

def fit_hmm(features, log_ret, n_regimes=N_REGIMES):
    """
    Gaussian HMM on a curated subset of regime features.
    Uses hmmlearn if available; falls back to GMM proxy otherwise.
    Temporal ordering is respected — regimes can only persist or
    transition, not jump arbitrarily.
    """
    hmm_cols = [c for c in [
        "spy_ret_21d", "rvol_21d", "vix_level",
        "spy_vs_ma200", "credit_spread_proxy",
        "spy_vs_tlt", "yield_curve", "turbulence",
    ] if c in features.columns]

    X = features[hmm_cols].dropna()
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        from hmmlearn.hmm import GaussianHMM
        log.info(f"  HMM: fitting {n_regimes}-state model on {len(X)} obs …")
        best, best_score = None, -np.inf
        for seed in range(HMM_N_INIT):
            try:
                m = GaussianHMM(
                    n_components   = n_regimes,
                    covariance_type= HMM_COVARIANCE_TYPE,
                    n_iter         = HMM_N_ITER,
                    random_state   = seed,
                    verbose        = False,
                )
                m.fit(X_scaled)
                s = m.score(X_scaled)
                if s > best_score:
                    best_score, best = s, m
            except Exception:
                continue
        labels = best.predict(X_scaled)
        log.info(f"  HMM log-likelihood: {best_score:.2f}")

    except ImportError:
        log.warning("  hmmlearn not installed — using GMM as HMM proxy.")
        best, best_score = None, np.inf
        for seed in range(HMM_N_INIT):
            m = GaussianMixture(
                n_components  = n_regimes,
                covariance_type="full",
                n_init        = 1,
                random_state  = seed,
                max_iter      = 300,
            )
            m.fit(X_scaled)
            s = m.bic(X_scaled)
            if s < best_score:
                best_score, best = s, m
        labels = best.predict(X_scaled)

    series = pd.Series(labels, index=X.index, name="hmm_regime")
    return _relabel_by_vol(series, log_ret)


# =============================================================================
# METHOD B: GMM with BIC model selection
# =============================================================================

def fit_gmm(features, log_ret, n_regimes=N_REGIMES):
    """
    Gaussian Mixture Model on return + volatility features.
    BIC used to confirm that n_regimes components is appropriate.
    Soft probabilities saved for downstream use.
    """
    gmm_cols = [c for c in [
        "spy_ret_21d", "rvol_21d", "vix_zscore",
        "spy_vs_ma200", "vol_ratio", "turbulence",
        "credit_spread_proxy", "spy_vs_tlt",
    ] if c in features.columns]

    X  = features[gmm_cols].dropna()
    sc = StandardScaler()
    Xs = sc.fit_transform(X)

    # BIC selection over k = 2..7
    bic = {}
    for k in range(2, min(8, max(3, len(X) // 80))):
        m = GaussianMixture(
            n_components=k, covariance_type="full",
            n_init=GMM_N_INIT, random_state=42,
        )
        m.fit(Xs)
        bic[k] = m.bic(Xs)
    opt_k = min(bic, key=bic.get)
    log.info(f"  GMM BIC: {bic}  →  optimal k={opt_k}")

    # Fit final model
    best, best_bic = None, np.inf
    for seed in range(GMM_N_INIT):
        m = GaussianMixture(
            n_components=n_regimes, covariance_type="full",
            n_init=1, random_state=seed, max_iter=400,
        )
        m.fit(Xs)
        b = m.bic(Xs)
        if b < best_bic:
            best_bic, best = b, m

    labels = best.predict(Xs)
    probs  = best.predict_proba(Xs)
    log.info(f"  GMM BIC (n={n_regimes}): {best_bic:.2f}")

    prob_df = pd.DataFrame(
        probs, index=X.index,
        columns=[f"gmm_prob_r{i}" for i in range(n_regimes)],
    )
    prob_df.to_parquet(os.path.join(DATA_DIR, "rq2_gmm_probs.parquet"))

    series = pd.Series(labels, index=X.index, name="gmm_regime")
    return _relabel_by_vol(series, log_ret)


# =============================================================================
# METHOD C: Bai-Perron structural breaks
# =============================================================================

def fit_structural_breaks(log_ret, target="SPY",
                           max_breaks=BP_MAX_BREAKS,
                           min_size=BP_MIN_SIZE):
    """
    Sequential Chow F-test with Bonferroni correction.
    Detects dates where the mean and variance of returns shift.
    Each segment between break-dates becomes a candidate regime.
    Segments are merged to N_REGIMES by volatility similarity.
    """
    log.info(f"  Bai-Perron: detecting structural breaks …")
    if target not in log_ret.columns:
        target = log_ret.columns[0]
    r = log_ret[target].dropna()
    n = len(r)

    def chow_f(series, bp):
        s1, s2 = series.iloc[:bp], series.iloc[bp:]
        if len(s1) < min_size or len(s2) < min_size:
            return 0.0
        rss_r = ((series - series.mean()) ** 2).sum()
        rss_u = (((s1 - s1.mean()) ** 2).sum()
                 + ((s2 - s2.mean()) ** 2).sum())
        df_num, df_den = 1, n - 2
        ms_num = (rss_r - rss_u) / df_num
        ms_den = rss_u / df_den
        return max(ms_num / ms_den, 0.0) if ms_den > 0 else 0.0

    breakpoints = []
    segments    = [(0, n)]
    bonf_thresh = ALPHA / max(max_breaks, 1)

    for _ in range(max_breaks):
        best_f, best_bp, best_seg = 0.0, None, None
        for si, (ss, se) in enumerate(segments):
            seg = r.iloc[ss:se]
            for bp in range(min_size, len(seg) - min_size):
                f = chow_f(seg, bp)
                if f > best_f:
                    best_f, best_bp, best_seg = f, ss + bp, si

        # Approximate p-value: F(1, n-2) tail
        try:
            p = float(stats.chi2.sf(best_f, df=1)) if hasattr(
                stats, "chi2") else np.exp(-best_f / 2)
        except Exception:
            p = np.exp(-best_f / 2)

        if best_f > 0 and p < bonf_thresh and best_bp is not None:
            breakpoints.append(best_bp)
            ss, se = segments.pop(best_seg)
            segments.insert(best_seg, (best_bp, se))
            segments.insert(best_seg, (ss, best_bp))
        else:
            break

    break_dates = sorted([r.index[bp] for bp in breakpoints])
    log.info(f"  Breaks detected: {len(break_dates)}")
    for bd in break_dates:
        log.info(f"    {bd.date()}")

    labels = pd.Series(0, index=r.index, name="bp_regime")
    for i, bd in enumerate(break_dates):
        labels.loc[bd:] = i + 1

    labels = _merge_to_n(labels, log_ret, N_REGIMES, target)
    return _relabel_by_vol(labels, log_ret)


# =============================================================================
# CONSENSUS
# =============================================================================

def compute_consensus(hmm, gmm, bp):
    """
    Majority vote across three methods.
    Returns consensus label at each date.
    Logs inter-method agreement rates.
    """
    df = pd.DataFrame({"hmm": hmm, "gmm": gmm, "bp": bp}).dropna()

    consensus = df.apply(
        lambda row: int(row.value_counts().idxmax()), axis=1
    )
    consensus.name = "consensus_regime"

    full_agree   = (df.nunique(axis=1) == 1).mean()
    two_of_three = (df.nunique(axis=1) <= 2).mean()
    log.info(f"\n  Full agreement (3/3)  : {full_agree*100:.1f}%")
    log.info(f"  Majority agree (2/3)  : {two_of_three*100:.1f}%")
    return consensus


# =============================================================================
# REGIME CHARACTERISATION
# =============================================================================

def characterise_regimes(log_ret, labels, features):
    """
    Compute descriptive statistics per regime and assign
    human-readable names (Bull / Bear / Sideways / Crisis).
    """
    primary = "SPY" if "SPY" in log_ret.columns else log_ret.columns[0]
    # Align returns to label index (labels covers feature subset of dates)
    r = log_ret[primary].reindex(labels.index)

    rows = []
    for reg in sorted(labels.unique()):
        mask  = (labels == reg)
        rsub  = r.loc[mask].dropna()
        fsub  = features.loc[mask].dropna(how="all")

        ann_r = rsub.mean() * 252
        ann_v = rsub.std()  * np.sqrt(252)
        sr    = ann_r / ann_v if ann_v > 0 else 0.0
        cum   = (1 + rsub).cumprod()
        mdd   = ((cum - cum.cummax()) / cum.cummax()).min()

        vix_avg  = (fsub["vix_level"].mean()
                    if "vix_level" in fsub.columns else np.nan)
        turb_avg = (fsub["turbulence"].mean()
                    if "turbulence" in fsub.columns else np.nan)
        n_runs   = _count_runs(mask)

        rows.append({
            "regime":       reg,
            "n_days":       int(mask.sum()),
            "n_periods":    n_runs,
            "avg_duration": round(mask.sum() / max(n_runs, 1), 1),
            "ann_ret":      round(ann_r, 4),
            "ann_vol":      round(ann_v, 4),
            "sharpe":       round(sr,    4),
            "max_dd":       round(mdd,   4),
            "vix_avg":      round(vix_avg,  2) if not np.isnan(vix_avg)  else np.nan,
            "turb_avg":     round(turb_avg, 4) if not np.isnan(turb_avg) else np.nan,
            "pct_time":     round(mask.sum() / len(r) * 100, 1),
        })

    df = pd.DataFrame(rows).set_index("regime")
    df["name"] = _name_regimes(df)

    log.info(f"\n── Regime Characteristics ────────────────────────────────")
    log.info(f"\n{df[['name','n_days','pct_time','ann_ret','ann_vol','sharpe','max_dd','vix_avg']].to_string()}")
    return df


def _name_regimes(df):
    """Heuristic naming: Bull / Bear / Sideways / Crisis by vol+VIX."""
    names = {}
    for r_id, row in df.iterrows():
        vix = row.get("vix_avg", np.nan)
        av  = row["ann_vol"]
        ar  = row["ann_ret"]
        if not np.isnan(vix) and (vix > 28 or av > 0.22):
            names[r_id] = "Crisis"
        elif ar > 0.05 and av < 0.15:
            names[r_id] = "Bull"
        elif ar < -0.03:
            names[r_id] = "Bear"
        else:
            names[r_id] = "Sideways"
    # Ensure uniqueness by suffixing duplicates
    seen = {}
    for k in list(names):
        v = names[k]
        seen[v] = seen.get(v, 0) + 1
        if seen[v] > 1:
            names[k] = f"{v}{seen[v]}"
    return pd.Series(names)


# =============================================================================
# VISUALISATION
# =============================================================================

REGIME_COLOURS = {0: "#B8D4F0", 1: "#F5C6C6", 2: "#D5E8D4", 3: "#FFE6CC"}
REGIME_NAMES   = ["Bull", "Bear", "Sideways", "Crisis"]


def _shade_regimes(ax, labels):
    prev, start = None, None
    for date, reg in labels.items():
        if reg != prev:
            if prev is not None:
                ax.axvspan(start, date, alpha=0.25,
                           color=REGIME_COLOURS.get(int(prev), "#DDDDDD"),
                           zorder=1)
            start, prev = date, reg
    if start is not None and prev is not None:
        ax.axvspan(start, labels.index[-1], alpha=0.25,
                   color=REGIME_COLOURS.get(int(prev), "#DDDDDD"),
                   zorder=1)


def plot_regime_timeline(prices, labels, regime_info, save_path=None):
    """Three-panel: price / VIX / turbulence, coloured by consensus regime."""
    features_path = os.path.join(DATA_DIR, "rq2_regime_features.parquet")
    feat = (pd.read_parquet(features_path)
            if os.path.exists(features_path) else pd.DataFrame())

    fig, axes = plt.subplots(
        3, 1, figsize=(15, 10),
        gridspec_kw={"height_ratios": [3, 1.5, 1.5]},
    )
    fig.suptitle("RQ2 — Regime Detection Timeline",
                 fontsize=14, fontweight="bold")

    spy_price = prices.get("SPY", prices.iloc[:, 0]).dropna()

    for ax in axes:
        _shade_regimes(ax, labels)

    # Panel 1: SPY price
    axes[0].plot(spy_price.index, spy_price.values,
                 color="#1B3A6B", linewidth=1.2, zorder=5)
    axes[0].set_ylabel("SPY Price (log scale)", fontsize=10)
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.3)

    patches = []
    for r_id, colour in REGIME_COLOURS.items():
        if r_id in regime_info.index:
            name = regime_info.loc[r_id, "name"]
            pct  = regime_info.loc[r_id, "pct_time"]
            patches.append(mpatches.Patch(
                color=colour,
                label=f"R{r_id}: {name} ({pct:.0f}%)",
            ))
    axes[0].legend(handles=patches, loc="upper left",
                   fontsize=8, ncol=2)

    # Panel 2: VIX
    if "^VIX" in prices.columns:
        vix = prices["^VIX"].dropna()
        axes[1].plot(vix.index, vix.values, color="#C0392B",
                     linewidth=1.0, zorder=5)
        axes[1].axhline(20, color="navy",    linewidth=0.8,
                        linestyle="--", label="VIX=20")
        axes[1].axhline(30, color="darkred", linewidth=0.8,
                        linestyle="--", label="VIX=30")
        axes[1].set_ylabel("VIX", fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)

    # Panel 3: Turbulence
    if "turbulence" in feat.columns:
        turb   = feat["turbulence"].dropna()
        thresh = turb.quantile(0.95)
        axes[2].plot(turb.index, turb.values, color="#8E44AD",
                     linewidth=0.8, alpha=0.7, zorder=5)
        axes[2].axhline(thresh, color="darkred", linewidth=1.0,
                        linestyle="--",
                        label=f"Crisis threshold (p95={thresh:.2f})")
        axes[2].set_ylabel("Turbulence", fontsize=10)
        axes[2].set_xlabel("Date", fontsize=10)
        axes[2].legend(fontsize=8)
        axes[2].grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved → {save_path}")
    plt.close()


def plot_agreement_heatmap(all_labels, save_path=None):
    """Confusion matrix of method agreement (HMM vs GMM vs BP)."""
    cols   = ["hmm", "gmm", "bp"]
    df     = all_labels[cols].dropna().astype(int)
    labels = sorted(df.stack().unique())
    k      = len(labels)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    pairs = [("hmm", "gmm"), ("hmm", "bp"), ("gmm", "bp")]
    titles = ["HMM vs GMM", "HMM vs BP", "GMM vs BP"]

    for ax, (c1, c2), title in zip(axes, pairs, titles):
        mat = np.zeros((k, k), dtype=int)
        for _, row in df.iterrows():
            i = labels.index(row[c1])
            j = labels.index(row[c2])
            mat[i, j] += 1
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        for i in range(k):
            for j in range(k):
                ax.text(j, i, str(mat[i, j]),
                        ha="center", va="center", fontsize=9,
                        color="white" if mat[i, j] > mat.max() * 0.5
                        else "black")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        lab_names = [f"R{l}" for l in labels]
        ax.set_xticklabels(lab_names, fontsize=8)
        ax.set_yticklabels(lab_names, fontsize=8)
        ax.set_xlabel(c2.upper(), fontsize=9)
        ax.set_ylabel(c1.upper(), fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        plt.colorbar(im, ax=ax)

    fig.suptitle("Method Agreement Heatmaps", fontsize=12,
                 fontweight="bold")
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
    log.info("  RQ2 STAGE 1 — Regime Detection")
    log.info("=" * 65)

    features, log_ret = load_data()
    prices_path = os.path.join(DATA_DIR, "prices.parquet")
    prices = (pd.read_parquet(prices_path)
              if os.path.exists(prices_path) else None)

    # ── A: HMM ────────────────────────────────────────────────────
    log.info("\n[A] Hidden Markov Model …")
    hmm_labels = fit_hmm(features, log_ret, N_REGIMES)

    # ── B: GMM ────────────────────────────────────────────────────
    log.info("\n[B] Gaussian Mixture Model …")
    gmm_labels = fit_gmm(features, log_ret, N_REGIMES)

    # ── C: Bai-Perron ─────────────────────────────────────────────
    log.info("\n[C] Bai-Perron Structural Breaks …")
    bp_labels  = fit_structural_breaks(log_ret)

    # ── D: Consensus ──────────────────────────────────────────────
    log.info("\n[D] Consensus …")
    consensus  = compute_consensus(hmm_labels, gmm_labels, bp_labels)

    # ── Save labels ───────────────────────────────────────────────
    all_labels = pd.DataFrame({
        "hmm":       hmm_labels,
        "gmm":       gmm_labels,
        "bp":        bp_labels,
        "consensus": consensus,
    })
    labels_path = os.path.join(DATA_DIR, "rq2_regime_labels.parquet")
    all_labels.to_parquet(labels_path)
    log.info(f"\nLabels saved → {labels_path}")

    # ── Characterise ──────────────────────────────────────────────
    log.info("\n[E] Characterising regimes …")
    regime_info = characterise_regimes(log_ret, consensus, features)
    regime_info.to_csv(
        os.path.join(OUTPUT_DIR, "rq2_regime_stats.csv")
    )

    # ── Plots ─────────────────────────────────────────────────────
    if prices is not None:
        plot_regime_timeline(
            prices, consensus, regime_info,
            save_path=os.path.join(OUTPUT_DIR,
                                   "rq2_regime_timeline.png"),
        )
    plot_agreement_heatmap(
        all_labels,
        save_path=os.path.join(OUTPUT_DIR,
                               "rq2_method_agreement.png"),
    )

    # Distribution of labels
    log.info("\n── Consensus label distribution ─────────────────────────")
    log.info(f"\n{consensus.value_counts().sort_index().to_string()}")

    log.info("\n✓ Stage 1 complete. Run rq2_stage2_anova.py next.\n")
    return all_labels, regime_info


if __name__ == "__main__":
    main()

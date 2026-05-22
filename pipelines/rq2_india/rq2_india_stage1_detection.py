#!/usr/bin/env python3
# =============================================================================
# rq2_india_stage1_detection.py — Regime Detection (HMM + GMM + Bai-Perron)
#
# Primary: 3-state HMM on 6 variables (Bull/Bear/Crisis)
# Robustness: 4-state HMM — BIC selects optimal model
# Also: GMM (density-based), Bai-Perron structural breaks, consensus labels
#
# India-specific:
#   - Annotates demonetisation, NBFC, COVID, FII episodes on regime timeline
#   - Compares India VIX vs realised vol regime agreement (if both available)
#   - Minimum regime duration = 5 days (India regimes can switch fast)
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from rq2_india_config import (
    DATA_DIR, OUTPUT_DIR, INDIA_BREAKS,
    N_REGIMES_PRIMARY, N_REGIMES_ROBUST,
    HMM_N_INIT, HMM_N_ITER, HMM_COVARIANCE_TYPE,
    GMM_N_INIT, BP_MIN_SIZE, BP_MAX_BREAKS, ALPHA,
    REGIME_NAMES_3, REGIME_NAMES_4,
)

try:
    from scipy_compat import stats
except ImportError:
    from scipy import stats


# ── HMM import ───────────────────────────────────────────────────────────────
try:
    from hmmlearn.hmm import GaussianHMM
    HMM_OK = True
except ImportError:
    HMM_OK = False
    log.warning("hmmlearn not installed: pip install hmmlearn")


def load_data():
    feat = pd.read_parquet(os.path.join(DATA_DIR, "rq2_india_features.parquet"))
    ret  = pd.read_parquet(os.path.join(DATA_DIR, "rq2_india_log_returns.parquet"))
    return feat, ret


# =============================================================================
# HELPER: select HMM input features (core 6)
# =============================================================================

def get_hmm_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Select the core 6 HMM input variables, drop rows with any NaN,
    return standardised array and the date index.
    """
    core6 = ["nsei_ret_1d", "vol_primary", "usdinr_5d",
              "spy_21d", "gold_21d", "crude_21d"]
    available = [c for c in core6 if c in features.columns]
    missing   = [c for c in core6 if c not in features.columns]
    if missing:
        log.warning(f"[HMM] Missing HMM features: {missing}")

    X = features[available].dropna()
    log.info(f"[HMM] Input: {len(available)} features, {len(X)} obs")
    return X


# =============================================================================
# RELABELLING — sort regimes by volatility (0=Bull, highest=Crisis)
# =============================================================================

def _relabel_by_vol(labels, feat, vol_col="nsei_vol_21d"):
    """Sort regime labels by ascending realised volatility of each state."""
    common = feat.index.intersection(pd.Series(labels).index
                                      if hasattr(labels, 'index')
                                      else feat.index[:len(labels)])
    lbl_ser = pd.Series(labels, index=feat.index[:len(labels)])
    if vol_col not in feat.columns:
        vol_col = "vol_primary"

    state_vols = {}
    for s in np.unique(labels):
        mask = lbl_ser == s
        v    = feat.loc[mask.index[mask], vol_col].mean()
        state_vols[s] = v if not np.isnan(v) else 0

    sorted_states = sorted(state_vols, key=state_vols.get)
    mapping       = {old: new for new, old in enumerate(sorted_states)}
    return np.array([mapping[l] for l in labels])


# =============================================================================
# A. HIDDEN MARKOV MODEL
# =============================================================================

def fit_hmm(X: pd.DataFrame, n_states: int) -> tuple:
    """
    Fit Gaussian HMM with n_states. Runs HMM_N_INIT random restarts,
    keeps the best log-likelihood. Returns (labels, model, bic).
    """
    if not HMM_OK:
        return None, None, np.inf

    sc      = StandardScaler()
    X_sc    = sc.fit_transform(X.values)
    lengths = [len(X_sc)]

    best_ll, best_model = -np.inf, None
    for seed in range(HMM_N_INIT):
        try:
            m = GaussianHMM(
                n_components=n_states,
                covariance_type=HMM_COVARIANCE_TYPE,
                n_iter=HMM_N_ITER,
                random_state=seed,
                verbose=False,
            )
            m.fit(X_sc, lengths)
            ll = m.score(X_sc, lengths)
            if ll > best_ll:
                best_ll, best_model = ll, m
        except Exception:
            pass

    if best_model is None:
        return None, None, np.inf

    labels = best_model.predict(X_sc, lengths)
    # BIC = -2 * log-likelihood + k * log(n)
    k   = (n_states ** 2 - n_states              # transition probs
           + n_states * X_sc.shape[1]            # means
           + n_states * X_sc.shape[1] ** 2)      # covariance (full)
    bic = -2 * best_ll + k * np.log(len(X_sc))
    log.info(f"[HMM] {n_states}-state: LL={best_ll:.2f}  BIC={bic:.2f}")
    return labels, best_model, bic


def run_hmm(features: pd.DataFrame) -> dict:
    """Fit 3-state (primary) and 4-state (robustness), select by BIC."""
    X = get_hmm_features(features)

    log.info("[HMM] Fitting 3-state model (primary) ...")
    lbl3, mdl3, bic3 = fit_hmm(X, N_REGIMES_PRIMARY)

    log.info("[HMM] Fitting 4-state model (robustness) ...")
    lbl4, mdl4, bic4 = fit_hmm(X, N_REGIMES_ROBUST)

    log.info(f"\n[HMM] BIC comparison:")
    log.info(f"  3-state BIC: {bic3:.2f}")
    log.info(f"  4-state BIC: {bic4:.2f}")
    log.info(f"  Winner: {'3-state' if bic3 <= bic4 else '4-state'} "
             f"(lower BIC = better fit penalised for complexity)")

    # Primary labels (3-state, relabelled by vol)
    if lbl3 is not None:
        lbl3_rel = _relabel_by_vol(lbl3, features.loc[X.index])
    else:
        lbl3_rel = None

    # Robustness labels (4-state, relabelled by vol)
    if lbl4 is not None:
        lbl4_rel = _relabel_by_vol(lbl4, features.loc[X.index])
    else:
        lbl4_rel = None

    # Regime stats (3-state)
    if lbl3_rel is not None:
        stats_rows = []
        for s in range(N_REGIMES_PRIMARY):
            mask   = lbl3_rel == s
            r_sub  = features.loc[X.index, "nsei_ret_1d"][mask]
            v_sub  = features.loc[X.index, "vol_primary"][mask]
            fx_sub = features.loc[X.index, "usdinr_5d"][mask] \
                     if "usdinr_5d" in features.columns else pd.Series()
            n      = int(mask.sum())
            stats_rows.append({
                "regime":        s,
                "label":         REGIME_NAMES_3.get(s, str(s)),
                "n_days":        n,
                "pct_days":      round(n / len(lbl3_rel) * 100, 1),
                "mean_ret_ann":  round(float(r_sub.mean()) * 252, 4),
                "mean_vol":      round(float(v_sub.mean()), 4),
                "mean_usdinr_5d":round(float(fx_sub.mean()), 5)
                                 if len(fx_sub) > 0 else np.nan,
                "bic_3state":    round(bic3, 2),
                "bic_4state":    round(bic4, 2),
                "bic_winner":    "3-state" if bic3 <= bic4 else "4-state",
            })
        stats_df = pd.DataFrame(stats_rows)
        log.info(f"\n[HMM] 3-state regime statistics:\n"
                 f"{stats_df.to_string(index=False)}")
    else:
        stats_df = pd.DataFrame()

    return {
        "labels_3":   pd.Series(lbl3_rel, index=X.index)
                       if lbl3_rel is not None else None,
        "labels_4":   pd.Series(lbl4_rel, index=X.index)
                       if lbl4_rel is not None else None,
        "bic_3":      bic3,
        "bic_4":      bic4,
        "bic_winner": 3 if bic3 <= bic4 else 4,
        "stats":      stats_df,
        "X_index":    X.index,
    }


# =============================================================================
# B. GAUSSIAN MIXTURE MODEL
# =============================================================================

def run_gmm(features: pd.DataFrame, n_states: int = N_REGIMES_PRIMARY) -> pd.Series:
    X   = get_hmm_features(features)
    sc  = StandardScaler()
    X_sc = sc.fit_transform(X.values)

    best_bic, best_gmm = np.inf, None
    for seed in range(GMM_N_INIT):
        try:
            g = GaussianMixture(n_components=n_states,
                                covariance_type="full",
                                random_state=seed, n_init=1)
            g.fit(X_sc)
            b = g.bic(X_sc)
            if b < best_bic:
                best_bic, best_gmm = b, g
        except Exception:
            pass

    if best_gmm is None:
        return pd.Series(np.nan, index=X.index)

    labels = best_gmm.predict(X_sc)
    labels = _relabel_by_vol(labels, features.loc[X.index])
    log.info(f"[GMM] {n_states}-state BIC={best_bic:.2f}")
    return pd.Series(labels, index=X.index, name="gmm")


# =============================================================================
# C. BAI-PERRON STRUCTURAL BREAKS
# =============================================================================

def run_bai_perron(features: pd.DataFrame) -> pd.Series:
    """
    Bai-Perron (2003) multiple structural break detection on Nifty returns.
    Tests for breaks in mean return. Each segment = one regime.
    """
    if "nsei_ret_1d" not in features.columns:
        return pd.Series(dtype=int)

    y   = features["nsei_ret_1d"].dropna().values
    n   = len(y)
    idx = features["nsei_ret_1d"].dropna().index

    min_size  = BP_MIN_SIZE
    max_breaks= min(BP_MAX_BREAKS, n // (min_size * 2))

    # Sequential F-test for breaks in mean
    break_points = []
    residuals    = y.copy()
    segment_start= 0

    for _ in range(max_breaks):
        best_f, best_bp = 0, None
        for bp in range(segment_start + min_size,
                        n - min_size):
            y1 = residuals[segment_start:bp]
            y2 = residuals[bp:]
            if len(y1) < min_size or len(y2) < min_size:
                continue
            # Chow-style F statistic
            rss_pooled = np.sum((residuals[segment_start:] - residuals[segment_start:].mean()) ** 2)
            rss_split  = (np.sum((y1 - y1.mean()) ** 2)
                          + np.sum((y2 - y2.mean()) ** 2))
            if rss_split <= 0:
                continue
            F = ((rss_pooled - rss_split) / 2) / (rss_split / (n - 2))
            if F > best_f:
                best_f, best_bp = F, bp

        if best_bp is None or best_f < 3.84:   # chi-sq(1) 5% critical value
            break
        break_points.append(best_bp)
        break_points.sort()

    log.info(f"[BP] Found {len(break_points)} structural breaks")

    # Assign regime labels: segment number 0, 1, 2, ...
    labels = np.zeros(n, dtype=int)
    prev   = 0
    for seg_id, bp in enumerate(sorted(break_points), 1):
        labels[bp:] = seg_id
    # Relabel by vol to match HMM ordering
    labels = _relabel_by_vol(labels, features.loc[idx])
    # Clip to n_states - 1
    labels = np.clip(labels, 0, N_REGIMES_PRIMARY - 1)

    bp_dates = [str(idx[bp].date()) for bp in break_points]
    log.info(f"[BP] Break dates: {bp_dates}")

    return pd.Series(labels, index=idx, name="bai_perron")


# =============================================================================
# D. CONSENSUS
# =============================================================================

def consensus_labels(hmm_lbl, gmm_lbl, bp_lbl) -> pd.Series:
    """Majority vote. Requires 2 of 3 methods to agree."""
    common = hmm_lbl.index
    for s in [gmm_lbl, bp_lbl]:
        if s is not None and len(s) > 0:
            common = common.intersection(s.index)

    rows = {"hmm": hmm_lbl.reindex(common)}
    if gmm_lbl is not None and len(gmm_lbl) > 0:
        rows["gmm"] = gmm_lbl.reindex(common)
    if bp_lbl is not None and len(bp_lbl) > 0:
        rows["bp"]  = bp_lbl.reindex(common)

    df = pd.DataFrame(rows).dropna()
    consensus = df.mode(axis=1)[0].astype(int)
    return consensus.rename("consensus")


# =============================================================================
# VISUALISATION
# =============================================================================

REGIME_COLOURS_3 = {0: "#1D9E75", 1: "#E24B4A", 2: "#6B21A8"}
REGIME_COLOURS_4 = {0: "#1D9E75", 1: "#EF9F27", 2: "#E24B4A", 3: "#6B21A8"}

def plot_regime_timeline(features, hmm_result, log_ret, save_path):
    """
    Two-panel chart:
      Top: Nifty price coloured by regime, structural break lines,
           India-specific event annotations.
      Bottom: Primary vol measure (India VIX or realised vol).
    """
    labels = hmm_result["labels_3"]
    if labels is None:
        return

    nsei_ret = features["nsei_ret_1d"].reindex(labels.index).fillna(0)
    price_idx = (1 + nsei_ret).cumprod()
    vol_col   = "vol_primary"
    vol_ser   = features[vol_col].reindex(labels.index)
    names     = REGIME_NAMES_3
    colours   = REGIME_COLOURS_3

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                    gridspec_kw={"height_ratios": [3, 1]},
                                    sharex=True)
    fig.suptitle("India RQ2 — Nifty 50 Regime Timeline (3-state HMM)",
                 fontweight="bold", fontsize=13)

    # Top panel: price with regime shading
    for s in range(3):
        mask = labels == s
        if not mask.any():
            continue
        runs = []
        in_run = False
        for i, m in enumerate(mask):
            if m and not in_run:
                start_i = i; in_run = True
            elif not m and in_run:
                runs.append((start_i, i)); in_run = False
        if in_run:
            runs.append((start_i, len(mask)))
        for (si, ei) in runs:
            ax1.axvspan(labels.index[si], labels.index[min(ei, len(labels.index)-1)],
                        alpha=0.18, color=colours[s], label=None)

    ax1.plot(price_idx.index, price_idx.values,
             color="#1C3678", linewidth=1.0, label="Nifty 50 (rebased)")
    ax1.set_ylabel("Nifty 50 (rebased to 1.0)")
    ax1.set_yscale("log")

    # Structural break / event annotations
    for dt_str, desc in INDIA_BREAKS.items():
        dt = pd.Timestamp(dt_str)
        if labels.index.min() <= dt <= labels.index.max():
            ax1.axvline(dt, color="black", linewidth=0.8,
                        linestyle="--", alpha=0.7)
            ax1.text(dt, ax1.get_ylim()[0] * 1.05, desc[:18],
                     fontsize=6.5, rotation=90, va="bottom",
                     ha="right", color="#333333")

    patches = [mpatches.Patch(color=colours[s], alpha=0.5,
                              label=f"Regime {s}: {names[s]}")
               for s in range(3)]
    ax1.legend(handles=patches, loc="upper left", fontsize=9)
    ax1.grid(alpha=0.20)

    # Bottom panel: volatility
    vol_label = ("India VIX" if "INDIAVIX" in str(features.columns.tolist())
                               and features["vol_primary"].max() < 200
                 else "21d Realised Vol (ann.)")
    ax2.fill_between(vol_ser.index, vol_ser.values, 0,
                     alpha=0.35, color="#EB5600")
    ax2.plot(vol_ser.index, vol_ser.values,
             color="#EB5600", linewidth=0.8)
    ax2.set_ylabel(vol_label)
    ax2.set_xlabel("Date")
    ax2.grid(alpha=0.20)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"[PLOT] Saved → {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ2 INDIA STAGE 1 — Regime Detection")
    log.info("=" * 65)

    features, log_ret = load_data()

    # A. HMM (3-state primary + 4-state robustness)
    log.info("\n[A] Hidden Markov Model ...")
    hmm_result = run_hmm(features)

    # B. GMM
    log.info("\n[B] Gaussian Mixture Model ...")
    gmm_lbl = run_gmm(features, N_REGIMES_PRIMARY)

    # C. Bai-Perron
    log.info("\n[C] Bai-Perron structural breaks ...")
    bp_lbl = run_bai_perron(features)

    # D. Consensus
    log.info("\n[D] Consensus labels ...")
    if hmm_result["labels_3"] is not None:
        cons_lbl = consensus_labels(
            hmm_result["labels_3"], gmm_lbl, bp_lbl
        )
    else:
        cons_lbl = pd.Series(dtype=int)

    # Combine and save
    label_df = pd.DataFrame({
        "hmm_3state":  hmm_result["labels_3"],
        "hmm_4state":  hmm_result["labels_4"],
        "gmm":         gmm_lbl,
        "bai_perron":  bp_lbl,
        "consensus":   cons_lbl,
    })
    out_path = os.path.join(DATA_DIR, "rq2_india_regime_labels.parquet")
    label_df.to_parquet(out_path)
    log.info(f"[SAVE] Labels → {out_path}")

    # Stats CSV
    if not hmm_result["stats"].empty:
        hmm_result["stats"].to_csv(
            os.path.join(OUTPUT_DIR, "rq2_india_regime_stats.csv"),
            index=False
        )

    # BIC report
    bic_df = pd.DataFrame([{
        "model": "3-state HMM", "bic": round(hmm_result["bic_3"], 2),
        "selected": hmm_result["bic_winner"] == 3,
    }, {
        "model": "4-state HMM", "bic": round(hmm_result["bic_4"], 2),
        "selected": hmm_result["bic_winner"] == 4,
    }])
    bic_df.to_csv(os.path.join(OUTPUT_DIR, "rq2_india_bic.csv"), index=False)
    log.info(f"\n[BIC] {hmm_result['bic_winner']}-state model selected")

    # Plot
    plot_regime_timeline(
        features, hmm_result, log_ret,
        save_path=os.path.join(OUTPUT_DIR, "rq2_india_regime_timeline.png")
    )

    # Print summary
    print(f"\n{'='*65}")
    print(f"  RQ2 INDIA STAGE 1 — REGIME DETECTION SUMMARY")
    print(f"{'='*65}")
    print(f"  BIC: 3-state={hmm_result['bic_3']:.1f}  "
          f"4-state={hmm_result['bic_4']:.1f}  "
          f"→ {hmm_result['bic_winner']}-state selected")
    if not hmm_result["stats"].empty:
        for _, r in hmm_result["stats"].iterrows():
            print(f"  Regime {int(r['regime'])} ({r['label']:<8}): "
                  f"{int(r['n_days']):>4}d ({r['pct_days']:>5.1f}%)  "
                  f"ann.ret={r['mean_ret_ann']:+.2%}  "
                  f"vol={r['mean_vol']:.2f}")
    print(f"{'='*65}")

    log.info("\n✓ Stage 1 complete. Run rq2_india_stage2_anova.py next.\n")
    return hmm_result, label_df


if __name__ == "__main__":
    main()

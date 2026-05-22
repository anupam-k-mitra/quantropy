#!/usr/bin/env python3
# =============================================================================
# rq3_stage3_ml.py
# Stage 3: ML Walk-Forward Comparison & Diebold-Mariano Test
#
# Core H0 test:
#   For each target asset, train TWO models in strict walk-forward:
#     Model 1 (null)        -- own-asset features only
#     Model 2 (alternative) -- own + cross-asset features
#
#   Diebold-Mariano (1995) test with HLN small-sample correction:
#     H0: E[d_t] = 0  where d_t = e1_t^2 - e2_t^2
#     H1: Model 2 significantly reduces MSFE  (one-sided, p < DM_ALPHA)
#
#   Also computes Campbell-Thompson OOS R^2.
#   SHAP importance identifies which cross-asset features drive predictions.
#
# Outputs
# -------
#   rq3_outputs/rq3_ml_results.csv
#   rq3_outputs/rq3_stage3_summary.txt
#   rq3_outputs/rq3_rmse_comparison.png
#   rq3_outputs/rq3_shap_heatmap.png
#   rq3_outputs/rq3_oos_errors.png
#
# Run: python rq3_stage3_ml.py
# =============================================================================

# ── ensure the script's own directory is on sys.path so that
#    scipy_compat.py and rq3_config.py are always found regardless
#    of which directory the user runs Python from ──────────────────
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
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
    DATA_DIR, OUTPUT_DIR, TARGET_TICKERS,
    TRAIN_WINDOW, TEST_WINDOW, SIGNAL_HORIZON, DM_ALPHA,
)
from scipy_compat import stats


# =============================================================================
# LOADER
# =============================================================================

def load_features(target: str):
    """Load own_only, cross_asset, and fwd_net for a target."""
    paths = {
        "own":   os.path.join(DATA_DIR, f"rq3_{target}_own_only.parquet"),
        "cross": os.path.join(DATA_DIR, f"rq3_{target}_cross_asset.parquet"),
        "fwd":   os.path.join(DATA_DIR, f"rq3_{target}_fwd_net.parquet"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run rq3_stage0_data.py first."
            )
    fwd_df = pd.read_parquet(paths["fwd"])
    # fwd_net was saved as a single-column DataFrame -- squeeze to Series
    if isinstance(fwd_df, pd.DataFrame):
        fwd_series = fwd_df.iloc[:, 0]
    else:
        fwd_series = fwd_df
    return (pd.read_parquet(paths["own"]),
            pd.read_parquet(paths["cross"]),
            fwd_series)


# =============================================================================
# DIEBOLD-MARIANO TEST
# =============================================================================

def diebold_mariano(e1: np.ndarray, e2: np.ndarray,
                     h: int = SIGNAL_HORIZON) -> dict:
    """
    Diebold & Mariano (1995) equal predictive accuracy test.
    Harvey, Leybourne & Newbold (1997) small-sample correction.

    d_t = L(e1_t) - L(e2_t)  where L = squared error loss
    H0: E[d_t] = 0
    H1: E[d_t] > 0  (model 2 is more accurate, one-sided)

    DM statistic = d_bar / sqrt(hat_V_d / n)
    HLN correction scales the variance estimate.
    """
    e1 = np.array(e1, float)
    e2 = np.array(e2, float)
    mask = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[mask], e2[mask]
    n = len(e1)

    empty = {"DM_stat": np.nan, "p_val": 1.0, "reject": False,
             "RMSE_1": np.nan, "RMSE_2": np.nan, "pct_reduction": np.nan}
    if n < 10:
        return empty

    d     = e1**2 - e2**2          # positive: model 1 is worse
    d_bar = d.mean()

    # Newey-West HAC variance of d
    gamma0 = np.var(d, ddof=1)
    gamma  = 0.0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag])[0, 1]
        gamma += 2.0 * cov
    var_d = (gamma0 + gamma) / n

    if var_d <= 0:
        return empty

    # HLN small-sample adjustment
    hln   = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    DM    = float(d_bar / np.sqrt(var_d) * hln)

    try:
        p_val = float(stats.norm.sf(DM))   # one-sided: d_bar > 0
    except Exception:
        p_val = float(np.exp(-DM**2 / 2) / 2) if DM > 0 else 0.5

    rmse1 = float(np.sqrt(np.mean(e1**2)))
    rmse2 = float(np.sqrt(np.mean(e2**2)))
    pct   = (rmse1 - rmse2) / rmse1 * 100 if rmse1 > 0 else 0.0

    return {
        "DM_stat":      round(DM,    4),
        "p_val":        round(p_val, 6),
        "reject":       DM > 0 and p_val < DM_ALPHA,
        "RMSE_1":       round(rmse1, 8),
        "RMSE_2":       round(rmse2, 8),
        "pct_reduction": round(pct, 3),
    }


# =============================================================================
# OOS R-SQUARED (Campbell & Thompson 2008)
# =============================================================================

def oos_r2(errors: np.ndarray, actuals: np.ndarray) -> float:
    """
    OOS R² = 1 - MSPE_model / MSPE_benchmark
    Benchmark: historical mean (predict zero return).
    Positive -> model beats the mean; negative -> worse than mean.
    """
    mspe_m = float(np.mean(errors**2))
    mspe_b = float(np.mean(actuals**2))
    return round(1.0 - mspe_m / mspe_b, 6) if mspe_b > 1e-15 else np.nan


# =============================================================================
# WALK-FORWARD PREDICTION ENGINE
# =============================================================================

def walkforward(X: pd.DataFrame, y: pd.Series,
                 model_type: str = "ridge") -> tuple:
    """
    Strict walk-forward: train on past TRAIN_WINDOW days,
    predict next TEST_WINDOW days. No future data ever used.

    Supported model_type values: 'ridge', 'xgboost'
    Falls back to GradientBoostingRegressor if xgboost unavailable.

    Returns (errors, actuals, pred_dates).
    """
    # Align
    common = X.index.intersection(y.index)
    X = X.loc[common].fillna(0.0)
    y = y.loc[common]

    dates = X.index
    n     = len(dates)

    all_err, all_act, all_idx = [], [], []

    for start in range(TRAIN_WINDOW, n - TEST_WINDOW, TEST_WINDOW):
        Xtr = X.iloc[start - TRAIN_WINDOW: start]
        ytr = y.iloc[start - TRAIN_WINDOW: start]
        Xte = X.iloc[start: start + TEST_WINDOW]
        yte = y.iloc[start: start + TEST_WINDOW]

        # Drop NaN rows
        tr_mask = ytr.notna() & Xtr.notna().all(axis=1)
        te_mask = yte.notna() & Xte.notna().all(axis=1)
        if tr_mask.sum() < 50 or te_mask.sum() < 1:
            continue

        Xtr_c, ytr_c = Xtr[tr_mask], ytr[tr_mask]
        Xte_c, yte_c = Xte[te_mask], yte[te_mask]

        sc    = StandardScaler()
        Xtr_s = sc.fit_transform(Xtr_c)
        Xte_s = sc.transform(Xte_c)

        if model_type == "ridge":
            m = Ridge(alpha=1.0)
            m.fit(Xtr_s, ytr_c)
            phat = m.predict(Xte_s)

        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                m = xgb.XGBRegressor(
                    n_estimators=200, max_depth=4,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.7,
                    random_state=42, verbosity=0, n_jobs=-1,
                )
                m.fit(Xtr_s, ytr_c,
                      eval_set=[(Xte_s, yte_c)],
                      verbose=False)
                phat = m.predict(Xte_s)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(
                    n_estimators=200, max_depth=3,
                    learning_rate=0.05, subsample=0.8,
                    random_state=42,
                )
                m.fit(Xtr_s, ytr_c)
                phat = m.predict(Xte_s)
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        errors = yte_c.values - phat
        all_err.extend(errors.tolist())
        all_act.extend(yte_c.values.tolist())
        all_idx.extend(yte_c.index.tolist())

    return (np.array(all_err,  float),
            np.array(all_act,  float),
            pd.DatetimeIndex(all_idx))


# =============================================================================
# SHAP / FEATURE IMPORTANCE
# =============================================================================

def get_feature_importance(X_train: pd.DataFrame,
                             y_train: pd.Series,
                             feature_names: list,
                             top_n: int = 20) -> pd.Series:
    """
    SHAP values (if xgboost+shap available) or
    GBM feature_importances_ as fallback.
    """
    sc  = StandardScaler()
    Xs  = sc.fit_transform(X_train.fillna(0))
    yv  = y_train.values

    try:
        import xgboost as xgb
        import shap
        m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                              random_state=42, verbosity=0)
        m.fit(Xs, yv)
        expl = shap.TreeExplainer(m)
        sv   = expl.shap_values(Xs)
        imp  = pd.Series(np.abs(sv).mean(axis=0),
                         index=feature_names)
    except ImportError:
        from sklearn.ensemble import GradientBoostingRegressor
        m = GradientBoostingRegressor(n_estimators=200, random_state=42)
        m.fit(Xs, yv)
        imp = pd.Series(m.feature_importances_, index=feature_names)
    except Exception:
        imp = pd.Series(np.zeros(len(feature_names)),
                        index=feature_names)

    return imp.sort_values(ascending=False).head(top_n)


# =============================================================================
# PER-TARGET ANALYSIS
# =============================================================================

def analyse_target(target: str) -> dict:
    """
    Run the full own-vs-cross walk-forward comparison for one target.
    Returns a results dict with DM stats, RMSE, R², SHAP.
    """
    log.info(f"\n{'='*50}")
    log.info(f"  Target: {target}")

    X_own, X_cross, y = load_features(target)
    results = {}

    for model_name in ["ridge", "xgboost"]:
        log.info(f"\n  Model: {model_name.upper()}")

        err_own,  act_own,  _     = walkforward(X_own,   y, model_name)
        err_crs,  act_crs,  idx_c = walkforward(X_cross, y, model_name)

        # Align arrays to same length
        n = min(len(err_own), len(err_crs))
        e1, e2, act = err_own[-n:], err_crs[-n:], act_crs[-n:]

        dm   = diebold_mariano(e1, e2, h=SIGNAL_HORIZON)
        r2_1 = oos_r2(e1, act)
        r2_2 = oos_r2(e2, act)

        log.info(f"    RMSE own  = {dm['RMSE_1']:.7f}")
        log.info(f"    RMSE cross= {dm['RMSE_2']:.7f}  "
                 f"(-{dm['pct_reduction']:.2f}%)")
        log.info(f"    DM stat   = {dm['DM_stat']:.3f}  "
                 f"p = {dm['p_val']:.5f}  "
                 f"{'REJECT H0' if dm['reject'] else 'fail'}")
        log.info(f"    OOS R2 own   = {r2_1:.5f}")
        log.info(f"    OOS R2 cross = {r2_2:.5f}  "
                 f"delta = {r2_2 - r2_1:+.5f}")

        results[model_name] = {
            "RMSE_own":        dm["RMSE_1"],
            "RMSE_cross":      dm["RMSE_2"],
            "pct_reduction":   dm["pct_reduction"],
            "DM_stat":         dm["DM_stat"],
            "DM_p_val":        dm["p_val"],
            "DM_reject":       dm["DM_reject"] if "DM_reject" in dm else dm["reject"],
            "R2_own":          r2_1,
            "R2_cross":        r2_2,
            "R2_delta":        round(r2_2 - r2_1, 6),
            "errors_own":      e1,
            "errors_cross":    e2,
            "idx":             idx_c[-n:],
        }

    # SHAP on last training block (cross model only)
    try:
        common = X_cross.index.intersection(y.index)
        Xlast  = X_cross.loc[common].fillna(0).iloc[-TRAIN_WINDOW:]
        ylast  = y.loc[common].iloc[-TRAIN_WINDOW:].dropna()
        Xlast  = Xlast.loc[ylast.index]
        shap_imp = get_feature_importance(
            Xlast, ylast, X_cross.columns.tolist()
        )
        results["shap_importance"] = shap_imp
    except Exception as e:
        log.debug(f"  SHAP failed: {e}")

    return results


# =============================================================================
# VISUALISATION
# =============================================================================

def plot_rmse_comparison(all_results: dict, save_path: str = None):
    """
    Side-by-side bar chart: own vs cross RMSE per target.
    Star marks DM-significant improvements.
    """
    targets = list(all_results.keys())
    models  = ["ridge", "xgboost"]
    n_t     = len(targets)

    fig, axes = plt.subplots(1, len(models), figsize=(13, 5))
    if len(models) == 1:
        axes = [axes]
    fig.suptitle("RQ3 -- OOS RMSE: Own-Asset vs Cross-Asset",
                 fontsize=13, fontweight="bold")

    for ax, model in zip(axes, models):
        x    = np.arange(n_t)
        r_o, r_c, dm_r = [], [], []
        for t in targets:
            m = all_results.get(t, {}).get(model, {})
            r_o.append(m.get("RMSE_own",   0) or 0)
            r_c.append(m.get("RMSE_cross", 0) or 0)
            dm_r.append(bool(m.get("DM_reject", False)))

        w  = 0.35
        ax.bar(x - w/2, r_o, w, label="Own-asset only",
               color="#AAAAAA", edgecolor="white")
        ax.bar(x + w/2, r_c, w, label="Cross-asset",
               color="#2E5FAC", edgecolor="white")

        for i, (ro, rc, sig) in enumerate(zip(r_o, r_c, dm_r)):
            if sig and rc < ro:
                ax.text(i + w/2, max(ro, rc) * 1.02, "*",
                        ha="center", fontsize=14,
                        color="#C0392B", fontweight="bold")

        ax.set_xticks(x)
        ax.set_xticklabels(targets, fontsize=9)
        ax.set_title(model.upper(), fontsize=11, fontweight="bold")
        ax.set_ylabel("OOS RMSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")

    fig.text(0.5, -0.02,
             "* = DM-significant improvement (p < {:.2f})".format(DM_ALPHA),
             ha="center", fontsize=9, color="#C0392B")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


def plot_shap_heatmap(all_results: dict, save_path: str = None):
    """
    Heatmap: rows = target assets, cols = top cross-asset features.
    Reveals which assets most inform which targets.
    """
    shap_data = {t: r["shap_importance"]
                 for t, r in all_results.items()
                 if "shap_importance" in r}
    if not shap_data:
        return

    all_feats = set()
    for s in shap_data.values():
        all_feats.update(s.index.tolist())
    cross_feats = [f for f in all_feats if not f.startswith("own_")][:30]
    if not cross_feats:
        return

    mat = pd.DataFrame(0.0, index=list(shap_data.keys()),
                       columns=cross_feats)
    for target, s in shap_data.items():
        for f in cross_feats:
            mat.loc[target, f] = float(s.get(f, 0.0))

    fig, ax = plt.subplots(figsize=(16, max(3, len(mat) * 0.8)))
    im = ax.imshow(mat.values, cmap="Blues", aspect="auto", vmin=0)
    plt.colorbar(im, ax=ax, label="Mean |SHAP|")
    ax.set_yticks(range(len(mat)))
    ax.set_yticklabels(mat.index.tolist(), fontsize=9)
    ax.set_xticks(range(len(mat.columns)))
    ax.set_xticklabels(mat.columns.tolist(), fontsize=7,
                       rotation=45, ha="right")
    ax.set_title("Cross-Asset SHAP Importance by Target",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved -> {save_path}")
    plt.close()


def plot_oos_errors(all_results: dict, save_path: str = None):
    """
    Cumulative squared error difference (own - cross) over time.
    Positive and rising -> cross-asset model gains over time.
    """
    targets = list(all_results.keys())
    n_t     = len(targets)
    if n_t == 0:
        return

    fig, axes = plt.subplots(1, n_t, figsize=(5 * n_t, 4), sharey=False)
    if n_t == 1:
        axes = [axes]
    fig.suptitle("Cumulative Error Reduction: Own - Cross (Ridge)\n"
                 "Rising = cross-asset model gaining",
                 fontsize=12, fontweight="bold")

    for ax, target in zip(axes, targets):
        m = all_results[target].get("ridge", {})
        e1 = m.get("errors_own",   np.array([]))
        e2 = m.get("errors_cross", np.array([]))
        idx = m.get("idx",         None)
        if len(e1) == 0 or len(e2) == 0:
            continue
        n   = min(len(e1), len(e2))
        d   = e1[-n:]**2 - e2[-n:]**2
        cum = np.cumsum(d)
        x   = range(len(cum))

        ax.plot(x, cum, linewidth=1.5, color="#2E5FAC")
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.fill_between(x, cum, 0,
                        where=(cum > 0), alpha=0.15, color="#2E5FAC",
                        label="Cross wins")
        ax.fill_between(x, cum, 0,
                        where=(cum < 0), alpha=0.15, color="#C0392B",
                        label="Own wins")
        ax.set_title(target, fontsize=11, fontweight="bold")
        ax.set_xlabel("OOS Fold")
        ax.set_ylabel("Cum. error reduction" if target == targets[0] else "")
        ax.legend(fontsize=7)
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
    log.info("  RQ3 STAGE 3 -- ML Walk-Forward & Diebold-Mariano")
    log.info("=" * 65)

    all_results = {}
    for target in TARGET_TICKERS:
        try:
            all_results[target] = analyse_target(target)
        except FileNotFoundError as e:
            log.warning(f"  Skipping {target}: {e}")
        except Exception as e:
            log.warning(f"  {target} failed: {e}")
            import traceback
            log.debug(traceback.format_exc())

    if not all_results:
        log.error("No results. Run rq3_stage0_data.py first.")
        return {}

    # Build summary table
    rows = []
    for target, res in all_results.items():
        for model in ["ridge", "xgboost"]:
            m = res.get(model, {})
            if not isinstance(m, dict) or "DM_stat" not in m:
                continue
            rows.append({
                "target":          target,
                "model":           model,
                "RMSE_own":        m.get("RMSE_own",       np.nan),
                "RMSE_cross":      m.get("RMSE_cross",     np.nan),
                "pct_reduction":   m.get("pct_reduction",  np.nan),
                "DM_stat":         m.get("DM_stat",        np.nan),
                "DM_p_val":        m.get("DM_p_val",       np.nan),
                "DM_reject":       m.get("DM_reject",      False),
                "R2_own":          m.get("R2_own",         np.nan),
                "R2_cross":        m.get("R2_cross",       np.nan),
                "R2_delta":        m.get("R2_delta",       np.nan),
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "rq3_ml_results.csv"), index=False)

    log.info(f"\n{df.to_string(index=False)}")

    n_reject = int(df["DM_reject"].sum()) if not df.empty else 0
    h1_dm    = n_reject > 0

    lines = [
        "=" * 65,
        "  RQ3 STAGE 3 RESULTS  --  Diebold-Mariano Test",
        "=" * 65,
        f"  Targets x models tested   : {len(df)}",
        f"  DM-reject (p < {DM_ALPHA}) : {n_reject}",
        f"  H1 (DM condition)         : "
        f"{'CONFIRMED' if h1_dm else 'NOT confirmed'}",
        "=" * 65,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "rq3_stage3_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # Plots
    plot_rmse_comparison(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, "rq3_rmse_comparison.png"),
    )
    plot_shap_heatmap(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, "rq3_shap_heatmap.png"),
    )
    plot_oos_errors(
        all_results,
        save_path=os.path.join(OUTPUT_DIR, "rq3_oos_errors.png"),
    )

    log.info("\n✓ Stage 3 complete. Run rq3_stage4_synthesis.py next.\n")
    return all_results


def main_with_transformer():
    """
    Extended main that runs Ridge + XGBoost then the TFT Transformer.
    Invoke with: python rq3_stage3_ml.py --transformer
    """
    ridge_xgb = main()

    log.info("\n" + "="*65)
    log.info("  RQ3 STAGE 3 — Adding Transformer / Attention Model")
    log.info("="*65)
    try:
        from rq3_stage3_transformer import run_transformer_stage
        transformer_results = run_transformer_stage(TARGET_TICKERS)
    except ImportError:
        log.error("rq3_stage3_transformer.py not found in the same directory.")
        log.error("Place it alongside rq3_stage3_ml.py and retry.")
    except Exception as e:
        log.error(f"Transformer stage failed: {e}")
        import traceback
        log.debug(traceback.format_exc())

    return ridge_xgb


if __name__ == "__main__":
    import sys
    if "--transformer" in sys.argv:
        main_with_transformer()
    else:
        main()

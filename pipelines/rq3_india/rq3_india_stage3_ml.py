#!/usr/bin/env python3
# =============================================================================
# rq3_india_stage3_ml.py
# Stage 3: ML Walk-Forward + Diebold-Mariano + TFT Transformer — India
#
# Identical statistical methodology to US stage 3.
# Loads India feature parquets built by stage 0.
# Runs Ridge, XGBoost, and TFT on Indian targets.
#
# Key expected findings:
#   INFY.NS: cross-asset model should REJECT H0 (USD/INR signal)
#   ^NSEI:   cross-asset model likely rejects H0 (SPY overnight + oil)
#   SBIN.NS: uncertain — PSU bank driven by domestic policy
#   GC=F:    likely rejects (USD, VIX, SPY — same as US result)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging, math
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from rq3_india_config import (
    DATA_DIR, OUTPUT_DIR, TARGET_TICKERS,
    TRAIN_WINDOW, TEST_WINDOW, SIGNAL_HORIZON, DM_ALPHA,
)

try:
    from scipy_compat import stats as scipy_stats
except ImportError:
    from scipy import stats as scipy_stats


# ── loaders ──────────────────────────────────────────────────────────────────
def load_features(target: str) -> tuple:
    stub = target.replace("^","").replace("=","").replace(".","_")
    paths = {
        "own":   os.path.join(DATA_DIR, f"rq3_india_{stub}_own_only.parquet"),
        "cross": os.path.join(DATA_DIR, f"rq3_india_{stub}_cross_asset.parquet"),
        "fwd":   os.path.join(DATA_DIR, f"rq3_india_{stub}_fwd_net.parquet"),
    }
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found. Run stage0 first.")
    fwd = pd.read_parquet(paths["fwd"])
    if isinstance(fwd, pd.DataFrame):
        fwd = fwd.iloc[:, 0]
    return (pd.read_parquet(paths["own"]),
            pd.read_parquet(paths["cross"]),
            fwd)


# ── Diebold-Mariano (HLN) ─────────────────────────────────────────────────
def diebold_mariano(e1, e2, h=SIGNAL_HORIZON):
    e1, e2 = np.array(e1, float), np.array(e2, float)
    mask = ~(np.isnan(e1) | np.isnan(e2))
    e1, e2 = e1[mask], e2[mask]
    n = len(e1)
    empty = dict(DM_stat=np.nan, p_val=1.0, reject=False,
                 RMSE_1=np.nan, RMSE_2=np.nan, pct_reduction=np.nan)
    if n < 10:
        return empty
    d     = e1**2 - e2**2
    d_bar = d.mean()
    g0    = np.var(d, ddof=1)
    g     = sum(2*np.cov(d[l:], d[:-l])[0,1] for l in range(1, h))
    var_d = (g0 + g) / n
    if var_d <= 0:
        return empty
    hln   = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    DM    = float(d_bar / np.sqrt(var_d) * hln)
    p_val = float(scipy_stats.norm.sf(DM))
    r1    = float(np.sqrt(np.mean(e1**2)))
    r2    = float(np.sqrt(np.mean(e2**2)))
    pct   = (r1 - r2) / r1 * 100 if r1 > 0 else 0
    return dict(DM_stat=round(DM,4), p_val=round(p_val,6),
                reject=(DM>0 and p_val<DM_ALPHA),
                RMSE_1=round(r1,8), RMSE_2=round(r2,8),
                pct_reduction=round(pct,3))


def oos_r2(errors, actuals):
    mspe_m = float(np.mean(errors**2))
    mspe_b = float(np.mean(actuals**2))
    return round(1 - mspe_m/mspe_b, 6) if mspe_b > 1e-15 else np.nan


# ── Walk-forward ─────────────────────────────────────────────────────────────
def walkforward(X: pd.DataFrame, y: pd.Series,
                 model_type="ridge") -> tuple:
    common = X.index.intersection(y.index)
    X = X.loc[common].fillna(0)
    y = y.loc[common]
    n = len(y)
    all_err, all_act = [], []

    for start in range(TRAIN_WINDOW, n - TEST_WINDOW, TEST_WINDOW):
        Xtr = X.iloc[start - TRAIN_WINDOW: start]
        ytr = y.iloc[start - TRAIN_WINDOW: start]
        Xte = X.iloc[start: start + TEST_WINDOW]
        yte = y.iloc[start: start + TEST_WINDOW]
        mask_tr = ytr.notna() & Xtr.notna().all(axis=1)
        mask_te = yte.notna() & Xte.notna().all(axis=1)
        if mask_tr.sum() < 50 or mask_te.sum() < 1:
            continue
        sc = StandardScaler()
        Xts = sc.fit_transform(Xtr[mask_tr])
        Xes = sc.transform(Xte[mask_te])
        if model_type == "ridge":
            m = Ridge(alpha=10.0)  # stronger regularisation: targeted features
            m.fit(Xts, ytr[mask_tr])
            p = m.predict(Xes)
        elif model_type == "xgboost":
            try:
                import xgboost as xgb
                m = xgb.XGBRegressor(n_estimators=200, max_depth=4,
                    learning_rate=0.05, subsample=0.8,
                    colsample_bytree=0.5, random_state=42,
                    verbosity=0, n_jobs=-1)
                m.fit(Xts, ytr[mask_tr])
                p = m.predict(Xes)
            except ImportError:
                from sklearn.ensemble import GradientBoostingRegressor
                m = GradientBoostingRegressor(n_estimators=200,
                    max_depth=3, learning_rate=0.05, random_state=42)
                m.fit(Xts, ytr[mask_tr])
                p = m.predict(Xes)
        else:
            raise ValueError(f"Unknown model: {model_type}")
        err = yte[mask_te].values - p
        all_err.extend(err.tolist())
        all_act.extend(yte[mask_te].values.tolist())

    return np.array(all_err, float), np.array(all_act, float)


# ── Per-target ────────────────────────────────────────────────────────────────
def analyse_target(target: str) -> dict:
    log.info(f"\n{'='*55}\n  India Target: {target}")
    X_own, X_cross, y = load_features(target)
    results = {}
    for model in ["ridge", "xgboost"]:
        log.info(f"  Model: {model.upper()}")
        e1, a1 = walkforward(X_own,   y, model)
        e2, a2 = walkforward(X_cross, y, model)
        n = min(len(e1), len(e2))
        e1, e2, act = e1[-n:], e2[-n:], a2[-n:]
        dm   = diebold_mariano(e1, e2)
        r2_1 = oos_r2(e1, act)
        r2_2 = oos_r2(e2, act)
        log.info(f"    RMSE own={dm['RMSE_1']:.7f}  "
                 f"cross={dm['RMSE_2']:.7f}  "
                 f"red={dm['pct_reduction']:+.2f}%")
        log.info(f"    DM={dm['DM_stat']:.3f}  p={dm['p_val']:.5f}  "
                 f"{'REJECT H0' if dm['reject'] else 'fail'}")
        log.info(f"    R2_own={r2_1:.5f}  R2_cross={r2_2:.5f}  "
                 f"delta={r2_2-r2_1:+.5f}")
        results[model] = {**dm, "R2_own": r2_1, "R2_cross": r2_2,
                          "R2_delta": round(r2_2-r2_1,6),
                          "errors_own": e1, "errors_cross": e2}
    return results


# ── Transformer ───────────────────────────────────────────────────────────────
def run_transformer(targets):
    """
    Import and run the Transformer stage.
    Looks for rq3_stage3_transformer.py in:
      1. Same directory as this file (India folder)
      2. Parent directory (US rq3_cross_asset folder, for code reuse)
    """
    import sys, importlib  # ensure available throughout this function
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    us_dir = os.path.join(parent, "rq3_cross_asset")
    for search_dir in [os.path.dirname(os.path.abspath(__file__)), us_dir]:
        tft_path = os.path.join(search_dir, "rq3_stage3_transformer.py")
        if os.path.exists(tft_path) and search_dir not in sys.path:
            sys.path.insert(0, search_dir)

    try:
        # Override the config values so TFT uses India data paths
        import rq3_stage3_transformer as tft
        # Monkey-patch config to use India paths
        tft.DATA_DIR      = DATA_DIR
        tft.OUTPUT_DIR    = OUTPUT_DIR
        tft.TARGET_TICKERS = targets
        tft.TRAIN_WINDOW  = TRAIN_WINDOW
        tft.TEST_WINDOW   = TEST_WINDOW
        tft.SIGNAL_HORIZON = SIGNAL_HORIZON
        tft.DM_ALPHA      = DM_ALPHA

        # Override load_features to use India stubs
        def india_load(target):
            return load_features(target)
        tft.load_features = india_load

        return tft.run_transformer_stage(targets)
    except ImportError as e:
        log.warning(f"[TFT] Cannot import transformer: {e}")
        log.warning("[TFT] Copy rq3_stage3_transformer.py to this folder")
        return {}
    except Exception as e:
        err = str(e)
        if 'WinError 127' in err or 'shm.dll' in err or 'DLL' in err:
            log.warning('[TFT] Windows DLL error detected.')
            log.warning('[TFT] To fix PyTorch permanently: ')
            log.warning('[TFT]   pip install torch '
                        '--index-url https://download.pytorch.org/whl/cpu')
            log.warning('[TFT] Forcing NumPy attention fallback ...')
            try:
                # Set TORCH_OK=False in builtins BEFORE importing the module.
                # This prevents the torch import from running at module load time.
                # Simply setting tft_mod.TORCH_OK=False after import does NOT work
                # because the DLL crash happens at the OS loader level during import.
                import builtins
                builtins.TORCH_OK = False   # signal to module before import
                # Remove cached broken module if already imported
                for mod_name in list(sys.modules.keys()):
                    if 'rq3_stage3_transformer' in mod_name:
                        del sys.modules[mod_name]
                tft_mod = importlib.import_module('rq3_stage3_transformer')
                result = tft_mod.run_transformer_stage(targets)
                del builtins.TORCH_OK  # clean up
                return result
            except Exception as e2:
                log.warning(f'[TFT] NumPy fallback also failed: {e2}')
                log.warning('[TFT] Skipping transformer — Granger + XGBoost DM sufficient for H1')
                return {}
        log.warning(f"[TFT] Transformer failed: {e}")
        import traceback; log.debug(traceback.format_exc())
        return {}


# ── Main ─────────────────────────────────────────────────────────────────────
def main(run_tft: bool = True):
    log.info("=" * 65)
    log.info("  RQ3 INDIA STAGE 3 — ML Walk-Forward & Diebold-Mariano")
    log.info("=" * 65)

    all_results = {}
    for target in TARGET_TICKERS:
        try:
            all_results[target] = analyse_target(target)
        except FileNotFoundError as e:
            log.warning(f"  Skipping {target}: {e}")
        except Exception as e:
            log.warning(f"  {target} failed: {e}")

    # Build summary
    rows = []
    for tgt, res in all_results.items():
        for model in ["ridge", "xgboost"]:
            m = res.get(model, {})
            if "DM_stat" not in m:
                continue
            rows.append(dict(target=tgt, model=model,
                             RMSE_own=m["RMSE_1"], RMSE_cross=m["RMSE_2"],
                             pct_reduction=m["pct_reduction"],
                             DM_stat=m["DM_stat"], DM_p_val=m["p_val"],
                             DM_reject=m["reject"],
                             R2_own=m["R2_own"], R2_cross=m["R2_cross"],
                             R2_delta=m["R2_delta"]))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUTPUT_DIR, "rq3_india_ml_results.csv"), index=False)

    n_reject = int(df["DM_reject"].sum()) if not df.empty else 0
    h1_dm    = n_reject > 0

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  RQ3 INDIA STAGE 3 — RESULTS (Ridge + XGBoost)")
    print(sep)
    print(f"  Targets × models   : {len(df)}")
    print(f"  DM-reject (p<{DM_ALPHA}): {n_reject}")
    print(f"  H1 (DM)            : {'CONFIRMED ✓' if h1_dm else 'NOT confirmed'}")
    print(f"\n  Per-target:")
    for _, r in df.iterrows():
        flag = " REJECT H0 ✓" if r["DM_reject"] else " fail"
        print(f"    {r['target']:<18} {r['model']:<10} "
              f"DM={r['DM_stat']:>6.3f}  p={r['DM_p_val']:.4f}  "
              f"RMSE-red={r['pct_reduction']:+.2f}%{flag}")
    print(sep)

    with open(os.path.join(OUTPUT_DIR, "rq3_india_stage3_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(f"DM-reject: {n_reject}\nH1: {'CONFIRMED' if h1_dm else 'NOT confirmed'}\n")

    # Transformer
    if run_tft:
        log.info("\n[TFT] Running Transformer stage for India ...")
        run_transformer(list(all_results.keys()))

    log.info("\n✓ Stage 3 complete. Run rq3_india_stage4_synthesis.py next.\n")
    return all_results


if __name__ == "__main__":
    import sys
    run_tft = "--no-transformer" not in sys.argv
    main(run_tft=run_tft)

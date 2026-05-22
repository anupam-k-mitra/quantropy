#!/usr/bin/env python3
# =============================================================================
# rq2_india_stage0_data.py — Data download and regime feature engineering
#
# India-specific features (6 variables for HMM input):
#   V1: ^NSEI daily log return       — primary state variable
#   V2: India VIX or 21d realised vol — vol regime (forward vs backward)
#   V3: USDINR=X 5d return           — INR stress (no US equivalent)
#   V4: SPY 21d return               — global coupling / FII proxy
#   V5: GC=F 21d return              — gold demand channel
#   V6: CL=F 21d return              — crude import dependency
# =============================================================================
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from rq2_india_config import (
    TICKERS, START_DATE, END_DATE, DATA_DIR, OUTPUT_DIR,
    TURBULENCE_TICKERS, TURBULENCE_LOOKBACK, TURBULENCE_CRISIS_PCT,
    INDIA_BREAKS,
)


# =============================================================================
# 1. DOWNLOAD — grouped for reliability
# =============================================================================

def _dl_group(tickers, start, end, label, delay=0.3):
    """Download a small group with individual-ticker fallback."""
    import time
    import yfinance as yf

    log.info(f"[DATA] Downloading group '{label}': {tickers}")
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False,
                          threads=False, timeout=30)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) \
                     else raw.iloc[:, :len(tickers)]
        else:
            prices = raw
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
    except Exception as e:
        log.warning(f"[DATA] Batch failed for '{label}': {e}")
        prices = pd.DataFrame()

    results = {}
    for t in tickers:
        col = prices[t] if t in prices.columns else None
        if col is not None and col.notna().mean() > 0.50:
            results[t] = col
            log.info(f"[DATA]   {t:<20}: OK ({col.notna().sum()} days)")
        else:
            time.sleep(delay)
            try:
                ind = yf.download(t, start=start, end=end,
                                   auto_adjust=True, progress=False, timeout=30)
                if isinstance(ind.columns, pd.MultiIndex):
                    ind = ind["Close"] if "Close" in ind.columns.get_level_values(0) \
                          else ind.iloc[:, 0]
                elif "Close" in ind.columns:
                    ind = ind["Close"]
                else:
                    ind = ind.iloc[:, 0]
                if isinstance(ind, pd.DataFrame):
                    ind = ind.squeeze()
                if ind.notna().mean() > 0.30:
                    results[t] = ind
                    log.info(f"[DATA]   {t:<20}: individual OK ({ind.notna().sum()} days)")
                else:
                    log.warning(f"[DATA]   {t:<20}: FAILED")
            except Exception as e2:
                log.warning(f"[DATA]   {t:<20}: {e2}")

    return pd.concat(results, axis=1) if results else pd.DataFrame()


def download_prices():
    cache = os.path.join(DATA_DIR, "prices_india_rq2.parquet")
    if os.path.exists(cache):
        cached = pd.read_parquet(cache)
        n_valid = (cached.isna().mean() < 0.50).sum()
        if n_valid >= 4:
            log.info(f"[DATA] Loading cache ({n_valid} valid cols): {cache}")
            return cached
        log.warning("[DATA] Corrupt cache — re-downloading")
        os.remove(cache)

    groups = [
        (["^NSEI", "^NSEBANK"],                    "NSE indices"),
        (["INFY.NS", "HDFCBANK.NS"],               "NSE equities"),
        (["SPY"],                                   "Global ETF"),
        (["USDINR=X"],                              "FX"),
        (["GC=F", "CL=F"],                         "Futures"),
        (["^INDIAVIX"],                             "India VIX"),
    ]
    dfs = [_dl_group(t, START_DATE, END_DATE, l)
           for t, l in groups]
    dfs = [d for d in dfs if not d.empty]

    prices = dfs[0]
    for d in dfs[1:]:
        prices = prices.join(d, how="outer")

    prices = prices.ffill(limit=3)
    prices.index.name = "Date"

    n_valid = (prices.isna().mean() < 0.50).sum()
    log.info(f"[DATA] Combined: {prices.shape}  valid cols: {n_valid}")
    prices.to_parquet(cache)
    return prices


def validate_prices(prices):
    prices = prices.replace(0, np.nan).ffill(limit=3)
    miss   = prices.isna().mean()
    drop   = miss[miss > 0.20].index.tolist()
    if drop:
        log.warning(f"[QC] Dropping >20% missing: {drop}")
        prices = prices.drop(columns=drop, errors="ignore")
    log.info(f"[QC] Clean: {prices.shape}  cols: {prices.columns.tolist()}")
    return prices.dropna(how="all")


# =============================================================================
# 2. REGIME FEATURE ENGINEERING
# =============================================================================

def build_regime_features(prices):
    """
    Build the 6-variable HMM input matrix plus additional diagnostic features.

    Core 6 (always present):
      nsei_ret_1d   : daily log return  (V1)
      vol_primary   : India VIX or 21d realised vol  (V2)
      usdinr_5d     : 5-day USD/INR return  (V3)
      spy_21d       : 21-day SPY return  (V4)
      gold_21d      : 21-day GC=F return  (V5)
      crude_21d     : 21-day CL=F return  (V6)

    Additional diagnostics (not used in HMM but saved for ANOVA):
      nsei_vol_21d  : realised vol (even if India VIX is primary)
      nsei_mom_63d  : 3-month Nifty momentum
      it_bank_spread: INFY vs NSEBANK — sector rotation signal
      nsei_dd       : rolling drawdown (regime confirmation)
    """
    log.info("[FEATURES] Building India regime features ...")
    feat  = pd.DataFrame(index=prices.index)
    nsei  = prices["^NSEI"] if "^NSEI" in prices.columns else None
    if nsei is None:
        raise RuntimeError("^NSEI not in prices — cannot build features")

    r_nsei = np.log(nsei / nsei.shift(1))

    # ── V1: Primary return ────────────────────────────────────────────
    feat["nsei_ret_1d"]  = r_nsei
    feat["nsei_ret_5d"]  = np.log(nsei / nsei.shift(5))
    feat["nsei_ret_21d"] = np.log(nsei / nsei.shift(21))
    feat["nsei_mom_63d"] = np.log(nsei / nsei.shift(63))

    # ── V2: Volatility — India VIX preferred, realised vol fallback ───
    ivix_ok = ("^INDIAVIX" in prices.columns
                and prices["^INDIAVIX"].notna().mean() > 0.80)
    if ivix_ok:
        ivix = prices["^INDIAVIX"]
        feat["vol_primary"]  = ivix
        feat["vol_chg_5d"]   = ivix.pct_change(5)
        feat["vol_zscore"]   = ((ivix - ivix.rolling(252).mean())
                                / ivix.rolling(252).std())
        feat["vol_regime"]   = pd.cut(ivix,
                                       bins=[0, 15, 25, 40, 9999],
                                       labels=[0, 1, 2, 3]).astype(float)
        log.info("[FEATURES] V2: India VIX (^INDIAVIX) — forward-looking ✓")
    else:
        rvol = r_nsei.rolling(21).std() * np.sqrt(252)
        feat["vol_primary"]  = rvol
        feat["vol_chg_5d"]   = rvol.pct_change(5)
        feat["vol_zscore"]   = ((rvol - rvol.rolling(252).mean())
                                / rvol.rolling(252).std())
        log.info("[FEATURES] V2: 21-day realised vol (India VIX unavailable)")

    # Always compute realised vol for ANOVA diagnostics
    feat["nsei_vol_21d"] = r_nsei.rolling(21).std() * np.sqrt(252)
    feat["nsei_vol_63d"] = r_nsei.rolling(63).std() * np.sqrt(252)
    feat["vol_ratio"]    = (feat["nsei_vol_21d"]
                            / feat["nsei_vol_63d"].replace(0, np.nan))

    # ── V3: USD/INR ───────────────────────────────────────────────────
    if "USDINR=X" in prices.columns:
        r_fx = np.log(prices["USDINR=X"] / prices["USDINR=X"].shift(1))
        feat["usdinr_5d"]    = r_fx.rolling(5).sum()   # V3
        feat["usdinr_21d"]   = r_fx.rolling(21).sum()
        feat["usdinr_trend"] = (prices["USDINR=X"].rolling(10).mean()
                                / prices["USDINR=X"].rolling(50).mean() - 1)
        feat["usdinr_vol"]   = r_fx.rolling(21).std() * np.sqrt(252)
        log.info("[FEATURES] V3: USDINR=X ✓")
    else:
        log.warning("[FEATURES] V3: USDINR=X not available")
        feat["usdinr_5d"] = np.nan

    # ── V4: SPY (global coupling / FII proxy) ─────────────────────────
    if "SPY" in prices.columns:
        r_spy = np.log(prices["SPY"] / prices["SPY"].shift(1))
        feat["spy_21d"]    = r_spy.rolling(21).sum()   # V4
        feat["spy_5d"]     = r_spy.rolling(5).sum()
        feat["nsei_vs_spy"]= (r_nsei.rolling(21).mean()
                               - r_spy.rolling(21).mean())
        log.info("[FEATURES] V4: SPY ✓")
    else:
        feat["spy_21d"] = np.nan

    # ── V5: Gold ──────────────────────────────────────────────────────
    if "GC=F" in prices.columns:
        r_gld = np.log(prices["GC=F"] / prices["GC=F"].shift(1))
        feat["gold_21d"]   = r_gld.rolling(21).sum()   # V5
        feat["gold_5d"]    = r_gld.rolling(5).sum()
        feat["gold_vol"]   = r_gld.rolling(21).std() * np.sqrt(252)
        log.info("[FEATURES] V5: GC=F (Gold) ✓")
    else:
        feat["gold_21d"] = np.nan

    # ── V6: Crude oil ─────────────────────────────────────────────────
    if "CL=F" in prices.columns:
        r_oil = np.log(prices["CL=F"] / prices["CL=F"].shift(1))
        feat["crude_21d"]  = r_oil.rolling(21).sum()   # V6
        feat["crude_5d"]   = r_oil.rolling(5).sum()
        feat["crude_vol"]  = r_oil.rolling(21).std() * np.sqrt(252)
        log.info("[FEATURES] V6: CL=F (Crude) ✓")
    else:
        feat["crude_21d"] = np.nan

    # ── Additional diagnostics ────────────────────────────────────────
    # IT-Bank spread: sector rotation regime signal
    if "INFY.NS" in prices.columns and "^NSEBANK" in prices.columns:
        r_it   = np.log(prices["INFY.NS"]   / prices["INFY.NS"].shift(1))
        r_bank = np.log(prices["^NSEBANK"]  / prices["^NSEBANK"].shift(1))
        feat["it_bank_spread"] = (r_it - r_bank).rolling(21).mean()

    # Rolling drawdown (regime confirmation)
    cum  = (1 + r_nsei.fillna(0)).cumprod()
    peak = cum.rolling(252, min_periods=21).max()
    feat["nsei_dd_252"] = ((cum - peak) / peak)
    feat["nsei_dd_acc"]  = feat["nsei_dd_252"].diff(5)

    # ── Clean ─────────────────────────────────────────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan)
    log.info(f"[FEATURES] Shape: {feat.shape}  "
             f"({feat.notna().any().sum()} non-null features)")
    return feat


# =============================================================================
# 3. KRITZMAN TURBULENCE — India cross-asset version
# =============================================================================

def compute_turbulence(prices, lookback=TURBULENCE_LOOKBACK):
    """
    Kritzman et al. (2012) turbulence index using India cross-asset returns:
    Nifty, Gold, Crude, USD/INR. Captures unusual co-movements that signal
    Indian market stress (e.g. all four moving adversely simultaneously).
    """
    log.info("[TURB] Computing Kritzman turbulence index ...")
    tickers = [t for t in TURBULENCE_TICKERS if t in prices.columns]
    if len(tickers) < 2:
        log.warning("[TURB] Insufficient tickers — skipping")
        return pd.Series(np.nan, index=prices.index, name="turbulence")

    log_ret = np.log(
        prices[tickers] / prices[tickers].shift(1)
    ).dropna(how="any")

    t_vals = np.full(len(log_ret), np.nan)
    for i in range(lookback, len(log_ret)):
        hist = log_ret.iloc[i - lookback: i].values
        r_t  = log_ret.iloc[i].values
        mu   = hist.mean(axis=0)
        try:
            Sig  = np.cov(hist, rowvar=False)
            diff = r_t - mu
            t_vals[i] = max(float(diff @ np.linalg.inv(Sig) @ diff), 0.0)
        except np.linalg.LinAlgError:
            pass

    turb      = pd.Series(t_vals, index=log_ret.index, name="turbulence")
    threshold = turb.quantile(TURBULENCE_CRISIS_PCT / 100)
    log.info(f"[TURB] p{TURBULENCE_CRISIS_PCT} threshold: {threshold:.4f}")
    return turb


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ2 INDIA STAGE 0 — Data & Regime Feature Engineering")
    log.info("=" * 65)
    log.info(f"  Universe    : {TICKERS}")
    log.info(f"  Period      : {START_DATE} → {END_DATE}")
    log.info(f"  RF rate     : 6.5%  Cost: 20 bps")
    log.info(f"  HMM primary : 3 states (BIC vs 4-state robustness)")

    prices_raw = download_prices()
    prices     = validate_prices(prices_raw)
    log_ret    = np.log(prices / prices.shift(1)).clip(-0.20, 0.20)
    features   = build_regime_features(prices)

    # Turbulence
    turb = compute_turbulence(prices)
    features["turbulence"] = turb.reindex(features.index)

    # Report structural break dates in the data window
    log.info("\n[INFO] India structural breaks covered:")
    for dt, desc in INDIA_BREAKS.items():
        in_data = prices.index.min() <= pd.Timestamp(dt) <= prices.index.max()
        log.info(f"  {dt}: {desc}  [{'IN DATA' if in_data else 'outside'}]")

    # Save
    for obj, fname in [
        (prices,   "prices_india_rq2.parquet"),
        (log_ret,  "rq2_india_log_returns.parquet"),
        (features, "rq2_india_features.parquet"),
    ]:
        path = os.path.join(DATA_DIR, fname)
        obj.to_parquet(path)
        log.info(f"[SAVE] {fname}")

    # Sanity check: core 6 features
    core6 = ["nsei_ret_1d", "vol_primary", "usdinr_5d",
              "spy_21d", "gold_21d", "crude_21d"]
    log.info("\n[QC] Core 6 HMM variables:")
    for v in core6:
        if v in features.columns:
            nn = features[v].notna().sum()
            log.info(f"  {v:<20}: {nn} non-null "
                     f"({'OK' if nn > 2000 else 'CHECK'})")
        else:
            log.warning(f"  {v:<20}: MISSING")

    log.info("\n✓ Stage 0 complete. Run rq2_india_stage1_detection.py next.\n")
    return prices, features, log_ret


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# =============================================================================
# rq3_india_stage0_data.py
# Stage 0: Data download and feature engineering for Indian markets.
#
# Structural differences vs US stage0:
#
# 1. NSE/BSE trading hours (IST) vs NYSE (EST).
#    yfinance aligns on calendar date automatically.
#    Weekend/holiday gaps filled with ffill(limit=3).
#
# 2. Circuit breaker artefacts — NSE has ±10/15/20% intraday limits.
#    Returns winsorised to [-0.20, +0.20] after download.
#    This prevents blow-up in Granger F-stats from limit-hit days.
#
# 3. USD/INR as a primary cross-asset signal.
#    USDINR=X from yfinance. A 1% INR depreciation → ~0.8% IT sector gain
#    with a 1-3 day lag (TCS/Infosys USD earnings reported in INR).
#    This relationship is the key India-specific hypothesis:
#      H_India: USD/INR Granger-causes IT sector returns (INFY, TCS)
#      This has no US equivalent and is uniquely testable here.
#
# 4. Sector rotation signals:
#    IT vs Bank divergence is a core India regime signal — when USD/INR rises,
#    IT outperforms and Banks underperform (import cost pressures on credit).
#    The cross-asset feature matrix explicitly captures this.
#
# 5. Global coupling:
#    Indian markets open ~5.5 hours after US close. The overnight S&P 500
#    return (SPY lag-1) is a meaningful predictor of Nifty open. Included
#    as a cross-asset feature for all Indian targets.
#
# Outputs
# -------
#   data_india/prices_india.parquet
#   data_india/rq3_india_log_returns.parquet
#   data_india/rq3_india_{TARGET}_own_only.parquet
#   data_india/rq3_india_{TARGET}_cross_asset.parquet
#   data_india/rq3_india_{TARGET}_fwd_net.parquet
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from rq3_india_config import (
    TICKERS, TARGET_TICKERS, START_DATE, END_DATE,
    COST_BPS, SIGNAL_HORIZON, DATA_DIR, RF_ANNUAL,
    INDIA_STRUCTURAL_BREAKS,
)


# =============================================================================
# 1. DOWNLOAD
# =============================================================================

def _download_group(tickers: list, start: str, end: str,
                    label: str) -> pd.DataFrame:
    """
    Download a small group of tickers with per-ticker fallback.
    yfinance batch downloads silently return NaN for tickers that fail
    (rate limits, wrong format, exchange mismatch). This function:
      1. Tries a batch download first (fast).
      2. Verifies each ticker has data.
      3. Re-downloads individually any ticker with >50% NaN.
    Returns a flat DataFrame with ticker names as columns.
    """
    import yfinance as yf
    import time

    log.info(f"[DATA] Downloading group '{label}': {tickers}")

    # Batch attempt
    try:
        raw    = yf.download(tickers, start=start, end=end,
                             auto_adjust=True, progress=False,
                             threads=False, timeout=30)
        if isinstance(raw.columns, pd.MultiIndex):
            # Extract Close prices; handle both (Price,Ticker) and (Ticker,Price)
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            elif "Close" in raw.columns.get_level_values(1):
                prices = raw.xs("Close", axis=1, level=1)
            else:
                prices = raw.iloc[:, :len(tickers)]
                prices.columns = tickers[:prices.shape[1]]
        else:
            prices = raw
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0])
    except Exception as e:
        log.warning(f"[DATA] Batch download failed for '{label}': {e}")
        prices = pd.DataFrame()

    # Check each ticker — re-download individually if >50% NaN
    results = {}
    for t in tickers:
        col_data = None
        if t in prices.columns:
            col_data = prices[t]
        elif prices.shape[1] == 1 and len(tickers) == 1:
            col_data = prices.iloc[:, 0]

        if col_data is not None and col_data.notna().mean() > 0.50:
            results[t] = col_data
            log.info(f"[DATA]   {t:<20}: OK ({col_data.notna().sum()} days)")
        else:
            # Individual retry
            log.info(f"[DATA]   {t:<20}: retrying individually ...")
            time.sleep(0.5)  # be polite to yfinance
            try:
                ind = yf.download(t, start=start, end=end,
                                   auto_adjust=True, progress=False,
                                   timeout=30)
                if isinstance(ind.columns, pd.MultiIndex):
                    if "Close" in ind.columns.get_level_values(0):
                        ind = ind["Close"]
                    else:
                        ind = ind.iloc[:, 0]
                elif "Close" in ind.columns:
                    ind = ind["Close"]
                elif not ind.empty:
                    ind = ind.iloc[:, 0]

                if isinstance(ind, pd.DataFrame):
                    ind = ind.squeeze()

                if ind.notna().mean() > 0.30:
                    results[t] = ind
                    log.info(f"[DATA]   {t:<20}: individual OK "
                             f"({ind.notna().sum()} days)")
                else:
                    log.warning(f"[DATA]   {t:<20}: FAILED "
                                f"({ind.notna().mean():.0%} data)")
            except Exception as e2:
                log.warning(f"[DATA]   {t:<20}: individual failed: {e2}")

    if not results:
        return pd.DataFrame()

    df = pd.concat(results, axis=1)
    df.index.name = "Date"
    return df


def download_prices() -> pd.DataFrame:
    """
    Download all India universe tickers using grouped, robust downloading.

    Why grouped download?
      yfinance batch downloads silently fail for some ticker types:
        - Index tickers (^NSEI, ^NSEBANK) fail in large batches
        - FX pairs (USDINR=X) need separate handling
        - Futures (CL=F, GC=F) often need individual download
        - NSE equities (.NS) are the most reliable in batch

    The cache is only written after verifying each group has data.
    If the cache exists but has <5 valid columns, it is deleted and
    re-downloaded — this handles the corrupt-cache case automatically.
    """
    cache = os.path.join(DATA_DIR, "prices_india.parquet")

    # Check for corrupt cache (saved from failed download)
    if os.path.exists(cache):
        try:
            cached = pd.read_parquet(cache)
            n_valid = (cached.isna().mean() < 0.50).sum()
            if n_valid >= 5:
                log.info(f"[DATA] Loading cached prices ({n_valid} valid cols): {cache}")
                return cached
            else:
                log.warning(f"[DATA] Corrupt cache detected ({n_valid} valid cols) "
                            f"— deleting and re-downloading")
                os.remove(cache)
        except Exception as e:
            log.warning(f"[DATA] Cache unreadable ({e}) — re-downloading")
            os.remove(cache)

    import yfinance as yf

    log.info(f"[DATA] Downloading India universe ({START_DATE} → {END_DATE})")

    # ── Group 1: NSE large-cap equities (most reliable) ──────────────────
    nse_equities = [
        "TCS.NS", "INFY.NS", "HDFCBANK.NS", "SBIN.NS",
        "SUNPHARMA.NS", "HINDUNILVR.NS", "COALINDIA.NS",
        "ONGC.NS", "TECHM.NS", "AXISBANK.NS",
    ]
    g1 = _download_group(nse_equities, START_DATE, END_DATE, "NSE equities")

    # ── Group 2: NSE indices (download individually — batch often fails) ──
    nse_indices = ["^NSEI", "^NSEBANK"]
    g2 = _download_group(nse_indices, START_DATE, END_DATE, "NSE indices")

    # ── Group 3: US/global (reliable) ────────────────────────────────────
    global_tickers = ["SPY"]
    g3 = _download_group(global_tickers, START_DATE, END_DATE, "Global")

    # ── Group 4: FX (USDINR — download individually) ─────────────────────
    fx_tickers = ["USDINR=X"]
    g4 = _download_group(fx_tickers, START_DATE, END_DATE, "FX")

    # ── Group 5: Commodities / futures (download individually) ───────────
    futures = ["GC=F", "CL=F"]
    g5 = _download_group(futures, START_DATE, END_DATE, "Futures")

    # ── Merge all groups on common date index ─────────────────────────────
    groups = [g for g in [g1, g2, g3, g4, g5] if not g.empty]
    if not groups:
        raise RuntimeError("[DATA] All downloads failed. Check internet connection.")

    prices = groups[0]
    for g in groups[1:]:
        prices = prices.join(g, how="outer")

    # Forward-fill short gaps (NSE holidays, weekends — max 3 days)
    prices = prices.ffill(limit=3)
    prices.index.name = "Date"

    n_valid = (prices.isna().mean() < 0.50).sum()
    log.info(f"[DATA] Combined shape: {prices.shape}  "
             f"Valid columns (>50% data): {n_valid}")
    log.info(f"[DATA] Columns: {prices.columns.tolist()}")

    if n_valid < 3:
        raise RuntimeError(
            f"[DATA] Only {n_valid} valid tickers after download. "
            "Check network / yfinance version."
        )

    prices.to_parquet(cache)
    log.info(f"[DATA] Saved → {cache}")
    return prices


def validate_and_clean(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Quality control:
      1. Flatten MultiIndex columns if yfinance returned (Price, Ticker) format
      2. Replace zeros with NaN (corrupt data points)
      3. Forward-fill short gaps (NSE holidays, weekends)
      4. Winsorise LOG RETURNS only — keeps original prices intact for
         MA / momentum features. This avoids the price-reconstruction
         bug where reindex(method=None) produced an all-NaN DataFrame.
      5. Drop tickers with >20% missing data
    """
    # ── Handle MultiIndex columns from yfinance ────────────────────────────
    # yfinance sometimes returns (Price, Ticker) MultiIndex columns.
    # Flatten to just the Ticker level so prices['^NSEI'] works correctly.
    if isinstance(prices.columns, pd.MultiIndex):
        log.info("[QC] Flattening MultiIndex columns from yfinance")
        # If top level is price fields (Open/High/Low/Close/Volume), take Close
        lvl0 = prices.columns.get_level_values(0).unique().tolist()
        if "Close" in lvl0:
            prices = prices["Close"]
        else:
            # Take the last price-like level
            prices = prices.iloc[:, prices.columns.get_level_values(0)
                                 .tolist().index(lvl0[-1]):]
            prices.columns = prices.columns.get_level_values(1)

    # ── Basic cleaning ────────────────────────────────────────────────────
    prices = prices.replace(0, np.nan)
    prices = prices.ffill(limit=3)   # fill short gaps (holidays, weekends)

    # ── Drop high-missing tickers ─────────────────────────────────────────
    miss = prices.isna().mean()
    drop = miss[miss > 0.20].index.tolist()
    if drop:
        log.warning(f"[QC] Dropping tickers >20% missing: {drop}")
        prices = prices.drop(columns=drop, errors="ignore")

    prices = prices.dropna(how="all")

    # ── Check for circuit-breaker artefacts (log, don't reconstruct) ──────
    # We winsorise log returns ONLY in build_features() at the point of use.
    # Keeping original prices intact here is critical — MA / pct_change()
    # features need the actual price levels, not a cumsum reconstruction.
    rets = prices.pct_change()
    bad_days = {col: int((rets[col].abs() > 0.20).sum())
                for col in prices.columns
                if (rets[col].abs() > 0.20).sum() > 0}
    if bad_days:
        log.info(f"[QC] Circuit-breaker days (>20% move): {bad_days}")
        log.info("[QC] These will be winsorised in log-return space inside build_features()")

    log.info(f"[QC] Clean prices shape: {prices.shape}")
    log.info(f"[QC] Tickers: {prices.columns.tolist()}")
    log.info(f"[QC] Date range: {prices.index[0]} → {prices.index[-1]}")

    log.info("[QC] Key India structural breaks in data window:")
    for date, desc in INDIA_STRUCTURAL_BREAKS.items():
        log.info(f"      {date}: {desc}")

    return prices


# =============================================================================
# 2. HELPERS
# =============================================================================

def _rsi(r: pd.Series, period: int = 14) -> pd.Series:
    gain = r.clip(lower=0).rolling(period).mean()
    loss = (-r.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _usdinr_sensitivity(log_ret: pd.DataFrame,
                         target: str) -> pd.Series:
    """
    Compute rolling 63-day beta of target returns to USD/INR.
    Used as an additional feature: how much does this asset
    co-move with currency movements right now?
    High beta to INR depreciation = USD-earner (IT sector).
    Low/negative beta = domestic-focused or import-cost-sensitive.
    """
    if "USDINR=X" not in log_ret.columns:
        return pd.Series(np.nan, index=log_ret.index)
    r_target = log_ret[target]
    r_fx     = log_ret["USDINR=X"]
    cov  = r_target.rolling(63).cov(r_fx)
    var_ = r_fx.rolling(63).var().replace(0, np.nan)
    return (cov / var_).shift(1).rename("usdinr_beta_63")


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def build_features(prices: pd.DataFrame) -> dict:
    """
    Builds own_only and cross_asset feature matrices per target.

    India-specific additions vs US version:
    ─────────────────────────────────────────
    A. USD/INR sensitivity beta (rolling 63-day) — key India cross-asset
       The USD/INR → IT sector relationship is explicitly captured here.
       INFY/TCS have high positive beta (INR depreciation → higher returns).
       Banks have low/negative beta (higher import costs, credit stress).

    B. IT-Bank spread: (INFY.NS return - NSEBANK return).rolling(5)
       Core India sector rotation signal. Positive spread = risk-on India IT,
       negative spread = defensives / domestics winning.

    C. Global overnight: SPY lag-1 return
       India opens ~5.5h after US close. S&P 500 overnight move is
       a significant predictor of Nifty gap-open direction.

    D. Oil impact: CL=F lag 1-3
       India is a large oil importer. Rising crude → CAD pressure →
       INR weakness → OMC losses → Nifty drag, especially on ONGC/BPCL.
       The lag structure (1-3 days) captures the news-absorption pattern.

    E. Relative momentum vs global: (^NSEI - SPY).rolling(21)
       India outperformance/underperformance relative to global. When this
       is positive, FII flows tend to be supportive; when negative, FIIs sell.
    """
    log.info("[FEATURES] Building India cross-asset feature matrices ...")
    # Compute log returns and winsorise to ±20% (NSE circuit breaker range)
    # This prevents extreme F-stats from limit-hit days corrupting Granger.
    # Applied here (not in validate_and_clean) so original prices stay intact
    # for MA cross and momentum pct_change() features.
    log_ret  = np.log(prices / prices.shift(1)).clip(-0.20, 0.20)
    features = {}
    COST     = COST_BPS / 10_000 * 2

    for target in TARGET_TICKERS:
        if target not in prices.columns:
            log.warning(f"[FEATURES] {target} not in prices — skipping")
            continue

        r = log_ret[target]
        p = prices[target]

        # ── OWN-ASSET features ────────────────────────────────────────────
        own = pd.DataFrame(index=prices.index)

        # AR lags
        for lag in [1, 2, 3, 5]:
            own[f"own_lag_{lag}"] = r.shift(lag)

        # Volatility
        own["own_vol_10d"]   = r.rolling(10).std() * np.sqrt(252)
        own["own_vol_21d"]   = r.rolling(21).std() * np.sqrt(252)
        own["own_vol_63d"]   = r.rolling(63).std() * np.sqrt(252)
        own["own_vol_ratio"] = (
            r.rolling(10).std() / r.rolling(63).std().replace(0, np.nan)
        ).shift(1)

        # Momentum
        own["own_mom_5d"]    = p.pct_change(5).shift(1)
        own["own_mom_21d"]   = p.pct_change(21).shift(1)
        own["own_mom_63d"]   = p.pct_change(63).shift(1)

        # Technicals
        own["own_ma_cross"]  = (
            p.rolling(10).mean() / p.rolling(50).mean() - 1
        ).shift(1)
        own["own_rsi"]       = _rsi(r, 14).shift(1)
        own["own_bb_z"]      = (
            (p - p.rolling(20).mean())
            / p.rolling(20).std().replace(0, np.nan)
        ).shift(1)

        # India-specific own features
        # Rolling Sharpe (own): return / vol (rf-adjusted for India)
        rf_daily = RF_ANNUAL / 252
        own["own_sharpe_21"] = (
            (r.rolling(21).mean() - rf_daily)
            / r.rolling(21).std().replace(0, np.nan)
        ).shift(1) * np.sqrt(252)

        # Drawdown depth (own)
        cum  = (1 + r.fillna(0)).cumprod()
        peak = cum.rolling(252, min_periods=21).max()
        own["own_dd_21"]     = ((cum - peak) / peak).shift(1)

        # USD/INR sensitivity (own asset's beta to currency moves)
        own["own_usdinr_beta"] = _usdinr_sensitivity(log_ret, target)

        # ── CROSS-ASSET features — TARGETED SET ──────────────────────────
        # Use targeted signals based on the Granger hypotheses, NOT all
        # pairwise lags. This avoids the dimensionality explosion (~145 features)
        # that dilutes the informative signal (USD/INR→IT, SPY→Nifty, etc.)
        # with noisy collinear features (NSE stock correlations with each other).
        #
        # Feature count: ~30 targeted vs ~145 pairwise → 5x better S/F ratio.
        # Ridge regression is far more effective with 1260/30 = 42 S/F ratio
        # than with 1260/145 = 8.7 which is near the overfitting boundary.
        cross  = own.copy()

        # ── 1. USD/INR signals (primary India hypothesis) ─────────────────
        if "USDINR=X" in log_ret.columns:
            r_fx = log_ret["USDINR=X"]
            for lag in [1, 2, 3, 5]:
                cross[f"usdinr_lag_{lag}"] = r_fx.shift(lag)
            cross["usdinr_5d"]    = r_fx.rolling(5).sum().shift(1)
            cross["usdinr_21d"]   = r_fx.rolling(21).sum().shift(1)
            cross["usdinr_vol"]   = r_fx.rolling(21).std().shift(1)
            cross["usdinr_trend"] = (
                r_fx.rolling(10).mean() / r_fx.rolling(50).mean() - 1
            ).shift(1)

        # ── 2. SPY overnight signal (global coupling) ─────────────────────
        if "SPY" in log_ret.columns:
            r_spy = log_ret["SPY"]
            for lag in [1, 2, 3]:
                cross[f"spy_lag_{lag}"] = r_spy.shift(lag)
            cross["spy_5d"]       = r_spy.rolling(5).sum().shift(1)
            cross["spy_vol_10d"]  = r_spy.rolling(10).std().shift(1)

        # ── 3. NSE Bank index (sector lead) ──────────────────────────────
        if "^NSEBANK" in log_ret.columns:
            r_bk = log_ret["^NSEBANK"]
            for lag in [1, 2, 3]:
                cross[f"nsebank_lag_{lag}"] = r_bk.shift(lag)
            cross["nsebank_5d"]   = r_bk.rolling(5).sum().shift(1)

        # ── 4. Gold signal ────────────────────────────────────────────────
        if "GC=F" in log_ret.columns:
            r_gld = log_ret["GC=F"]
            for lag in [1, 2, 3]:
                cross[f"gold_lag_{lag}"] = r_gld.shift(lag)
            cross["gold_5d"]      = r_gld.rolling(5).sum().shift(1)

        # ── 5. Oil signal ─────────────────────────────────────────────────
        if "CL=F" in log_ret.columns:
            r_oil = log_ret["CL=F"]
            for lag in [1, 2, 3]:
                cross[f"oil_lag_{lag}"] = r_oil.shift(lag)
            cross["oil_5d"]       = r_oil.rolling(5).sum().shift(1)

        # ── 6. Rolling correlation with KEY cross-asset signals only ──────
        # Only compute correlations for economically motivated pairs,
        # not all pairwise (which are noisy and collinear for NSE stocks).
        key_cross = {
            "USDINR=X": "usdinr", "SPY": "spy",
            "GC=F": "gold",       "CL=F": "oil",
        }
        for ticker, stub in key_cross.items():
            if ticker in log_ret.columns and ticker != target:
                r_o = log_ret[ticker]
                cross[f"corr_{stub}_63"] = r.rolling(63).corr(r_o).shift(1)

        # ── India-specific CROSS-ASSET features ──────────────────────────

        # A. USD/INR → IT sector sensitivity (primary India hypothesis)
        if "USDINR=X" in log_ret.columns:
            r_fx = log_ret["USDINR=X"]
            # Is INR weakening this week? (positive = INR depreciation)
            cross["usdinr_5d_chg"]  = r_fx.rolling(5).sum().shift(1)
            cross["usdinr_21d_chg"] = r_fx.rolling(21).sum().shift(1)
            # FX momentum: is USD/INR trending up (INR weakening)?
            cross["usdinr_trend"]   = (
                r_fx.rolling(10).mean() / r_fx.rolling(50).mean() - 1
            ).shift(1)

        # B. IT-Bank spread (sector rotation)
        if ("INFY.NS" in log_ret.columns
                and "HDFCBANK.NS" in log_ret.columns):
            it_ret   = log_ret["INFY.NS"]
            bank_ret = log_ret["HDFCBANK.NS"]
            cross["it_bank_spread_5"]  = (
                it_ret.rolling(5).mean() - bank_ret.rolling(5).mean()
            ).shift(1)
            cross["it_bank_spread_21"] = (
                it_ret.rolling(21).mean() - bank_ret.rolling(21).mean()
            ).shift(1)

        # C. Global overnight: SPY lag-1 (India opens after US closes)
        if "SPY" in log_ret.columns:
            cross["spy_overnight"]    = log_ret["SPY"].shift(1)
            cross["spy_mom_5d_India"] = log_ret["SPY"].rolling(5).sum().shift(1)

        # D. Oil impact on India (crude → CAD → INR → costs)
        if "CL=F" in log_ret.columns:
            r_oil = log_ret["CL=F"]
            cross["oil_lag1"] = r_oil.shift(1)
            cross["oil_lag3"] = r_oil.shift(3)
            cross["oil_5d"]   = r_oil.rolling(5).sum().shift(1)

        # E. India vs global relative momentum
        if "SPY" in log_ret.columns and "^NSEI" in log_ret.columns:
            r_nsei = log_ret["^NSEI"]
            r_spy  = log_ret["SPY"]
            cross["india_vs_global_21"] = (
                r_nsei.rolling(21).mean() - r_spy.rolling(21).mean()
            ).shift(1)

        # ── Target: forward net return ────────────────────────────────────
        fwd_gross = r.shift(-SIGNAL_HORIZON).rolling(SIGNAL_HORIZON).sum()
        fwd_net   = fwd_gross - COST

        # ── Clean ─────────────────────────────────────────────────────────
        own   = own.replace([np.inf, -np.inf], np.nan)
        cross = cross.replace([np.inf, -np.inf], np.nan)

        features[target] = {
            "own_only":    own,
            "cross_asset": cross,
            "fwd_net":     fwd_net,
            "fwd_gross":   fwd_gross,
            "log_ret":     r,
        }

        log.info(f"[FEATURES] {target}: own={own.shape[1]} features, "
                 f"cross={cross.shape[1]} features, n_rows={own.shape[0]}")

    return features, log_ret


# =============================================================================
# 4. MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ3 INDIA STAGE 0 — Data & Cross-Asset Feature Engineering")
    log.info("=" * 65)
    log.info(f"  Universe  : {len(TICKERS)} tickers (NSE + global proxies)")
    log.info(f"  Targets   : {TARGET_TICKERS}")
    log.info(f"  Period    : {START_DATE} → {END_DATE}")
    log.info(f"  RF rate   : {RF_ANNUAL:.1%} (RBI repo)")
    log.info(f"  Cost      : {COST_BPS} bps one-way")

    prices_raw            = download_prices()
    prices                = validate_and_clean(prices_raw)
    features, log_ret     = build_features(prices)

    # Save prices and log returns
    prices.to_parquet(os.path.join(DATA_DIR, "prices_india.parquet"))
    log_ret.to_parquet(os.path.join(DATA_DIR, "rq3_india_log_returns.parquet"))

    # Save per-target feature sets
    saved = 0
    for target, fd in features.items():
        stub = target.replace("^", "").replace("=", "").replace(".", "_")
        for key in ["own_only", "cross_asset"]:
            fd[key].to_parquet(
                os.path.join(DATA_DIR, f"rq3_india_{stub}_{key}.parquet")
            )
        fwd = fd["fwd_net"]
        if hasattr(fwd, "to_frame"):
            fwd = fwd.to_frame(name="fwd_net")
        fwd.to_parquet(
            os.path.join(DATA_DIR, f"rq3_india_{stub}_fwd_net.parquet")
        )
        saved += 1
        log.info(f"[SAVE] {target}: own={fd['own_only'].shape[1]}f  "
                 f"cross={fd['cross_asset'].shape[1]}f  saved.")

    log.info(f"\n[SAVE] {saved} target sets saved to {DATA_DIR}/")

    # Sanity check: USD/INR beta for INFY (should be positive — USD earner)
    if "^NSEI" in features:
        sample = (features["^NSEI"]["cross_asset"]
                  .dropna(how="all")
                  .iloc[-3:, :6])
        log.info(f"\n[QC] Sample (^NSEI cross, last 3 rows, first 6 cols):\n"
                 f"{sample.round(5).to_string()}")

    # Sanity check: verify features are not all NaN
    log.info("[QC] Feature sanity check:")
    for target, fd in features.items():
        own_ok   = fd["own_only"].notna().any().any()
        cross_ok = fd["cross_asset"].notna().any().any()
        fwd_ok   = fd["fwd_net"].notna().any()
        status   = "OK" if (own_ok and cross_ok and fwd_ok) else "ALL NaN — CHECK"
        n_own_valid = int(fd["own_only"].dropna(how="all").shape[0])
        log.info(f"  {target:<20} own={own_ok}  cross={cross_ok}  "
                 f"fwd={fwd_ok}  valid_rows={n_own_valid}  [{status}]")

    log.info("\n✓ Stage 0 complete. Run rq3_india_stage1_granger.py next.\n")
    return prices, features


if __name__ == "__main__":
    main()

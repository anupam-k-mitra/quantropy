#!/usr/bin/env python3
# =============================================================================
# rq2_stage0_data.py
# Stage 0: Data download, validation, regime feature engineering,
#          and Kritzman turbulence index computation.
#
# Outputs:
#   data/prices.parquet              — adjusted OHLCV close prices
#   data/rq2_log_returns.parquet     — daily log returns
#   data/rq2_regime_features.parquet — 25 regime detection features
#
# Run: python rq2_stage0_data.py
# =============================================================================

import os
import sys
import warnings
import logging

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from regime_config import (
    TICKERS, START_DATE, END_DATE, DATA_DIR,
    TURBULENCE_LOOKBACK, TURBULENCE_CRISIS_PCT,
)


# =============================================================================
# 1.  DOWNLOAD
# =============================================================================

def download_prices(tickers, start, end):
    """
    Download adjusted close prices via yfinance.
    Reuses cached parquet if already present.
    """
    cache = os.path.join(DATA_DIR, "prices.parquet")
    if os.path.exists(cache):
        log.info(f"Reusing cached prices: {cache}")
        return pd.read_parquet(cache)

    try:
        import yfinance as yf
        log.info(f"Downloading {len(tickers)} tickers ({start} to {end}) …")
        raw    = yf.download(
            tickers, start=start, end=end,
            auto_adjust=True, progress=True, threads=True,
        )
        prices = raw["Close"].ffill(limit=3).dropna(how="all")
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices.to_parquet(cache)
        log.info(f"Saved: {cache}")
        return prices
    except Exception as e:
        raise RuntimeError(
            f"Download failed: {e}\n"
            "Place prices.parquet manually in the data/ folder."
        ) from e


def validate_prices(prices):
    """Quality checks: missing values, zero prices, final cleaning."""
    log.info(f"\nRaw shape: {prices.shape}")
    miss = prices.isna().mean().sort_values(ascending=False)
    log.info(f"Missing %:\n{miss.to_string()}")

    prices = prices.replace(0, np.nan).ffill(limit=3)
    drop   = miss[miss > 0.15].index.tolist()
    if drop:
        log.warning(f"Dropping tickers >15% missing: {drop}")
        prices = prices.drop(columns=drop)

    prices = prices.dropna(how="all")
    log.info(f"Clean shape: {prices.shape}")
    log.info(f"Tickers kept: {prices.columns.tolist()}")
    return prices


# =============================================================================
# 2.  REGIME FEATURE ENGINEERING
# =============================================================================

def build_regime_features(prices):
    """
    Construct 25 regime features across 5 families.
    All normalised via rolling 252-day z-score (no look-ahead).

    Families
    --------
    1. Return / Trend     — where in the market cycle?
    2. Volatility         — how uncertain is the environment?
    3. Credit / Stress    — is credit risk building?
    4. Cross-Asset Risk   — risk-on vs risk-off positioning?
    5. Macro / Rates      — monetary policy and yield curve regime?
    """
    log.info("Building regime feature matrix …")
    feat = pd.DataFrame(index=prices.index)

    # ── Helper: safe log return ──────────────────────────────────
    def lr(col, shift=1):
        p = prices[col] if col in prices.columns else None
        if p is None:
            return None
        return np.log(p / p.shift(shift))

    # ── FAMILY 1: Return / Trend ──────────────────────────────────
    if "SPY" in prices.columns:
        spy = prices["SPY"]
        r1  = np.log(spy / spy.shift(1))
        feat["spy_ret_1d"]    = r1
        feat["spy_ret_5d"]    = np.log(spy / spy.shift(5))
        feat["spy_ret_21d"]   = np.log(spy / spy.shift(21))
        feat["spy_ret_63d"]   = np.log(spy / spy.shift(63))
        feat["spy_vs_ma50"]   = spy / spy.rolling(50).mean() - 1
        feat["spy_vs_ma200"]  = spy / spy.rolling(200).mean() - 1
        feat["ma50_vs_ma200"] = (spy.rolling(50).mean()
                                 / spy.rolling(200).mean() - 1)

    # ── FAMILY 2: Volatility ──────────────────────────────────────
    if "SPY" in prices.columns:
        r1 = np.log(prices["SPY"] / prices["SPY"].shift(1))
        for w in [10, 21, 63]:
            feat[f"rvol_{w}d"] = r1.rolling(w).std() * np.sqrt(252)
        feat["vol_ratio"] = (feat["rvol_10d"]
                             / feat["rvol_63d"].replace(0, np.nan))

    if "^VIX" in prices.columns:
        vix = prices["^VIX"]
        feat["vix_level"]   = vix
        feat["vix_chg_5d"]  = vix.pct_change(5)
        feat["vix_zscore"]  = ((vix - vix.rolling(252).mean())
                               / vix.rolling(252).std())
        feat["vix_regime"]  = (pd.cut(vix,
                                       bins=[0, 15, 20, 30, 9999],
                                       labels=[0, 1, 2, 3])
                                .astype(float))

    # ── FAMILY 3: Credit / Stress ─────────────────────────────────
    if "HYG" in prices.columns and "IEF" in prices.columns:
        hyg_r = np.log(prices["HYG"] / prices["HYG"].shift(1))
        ief_r = np.log(prices["IEF"] / prices["IEF"].shift(1))
        feat["credit_spread_proxy"] = (hyg_r - ief_r).rolling(21).mean()

    if "TLT" in prices.columns and "SPY" in prices.columns:
        tlt_r = np.log(prices["TLT"] / prices["TLT"].shift(1))
        spy_r = np.log(prices["SPY"] / prices["SPY"].shift(1))
        feat["spy_vs_tlt"]  = (spy_r - tlt_r).rolling(21).mean()
        feat["tlt_mom_21"]  = np.log(prices["TLT"] / prices["TLT"].shift(21))

    # ── FAMILY 4: Cross-Asset Risk-On/Off ─────────────────────────
    if "GLD" in prices.columns and "SPY" in prices.columns:
        gld_r = np.log(prices["GLD"] / prices["GLD"].shift(1))
        spy_r = np.log(prices["SPY"] / prices["SPY"].shift(1))
        feat["eq_vs_gold"] = (spy_r - gld_r).rolling(21).mean()

    if "DX-Y.NYB" in prices.columns:
        dxy = prices["DX-Y.NYB"]
        feat["dxy_mom_21"] = np.log(dxy / dxy.shift(21))

    if "IWM" in prices.columns and "SPY" in prices.columns:
        iwm_r = np.log(prices["IWM"] / prices["IWM"].shift(1))
        spy_r = np.log(prices["SPY"] / prices["SPY"].shift(1))
        feat["small_large_spread"] = (iwm_r - spy_r).rolling(21).mean()

    # ── FAMILY 5: Macro / Rates ───────────────────────────────────
    if "^TNX" in prices.columns:
        tnx = prices["^TNX"]
        feat["yield_10y"]    = tnx
        feat["yield_chg_21"] = tnx.diff(21)

    if "^TNX" in prices.columns and "^IRX" in prices.columns:
        feat["yield_curve"] = prices["^TNX"] - prices["^IRX"]
        feat["curve_chg"]   = feat["yield_curve"].diff(21)

    # ── Clean and rolling z-score normalise ───────────────────────
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat_norm = feat.copy()
    skip_norm = {"vix_regime"}
    for col in feat.columns:
        if col in skip_norm:
            continue
        mu  = feat[col].rolling(252, min_periods=63).mean()
        std = feat[col].rolling(252, min_periods=63).std()
        feat_norm[col] = (feat[col] - mu) / std.replace(0, np.nan)
    feat_norm = feat_norm.replace([np.inf, -np.inf], np.nan)

    log.info(f"Feature matrix shape: {feat_norm.shape}  "
             f"({feat_norm.shape[1]} features)")
    return feat_norm


# =============================================================================
# 3.  KRITZMAN TURBULENCE INDEX
# =============================================================================

def compute_turbulence(prices, lookback=TURBULENCE_LOOKBACK):
    """
    Kritzman, Li, Page & Rigobon (2012) turbulence index.

    T_t = (r_t - mu)' * Sigma^{-1} * (r_t - mu)

    High turbulence signals unusual cross-asset co-movements
    consistent with stress or crisis regimes.
    """
    log.info("Computing Kritzman turbulence index …")
    turb_tickers = [t for t in ["SPY", "TLT", "GLD", "HYG"]
                    if t in prices.columns]
    if len(turb_tickers) < 2:
        log.warning("Insufficient tickers for turbulence; skipping.")
        return pd.Series(np.nan, index=prices.index, name="turbulence")

    log_ret = np.log(
        prices[turb_tickers] / prices[turb_tickers].shift(1)
    ).dropna(how="any")

    t_vals = np.full(len(log_ret), np.nan)

    for i in range(lookback, len(log_ret)):
        hist    = log_ret.iloc[i - lookback: i].values
        r_t     = log_ret.iloc[i].values
        mu      = hist.mean(axis=0)
        try:
            Sigma   = np.cov(hist, rowvar=False)
            Sig_inv = np.linalg.inv(Sigma)
            diff    = r_t - mu
            t_vals[i] = max(float(diff @ Sig_inv @ diff), 0.0)
        except np.linalg.LinAlgError:
            pass

    turb = pd.Series(t_vals, index=log_ret.index, name="turbulence")
    threshold = turb.quantile(TURBULENCE_CRISIS_PCT / 100)
    log.info(f"  Crisis threshold (p{TURBULENCE_CRISIS_PCT}): {threshold:.4f}")
    return turb


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ2 STAGE 0 — Data Download & Feature Engineering")
    log.info("=" * 65)

    # Download & validate
    prices_raw = download_prices(TICKERS, START_DATE, END_DATE)
    prices     = validate_prices(prices_raw)

    # Log returns
    log_ret = np.log(prices / prices.shift(1)).dropna(how="all")

    # Regime features
    features = build_regime_features(prices)

    # Turbulence index — appended to feature matrix
    turb = compute_turbulence(prices)
    features["turbulence"] = turb.reindex(features.index)

    # Save
    prices_path  = os.path.join(DATA_DIR, "prices.parquet")
    ret_path     = os.path.join(DATA_DIR, "rq2_log_returns.parquet")
    feat_path    = os.path.join(DATA_DIR, "rq2_regime_features.parquet")

    prices.to_parquet(prices_path)
    log_ret.to_parquet(ret_path)
    features.to_parquet(feat_path)

    log.info("\n── Saved ─────────────────────────────────────────────────")
    log.info(f"  Prices   → {prices_path}")
    log.info(f"  Returns  → {ret_path}")
    log.info(f"  Features → {feat_path}")

    # Sanity check
    log.info("\n── Feature sample (last 5 rows, first 4 cols) ────────────")
    log.info(f"\n{features.iloc[-5:, :4].round(3).to_string()}")

    log.info("\n✓ Stage 0 complete. Run rq2_stage1_detection.py next.\n")
    return prices, features, log_ret


if __name__ == "__main__":
    main()

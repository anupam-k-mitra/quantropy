#!/usr/bin/env python3
# =============================================================================
# rq3_stage0_data.py
# Stage 0: Data download and feature engineering.
#
# Builds TWO feature matrices per target asset -- this split is the
# structural core of the H0 test:
#
#   own_only    -- AR lags + own technicals  (null model input)
#   cross_asset -- own + cross-asset lags    (alternative model input)
#
# The RMSE difference between models trained on these two feature sets
# is what the Diebold-Mariano test evaluates in Stage 3.
#
# Outputs
# -------
#   data/prices.parquet
#   data/rq3_log_returns.parquet
#   data/rq3_{TARGET}_own_only.parquet
#   data/rq3_{TARGET}_cross_asset.parquet
#   data/rq3_{TARGET}_fwd_net.parquet
#
# Run: python rq3_stage0_data.py
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

from rq3_config import (
    TICKERS, TARGET_TICKERS, START_DATE, END_DATE,
    COST_BPS, SIGNAL_HORIZON, DATA_DIR,
)


# =============================================================================
# 1.  DOWNLOAD
# =============================================================================

def download_prices():
    """Download adjusted close prices. Reuses cache if present."""
    cache = os.path.join(DATA_DIR, "prices.parquet")
    if os.path.exists(cache):
        log.info(f"Reusing cached prices: {cache}")
        return pd.read_parquet(cache)

    try:
        import yfinance as yf
        tickers = list(TICKERS.keys())
        log.info(f"Downloading {len(tickers)} tickers "
                 f"({START_DATE} to {END_DATE}) ...")
        raw    = yf.download(
            tickers, start=START_DATE, end=END_DATE,
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
            "Place prices.parquet in data/ folder manually."
        ) from e


def validate_prices(prices):
    """Light QC: fill short gaps, drop tickers >15% missing."""
    miss = prices.isna().mean()
    prices = prices.replace(0, np.nan).ffill(limit=3)
    drop  = miss[miss > 0.15].index.tolist()
    if drop:
        log.warning(f"Dropping tickers (>15% missing): {drop}")
        prices = prices.drop(columns=drop, errors="ignore")
    prices = prices.dropna(how="all")
    log.info(f"Clean shape: {prices.shape}  "
             f"tickers: {prices.columns.tolist()}")
    return prices


# =============================================================================
# 2.  HELPER: RSI
# =============================================================================

def _rsi(r: pd.Series, period: int = 14) -> pd.Series:
    gain = r.clip(lower=0).rolling(period).mean()
    loss = (-r.clip(upper=0)).rolling(period).mean()
    rs   = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


# =============================================================================
# 3.  FEATURE ENGINEERING
# =============================================================================

def build_features(prices: pd.DataFrame) -> dict:
    """
    For each TARGET_TICKER build:

    own_only  (null model)
    ---------
    * Own AR lags: lag-1, 2, 3, 5 of daily log return
    * Own technicals: 10/21-day realised vol, 5/21-day momentum,
      MA cross (10/50), RSI-14, Bollinger Z-score

    cross_asset  (alternative model)
    -----------
    * Everything in own_only, PLUS for every OTHER asset:
      - Lagged returns at lag 1, 2, 3, 5  (the Granger predictors)
      - 21-day rolling correlation with target (DCC proxy)
      - Relative momentum (other - target, 21-day)

    Key design rule: every feature is shifted by at least 1 day so
    that no future information leaks into the model inputs.

    Returns dict keyed by target ticker.
    """
    log.info("Building cross-asset feature matrices ...")
    log_ret  = np.log(prices / prices.shift(1))
    features = {}
    COST     = COST_BPS / 10_000 * 2      # round-trip cost

    for target in TARGET_TICKERS:
        if target not in prices.columns:
            log.warning(f"  {target} not in prices. Skipping.")
            continue

        r   = log_ret[target]
        p   = prices[target]

        # ── OWN-ASSET features ────────────────────────────────────
        own = pd.DataFrame(index=prices.index)

        for lag in [1, 2, 3, 5]:
            own[f"own_lag_{lag}"] = r.shift(lag)

        own["own_vol_10d"]  = r.rolling(10).std() * np.sqrt(252)
        own["own_vol_21d"]  = r.rolling(21).std() * np.sqrt(252)
        own["own_mom_5d"]   = p.pct_change(5).shift(1)
        own["own_mom_21d"]  = p.pct_change(21).shift(1)
        own["own_ma_cross"] = (
            p.rolling(10).mean() / p.rolling(50).mean() - 1
        ).shift(1)
        own["own_rsi"]      = _rsi(r, 14).shift(1)
        own["own_bb_z"]     = (
            (p - p.rolling(20).mean())
            / p.rolling(20).std().replace(0, np.nan)
        ).shift(1)
        own["own_vol_ratio"] = (
            r.rolling(10).std() / r.rolling(63).std().replace(0, np.nan)
        ).shift(1)

        # ── CROSS-ASSET features ──────────────────────────────────
        cross  = own.copy()
        others = [t for t in prices.columns if t != target]

        for other in others:
            r_o = log_ret[other]

            # Granger predictors: lagged returns of other asset
            for lag in [1, 2, 3, 5]:
                cross[f"{other}_lag_{lag}"] = r_o.shift(lag)

            # Time-varying correlation proxy (shifted to avoid look-ahead)
            cross[f"corr_{other}_21"] = (
                r.rolling(21).corr(r_o).shift(1)
            )

            # Relative momentum: other leads or lags target
            cross[f"relmom_{other}_5"]  = (
                (r_o.rolling(5).mean()
                 - r.rolling(5).mean()).shift(1)
            )
            cross[f"relmom_{other}_21"] = (
                (r_o.rolling(21).mean()
                 - r.rolling(21).mean()).shift(1)
            )

        # ── Target: forward net return ────────────────────────────
        fwd_gross = (r.shift(-SIGNAL_HORIZON)
                      .rolling(SIGNAL_HORIZON).sum())
        fwd_net   = fwd_gross - COST

        # ── Clean ─────────────────────────────────────────────────
        own   = own.replace([np.inf, -np.inf], np.nan)
        cross = cross.replace([np.inf, -np.inf], np.nan)

        features[target] = {
            "own_only":    own,
            "cross_asset": cross,
            "fwd_net":     fwd_net,
            "fwd_gross":   fwd_gross,
            "log_ret":     r,
        }
        log.info(f"  {target}: own={own.shape[1]} features, "
                 f"cross={cross.shape[1]} features, "
                 f"n_rows={own.shape[0]}")

    return features


# =============================================================================
# 4.  MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ3 STAGE 0 -- Data & Cross-Asset Feature Engineering")
    log.info("=" * 65)

    prices_raw = download_prices()
    prices     = validate_prices(prices_raw)
    log_ret    = np.log(prices / prices.shift(1)).dropna(how="all")
    features   = build_features(prices)

    # Save prices and log returns
    prices.to_parquet(os.path.join(DATA_DIR, "prices.parquet"))
    log_ret.to_parquet(os.path.join(DATA_DIR, "rq3_log_returns.parquet"))

    # Save per-target feature sets
    # NOTE: own_only and cross_asset are DataFrames; fwd_net is a Series.
    # Series.to_parquet() does not exist -- convert to DataFrame first.
    saved = 0
    for target, fd in features.items():
        for key in ["own_only", "cross_asset"]:
            fd[key].to_parquet(
                os.path.join(DATA_DIR, f"rq3_{target}_{key}.parquet")
            )
        # fwd_net is a pd.Series -- save as single-column DataFrame
        fwd_df = fd["fwd_net"]
        if hasattr(fwd_df, "to_frame"):
            fwd_df = fwd_df.to_frame(name="fwd_net")
        fwd_df.to_parquet(
            os.path.join(DATA_DIR, f"rq3_{target}_fwd_net.parquet")
        )
        saved += 1

    log.info(f"\nSaved {saved} target feature sets to {DATA_DIR}")

    # Sanity check
    log.info("\n-- Feature sample (SPY cross_asset, last 3 rows, first 5 cols) --")
    if "SPY" in features:
        sample = (features["SPY"]["cross_asset"]
                  .dropna(how="all")
                  .iloc[-3:, :5])
        log.info(f"\n{sample.round(4).to_string()}")

    log.info("\n✓ Stage 0 complete. Run rq3_stage1_granger.py next.\n")
    return prices, features


if __name__ == "__main__":
    main()

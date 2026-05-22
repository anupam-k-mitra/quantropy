#!/usr/bin/env python3
"""
preprocess.py — Download and clean market data for all RQs.

Usage:
    python src/data/preprocess.py --market us       # US ETF universe
    python src/data/preprocess.py --market india    # India NSE universe
    python src/data/preprocess.py --market both     # Both (default)
"""
import argparse, os, sys, logging, time
import numpy as np
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# ── Universe definitions ───────────────────────────────────────────────────
US_UNIVERSE = {
    "equities": ["SPY","QQQ","IWM","GLD","TLT","HYG","IEF","USO","XLF","XLE","XLK"],
    "crypto":   ["BTC-USD","ETH-USD"],
    "fx":       ["EURUSD=X","GBPUSD=X","USDJPY=X","DX-Y.NYB"],
    "macro":    ["^VIX","^TNX","^IRX","GC=F","CL=F"],
}
INDIA_UNIVERSE = {
    "equities": ["TCS.NS","INFY.NS","WIPRO.NS","HCLTECH.NS","TECHM.NS",
                 "HDFCBANK.NS","ICICIBANK.NS","AXISBANK.NS","KOTAKBANK.NS",
                 "SBIN.NS","SBILIFE.NS",
                 "SUNPHARMA.NS","DRREDDY.NS","CIPLA.NS",
                 "ONGC.NS","COALINDIA.NS","BPCL.NS",
                 "HINDUNILVR.NS","ITC.NS","TITAN.NS",
                 "MARUTI.NS","LT.NS","ASIANPAINT.NS",
                 "^NSEI","^NSEBANK"],
    "crypto":   ["BTC-USD"],
    "fx":       ["USDINR=X"],
    "macro":    ["GC=F","CL=F","SPY","^INDIAVIX","^VIX"],
}
DATES = {
    "us":    ("2010-01-01","2024-12-31"),
    "india": ("2010-01-01","2026-04-30"),
}

def _dl(tickers, start, end, label):
    """Download a group of tickers with individual fallback."""
    log.info(f"  Downloading '{label}': {tickers}")
    try:
        raw = yf.download(tickers, start=start, end=end,
                          auto_adjust=True, progress=False,
                          threads=False, timeout=30)
        if isinstance(raw.columns, pd.MultiIndex):
            prices = raw["Close"] if "Close" in raw.columns.get_level_values(0) \
                     else raw.iloc[:, :len(tickers)]
        else:
            prices = raw if isinstance(raw, pd.DataFrame) \
                     else raw.to_frame(name=tickers[0])
    except Exception as e:
        log.warning(f"  Batch failed: {e} — retrying individually")
        prices = pd.DataFrame()

    results = {}
    for t in tickers:
        col = prices[t] if t in prices.columns else None
        if col is not None and col.notna().mean() > 0.50:
            results[t] = col
        else:
            time.sleep(0.5)
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
                ind = ind.squeeze()
                if ind.notna().mean() > 0.30:
                    results[t] = ind
                    log.info(f"    {t}: individual OK ({ind.notna().sum()} days)")
                else:
                    log.warning(f"    {t}: FAILED")
            except Exception as e2:
                log.warning(f"    {t}: {e2}")
    return pd.DataFrame(results)


def download_market(market: str, base_dir: str = "data"):
    universe = US_UNIVERSE if market == "us" else INDIA_UNIVERSE
    start, end = DATES[market]
    log.info(f"Downloading {market.upper()} universe ({start} → {end})")

    all_frames = []
    for cat, tickers in universe.items():
        raw_dir = os.path.join(base_dir, "raw", cat)
        os.makedirs(raw_dir, exist_ok=True)
        df = _dl(tickers, start, end, f"{market}/{cat}")
        if df.empty:
            continue
        df = df.ffill(limit=3).replace(0, np.nan)
        # Save per-category CSV
        df.to_csv(os.path.join(raw_dir, f"{market}_{cat}.csv"))
        all_frames.append(df)
        log.info(f"  {cat}: {df.shape[1]} tickers saved")

    if all_frames:
        combined = all_frames[0]
        for f in all_frames[1:]:
            combined = combined.join(f, how="outer")
        combined = combined.ffill(limit=3)
        proc_dir = os.path.join(base_dir, "processed")
        os.makedirs(proc_dir, exist_ok=True)
        combined.to_parquet(os.path.join(proc_dir, f"prices_{market}.parquet"))
        log.info(f"Combined {market}: {combined.shape} saved to data/processed/")
    return combined


def build_features(prices: pd.DataFrame, market: str) -> pd.DataFrame:
    """Compute the 38 candidate signals across 5 families."""
    log_ret = np.log(prices / prices.shift(1)).clip(-0.20, 0.20)
    feat = pd.DataFrame(index=prices.index)

    for col in log_ret.columns:
        r = log_ret[col]
        p = prices[col]
        stub = col.replace("^","").replace("=X","").replace(".","_").replace("-","_")
        # Momentum family
        for d in [5,21,63,126,252]:
            feat[f"{stub}_mom_{d}d"] = np.log(p/p.shift(d)).shift(1)
        # Mean-reversion
        feat[f"{stub}_rev_1d"] = r.shift(1)
        feat[f"{stub}_rev_5d"] = r.rolling(5).sum().shift(1)
        # RSI
        g = r.clip(lower=0).rolling(14).mean()
        l_ = (-r.clip(upper=0)).rolling(14).mean()
        feat[f"{stub}_rsi"] = (100 - 100/(1+g/(l_+1e-9))).shift(1)
        # Volatility
        feat[f"{stub}_vol21"] = r.rolling(21).std().shift(1) * (252**0.5)
        feat[f"{stub}_vol63"] = r.rolling(63).std().shift(1) * (252**0.5)

    proc_dir = "data/processed"
    os.makedirs(proc_dir, exist_ok=True)
    feat.to_parquet(os.path.join(proc_dir, "features.parquet"))
    log.info(f"Features: {feat.shape} saved to data/processed/features.parquet")
    return feat


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess market data")
    parser.add_argument("--market", choices=["us","india","both"], default="both")
    parser.add_argument("--base-dir", default="data")
    args = parser.parse_args()

    markets = ["us","india"] if args.market == "both" else [args.market]
    for m in markets:
        prices = download_market(m, args.base_dir)
        build_features(prices, m)
    log.info("Done. Run pipelines/ scripts or open notebooks/ next.")


if __name__ == "__main__":
    main()

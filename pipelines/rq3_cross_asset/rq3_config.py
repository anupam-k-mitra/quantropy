#!/usr/bin/env python3
# =============================================================================
# rq3_config.py
# Central configuration for RQ3 -- Cross-Asset Dependencies pipeline.
# =============================================================================

import os

# ── Asset Universe ─────────────────────────────────────────────────────────
TICKERS = {
    "SPY":      "S&P 500 ETF",
    "QQQ":      "Nasdaq 100 ETF",
    "IWM":      "Russell 2000 ETF",
    "EEM":      "Emerging Markets ETF",
    "TLT":      "20Y+ Treasury ETF",
    "HYG":      "High Yield Bond ETF",
    "GLD":      "Gold ETF",
    "USO":      "Oil ETF",
    "BTC-USD":  "Bitcoin",
    "^VIX":     "CBOE VIX",
    "^TNX":     "10Y Treasury Yield",
    "DX-Y.NYB": "US Dollar Index",
}

TARGET_TICKERS   = ["SPY", "QQQ", "TLT", "GLD"]

START_DATE       = "2010-01-01"
END_DATE         = "2024-12-31"
COST_BPS         = 5.0
RF_ANNUAL        = 0.04

GRANGER_MAX_LAG  = 5
GRANGER_ALPHA    = 0.05
MIN_CAUSAL_PAIRS = 3

DCC_LOOKBACK     = 252

TRAIN_WINDOW     = 504
TEST_WINDOW      = 21
SIGNAL_HORIZON   = 5

DM_ALPHA         = 0.05

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "rq3_outputs")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

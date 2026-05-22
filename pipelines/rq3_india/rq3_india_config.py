#!/usr/bin/env python3
# =============================================================================
# rq3_india_config.py
# Central configuration for RQ3 India — Cross-Asset Dependencies (NSE)
#
# Key structural differences vs US config:
#   1. Risk-free rate: 6.5% (RBI repo) vs 4% (US T-bill)
#   2. Cost: 20 bps vs 5 bps (STT + brokerage + NSE fees)
#   3. No liquid INR bond ETF equivalent to TLT/HYG — use G-Sec yield proxy
#   4. USD/INR is a primary cross-asset driver (not present in US universe)
#   5. Sector indices capture rotation patterns unique to Indian market
#   6. Data from 2010 — covers GFC recovery, demonetisation, COVID, FII cycles
# =============================================================================

import os

# ── Indian asset universe ──────────────────────────────────────────────────
# Format: ticker -> description
# yfinance tickers: NSE equities use .NS suffix, indices use ^ prefix
TICKERS = {
    # ── NSE Broad Indices (primary cross-asset signals) ──
    "^NSEI":       "Nifty 50 Index",
    "^NSEBANK":    "Nifty Bank Index",

    # ── Sector ETFs / Proxies (NSE sector rotation signals) ──
    # IT sector — USD/INR sensitive (TCS, Infosys earn in USD)
    "INFY.NS":     "Infosys (IT sector proxy)",
    "TCS.NS":      "TCS (IT sector proxy)",
    # Financials
    "HDFCBANK.NS": "HDFC Bank (private banking)",
    "SBIN.NS":     "SBI (PSU banking / rate proxy)",
    # Commodities / Energy
    "ONGC.NS":     "ONGC (crude oil proxy)",
    "COALINDIA.NS":"Coal India (commodity)",
    # Pharma (defensive / export sector)
    "SUNPHARMA.NS":"Sun Pharma (pharma sector)",
    # Consumer
    "HINDUNILVR.NS":"HUL (consumer staples / defensive)",

    # ── Macro / Cross-Asset Signals ──
    "USDINR=X":    "USD/INR exchange rate (dominant macro driver)",
    "GC=F":        "Gold futures USD (global safe-haven)",
    "CL=F":        "Crude oil futures USD (energy/CAD)",
    "BTC-USD":     "Bitcoin (global risk appetite)",

    # ── Global context (India is globally coupled) ──
    "SPY":         "S&P 500 ETF (global risk-on/off)",
    "^VIX":        "CBOE VIX (global fear index)",
}

# ── Target assets for DM test (own vs cross prediction) ─────────────────────
# These are the assets we try to PREDICT better using cross-asset information.
# Chosen to represent the key RQ3 hypotheses:
#   ^NSEI    — broad market: should benefit from all cross-asset signals
#   INFY.NS  — IT sector: strong USD/INR dependency hypothesis
#   SBIN.NS  — PSU bank: RBI rate + global credit spread dependency
#   GC=F     — Gold: USD, VIX, real-yield proxy dependency
TARGET_TICKERS = ["^NSEI", "INFY.NS", "SBIN.NS", "GC=F"]

# ── Date range ───────────────────────────────────────────────────────────────
# Start 2010: post-GFC recovery. Captures demonetisation (2016),
# NBFC crisis (2018), COVID crash (2020), FII outflow episode (2021-22).
START_DATE = "2010-01-01"
END_DATE   = "2026-04-30"

# ── Market parameters ────────────────────────────────────────────────────────
COST_BPS   = 20.0    # NSE STT + brokerage + SEBI charges
RF_ANNUAL  = 0.065   # RBI repo rate

# ── Granger test parameters ──────────────────────────────────────────────────
GRANGER_MAX_LAG  = 5     # up to 5-day lag (one trading week)
GRANGER_ALPHA    = 0.05
MIN_CAUSAL_PAIRS = 3     # H1 condition: >= 3 BH-significant pairs

# ── DCC-GARCH parameters ─────────────────────────────────────────────────────
DCC_LOOKBACK = 252       # 1-year rolling estimation window

# ── ML walk-forward parameters ───────────────────────────────────────────────
TRAIN_WINDOW   = 756     # 3 years: more folds for DM power; 756/30 feats = 25 S/F ratio
TEST_WINDOW    = 63      # quarterly
SIGNAL_HORIZON = 5       # 5-day forward return prediction
DM_ALPHA       = 0.05

# ── Key Indian structural break dates (for Bai-Perron / Chow tests) ─────────
# Used in RQ2 regime analysis and referenced in RQ3 DCC commentary.
INDIA_STRUCTURAL_BREAKS = {
    "2013-06-01": "Taper Tantrum — INR collapsed 15% in 3 months, FII outflows",
    "2016-11-08": "Demonetisation — 86% of currency notes withdrawn overnight",
    "2018-09-01": "NBFC crisis — IL&FS default triggered credit market freeze",
    "2020-03-23": "COVID crash — Nifty -38% peak-to-trough in 40 days",
    "2022-01-01": "FII outflow cycle — global rate shock, FIIs sold $17B in 6M",
}

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "data_india")
OUTPUT_DIR = os.path.join(BASE_DIR, "rq3_india_outputs")
os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

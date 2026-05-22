#!/usr/bin/env python3
# =============================================================================
# rq2_india_config.py — Central configuration for RQ2 India
#
# Design decisions (documented):
#   States  : 3 primary (Bull/Bear/Crisis), 4 robustness check
#   Vol     : India VIX (^INDIAVIX) preferred; 21d realised vol fallback
#   Variables: 6 — Nifty return, vol, USDINR, SPY, Gold, Crude
#   Gold/Crude: included because India is world's 2nd gold consumer
#               and imports 85% of crude — both define distinct regimes
#   SPY     : global FII coupling channel (US risk-off → India selling)
# =============================================================================
import os

# ── India universe (regime detection) ─────────────────────────────────────
TICKERS = [
    "^NSEI",       # Nifty 50          — primary strategy asset
    "^NSEBANK",    # Nifty Bank        — domestic credit / rate regime
    "^INDIAVIX",   # India VIX         — NSE implied vol (forward-looking)
    "USDINR=X",    # USD/INR           — FX stress, India-specific driver
    "SPY",         # S&P 500 ETF       — global risk-on/off / FII proxy
    "GC=F",        # Gold futures      — India gold demand + safe-haven
    "CL=F",        # Crude oil futures — 85% import dependency
    "INFY.NS",     # Infosys           — IT sector / USD/INR sensitivity
    "HDFCBANK.NS", # HDFC Bank         — private banking / domestic growth
]

# ── Date range ────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2026-04-30"

# ── Market parameters ─────────────────────────────────────────────────────
COST_BPS          = 20.0     # NSE STT + brokerage
RISK_FREE_ANNUAL  = 0.065    # RBI repo rate

# ── Regime detection ──────────────────────────────────────────────────────
N_REGIMES_PRIMARY    = 3     # Bull / Bear / Crisis (primary model)
N_REGIMES_ROBUST     = 4     # + Sideways (robustness / BIC check)
HMM_COVARIANCE_TYPE  = "full"
HMM_N_ITER           = 300   # more iterations for 6-var model
HMM_N_INIT           = 15    # EM restarts (India data noisier)
GMM_N_INIT           = 15

# ── Structural break detection ────────────────────────────────────────────
BP_MIN_SIZE   = 63           # minimum segment = 1 quarter
BP_MAX_BREAKS = 6            # India has more breaks than US

# ── India structural breaks (Bai-Perron reference dates) ─────────────────
INDIA_BREAKS = {
    "2013-06-01": "Taper Tantrum — INR -15%, FII outflows",
    "2016-11-08": "Demonetisation — 86% of currency notes withdrawn",
    "2018-09-01": "NBFC crisis — IL&FS default, credit market freeze",
    "2020-03-23": "COVID crash — Nifty -38% peak-to-trough in 40 days",
    "2022-01-01": "FII outflow cycle — global rate shock, $17B sold",
}

# ── Walk-forward windows ──────────────────────────────────────────────────
TRAIN_WINDOW    = 504        # 2 years (consistent with US RQ2)
TEST_WINDOW     = 63         # 1 quarter OOS
MIN_REGIME_OBS  = 20         # minimum obs per regime per fold

# ── Statistical thresholds ────────────────────────────────────────────────
ALPHA           = 0.05
MIN_SHARPE_GAIN = 0.20       # H1: delta Sharpe >= 0.20 for regime strategy
MAX_REGIME_DD   = -0.30

# ── Regime labels (3-state primary) ──────────────────────────────────────
# Ordered by ascending realised volatility (0=lowest, 2=highest)
REGIME_NAMES_3 = {0: "Bull", 1: "Bear", 2: "Crisis"}
REGIME_NAMES_4 = {0: "Bull", 1: "Sideways", 2: "Bear", 3: "Crisis"}

# ── Position scaling per regime (adaptive strategy) ───────────────────────
REGIME_SCALES_3 = {
    0: 1.00,   # Bull:   full exposure
    1: 0.35,   # Bear:   reduced (India bear can be sharp)
    2: 0.00,   # Crisis: full exit
}
REGIME_SCALES_4 = {
    0: 1.00,   # Bull:     full exposure
    1: 0.60,   # Sideways: partial (mean-reversion partial position)
    2: 0.30,   # Bear:     reduced
    3: 0.00,   # Crisis:   full exit
}

# ── Turbulence index tickers (India cross-asset) ──────────────────────────
TURBULENCE_TICKERS  = ["^NSEI", "GC=F", "CL=F", "USDINR=X"]
TURBULENCE_LOOKBACK = 252
TURBULENCE_CRISIS_PCT = 95

# ── Output dirs ───────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "rq2_india_outputs")
DATA_DIR   = os.path.join(BASE_DIR, "data_india_rq2")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

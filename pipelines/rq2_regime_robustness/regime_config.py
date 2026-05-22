#!/usr/bin/env python3
# =============================================================================
# regime_config.py
# Central configuration for RQ2 — Regime Robustness pipeline.
# Edit this file to adjust tickers, windows, thresholds.
# =============================================================================

import os

# ── Asset Universe ─────────────────────────────────────────────────────────
TICKERS = [
    "SPY",        # S&P 500 ETF           — primary strategy asset
    "QQQ",        # Nasdaq 100 ETF        — tech / growth regime signal
    "IWM",        # Russell 2000 ETF      — risk appetite (small-cap)
    "TLT",        # 20Y+ Treasury ETF     — flight-to-quality signal
    "IEF",        # 7-10Y Treasury ETF    — intermediate duration
    "HYG",        # High Yield Bond ETF   — credit stress signal
    "GLD",        # Gold ETF              — uncertainty / inflation hedge
    "USO",        # Oil ETF               — macro / growth proxy
    "^VIX",       # CBOE VIX              — PRIMARY regime feature
    "^TNX",       # 10Y Treasury Yield    — rate regime
    "^IRX",       # 3M T-Bill             — yield curve shape
    "DX-Y.NYB",   # US Dollar Index       — global risk-off indicator
]

# ── Date Range ─────────────────────────────────────────────────────────────
START_DATE = "2010-01-01"
END_DATE   = "2024-12-31"

# ── Transaction Costs ──────────────────────────────────────────────────────
COST_BPS       = 5.0            # basis points one-way
RISK_FREE_ANNUAL = 0.04         # 4% annual risk-free rate

# ── Regime Detection ───────────────────────────────────────────────────────
N_REGIMES           = 4         # Bull / Bear / Sideways / Crisis
HMM_COVARIANCE_TYPE = "full"
HMM_N_ITER          = 200
HMM_N_INIT          = 10        # EM restarts to avoid local optima
GMM_N_INIT          = 10

# ── Structural Break Detection (Bai-Perron) ────────────────────────────────
BP_MIN_SIZE   = 63              # minimum segment length (1 quarter)
BP_MAX_BREAKS = 8               # maximum breaks to test

# ── Walk-Forward Windows ───────────────────────────────────────────────────
TRAIN_WINDOW   = 504            # 2 years
TEST_WINDOW    = 63             # 1 quarter out-of-sample
MIN_REGIME_OBS = 30             # minimum obs per regime per fold

# ── Statistical Thresholds ─────────────────────────────────────────────────
ALPHA           = 0.05
MIN_SHARPE_GAIN = 0.20          # H1 requires delta Sharpe >= 0.20
MAX_REGIME_DD   = -0.30         # catastrophic drawdown threshold

# ── Turbulence Index (Kritzman et al. 2012) ────────────────────────────────
TURBULENCE_LOOKBACK   = 252
TURBULENCE_CRISIS_PCT = 95

# ── Regime Scaling Rules ───────────────────────────────────────────────────
# Position size multiplier per detected regime.
# 0=Bull, 1=Bear, 2=Sideways, 3=Crisis (canonical ordering by frequency)
REGIME_SCALES = {
    0: 1.00,    # Bull:     full exposure
    1: 0.30,    # Bear:     heavily reduced
    2: 0.60,    # Sideways: partial exposure
    3: 0.00,    # Crisis:   full exit
}

# ── Output Directories ─────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "rq2_outputs")
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR,   exist_ok=True)

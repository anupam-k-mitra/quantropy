# Quantropy — Quantitative Investment / Trading System

**QM640 Data Analytics Capstone · Walsh College · DBA Program · Anupam K Mitra · May 2026**

A multi-layered quantitative investment framework addressing four interconnected research questions
across two markets: the **US ETF universe** (12 assets, 2010–2024) and the **Indian NSE large-cap
universe** (25 assets, 2010–2026).

---

## Results Summary

| RQ | Research Question | US Market | India NSE |
|---|---|---|---|
| RQ1 | Signal Predictability | Fail to reject H0 (t=1.87) | **REJECT H0 ✓** (ICIR=0.738) |
| RQ2 | Regime Robustness | Fail to reject H0 | **REJECT H0 ✓** (Levene p=0.000) |
| RQ3 | Cross-Asset Dependencies | **REJECT H0 ✓** (SPY+GLD) | **REJECT H0 ✓** (GC=F) |
| RQ4 | Risk & Drawdown Control | **REJECT H0 ✓** (MDD −71.9%) | **REJECT H0 ✓** (MDD −72.4%) |

**Key finding:** Gold (GLD/GC=F) rejects H0 via both XGBoost and TFT independently in both
markets — same asset class, two model families, two independent datasets.

---

## Repository Structure

```
quantropy/
├── data/
│   ├── raw/
│   │   ├── equities/      # Daily OHLCV — US ETFs and NSE large-caps
│   │   ├── crypto/        # BTC-USD, ETH-USD
│   │   ├── fx/            # USDINR=X, EURUSD=X, DX-Y.NYB
│   │   └── macro/         # ^VIX, ^INDIAVIX, ^TNX, ^IRX, GC=F, CL=F
│   └── processed/
│       ├── features.parquet     # Master feature table (38 signals, 5 families)
│       └── regimes.parquet      # HMM/GMM regime labels (US + India)
├── notebooks/
│   ├── 01_EDA.ipynb             # Exploratory data analysis
│   ├── 03_RQ1_Signals.ipynb     # Signal discovery and alpha testing
│   ├── 04_RQ2_Regimes.ipynb     # Regime detection and robustness
│   ├── 05_RQ3_CrossAsset.ipynb  # Granger, DCC-GARCH, TFT attention
│   └── 06_RQ4_RiskControl.ipynb # CVaR, kill-switch, DRL
├── pipelines/
│   ├── rq1_signals/             # RQ1 four-stage pipeline (US)
│   ├── rq2_regime_robustness/   # RQ2 pipeline (US)
│   ├── rq2_india/               # RQ2 pipeline (India NSE)
│   ├── rq3_cross_asset/         # RQ3 pipeline (US) + TFT transformer
│   ├── rq3_india/               # RQ3 pipeline (India NSE)
│   └── rq4_risk_control/        # RQ4 two-zone kill-switch pipeline
├── src/
│   ├── data/preprocess.py       # Cleaning, feature computation
│   ├── models/signal_model.py   # RQ1: XGBoost + Fama-MacBeth
│   ├── models/regime_model.py   # RQ2: HMM/GMM regime detection
│   ├── models/attention_model.py# RQ3: TFT Transformer cross-asset
│   ├── models/risk_model.py     # RQ4: CVaR, kill-switch, DRL
│   └── backtest/engine.py       # Walk-forward backtest engine
└── results/
    └── figures/                 # All generated charts and plots
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/anupam-k-mitra/quantropy.git
cd quantropy
pip install -r requirements.txt
# or with conda:
conda env create -f environment.yml
conda activate quantropy
```

### 2. Download data

```bash
python src/data/preprocess.py --market us    # US ETF universe
python src/data/preprocess.py --market india # India NSE universe
```

### 3. Run individual pipelines

```bash
# US pipelines
cd pipelines/rq3_cross_asset && python rq3_run_pipeline.py
cd pipelines/rq4_risk_control && python rq4_pipeline.py --market both

# India pipelines
cd pipelines/rq2_india       && python rq2_india_run_pipeline.py
cd pipelines/rq3_india       && python rq3_india_run_pipeline.py
```

### 4. Run notebooks

```bash
jupyter lab
# Open notebooks/ in order: 01 → 03 → 04 → 05 → 06
```

---

## Data

All data are downloaded via [yfinance](https://github.com/ranaroussi/yfinance).
**Raw data files are excluded from this repository** (see `.gitignore`) due to
Yahoo Finance terms of service. Run the download scripts above to recreate them.

Processed features and regime labels (`data/processed/*.parquet`) are generated
by the pipeline scripts and are also excluded from version control.

---

## Key Parameters

| Parameter | US Market | India NSE |
|---|---|---|
| Universe | 12 ETFs | 25 NSE large-caps |
| Period | Jan 2010 – Dec 2024 | Jan 2010 – Apr 2026 |
| Risk-free rate | 4.0% (T-bill) | 6.5% (RBI repo) |
| Transaction cost | 5 bps one-way | 20 bps one-way |
| Benchmark | SPY | ^NSEI (Nifty 50) |
| Train window | 504 # days (2yr) | 756 * days (3yr) |
| Test window | 63 # days (qtr) | 63 * days (qtr) |
# Walk forward validation uses a 2 year rolling training window for US data stepped forward by a quarterly test window in each fold(yields 50 OOS test folds).
* Walk forward validation uses a 3 year rolling training window for India data stepped forward by a quarterly test window in each fold(yields 40 OOS test folds).
Rolling window ensures model parameters reflect prevailing market conditions rather than a long term average that might be obsolete. All 14+ years data are evaluated OOS.
---

## Dependencies

- Python 3.10+
- yfinance, pandas, numpy, scipy, statsmodels
- scikit-learn, xgboost, hmmlearn
- torch (PyTorch ≥ 2.0 for TFT; CPU build sufficient)
- matplotlib, seaborn
- python-docx, python-pptx (report generation)

See `requirements.txt` for pinned versions.

---

## Citation / Academic Context

This work is submitted as the capstone project for QM640: Data Analytics at
Walsh College, Doctor of Business Administration program.

Key references:
- Gu, Kelly & Xiu (2020) — ML asset pricing, nonlinearity
- Hamilton (1989) — Hidden Markov Model regime switching
- Lim et al. (2021) — Temporal Fusion Transformer
- Schulman et al. (2017) — PPO / DRL position sizing
- Kupiec (1995) — VaR coverage test

---

*Anupam K Mitra · DBA, Walsh College · QM640 · May 2026*

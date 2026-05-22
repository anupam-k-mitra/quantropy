# data/raw/macro/

Daily macro indicators: volatility indices, yield curves, commodity futures.

**Files generated here after running `preprocess.py`:**

| File | Tickers | Role |
|---|---|---|
| `us_macro.csv` | ^VIX, ^TNX, ^IRX, GC=F, CL=F | Fear index; yield curve; gold; crude |
| `india_macro.csv` | ^INDIAVIX, ^VIX, GC=F, CL=F, SPY | India VIX preferred over CBOE VIX |

> **Note on India VIX:** `^INDIAVIX` correctly captures India-specific crises (e.g. demonetisation Nov 2016: India VIX spiked to 33 while CBOE VIX barely moved). If `^INDIAVIX` is unavailable on your yfinance version, the pipeline automatically falls back to 21-day realised volatility computed from `^NSEI`.

> Excluded from version control via `.gitignore`.

# data/raw/equities/

Daily OHLCV data for equity assets, downloaded via yfinance.

**Files generated here after running:**
```bash
python src/data/preprocess.py --market us
python src/data/preprocess.py --market india
```

| File | Contents |
|---|---|
| `us_equities.csv` | SPY, QQQ, IWM, GLD, TLT, HYG, IEF, USO, XLF, XLE, XLK — daily adjusted close |
| `india_equities.csv` | 25 NSE large-caps (.NS suffix) + ^NSEI, ^NSEBANK — daily adjusted close |

> Data files are excluded from version control (`.gitignore`) in accordance with Yahoo Finance terms of service.

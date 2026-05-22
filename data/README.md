# Data Directory

Raw data files are **not committed** to this repository (Yahoo Finance Terms of Service).

## Regenerate data

```bash
# From the repo root:
python src/data/preprocess.py --market us     # downloads to data/raw/ and data/processed/
python src/data/preprocess.py --market india
```

## Structure after download

```
data/
├── raw/
│   ├── equities/   us_equities.csv, india_equities.csv
│   ├── crypto/     us_crypto.csv, india_crypto.csv
│   ├── fx/         us_fx.csv, india_fx.csv
│   └── macro/      us_macro.csv, india_macro.csv
└── processed/
    ├── prices_us.parquet
    ├── prices_india.parquet
    ├── features.parquet      # 38-signal feature matrix
    └── regimes.parquet       # HMM/GMM regime labels
```

#!/usr/bin/env python3
"""
Standalone TFT runner for India — bypasses the environment mismatch.

Run this script DIRECTLY:
    python run_tft_india.py

It diagnoses the torch environment, prints the exact Python and torch
path being used, and runs the transformer with full error reporting.
Place this file in the same folder as rq3_stage3_transformer.py
and rq3_india_stage3_ml.py.
"""
import sys, os

# ── Step 1: show exactly which Python is running ──────────────────────────
print(f"Python executable : {sys.executable}")
print(f"Python version    : {sys.version.split()[0]}")
print(f"Script location   : {os.path.abspath(__file__)}")
print()

# ── Step 2: add script folder to path ─────────────────────────────────────
here = os.path.dirname(os.path.abspath(__file__))
for folder in [here,
               os.path.join(here, '..', 'rq3_cross_asset'),
               os.path.join(here, '..', 'rq3_india')]:
    p = os.path.normpath(folder)
    if os.path.isdir(p) and p not in sys.path:
        sys.path.insert(0, p)

# ── Step 3: try torch import with full diagnostics ─────────────────────────
print("Testing PyTorch import ...")
try:
    import torch
    print(f"  torch version : {torch.__version__}")
    print(f"  torch location: {torch.__file__}")
    t = torch.tensor([1.0])
    print(f"  test tensor   : {t}  ← OK")
    TORCH_OK = True
except (ImportError, OSError) as e:
    print(f"  FAILED: {e}")
    print()
    print("  The torch that this Python can see is broken.")
    print(f"  Fix: run THIS command in your terminal:")
    print(f"    {sys.executable} -m pip install torch "
          "--index-url https://download.pytorch.org/whl/cpu")
    print()
    print("  Then rerun: python run_tft_india.py")
    print()

    # Try importing anyway with the NumPy fallback explicitly
    TORCH_OK = False
    # Signal to transformer module before it loads
    import builtins
    builtins.TORCH_OK = False

print()

# ── Step 4: load India config ─────────────────────────────────────────────
try:
    from rq3_india_config import (
        DATA_DIR, OUTPUT_DIR, TARGET_TICKERS,
        TRAIN_WINDOW, TEST_WINDOW, SIGNAL_HORIZON, DM_ALPHA,
    )
    print(f"India config loaded. Targets: {TARGET_TICKERS}")
except ImportError as e:
    print(f"Cannot load rq3_india_config: {e}")
    print("Make sure this script is in the rq3_india folder.")
    sys.exit(1)

# ── Step 5: load features loader ──────────────────────────────────────────
def load_features(target):
    stub = target.replace("^","").replace("=","").replace(".","_")
    paths = {
        "own":   os.path.join(DATA_DIR, f"rq3_india_{stub}_own_only.parquet"),
        "cross": os.path.join(DATA_DIR, f"rq3_india_{stub}_cross_asset.parquet"),
        "fwd":   os.path.join(DATA_DIR, f"rq3_india_{stub}_fwd_net.parquet"),
    }
    import pandas as pd
    for k, p in paths.items():
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found — run stage 0 first")
    fwd = pd.read_parquet(paths["fwd"])
    if isinstance(fwd, pd.DataFrame):
        fwd = fwd.iloc[:, 0]
    return (pd.read_parquet(paths["own"]),
            pd.read_parquet(paths["cross"]),
            fwd)

# ── Step 6: run transformer ────────────────────────────────────────────────
print("Loading transformer module ...")
try:
    import importlib

    # Remove any cached broken version
    for mod in list(sys.modules.keys()):
        if 'transformer' in mod:
            del sys.modules[mod]

    tft = importlib.import_module('rq3_stage3_transformer')

    # Override config to point at India data
    tft.DATA_DIR        = DATA_DIR
    tft.OUTPUT_DIR      = OUTPUT_DIR
    tft.TARGET_TICKERS  = TARGET_TICKERS
    tft.TRAIN_WINDOW    = TRAIN_WINDOW
    tft.TEST_WINDOW     = TEST_WINDOW
    tft.SIGNAL_HORIZON  = SIGNAL_HORIZON
    tft.DM_ALPHA        = DM_ALPHA
    tft.load_features   = load_features   # use India loader

    print(f"Transformer backend: "
          f"{'TFT (PyTorch)' if tft.TORCH_OK else 'NumPy attention (fallback)'}")
    print()

    results = tft.run_transformer_stage(TARGET_TICKERS)

    # Save results to India output folder
    import pandas as pd
    rows = []
    for target, r in results.items():
        rows.append(dict(
            target=target, model="transformer",
            RMSE_own=r.get("RMSE_own"), RMSE_cross=r.get("RMSE_cross"),
            pct_reduction=r.get("pct_reduction"),
            DM_stat=r.get("DM_stat"), DM_p_val=r.get("DM_p_val"),
            DM_reject=r.get("DM_reject"),
            R2_own=r.get("R2_own"), R2_cross=r.get("R2_cross"),
            R2_delta=r.get("R2_delta"),
        ))
    if rows:
        out = os.path.join(OUTPUT_DIR, "rq3_transformer_results.csv")
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nResults saved → {out}")

except Exception as e:
    import traceback
    print(f"Transformer failed: {e}")
    traceback.print_exc()

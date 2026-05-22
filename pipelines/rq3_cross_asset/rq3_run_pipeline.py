#!/usr/bin/env python3
# =============================================================================
# rq3_run_pipeline.py
# Master runner for the RQ3 Cross-Asset Dependencies pipeline.
#
# Usage
# -----
#   python rq3_run_pipeline.py             # run all stages
#   python rq3_run_pipeline.py --from 1    # resume from Stage 1
#   python rq3_run_pipeline.py --only 3    # run Stage 3 only
#   python rq3_run_pipeline.py --stop-on-fail
#
# Stages
# ------
#   0 -- Data download & feature engineering
#   1 -- Granger causality network (all N*(N-1) pairs)
#   2 -- DCC correlations & Diebold-Yilmaz spillover
#   3 -- ML walk-forward + Diebold-Mariano test
#   4 -- Final H0 synthesis & scorecard
# =============================================================================

import sys
import time
import argparse
import logging
import traceback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

STAGES = {
    0: ("rq3_stage0_data",       "Data & Cross-Asset Feature Engineering"),
    1: ("rq3_stage1_granger",    "Granger Causality Network"),
    2: ("rq3_stage2_dcc",        "DCC Correlations & Spillover Index"),
    3: ("rq3_stage3_ml",         "ML Walk-Forward & Diebold-Mariano Test"),
    4: ("rq3_stage4_synthesis",  "Final H0 Synthesis & Scorecard"),
}

BANNER = """
+===============================================================+
|  RQ3 PIPELINE  --  Cross-Asset Dependencies                  |
|                                                               |
|  H0: Assets informationally independent                      |
|  H1: Cross-asset signals improve OOS RMSE (DM p < 0.05)     |
+===============================================================+
"""


def run_stage(n: int) -> bool:
    module, desc = STAGES[n]
    log.info("\n" + "#" * 65)
    log.info(f"  Stage {n}: {desc}")
    log.info("#" * 65)
    t0 = time.time()
    try:
        mod = __import__(module)
        mod.main()
        log.info(f"  Stage {n} OK ({time.time()-t0:.1f}s)")
        return True
    except Exception as e:
        log.error(f"\n  Stage {n} FAILED: {e}")
        log.error(traceback.format_exc())
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="RQ3 Cross-Asset Dependencies pipeline."
    )
    p.add_argument("--from",  dest="from_stage", type=int, default=0,
                   help="Start from stage N (default: 0)")
    p.add_argument("--only",  dest="only_stage", type=int, default=None,
                   help="Run only stage N")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Stop if any stage fails")
    return p.parse_args()


def main():
    print(BANNER)
    args   = parse_args()
    stages = ([args.only_stage]
              if args.only_stage is not None
              else list(range(args.from_stage, len(STAGES))))

    log.info(f"Stages to run: {stages}")
    results = {}
    t_start = time.time()

    for n in stages:
        if n not in STAGES:
            log.warning(f"Unknown stage {n}. Skipping.")
            continue
        ok = run_stage(n)
        results[n] = ok
        if not ok and args.stop_on_fail:
            log.error(f"Stopping at Stage {n}.")
            break

    # Summary
    elapsed = time.time() - t_start
    print("\n" + "=" * 65)
    print("  RQ3 PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    for n, ok in results.items():
        _, desc = STAGES[n]
        print(f"  Stage {n}: {desc:<45s}  {'OK' if ok else 'FAILED'}")
    print(f"\n  Total time: {elapsed/60:.1f} minutes")
    print("=" * 65)

    # List output files
    from rq3_config import OUTPUT_DIR
    import os
    if os.path.exists(OUTPUT_DIR):
        files = sorted(os.listdir(OUTPUT_DIR))
        if files:
            print(f"\n  Outputs in {OUTPUT_DIR}/")
            for f in files:
                sz = os.path.getsize(os.path.join(OUTPUT_DIR, f))
                print(f"    {f:<52s}  {sz/1024:.1f} KB")

    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()

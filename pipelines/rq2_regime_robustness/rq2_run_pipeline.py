#!/usr/bin/env python3
# =============================================================================
# rq2_run_pipeline.py
# Master runner for the RQ2 Regime Robustness pipeline.
#
# Usage
# -----
#   python rq2_run_pipeline.py            # run all stages
#   python rq2_run_pipeline.py --from 2   # resume from Stage 2
#   python rq2_run_pipeline.py --only 3   # run one stage only
#   python rq2_run_pipeline.py --stop-on-fail
#
# Stages
# ------
#   0 -- Data download & regime feature engineering
#   1 -- Regime detection (HMM / GMM / Bai-Perron / Consensus)
#   2 -- Statistical tests (ANOVA / Kruskal-Wallis / Levene)
#   3 -- Adaptive strategy & Sharpe improvement test
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
    0: ("rq2_stage0_data",       "Data Download & Feature Engineering"),
    1: ("rq2_stage1_detection",  "Regime Detection (HMM / GMM / BP)"),
    2: ("rq2_stage2_anova",      "Statistical Tests (ANOVA / KW / Levene)"),
    3: ("rq2_stage3_adaptive",   "Adaptive Strategy & Sharpe Test"),
    4: ("rq2_stage4_synthesis",  "Final H0 Synthesis & Scorecard"),
}

BANNER = """
+===============================================================+
|  RQ2 PIPELINE  --  Regime Robustness                         |
|                                                               |
|  H0: Strategy performance invariant across market regimes    |
|  H1: Regime conditioning improves Sharpe >= 0.20             |
+===============================================================+
"""


def run_stage(n):
    module, desc = STAGES[n]
    log.info("\n" + "#" * 65)
    log.info(f"  Stage {n}: {desc}")
    log.info("#" * 65)
    t0 = time.time()
    try:
        mod = __import__(module)
        mod.main()
        elapsed = time.time() - t0
        log.info(f"  Stage {n} completed in {elapsed:.1f}s  [OK]\n")
        return True
    except Exception as e:
        log.error(f"\n  Stage {n} FAILED: {e}")
        log.error(traceback.format_exc())
        return False


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the RQ2 Regime Robustness pipeline."
    )
    p.add_argument("--from",  dest="from_stage", type=int, default=0,
                   help="Start from this stage (default: 0)")
    p.add_argument("--only",  dest="only_stage", type=int, default=None,
                   help="Run only this stage")
    p.add_argument("--stop-on-fail", action="store_true",
                   help="Stop if a stage fails")
    return p.parse_args()


def main():
    print(BANNER)
    args = parse_args()

    stages = (
        [args.only_stage]
        if args.only_stage is not None
        else list(range(args.from_stage, len(STAGES)))
    )
    log.info(f"Stages to run: {stages}")

    results   = {}
    t_start   = time.time()

    for n in stages:
        if n not in STAGES:
            log.warning(f"Unknown stage {n}. Skipping.")
            continue
        ok = run_stage(n)
        results[n] = ok
        if not ok and args.stop_on_fail:
            log.error(f"Stopping pipeline at Stage {n}.")
            break

    # Summary
    total = time.time() - t_start
    print("\n" + "=" * 65)
    print("  RQ2 PIPELINE EXECUTION SUMMARY")
    print("=" * 65)
    for n, ok in results.items():
        _, desc = STAGES[n]
        status  = "OK" if ok else "FAILED"
        print(f"  Stage {n}: {desc:<45s}  {status}")
    print(f"\n  Total time: {total/60:.1f} minutes")
    print("=" * 65)

    # List output files
    from regime_config import OUTPUT_DIR
    import os
    if os.path.exists(OUTPUT_DIR):
        files = sorted(os.listdir(OUTPUT_DIR))
        if files:
            print(f"\n  Outputs in {OUTPUT_DIR}/")
            for f in files:
                size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
                print(f"    {f:<50s}  {size/1024:.1f} KB")

    all_ok = all(results.values())
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

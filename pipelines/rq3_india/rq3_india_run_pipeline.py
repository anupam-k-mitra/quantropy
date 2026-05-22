#!/usr/bin/env python3
"""
RQ3 India — Full Pipeline Runner
Usage:
  python rq3_india_run_pipeline.py                # all stages
  python rq3_india_run_pipeline.py --no-transformer  # skip TFT (faster)
  python rq3_india_run_pipeline.py --stage 0      # single stage
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse, time

def run():
    parser = argparse.ArgumentParser(description="RQ3 India Cross-Asset Pipeline")
    parser.add_argument("--stage", type=int, default=None,
                        help="Run single stage (0-4)")
    parser.add_argument("--no-transformer", action="store_true",
                        help="Skip TFT Transformer in stage 3")
    args = parser.parse_args()

    stages = {
        0: ("Data Download & Feature Engineering",
            lambda: __import__("rq3_india_stage0_data").main()),
        1: ("Granger Causality Network",
            lambda: __import__("rq3_india_stage1_granger").main()),
        2: ("DCC-GARCH Time-Varying Correlations",
            lambda: __import__("rq3_india_stage2_dcc").main()),
        3: ("ML Walk-Forward + Diebold-Mariano",
            lambda: __import__("rq3_india_stage3_ml").main(
                        run_tft=not args.no_transformer)),
        4: ("Final H0 Synthesis",
            lambda: __import__("rq3_india_stage4_synthesis").main()),
    }

    to_run = [args.stage] if args.stage is not None else list(stages.keys())

    print("\n" + "="*65)
    print("  RQ3 INDIA — CROSS-ASSET DEPENDENCIES PIPELINE")
    print("="*65)
    t0 = time.time()

    for s in to_run:
        name, fn = stages[s]
        print(f"\n{'█'*65}")
        print(f"  Stage {s}: {name}")
        print(f"{'█'*65}")
        st = time.time()
        try:
            fn()
            print(f"  Stage {s} completed in {time.time()-st:.1f}s ✓")
        except Exception as e:
            print(f"  Stage {s} FAILED: {e}")
            import traceback; traceback.print_exc()

    print(f"\n{'='*65}")
    print(f"  Total time: {(time.time()-t0)/60:.1f} minutes")
    print("="*65)

if __name__ == "__main__":
    run()

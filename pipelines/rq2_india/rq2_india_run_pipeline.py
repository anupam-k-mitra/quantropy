#!/usr/bin/env python3
"""
India RQ2 Pipeline Runner
Usage:
  python rq2_india_run_pipeline.py             # all stages
  python rq2_india_run_pipeline.py --stage 1   # single stage
"""
import sys, os, argparse, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run():
    parser = argparse.ArgumentParser(description="RQ2 India Regime Pipeline")
    parser.add_argument("--stage", type=int, default=None)
    args = parser.parse_args()

    stages = {
        0: ("Data & Feature Engineering",
            lambda: __import__("rq2_india_stage0_data").main()),
        1: ("Regime Detection (HMM+GMM+BP)",
            lambda: __import__("rq2_india_stage1_detection").main()),
        2: ("ANOVA & Structural Break Tests",
            lambda: __import__("rq2_india_stage2_anova").main()),
        3: ("Adaptive Strategy & H0 Test",
            lambda: __import__("rq2_india_stage3_adaptive").main()),
        4: ("Final H0 Synthesis",
            lambda: __import__("rq2_india_stage4_synthesis").main()),
    }

    to_run = [args.stage] if args.stage is not None else list(stages.keys())
    print("\n" + "="*65)
    print("  RQ2 INDIA — REGIME ROBUSTNESS PIPELINE")
    print("  H0: strategy performance identical across regimes")
    print("  H1: regime-adaptive strategy improves Sharpe >= 0.20")
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
    print(f"  Total: {(time.time()-t0)/60:.1f} minutes")
    print("="*65)

if __name__ == "__main__":
    run()

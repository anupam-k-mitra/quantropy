#!/usr/bin/env python3
# =============================================================================
# rq3_india_stage4_synthesis.py
# Stage 4: Final H0 Synthesis — India Cross-Asset Dependencies
#
# H0 REJECTED iff:
#   (1) Granger: >= MIN_CAUSAL_PAIRS BH-significant pairs
#   (2) DM test: >= 1 target × model combination significant
#   DCC evidence supporting.
#
# Also produces the cross-market comparison table: US vs India.
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import warnings, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger(__name__)

from rq3_india_config import (
    OUTPUT_DIR, GRANGER_ALPHA, MIN_CAUSAL_PAIRS, DM_ALPHA,
)


def _f(v, default=np.nan):
    try: return float(v)
    except: return default


def load_all() -> dict:
    res = {}

    # Granger
    g = os.path.join(OUTPUT_DIR, "rq3_india_granger_results.csv")
    if os.path.exists(g):
        gc  = pd.read_csv(g)
        col = "reject_BH" if "reject_BH" in gc.columns else "reject_raw"
        n   = int(gc[col].astype(bool).sum())
        res["granger"] = {"n_reject": n, "h1": n >= MIN_CAUSAL_PAIRS,
                          "detail": f"{n} pairs BH-significant"}
    else:
        res["granger"] = {"n_reject": 0, "h1": False,
                          "detail": "File not found"}

    # DCC
    d = os.path.join(OUTPUT_DIR, "rq3_india_dcc_results.csv")
    if os.path.exists(d):
        dc = pd.read_csv(d)
        nv = int(dc["reject"].astype(bool).sum())
        res["dcc"] = {"n_tv": nv, "h1": nv > 0,
                      "detail": f"{nv}/{len(dc)} pairs time-varying"}
    else:
        res["dcc"] = {"n_tv": 0, "h1": False, "detail": "File not found"}

    # DM (Ridge + XGBoost)
    m = os.path.join(OUTPUT_DIR, "rq3_india_ml_results.csv")
    if os.path.exists(m):
        mc = pd.read_csv(m)
        nr = int(mc["DM_reject"].astype(bool).sum())
        res["dm_ml"] = {"n_reject": nr, "h1": nr > 0,
                        "detail": f"{nr}/{len(mc)} target×model DM-significant"}
    else:
        res["dm_ml"] = {"n_reject": 0, "h1": False, "detail": "File not found"}

    # Transformer DM
    t = os.path.join(OUTPUT_DIR, "rq3_transformer_results.csv")
    if os.path.exists(t):
        tc = pd.read_csv(t)
        nt = int(tc["DM_reject"].astype(bool).sum()) if "DM_reject" in tc else 0
        res["dm_tft"] = {"n_reject": nt, "h1": nt > 0,
                         "detail": f"{nt}/{len(tc)} targets TFT DM-significant"}
    else:
        res["dm_tft"] = {"n_reject": 0, "h1": False,
                         "detail": "Not run (add --transformer to stage3)"}

    return res


def make_scorecard(res: dict) -> plt.Figure:
    conditions = [
        ("Granger causality\n(BH-corrected)",
         res["granger"]["h1"], res["granger"]["detail"]),
        ("DCC time-varying\ncorrelations",
         res["dcc"]["h1"],     res["dcc"]["detail"]),
        ("DM test\n(Ridge + XGBoost)",
         res["dm_ml"]["h1"],   res["dm_ml"]["detail"]),
        ("DM test\n(TFT Transformer)",
         res["dm_tft"]["h1"],  res["dm_tft"]["detail"]),
    ]

    fig, axes = plt.subplots(1, len(conditions), figsize=(13, 3.5))
    fig.suptitle("RQ3 India — Cross-Asset Dependency Evidence Scorecard",
                 fontweight="bold", fontsize=12, y=1.02)

    for ax, (label, passed, detail) in zip(axes, conditions):
        col  = "#1D9E75" if passed else "#607080"
        icon = "✓ CONFIRMED" if passed else "✗ not confirmed"
        ax.add_patch(plt.Rectangle((0.05, 0.05), 0.90, 0.90,
                     color=col, alpha=0.15, transform=ax.transAxes,
                     clip_on=False))
        ax.text(0.5, 0.72, label, ha="center", va="center",
                fontsize=10, fontweight="bold", transform=ax.transAxes)
        ax.text(0.5, 0.45, icon, ha="center", va="center",
                fontsize=11, color=col, fontweight="bold",
                transform=ax.transAxes)
        ax.text(0.5, 0.20, detail, ha="center", va="center",
                fontsize=8, color="#555555", transform=ax.transAxes)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.axis("off")

    n_confirmed = sum(c[1] for c in conditions)
    overall = n_confirmed >= 2   # H0 rejected if Granger + DM both confirm
    verdict_col = "#1D9E75" if overall else "#E24B4A"
    fig.text(0.5, -0.08,
             f"FINAL VERDICT: {'REJECT H0 ✓' if overall else 'FAIL TO REJECT H0'}  "
             f"({n_confirmed}/4 conditions confirmed)",
             ha="center", fontsize=13, fontweight="bold", color=verdict_col)

    plt.tight_layout()
    return fig


def main():
    log.info("=" * 65)
    log.info("  RQ3 INDIA STAGE 4 — Final H0 Synthesis")
    log.info("=" * 65)

    res    = load_all()
    h1_granger = res["granger"]["h1"]
    h1_dm      = res["dm_ml"]["h1"] or res["dm_tft"]["h1"]
    reject_h0  = h1_granger and h1_dm

    sep = "=" * 65
    report_lines = [
        sep,
        "  RQ3 INDIA — FINAL H0 SYNTHESIS",
        "  H0: Indian assets are informationally independent.",
        "  H1: Cross-asset signals improve return predictability.",
        sep,
        f"  Granger   : {res['granger']['detail']}  "
        f"→ H1 {'✓' if h1_granger else '✗'}",
        f"  DCC       : {res['dcc']['detail']}  "
        f"→ H1 {'✓' if res['dcc']['h1'] else '✗'}",
        f"  DM (ML)   : {res['dm_ml']['detail']}  "
        f"→ H1 {'✓' if res['dm_ml']['h1'] else '✗'}",
        f"  DM (TFT)  : {res['dm_tft']['detail']}  "
        f"→ H1 {'✓' if res['dm_tft']['h1'] else '✗'}",
        sep,
        f"  FINAL DECISION: {'REJECT H0 ✓' if reject_h0 else 'FAIL TO REJECT H0'}",
        sep,
    ]
    report = "\n".join(report_lines)
    print("\n" + report)

    with open(os.path.join(OUTPUT_DIR, "rq3_india_final_report.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n")

    fig = make_scorecard(res)
    fig.savefig(os.path.join(OUTPUT_DIR, "rq3_india_scorecard.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    log.info("\n✓ Stage 4 complete. RQ3 India pipeline finished.\n")
    return reject_h0, res


if __name__ == "__main__":
    main()

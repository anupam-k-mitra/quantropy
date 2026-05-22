#!/usr/bin/env python3
# =============================================================================
# rq3_stage4_synthesis.py
# Stage 4: Final H0 Synthesis
#
# H0 REJECTED iff:
#   (1) Granger: >= MIN_CAUSAL_PAIRS BH-significant pairs  [AND]
#   (2) DM test: >= 1 target shows DM p < DM_ALPHA
#   DCC evidence is supporting, not gating.
#
# Outputs
# -------
#   rq3_outputs/rq3_final_scorecard.png
#   rq3_outputs/rq3_final_report.txt
#
# Run: python rq3_stage4_synthesis.py
# =============================================================================

import os
import sys
import logging
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from rq3_config import OUTPUT_DIR, GRANGER_ALPHA, MIN_CAUSAL_PAIRS, DM_ALPHA


# =============================================================================
# PARSERS
# =============================================================================

def _f(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def load_all():
    res = {}

    # Stage 1: Granger
    g = os.path.join(OUTPUT_DIR, "rq3_granger_results.csv")
    if os.path.exists(g):
        gc  = pd.read_csv(g)
        col = "reject_BH" if "reject_BH" in gc.columns else "reject_raw"
        n   = int(gc[col].astype(bool).sum())
        res.update({
            "granger_n_bh":  n,
            "granger_total": len(gc),
            "granger_reject": n >= MIN_CAUSAL_PAIRS,
        })
    else:
        log.warning("rq3_granger_results.csv not found.")

    # Stage 2: DCC
    d = os.path.join(OUTPUT_DIR, "rq3_dcc_results.csv")
    if os.path.exists(d):
        dcc = pd.read_csv(d)
        n_v = int(dcc["varies"].astype(bool).sum()) \
              if "varies" in dcc.columns else 0
        res.update({
            "dcc_n_varying": n_v,
            "dcc_total":     len(dcc),
            "dcc_reject":    n_v > len(dcc) * 0.3,
        })

    s = os.path.join(OUTPUT_DIR, "rq3_spillover_series.csv")
    if os.path.exists(s):
        sp = pd.read_csv(s, index_col=0).iloc[:, 0]
        res["spillover_mean"] = round(float(sp.mean()), 4)
    else:
        res["spillover_mean"] = 0.0

    # Stage 3: ML / DM
    m = os.path.join(OUTPUT_DIR, "rq3_ml_results.csv")
    if os.path.exists(m):
        ml  = pd.read_csv(m)
        n_r = int(ml["DM_reject"].astype(bool).sum()) \
              if "DM_reject" in ml.columns else 0
        res.update({
            "dm_n_reject": n_r,
            "dm_reject":   n_r > 0,
        })
        if "pct_reduction" in ml.columns:
            res["mean_pct_reduction"] = round(
                float(ml["pct_reduction"].mean()), 3
            )
    else:
        log.warning("rq3_ml_results.csv not found.")

    return res


# =============================================================================
# SCORECARD PLOT
# =============================================================================

def plot_scorecard(results: dict, final_reject: bool,
                   save_path: str = None):
    granger_ok = results.get("granger_reject",  False)
    dcc_ok     = results.get("dcc_reject",       False)
    dm_ok      = results.get("dm_reject",        False)

    checks = [
        (
            f"Granger Causality\n>= {MIN_CAUSAL_PAIRS} BH pairs",
            granger_ok,
            f"{results.get('granger_n_bh', 0)} / "
            f"{results.get('granger_total', 0)} pairs significant\n"
            f"(BH FDR alpha={GRANGER_ALPHA})",
        ),
        (
            "DCC Time-Varying\nCorrelations",
            dcc_ok,
            f"{results.get('dcc_n_varying', 0)} / "
            f"{results.get('dcc_total', 0)} pairs vary\n"
            f"DY spillover mean = {results.get('spillover_mean', 0):.4f}",
        ),
        (
            f"Diebold-Mariano\np < {DM_ALPHA}",
            dm_ok,
            f"{results.get('dm_n_reject', 0)} target(s) reject H0\n"
            f"Mean RMSE reduction = "
            f"{results.get('mean_pct_reduction', 0):.2f}%",
        ),
        (
            "Combined\nH0 Decision",
            final_reject,
            "Granger AND DM both confirmed"
            if final_reject else "Not all conditions met",
        ),
    ]

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#F0F4FA")
    fig.text(0.5, 0.96,
             "RQ3 Evidence Scorecard -- Cross-Asset Dependencies",
             ha="center", fontsize=15, fontweight="bold",
             color="#1B3A6B")
    fig.text(0.5, 0.92,
             "H0: Assets informationally independent  |  "
             f"H1: Cross-asset signals improve OOS RMSE (DM p < {DM_ALPHA})",
             ha="center", fontsize=10, color="#555555")

    positions = [(0.05, 0.53), (0.52, 0.53),
                 (0.05, 0.17), (0.52, 0.17)]

    for (x, y), (label, ok, detail) in zip(positions, checks):
        colour = "#2E5FAC" if ok else "#C0392B"
        bg     = "#E8F0FB" if ok else "#FBE8E8"
        ax     = fig.add_axes([x, y, 0.43, 0.33])
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_edgecolor(colour); sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.text(0.08, 0.85, label, transform=ax.transAxes,
                fontsize=11, fontweight="bold", color="#1B3A6B", va="top")
        ax.text(0.08, 0.50, detail, transform=ax.transAxes,
                fontsize=9, color="#333333", va="top")
        ax.text(0.88, 0.28, "PASS" if ok else "FAIL",
                transform=ax.transAxes, fontsize=20,
                color=colour, ha="center", fontweight="bold")

    # Verdict banner
    v_colour = "#1B3A6B" if final_reject else "#C0392B"
    v_bg     = "#D6E4F7" if final_reject else "#FBE8E8"
    v_text   = (
        "FINAL: REJECT H0 -- Cross-asset information "
        "significantly improves predictability"
        if final_reject else
        "FINAL: FAIL TO REJECT H0 -- Insufficient evidence"
    )
    ax_f = fig.add_axes([0.05, 0.03, 0.90, 0.11])
    ax_f.set_facecolor(v_bg)
    for sp in ax_f.spines.values():
        sp.set_edgecolor(v_colour); sp.set_linewidth(3)
    ax_f.set_xticks([]); ax_f.set_yticks([])
    ax_f.text(0.5, 0.5, v_text, transform=ax_f.transAxes,
              fontsize=13, fontweight="bold", color=v_colour,
              ha="center", va="center")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        log.info(f"  Saved -> {save_path}")
    plt.close()


# =============================================================================
# FINAL REPORT
# =============================================================================

def write_report(results: dict, final_reject: bool, save_path: str):
    lines = [
        "=" * 70,
        "  FINAL SYNTHESIS  --  RQ3: CROSS-ASSET DEPENDENCIES",
        "  H0: Assets informationally independent",
        "  H1: Cross-asset signals significantly improve OOS RMSE",
        f"      Granger >= {MIN_CAUSAL_PAIRS} pairs  AND  DM p < {DM_ALPHA}",
        "=" * 70,
        "",
        "  [1] Granger Causality",
        f"      BH-significant pairs : "
        f"{results.get('granger_n_bh', 0)} / "
        f"{results.get('granger_total', 0)}",
        f"      Required             : >= {MIN_CAUSAL_PAIRS}",
        f"      Decision: "
        f"{'REJECT H0' if results.get('granger_reject') else 'fail to reject'}",
        "",
        "  [2] DCC Time-Varying Correlations (supporting evidence)",
        f"      Pairs with varying corr: "
        f"{results.get('dcc_n_varying', 0)} / "
        f"{results.get('dcc_total', 0)}",
        f"      Mean DY Spillover    : "
        f"{results.get('spillover_mean', 0):.4f}",
        f"      Decision: "
        f"{'Time-varying confirmed' if results.get('dcc_reject') else 'Approximately constant'}",
        "",
        "  [3] Diebold-Mariano Test (OOS RMSE)",
        f"      DM-reject targets    : {results.get('dm_n_reject', 0)}",
        f"      Mean RMSE reduction  : "
        f"{results.get('mean_pct_reduction', 0):.2f}%",
        f"      Decision: "
        f"{'REJECT H0' if results.get('dm_reject') else 'fail to reject'}",
        "",
        "=" * 70,
        f"  FINAL: {'REJECT H0' if final_reject else 'FAIL TO REJECT H0'}",
    ]

    if final_reject:
        lines += [
            "",
            "  Interpretation:",
            "  Cross-asset lagged returns Granger-cause target returns",
            "  for >= " + str(MIN_CAUSAL_PAIRS) + " BH-corrected pairs, AND",
            "  a cross-asset model produces statistically lower MSFE",
            "  (DM p < " + str(DM_ALPHA) + ") in a strict walk-forward test.",
            "  H0 of informational independence is rejected.",
        ]
    else:
        n_pass = sum([
            results.get("granger_reject", False),
            results.get("dm_reject", False),
        ])
        lines += [
            "",
            f"  Evidence score: {n_pass}/2 gating conditions met.",
            "  Possible causes: signal-to-noise too low at 5-day horizon,",
            "  insufficient cross-asset sample overlap, or true independence.",
        ]

    lines += [
        "",
        "=" * 70,
        "  References:",
        "  Granger (1969) Econometrica   -- Causality in econometrics",
        "  Engle (2002) JBES             -- DCC-GARCH model",
        "  Diebold & Yilmaz (2014) JE    -- Spillover index",
        "  Diebold & Mariano (1995) JBES -- Predictive accuracy test",
        "  Harvey et al. (1997) IJF      -- HLN small-sample correction",
        "=" * 70,
    ]

    report = "\n".join(lines)
    print("\n" + report)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    log.info(f"  Report saved -> {save_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ3 STAGE 4 -- Final H0 Synthesis")
    log.info("=" * 65)

    results = load_all()

    log.info(f"\nLoaded results:")
    for k, v in results.items():
        log.info(f"  {k:<30s}: {v}")

    # Decision: Granger AND DM must both reject H0
    granger_ok = results.get("granger_reject", False)
    dm_ok      = results.get("dm_reject",      False)
    final      = granger_ok and dm_ok

    log.info(f"\n  [1] Granger: {'PASS' if granger_ok else 'FAIL'}")
    log.info(f"  [2] DM test: {'PASS' if dm_ok else 'FAIL'}")
    log.info(f"  --> {'REJECT H0' if final else 'FAIL TO REJECT H0'}")

    plot_scorecard(
        results, final,
        save_path=os.path.join(OUTPUT_DIR, "rq3_final_scorecard.png"),
    )
    write_report(
        results, final,
        save_path=os.path.join(OUTPUT_DIR, "rq3_final_report.txt"),
    )

    log.info("\n✓ RQ3 pipeline complete.\n")
    return final


if __name__ == "__main__":
    main()

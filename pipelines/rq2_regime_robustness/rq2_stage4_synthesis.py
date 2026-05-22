#!/usr/bin/env python3
# =============================================================================
# rq2_stage4_synthesis.py
# Stage 4: Final H0 Synthesis — consolidate all evidence.
#
# H0 REJECTED iff ALL four conditions hold:
#   1. ANOVA F-test p < 0.05        (means differ across regimes)
#   2. Kruskal-Wallis p < 0.05      (non-parametric confirmation)
#   3. Delta Sharpe >= MIN_SHARPE_GAIN (economically exploitable)
#   4. No catastrophic drawdown     (strategy survives all regimes)
#
# Outputs
# -------
#   rq2_outputs/rq2_final_scorecard.png
#   rq2_outputs/rq2_final_report.txt
#
# Run: python rq2_stage4_synthesis.py
# =============================================================================

import os
import re
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

from regime_config import (
    OUTPUT_DIR, ALPHA, MIN_SHARPE_GAIN, MAX_REGIME_DD,
)


# =============================================================================
# PARSERS
# =============================================================================

def _float(s, default=np.nan):
    try:
        return float(s)
    except Exception:
        return default


def _bool_from_csv(series, key, default=False):
    val = series.get(key, str(default))
    return str(val).strip().lower() in ("true", "1", "1.0", "yes")


def parse_stage2():
    path = os.path.join(OUTPUT_DIR, "rq2_stage2_summary.csv")
    if not os.path.exists(path):
        log.warning("rq2_stage2_summary.csv not found.")
        return {}
    s = pd.read_csv(path, index_col=0, header=None).squeeze()
    return {
        "anova_F":       _float(s.get("anova_F", np.nan)),
        "anova_p":       _float(s.get("anova_p", 1.0)),
        "anova_eta2":    _float(s.get("anova_eta2", 0.0)),
        "anova_reject":  _bool_from_csv(s, "anova_reject"),
        "kw_H":          _float(s.get("kw_H", 0.0)),
        "kw_p":          _float(s.get("kw_p", 1.0)),
        "kw_reject":     _bool_from_csv(s, "kw_reject"),
        "levene_W":      _float(s.get("levene_W", 0.0)),
        "levene_p":      _float(s.get("levene_p", 1.0)),
        "levene_reject": _bool_from_csv(s, "levene_reject"),
        "pairs_reject":  int(_float(s.get("pairs_reject", 0))),
    }


def parse_stage3():
    path = os.path.join(OUTPUT_DIR, "rq2_stage3_summary.txt")
    if not os.path.exists(path):
        log.warning("rq2_stage3_summary.txt not found.")
        return {}
    with open(path, encoding="utf-8") as f:
        txt = f.read()

    m_delta = re.search(r"Delta SR\s*=\s*([+-]?[\d.]+)", txt)
    delta   = float(m_delta.group(1)) if m_delta else 0.0
    h1_sharpe = delta >= MIN_SHARPE_GAIN
    no_cat    = "YES" in txt and "NO" not in txt.split("No Catastrophe")[-1]

    # Also check per-regime CSV
    reg_path = os.path.join(OUTPUT_DIR, "rq2_per_regime_perf.csv")
    worst_dd = 0.0
    if os.path.exists(reg_path):
        df = pd.read_csv(reg_path)
        adapt = df[df["strategy"] == "adaptive"]["max_dd"]
        if not adapt.empty:
            worst_dd = float(adapt.min())
    no_catastrophe = worst_dd > MAX_REGIME_DD

    return {
        "delta_sharpe":  round(delta, 4),
        "h1_sharpe":     h1_sharpe,
        "no_catastrophe": no_catastrophe,
        "worst_dd":      round(worst_dd, 4),
    }


# =============================================================================
# SCORECARD PLOT
# =============================================================================

def plot_scorecard(results, final_reject, save_path=None):
    """
    Four-card visual evidence summary with final verdict banner.
    """
    checks = [
        (
            "ANOVA F-test\np < 0.05",
            results.get("anova_p", 1.0) < ALPHA,
            f"F={results.get('anova_F', 0):.2f}  "
            f"p={results.get('anova_p', 1):.5f}\n"
            f"eta^2={results.get('anova_eta2', 0):.4f}",
        ),
        (
            "Kruskal-Wallis\n(non-parametric)",
            results.get("kw_reject", False),
            f"H={results.get('kw_H', 0):.2f}  "
            f"p={results.get('kw_p', 1):.5f}",
        ),
        (
            f"Delta Sharpe\n>= {MIN_SHARPE_GAIN}",
            results.get("h1_sharpe", False),
            f"delta = {results.get('delta_sharpe', 0):+.4f}",
        ),
        (
            f"No Catastrophic DD\n(worst > {MAX_REGIME_DD*100:.0f}%)",
            results.get("no_catastrophe", True),
            f"worst MDD = {results.get('worst_dd', 0)*100:.1f}%",
        ),
    ]

    fig = plt.figure(figsize=(13, 9))
    fig.patch.set_facecolor("#F0F4FA")
    fig.text(0.5, 0.96,
             "RQ2 Evidence Scorecard — Regime Robustness",
             ha="center", fontsize=15, fontweight="bold",
             color="#1B3A6B")
    fig.text(0.5, 0.92,
             "H0: Performance invariant across regimes  |  "
             f"H1: Regime conditioning improves Sharpe >= {MIN_SHARPE_GAIN}",
             ha="center", fontsize=10, color="#555555")

    positions = [(0.05, 0.53), (0.52, 0.53),
                 (0.05, 0.17), (0.52, 0.17)]

    for (x, y), (label, ok, detail) in zip(positions, checks):
        colour = "#2E5FAC" if ok else "#C0392B"
        bg     = "#E8F0FB" if ok else "#FBE8E8"

        ax = fig.add_axes([x, y, 0.43, 0.33])
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_edgecolor(colour)
            sp.set_linewidth(2)
        ax.set_xticks([]); ax.set_yticks([])

        ax.text(0.08, 0.85, label,
                transform=ax.transAxes, fontsize=11,
                fontweight="bold", color="#1B3A6B", va="top")
        ax.text(0.08, 0.50, detail,
                transform=ax.transAxes, fontsize=9,
                color="#333333", va="top")
        ax.text(0.88, 0.30,
                "PASS" if ok else "FAIL",
                transform=ax.transAxes, fontsize=20,
                color=colour, ha="center", fontweight="bold")

    # Final verdict banner
    v_colour = "#1B3A6B" if final_reject else "#C0392B"
    v_bg     = "#D6E4F7" if final_reject else "#FBE8E8"
    v_text   = (
        "FINAL: REJECT H0 — Regimes are real; "
        "adaptive conditioning improves strategy"
        if final_reject else
        "FINAL: FAIL TO REJECT H0 — Insufficient evidence"
    )
    ax_f = fig.add_axes([0.05, 0.03, 0.90, 0.11])
    ax_f.set_facecolor(v_bg)
    for sp in ax_f.spines.values():
        sp.set_edgecolor(v_colour); sp.set_linewidth(3)
    ax_f.set_xticks([]); ax_f.set_yticks([])
    ax_f.text(0.5, 0.5, v_text,
              transform=ax_f.transAxes,
              fontsize=13, fontweight="bold",
              color=v_colour, ha="center", va="center")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        log.info(f"  Scorecard saved -> {save_path}")
    plt.close()


# =============================================================================
# FINAL REPORT
# =============================================================================

def write_report(results, final_reject, save_path):
    lines = [
        "=" * 70,
        "  FINAL SYNTHESIS  --  RQ2: REGIME ROBUSTNESS",
        "  H0: Strategy performance invariant across market regimes",
        "  H1: Performance varies; adaptive conditioning improves",
        f"      Sharpe by >= {MIN_SHARPE_GAIN}",
        "=" * 70,
        "",
        "  [1] ANOVA F-test (means differ across regimes)",
        f"      F  = {results.get('anova_F', 0):.4f}   "
        f"p  = {results.get('anova_p', 1):.6f}   "
        f"eta^2 = {results.get('anova_eta2', 0):.4f}",
        f"      Decision: {'REJECT H0' if results.get('anova_reject') else 'fail to reject'}",
        "",
        "  [2] Kruskal-Wallis (non-parametric confirmation)",
        f"      H  = {results.get('kw_H', 0):.4f}   "
        f"p  = {results.get('kw_p', 1):.6f}",
        f"      Decision: {'REJECT H0' if results.get('kw_reject') else 'fail to reject'}",
        "",
        "  [3] Levene test (variance homogeneity)",
        f"      W  = {results.get('levene_W', 0):.4f}   "
        f"p  = {results.get('levene_p', 1):.6f}",
        f"      Decision: {'Variances differ' if results.get('levene_reject') else 'equal variances'}",
        "",
        f"  [4] Sharpe improvement (adaptive vs baseline)",
        f"      Delta Sharpe = {results.get('delta_sharpe', 0):+.4f}   "
        f"Threshold = {MIN_SHARPE_GAIN}",
        f"      Decision: {'CONFIRMED H1' if results.get('h1_sharpe') else 'not confirmed'}",
        "",
        "  [5] Catastrophic failure check",
        f"      Worst regime MDD = {results.get('worst_dd', 0)*100:.1f}%   "
        f"Threshold = {MAX_REGIME_DD*100:.0f}%",
        f"      Decision: {'PASS' if results.get('no_catastrophe', True) else 'FAIL'}",
        "",
        "=" * 70,
        f"  FINAL: {'REJECT H0' if final_reject else 'FAIL TO REJECT H0'}",
    ]

    if final_reject:
        lines += [
            "",
            "  Interpretation:",
            "  All four conditions are met. Market regimes are statistically",
            "  real and persistent. The adaptive strategy exploits regime",
            "  information to improve risk-adjusted returns by at least",
            f"  {MIN_SHARPE_GAIN} Sharpe units while avoiding catastrophic",
            "  drawdowns in any single regime.",
            "  Proceed to RQ3 (Cross-Asset Dependencies) and RQ4 (Risk Control).",
        ]
    else:
        n_pass = sum([
            results.get("anova_p",   1.0) < ALPHA,
            results.get("kw_reject", False),
            results.get("h1_sharpe", False),
            results.get("no_catastrophe", True),
        ])
        lines += [
            "",
            f"  Evidence score: {n_pass}/4 conditions met.",
            "  Possible causes: insufficient data, regime labels too noisy,",
            "  or scaling rules require calibration.",
        ]

    lines += [
        "",
        "=" * 70,
        "  REFERENCES",
        "=" * 70,
        "  Hamilton (1989) Econometrica -- Regime-switching models",
        "  Ang & Bekaert (2002) RFS     -- Returns vary across regimes",
        "  Bai & Perron (2003) JAE      -- Multiple structural breaks",
        "  Kritzman et al. (2012) FAJ   -- Turbulence index",
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
    log.info("  RQ2 STAGE 4 -- Final H0 Synthesis")
    log.info("=" * 65)

    # Load all stage results
    s2 = parse_stage2()
    s3 = parse_stage3()
    results = {**s2, **s3}

    log.info(f"\nParsed results:")
    for k, v in results.items():
        log.info(f"  {k:<25s}: {v}")

    # Decision: ALL four conditions must hold
    cond1 = results.get("anova_p",       1.0) < ALPHA
    cond2 = results.get("kw_reject",     False)
    cond3 = results.get("h1_sharpe",     False)
    cond4 = results.get("no_catastrophe", True)

    final_reject = cond1 and cond2 and cond3 and cond4

    log.info(f"\nCondition checks:")
    log.info(f"  [1] ANOVA p < {ALPHA}    : {'PASS' if cond1 else 'FAIL'}")
    log.info(f"  [2] KW reject H0     : {'PASS' if cond2 else 'FAIL'}")
    log.info(f"  [3] Delta Sharpe OK  : {'PASS' if cond3 else 'FAIL'}")
    log.info(f"  [4] No catastrophe   : {'PASS' if cond4 else 'FAIL'}")
    log.info(f"\n  --> {'REJECT H0' if final_reject else 'FAIL TO REJECT H0'}")

    plot_scorecard(
        results, final_reject,
        save_path=os.path.join(OUTPUT_DIR, "rq2_final_scorecard.png"),
    )
    write_report(
        results, final_reject,
        save_path=os.path.join(OUTPUT_DIR, "rq2_final_report.txt"),
    )

    log.info("\n✓ RQ2 pipeline complete.\n")
    return final_reject


if __name__ == "__main__":
    main()

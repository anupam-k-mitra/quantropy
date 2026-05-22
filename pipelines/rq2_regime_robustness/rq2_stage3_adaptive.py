#!/usr/bin/env python3
# =============================================================================
# rq2_stage3_adaptive.py
# Stage 3: Adaptive strategy — does regime conditioning improve Sharpe >= 0.20?
#
# Three strategy variants compared in strict walk-forward:
#   Baseline  — fixed allocation, no regime awareness
#   Adaptive  — position scaled by REGIME_SCALES[regime_t-1]
#   Defensive — binary: full exposure in Bull/Sideways, cash in Bear/Crisis
#
# Decision rule for H1 (Sharpe condition):
#   delta_Sharpe = Sharpe(adaptive) - Sharpe(baseline) >= MIN_SHARPE_GAIN
#
# Catastrophic failure check:
#   worst per-regime MDD of adaptive strategy must be > MAX_REGIME_DD
#
# Outputs
# -------
#   rq2_outputs/rq2_stage3_summary.txt
#   rq2_outputs/rq2_adaptive_returns.csv
#   rq2_outputs/rq2_per_regime_perf.csv
#   rq2_outputs/rq2_strategy_comparison.png
#
# Run: python rq2_stage3_adaptive.py
# =============================================================================

import os
import sys
import warnings
import logging

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
    DATA_DIR, OUTPUT_DIR, COST_BPS, RISK_FREE_ANNUAL,
    TRAIN_WINDOW, TEST_WINDOW, MIN_SHARPE_GAIN, MAX_REGIME_DD,
    REGIME_SCALES,
)
from scipy_compat import sm


# =============================================================================
# LOADER
# =============================================================================

def load_data():
    for fname in ["rq2_regime_labels.parquet",
                  "rq2_log_returns.parquet",
                  "prices.parquet"]:
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} not found. Run earlier stages first."
            )
    labels  = pd.read_parquet(
        os.path.join(DATA_DIR, "rq2_regime_labels.parquet"))
    log_ret = pd.read_parquet(
        os.path.join(DATA_DIR, "rq2_log_returns.parquet"))
    prices  = pd.read_parquet(
        os.path.join(DATA_DIR, "prices.parquet"))
    return labels, log_ret, prices


# =============================================================================
# PERFORMANCE METRICS
# =============================================================================

def compute_metrics(returns, rf=RISK_FREE_ANNUAL):
    """Full performance analytics for a return series."""
    r  = returns.dropna()
    n  = len(r)
    if n < 10:
        return {}

    ann_r = r.mean() * 252
    ann_v = r.std()  * np.sqrt(252)
    sr    = (ann_r - rf) / ann_v if ann_v > 0 else 0.0

    downside = r[r < rf / 252].std() * np.sqrt(252)
    sortino  = (ann_r - rf) / downside if downside > 0 else 0.0

    cum  = (1 + r).cumprod()
    peak = cum.cummax()
    dd   = (cum - peak) / peak
    mdd  = float(dd.min())
    calmar = ann_r / abs(mdd) if mdd < 0 else 0.0

    # Alpha via HAC OLS
    ones = sm.add_constant(
        pd.Series(np.ones(n), index=r.index), has_constant="add"
    )
    try:
        ols     = sm.OLS(r, ones).fit(cov_type="HAC",
                                       cov_kwds={"maxlags": 5})
        alpha_t = ols.tvalues.get("const", np.nan)
        alpha_p = ols.pvalues.get("const", np.nan)
    except Exception:
        se      = r.std() / np.sqrt(n) if r.std() > 0 else 1e-9
        alpha_t = r.mean() / se
        alpha_p = float("nan")

    return {
        "ann_return":  round(ann_r,   4),
        "ann_vol":     round(ann_v,   4),
        "sharpe":      round(sr,      4),
        "sortino":     round(sortino, 4),
        "calmar":      round(calmar,  4),
        "max_dd":      round(mdd,     4),
        "hit_rate":    round((r > 0).mean(), 4),
        "alpha_tstat": round(float(alpha_t), 4)
                       if not np.isnan(float(alpha_t)) else np.nan,
        "alpha_pval":  round(float(alpha_p), 6)
                       if alpha_p is not None and
                          not np.isnan(float(alpha_p)) else np.nan,
        "n":           n,
    }


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def walkforward_adaptive(log_ret, labels, ticker="SPY"):
    """
    Strict walk-forward comparison of three strategy variants.

    KEY DESIGN: the regime used for position sizing on day t is
    the regime observed at close of day t-1 (one-day lag).
    This prevents any look-ahead bias — you can only trade on
    information that was available before the trading session.

    Strategies
    ----------
    baseline  — always 1.0 weight regardless of regime
    adaptive  — weight = REGIME_SCALES[regime at t-1]
    defensive — 1.0 in Bull/Sideways (R0/R2), 0.0 in Bear/Crisis (R1/R3)
    """
    if ticker not in log_ret.columns:
        ticker = log_ret.columns[0]

    r      = log_ret[ticker].dropna()
    regime = labels["consensus"].reindex(r.index).ffill()
    COST   = COST_BPS / 10_000   # one-way decimal

    dates  = r.index
    b_rets, a_rets, d_rets, all_dates = [], [], [], []

    prev_a_scale = 1.0
    prev_d_scale = 1.0

    for start in range(TRAIN_WINDOW, len(dates) - TEST_WINDOW, TEST_WINDOW):
        test_slice = dates[start: start + TEST_WINDOW]

        # ── Regime at END of training period (lagged) ─────────────
        regime_t = int(regime.iloc[start - 1])

        adaptive_scale  = REGIME_SCALES.get(regime_t, 0.6)
        defensive_scale = 1.0 if regime_t in (0, 2) else 0.0

        # Transaction costs proportional to change in weight
        a_cost = abs(adaptive_scale  - prev_a_scale)  * COST
        d_cost = abs(defensive_scale - prev_d_scale) * COST
        prev_a_scale  = adaptive_scale
        prev_d_scale  = defensive_scale

        # Daily cost spread over test window
        a_daily_cost = a_cost / TEST_WINDOW
        d_daily_cost = d_cost / TEST_WINDOW

        for d in test_slice:
            if d not in r.index:
                continue
            raw = float(r.loc[d])
            b_rets.append(raw)
            a_rets.append(raw * adaptive_scale  - a_daily_cost)
            d_rets.append(raw * defensive_scale - d_daily_cost)
            all_dates.append(d)

    idx = pd.DatetimeIndex(all_dates)
    return {
        "baseline":  pd.Series(b_rets, index=idx, name="baseline"),
        "adaptive":  pd.Series(a_rets, index=idx, name="adaptive"),
        "defensive": pd.Series(d_rets, index=idx, name="defensive"),
    }


# =============================================================================
# PER-REGIME PERFORMANCE
# =============================================================================

def per_regime_performance(returns_dict, labels):
    """Per-regime Sharpe, return, vol, MDD for each strategy variant."""
    rows = []
    for strat, ret in returns_dict.items():
        # Align on common index
        common  = ret.index.intersection(labels.index)
        ret_aln = ret.loc[common]
        lbl_aln = labels.loc[common]
        aligned = pd.DataFrame({
            "ret":    ret_aln,
            "regime": lbl_aln,
        }).dropna()

        for reg in sorted(aligned["regime"].unique()):
            sub = aligned.loc[aligned["regime"] == reg, "ret"]
            if len(sub) < 20:
                continue
            ar  = sub.mean() * 252
            av  = sub.std()  * np.sqrt(252)
            sr  = ar / av if av > 0 else 0.0
            cum = (1 + sub).cumprod()
            mdd = float(((cum - cum.cummax()) / cum.cummax()).min())

            rows.append({
                "strategy": strat, "regime": int(reg),
                "n":        len(sub),
                "ann_ret":  round(ar,  4),
                "ann_vol":  round(av,  4),
                "sharpe":   round(sr,  4),
                "max_dd":   round(mdd, 4),
            })

    return pd.DataFrame(rows)


# =============================================================================
# VISUALISATION
# =============================================================================

COLOURS = {"baseline": "#AAAAAA", "adaptive": "#2E5FAC",
           "defensive": "#27AE60"}
REG_NAMES = {0: "Bull", 1: "Bear", 2: "Sideways", 3: "Crisis"}


def plot_comparison(returns_dict, labels, perf_df, save_path=None):
    """
    Four-panel chart:
      1. Cumulative equity curves
      2. Rolling 63-day Sharpe
      3. Per-regime Sharpe bars
      4. Drawdown comparison
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("RQ2 — Adaptive vs Baseline Strategy Comparison",
                 fontsize=14, fontweight="bold")

    # Panel 1: Equity curves
    ax = axes[0, 0]
    for name, r in returns_dict.items():
        cum = (1 + r).cumprod()
        ax.plot(cum.index, cum.values, linewidth=1.5,
                color=COLOURS[name], label=name.capitalize())
    ax.set_title("Cumulative Returns", fontsize=11, fontweight="bold")
    ax.set_ylabel("Portfolio Value (start=1)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Rolling Sharpe
    ax = axes[0, 1]
    for name, r in returns_dict.items():
        roll = (r.rolling(63).mean() /
                r.rolling(63).std() * np.sqrt(252))
        ax.plot(roll.index, roll.values, linewidth=1.0,
                color=COLOURS[name], label=name.capitalize(), alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(MIN_SHARPE_GAIN, color="red", linewidth=0.8,
               linestyle="-.",
               label=f"Delta Sharpe threshold ({MIN_SHARPE_GAIN})")
    ax.set_title("Rolling 63-Day Sharpe", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 3: Per-regime Sharpe bars
    ax = axes[1, 0]
    if not perf_df.empty:
        strats  = perf_df["strategy"].unique()
        regs    = sorted(perf_df["regime"].unique())
        x       = np.arange(len(regs))
        width   = 0.8 / max(len(strats), 1)
        for i, strat in enumerate(strats):
            vals = [
                perf_df.loc[(perf_df["strategy"] == strat) &
                            (perf_df["regime"]   == reg), "sharpe"
                ].values[0]
                if len(perf_df.loc[(perf_df["strategy"] == strat) &
                                   (perf_df["regime"]   == reg)]) > 0
                else 0.0
                for reg in regs
            ]
            ax.bar(x + i * width, vals, width,
                   label=strat.capitalize(),
                   color=list(COLOURS.values())[i % 3],
                   edgecolor="white", alpha=0.85)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xticks(x + width)
        ax.set_xticklabels(
            [f"R{r}\n{REG_NAMES.get(r, '')}" for r in regs],
            fontsize=8,
        )
        ax.set_title("Per-Regime Sharpe", fontsize=11, fontweight="bold")
        ax.set_ylabel("Sharpe Ratio")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3, axis="y")

    # Panel 4: Drawdowns
    ax = axes[1, 1]
    for name, r in returns_dict.items():
        cum  = (1 + r).cumprod()
        peak = cum.cummax()
        dd   = (cum - peak) / peak
        ax.fill_between(dd.index, dd.values, 0,
                        alpha=0.3, color=COLOURS[name])
        ax.plot(dd.index, dd.values, linewidth=0.8,
                color=COLOURS[name], label=name.capitalize())
    ax.axhline(MAX_REGIME_DD, color="darkred", linewidth=1.0,
               linestyle="--",
               label=f"Catastrophic ({MAX_REGIME_DD*100:.0f}%)")
    ax.set_title("Drawdown Comparison", fontsize=11, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved → {save_path}")
    plt.close()


def plot_regime_returns(returns_dict, labels, save_path=None):
    """Box plots of per-regime daily returns for each strategy."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    fig.suptitle("Per-Regime Return Distributions by Strategy",
                 fontsize=13, fontweight="bold")

    regs = sorted(labels.unique())
    for ax, (name, r) in zip(axes, returns_dict.items()):
        common  = r.index.intersection(labels.index)
        aligned = pd.DataFrame({"ret": r.loc[common], "regime": labels.loc[common]}).dropna()
        data    = [aligned.loc[aligned["regime"] == reg, "ret"].values * 252
                   for reg in regs]
        bp = ax.boxplot(data, patch_artist=True, notch=False)
        reg_colours = ["#B8D4F0", "#F5C6C6", "#D5E8D4", "#FFE6CC"]
        for patch, col in zip(bp["boxes"], reg_colours):
            patch.set_facecolor(col)
        ax.set_xticklabels(
            [f"R{r}\n{REG_NAMES.get(r,'')}" for r in regs],
            fontsize=8,
        )
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.set_title(name.capitalize(), fontsize=11, fontweight="bold",
                     color=COLOURS[name])
        ax.set_ylabel("Annualised Return" if name == "baseline" else "")
        ax.grid(alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        log.info(f"  Saved → {save_path}")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 65)
    log.info("  RQ2 STAGE 3 — Adaptive Strategy & Sharpe Improvement")
    log.info("=" * 65)

    labels, log_ret, prices = load_data()
    consensus = labels["consensus"]
    ticker    = "SPY" if "SPY" in log_ret.columns else log_ret.columns[0]

    # ── Walk-forward ──────────────────────────────────────────────
    log.info(f"\nRunning walk-forward (ticker={ticker}) …")
    returns_dict = walkforward_adaptive(log_ret, labels, ticker)

    # ── Overall metrics ───────────────────────────────────────────
    log.info("\n── Overall Performance ───────────────────────────────────")
    perf = {}
    for name, r in returns_dict.items():
        p = compute_metrics(r)
        perf[name] = p
        log.info(
            f"  {name:12s}  SR={p.get('sharpe',0):.3f}  "
            f"Ann={p.get('ann_return',0)*100:.1f}%  "
            f"Vol={p.get('ann_vol',0)*100:.1f}%  "
            f"MDD={p.get('max_dd',0)*100:.1f}%  "
            f"alpha-t={p.get('alpha_tstat','n/a')}"
        )

    # ── Sharpe improvement ────────────────────────────────────────
    base_sr   = perf.get("baseline", {}).get("sharpe", 0.0)
    adapt_sr  = perf.get("adaptive", {}).get("sharpe", 0.0)
    delta_sr  = adapt_sr - base_sr
    h1_sharpe = delta_sr >= MIN_SHARPE_GAIN

    log.info(f"\n── Sharpe Improvement ───────────────────────────────────────")
    log.info(f"  Baseline Sharpe : {base_sr:.4f}")
    log.info(f"  Adaptive Sharpe : {adapt_sr:.4f}")
    log.info(f"  Delta Sharpe    : {delta_sr:+.4f}  "
             f"(threshold >= {MIN_SHARPE_GAIN})")
    log.info(f"  H1 (Sharpe)     : {'CONFIRMED' if h1_sharpe else 'NOT confirmed'}")

    # ── Per-regime performance ────────────────────────────────────
    # Reindex consensus to the OOS return dates (returns_dict values cover OOS)
    oos_idx   = list(returns_dict.values())[0].index
    consensus_oos = consensus.reindex(oos_idx).ffill().dropna()
    perf_df = per_regime_performance(returns_dict, consensus_oos)

    # ── Catastrophic failure check ────────────────────────────────
    log.info(f"\n── Catastrophic Failure Check (threshold MDD > {MAX_REGIME_DD*100:.0f}%) ──")
    no_catastrophe = True
    for name in ["baseline", "adaptive", "defensive"]:
        sub = perf_df[perf_df["strategy"] == name]
        if sub.empty:
            continue
        worst_dd = sub["max_dd"].min()
        fail     = worst_dd < MAX_REGIME_DD
        if fail:
            no_catastrophe = False
        log.info(f"  {name:12s}  Worst MDD = {worst_dd*100:.1f}%  "
                 f"{'CATASTROPHIC FAILURE' if fail else 'OK'}")

    # ── Save ──────────────────────────────────────────────────────
    pd.DataFrame(returns_dict).to_csv(
        os.path.join(OUTPUT_DIR, "rq2_adaptive_returns.csv")
    )
    perf_df.to_csv(
        os.path.join(OUTPUT_DIR, "rq2_per_regime_perf.csv"), index=False
    )

    # Text summary
    lines = [
        "=" * 65,
        "  RQ2 STAGE 3 RESULTS  —  Adaptive Strategy",
        "=" * 65,
        f"  Baseline  Sharpe = {base_sr:.4f}",
        f"  Adaptive  Sharpe = {adapt_sr:.4f}",
        f"  Delta SR         = {delta_sr:+.4f}  "
        f"{'CONFIRMED H1' if h1_sharpe else 'H1 not confirmed'}",
        f"  Threshold        : delta >= {MIN_SHARPE_GAIN}",
        f"  No Catastrophe   : {'YES' if no_catastrophe else 'NO — failure detected'}",
        "=" * 65,
    ]
    report = "\n".join(lines)
    print("\n" + report)
    with open(os.path.join(OUTPUT_DIR, "rq2_stage3_summary.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # ── Plots ─────────────────────────────────────────────────────
    plot_comparison(
        returns_dict, consensus_oos, perf_df,
        save_path=os.path.join(OUTPUT_DIR,
                               "rq2_strategy_comparison.png"),
    )
    plot_regime_returns(
        returns_dict, consensus_oos,
        save_path=os.path.join(OUTPUT_DIR,
                               "rq2_regime_return_boxes.png"),
    )

    log.info("\n✓ Stage 3 complete. Run rq2_stage4_synthesis.py next.\n")
    return perf, delta_sr, no_catastrophe


if __name__ == "__main__":
    main()

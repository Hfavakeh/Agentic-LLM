"""Q2 — do per-epoch training-curve summaries improve the LLM or the rule-based
controller? Compares curves-OFF (P0) vs curves-ON (P1) for both arms across the
model ladder and emits a CSV + figure.

P0 folders are the Q3 `q3-none-<model>` runs (history=none + curves off =
defaults); P1 folders are the `curve-<tag>` runs (--payload-curves). Reads the
per-arm test RMSE from final_evaluation_run1.json and the LLM exploration
metrics from protocol_log_run1.json.

Run: python analyze_q2_curves.py --out analysis/q2_curves
"""
from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analyze_history_ablation import _final_rmse, _seed_metrics

BASELINE = 0.2308  # radar fixed-reference baseline (model-independent)

# model label -> (P0 folder, P1 folder). All pairs use the SAME model on both
# sides (clean A/B): llama runs are all llama3:8b, etc.
# Paths are relative to --root (default: the repo's results/ directory).
PAIRS: Dict[str, Tuple[str, str]] = {
    "llama3:8b":          ("history-use/history-none-llama3",
                           "curve-summaries/curve-llama3"),
    "nemotron-3-nano:4b": ("history-use/history-none-nemotron",
                           "curve-summaries/curve-nemotron"),
    "phi4:14b":           ("history-use/history-none-phi",
                           "curve-summaries/curve-phi"),
}


def _collect(folder: str) -> Dict[str, List[float]]:
    out = {"distinct": [], "rmse_llm": [], "rmse_rule": []}
    for sd in sorted(glob.glob(os.path.join(folder, "seed_*"))):
        pl = glob.glob(os.path.join(sd, "protocol_log_run1.json"))
        if pl:
            out["distinct"].append(_seed_metrics(pl[0])["distinct"])
        fe = glob.glob(os.path.join(sd, "final_evaluation_run1.json"))
        if fe:
            r = _final_rmse(fe[0])
            out["rmse_llm"].append(r.get("rmse_llm"))
            out["rmse_rule"].append(r.get("rmse_rule_based"))
    return out


def _mean_std(vals: List[float]) -> Tuple[float, float]:
    a = np.array([v for v in vals if isinstance(v, (int, float))], float)
    if not len(a):
        return float("nan"), 0.0
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=str(Path(__file__).resolve().parent.parent / "results"),
                    help="Directory holding the experiment groups (default: repo results/).")
    ap.add_argument("--out", default="analysis/curve_summaries")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    data = {
        m: {"P0": _collect(os.path.join(args.root, p0)),
            "P1": _collect(os.path.join(args.root, p1))}
        for m, (p0, p1) in PAIRS.items()
    }
    models = list(PAIRS.keys())

    # ── CSV ──
    csv = os.path.join(args.out, "q2_summary.csv")
    with open(csv, "w") as f:
        f.write("model,arm,metric,curves_off,curves_on,delta\n")
        for m in models:
            for key, arm, metric in [("rmse_llm", "llm", "test_rmse"),
                                     ("rmse_rule", "rule_based", "test_rmse"),
                                     ("distinct", "llm", "distinct_settings")]:
                m0, _ = _mean_std(data[m]["P0"][key])
                m1, _ = _mean_std(data[m]["P1"][key])
                f.write(f"{m},{arm},{metric},{m0:.4f},{m1:.4f},{m1 - m0:+.4f}\n")
    print(f"  summary -> {csv}")

    # ── Figure ──
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(models))
    w = 0.2
    fig, (axR, axE) = plt.subplots(1, 2, figsize=(13, 4.6))

    def bars(ax, key, off_lab, on_lab, c_off, c_on, offset):
        m0 = [_mean_std(data[m]["P0"][key])[0] for m in models]
        s0 = [_mean_std(data[m]["P0"][key])[1] for m in models]
        m1 = [_mean_std(data[m]["P1"][key])[0] for m in models]
        s1 = [_mean_std(data[m]["P1"][key])[1] for m in models]
        ax.bar(x + offset - w, m0, w, yerr=s0, capsize=3, color=c_off, label=off_lab)
        ax.bar(x + offset,     m1, w, yerr=s1, capsize=3, color=c_on,  label=on_lab)
        return m0, m1

    # RMSE panel: LLM (off/on) + Rule-based (off/on)
    bars(axR, "rmse_llm",  "LLM curves-off",  "LLM curves-on",  "#9ecae1", "#2171b5", -w)
    bars(axR, "rmse_rule", "Rule curves-off", "Rule curves-on", "#fdae6b", "#e6550d", +w + 0.02)
    axR.axhline(BASELINE, ls="--", color="k", lw=1, label=f"baseline {BASELINE:.3f}")
    axR.set_xticks(x); axR.set_xticklabels(models, fontsize=8)
    axR.set_title("Test RMSE (m) — curves OFF vs ON, per arm")
    allv = [v for m in models for k in ("rmse_llm", "rmse_rule")
            for v in (_mean_std(data[m]["P0"][k])[0], _mean_std(data[m]["P1"][k])[0])]
    lo, hi = min(allv + [BASELINE]), max(allv + [BASELINE])
    axR.set_ylim(lo - (hi - lo) * 0.3, hi + (hi - lo) * 0.15)
    axR.legend(fontsize=7, ncol=2)

    # Exploration panel: LLM distinct settings off vs on
    m0 = [_mean_std(data[m]["P0"]["distinct"])[0] for m in models]
    m1 = [_mean_std(data[m]["P1"]["distinct"])[0] for m in models]
    s0 = [_mean_std(data[m]["P0"]["distinct"])[1] for m in models]
    s1 = [_mean_std(data[m]["P1"]["distinct"])[1] for m in models]
    axE.bar(x - w / 2, m0, w, yerr=s0, capsize=3, color="#9ecae1", label="curves-off")
    axE.bar(x + w / 2, m1, w, yerr=s1, capsize=3, color="#2171b5", label="curves-on")
    axE.set_xticks(x); axE.set_xticklabels(models, fontsize=8)
    axE.set_title("LLM distinct settings explored / 25")
    axE.legend(fontsize=8)

    if not os.environ.get("FIG_NO_TITLE"):
        fig.suptitle("Q2 — per-epoch curves help the rule-based controller most "
                     "(only curve-aware rule-based beats baseline); the LLM improves "
                     "slightly but explores less", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1) if os.environ.get("FIG_NO_TITLE") else (0, 0, 1, 0.94))
    out_png = os.path.join(args.out, "q2_curves.png")
    fig.savefig(out_png, dpi=150)
    print(f"  figure -> {out_png}")


if __name__ == "__main__":
    main()

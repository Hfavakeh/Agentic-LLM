"""Q4 — why does the LLM explore poorly: prompt, model, or search space?

Compares how much of the search grid each arm covers in its 25 attempts, using
the per_setting_report CSVs from the standard (curves-off, real-history) runs.
random / Optuna / rule-based are model-independent (averaged across the model
folders); the LLM is shown per model. Emits a CSV + figure.

Run: python analyze_q4_exploration.py --out analysis/q4_exploration
"""
from __future__ import annotations

import sys as _sys, pathlib as _pathlib
_REPO_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import glob
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from pipeline import HP_GRID

HPS = ["learning_rate", "weight_decay", "dropout", "batch_size", "lstm_hidden",
       "lstm_layers", "window_size", "optimizer_choice", "patience"]
GRIDSIZE = {k: len(HP_GRID[k]) for k in HPS}

# Standard P0 runs (curves off, real history) per model.
MODEL_FOLDERS = {
    "llama3:8b":          "history-use/history-none-llama3",
    "nemotron-3-nano:4b": "history-use/history-none-nemotron",
    "phi4:14b":           "history-use/history-none-phi",
}
REF_ARMS = ["random", "optuna", "rule_based"]   # model-independent reference methods

# Q4 prompt-variant (--explore-prompt) runs, paired with the default-prompt runs
# above. Used by the --variant comparison.
EXPLORE_FOLDERS = {
    "llama3:8b":          "prompt-ablation/prompt-llama3",
    "nemotron-3-nano:4b": "prompt-ablation/prompt-nemotron",
    "phi4:14b":           "prompt-ablation/prompt-phi",
}
BASELINE_RMSE = 0.2308   # radar fixed-reference baseline
RULE_RMSE     = 0.2264   # curve-aware rule-based (best arm), for reference


def _seed_arm_stats(csv: str, arm: str):
    df = pd.read_csv(csv)
    df = df[(df["arm"] == arm) & (df["trained"] == True)]  # noqa: E712
    if not len(df):
        return None
    distinct = df[HPS].drop_duplicates().shape[0]
    # per-HP coverage capped at 1.0 (guards float-precision over-counts)
    cov = np.mean([min(df[h].nunique() / GRIDSIZE[h], 1.0) for h in HPS])
    return distinct, float(cov)


def collect() -> pd.DataFrame:
    rows: List[Dict] = []
    # LLM per model
    for model, fold in MODEL_FOLDERS.items():
        d, c = [], []
        for csv in glob.glob(os.path.join(fold, "seed_*", "per_setting_report_run1.csv")):
            s = _seed_arm_stats(csv, "llm")
            if s:
                d.append(s[0]); c.append(s[1])
        if d:
            rows.append({"arm": f"LLM — {model}", "kind": "llm",
                         "distinct": np.mean(d), "coverage": np.mean(c)})
    # reference methods, averaged across all model folders (they don't depend on the LLM)
    for arm in REF_ARMS:
        d, c = [], []
        for fold in MODEL_FOLDERS.values():
            for csv in glob.glob(os.path.join(fold, "seed_*", "per_setting_report_run1.csv")):
                s = _seed_arm_stats(csv, arm)
                if s:
                    d.append(s[0]); c.append(s[1])
        if d:
            label = {"random": "random", "optuna": "Optuna", "rule_based": "rule-based"}[arm]
            rows.append({"arm": label, "kind": "ref",
                         "distinct": np.mean(d), "coverage": np.mean(c)})
    return pd.DataFrame(rows)


def _llm_stats(folder: str):
    """(mean distinct, mean coverage, n_seeds) for the LLM arm in one folder."""
    D, C = [], []
    for csv in sorted(glob.glob(os.path.join(folder, "seed_*", "per_setting_report_run1.csv"))):
        s = _seed_arm_stats(csv, "llm")
        if s:
            D.append(s[0]); C.append(s[1])
    return (np.mean(D) if D else np.nan, np.mean(C) if C else np.nan, len(D))


def _llm_rmse(folder: str):
    from analyze_q3_ablation import _final_rmse
    v = []
    for fe in sorted(glob.glob(os.path.join(folder, "seed_*", "final_evaluation_run1.json"))):
        r = _final_rmse(fe).get("rmse_llm")
        if isinstance(r, (int, float)):
            v.append(r)
    return (np.mean(v) if v else np.nan, len(v))


def variant(out: str):
    """Q4 prompt-variant: default prompt vs --explore-prompt, per model —
    grid coverage (does it rise toward random's ~1.0?) and test RMSE (does the
    extra exploration buy better decisions?)."""
    models = list(EXPLORE_FOLDERS.keys())
    rows = []
    for m in models:
        d_dist, d_cov, d_n = _llm_stats(MODEL_FOLDERS[m])
        e_dist, e_cov, e_n = _llm_stats(EXPLORE_FOLDERS[m])
        d_rmse, dr_n = _llm_rmse(MODEL_FOLDERS[m])
        e_rmse, er_n = _llm_rmse(EXPLORE_FOLDERS[m])
        rows.append({"model": m, "cov_default": d_cov, "cov_explore": e_cov,
                     "dist_default": d_dist, "dist_explore": e_dist,
                     "rmse_default": d_rmse, "rmse_explore": e_rmse,
                     "n_explore_cov": e_n, "n_explore_rmse": er_n})
    df = pd.DataFrame(rows)
    csv = os.path.join(out, "q4_prompt_variant.csv")
    df.to_csv(csv, index=False)
    print(f"  summary -> {csv}")
    pd.set_option("display.width", 160)
    print(df.round(4).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(len(models))
    w = 0.36
    fig, (axC, axR) = plt.subplots(1, 2, figsize=(13, 4.8))

    # Coverage panel
    axC.bar(x - w / 2, df["cov_default"], w, color="#9ecae1", label="default prompt")
    axC.bar(x + w / 2, df["cov_explore"], w, color="#08519c", label="explore prompt")
    axC.axhline(1.0, ls="--", color="grey", lw=1, label="random / Optuna ~1.0")
    axC.set_xticks(x); axC.set_xticklabels(models, fontsize=8)
    axC.set_ylim(0, 1.1); axC.set_ylabel("fraction of grid covered")
    axC.set_title("Grid coverage — default vs explore prompt")
    axC.legend(fontsize=8)
    for xi, a, b in zip(x, df["cov_default"], df["cov_explore"]):
        axC.text(xi - w / 2, a + 0.02, f"{a:.2f}", ha="center", fontsize=8)
        axC.text(xi + w / 2, b + 0.02, f"{b:.2f}", ha="center", fontsize=8)

    # RMSE panel
    axR.bar(x - w / 2, df["rmse_default"], w, color="#fdae6b", label="default prompt")
    axR.bar(x + w / 2, df["rmse_explore"], w, color="#a63603", label="explore prompt")
    axR.axhline(BASELINE_RMSE, ls="--", color="k", lw=1, label=f"baseline {BASELINE_RMSE:.3f}")
    axR.axhline(RULE_RMSE, ls=":", color="green", lw=1.2, label=f"curve-rule {RULE_RMSE:.3f}")
    axR.set_xticks(x); axR.set_xticklabels(models, fontsize=8)
    vals = list(df["rmse_default"]) + list(df["rmse_explore"]) + [BASELINE_RMSE, RULE_RMSE]
    axR.set_ylim(min(vals) - 0.004, max(vals) + 0.003)
    axR.set_ylabel("test RMSE (m)")
    axR.set_title("Test RMSE — default vs explore prompt")
    axR.legend(fontsize=7)
    for xi, a, b in zip(x, df["rmse_default"], df["rmse_explore"]):
        axR.text(xi - w / 2, a + 0.0004, f"{a:.3f}", ha="center", fontsize=7)
        axR.text(xi + w / 2, b + 0.0004, f"{b:.3f}", ha="center", fontsize=7)

    if not os.environ.get("FIG_NO_TITLE"):
        fig.suptitle("Q4 prompt variant — dropping the anchor instruction lifts exploration "
                     "toward random's, but test RMSE stays flat (exploration was not the bottleneck)",
                     fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 1) if os.environ.get("FIG_NO_TITLE") else (0, 0, 1, 0.94))
    out_png = os.path.join(out, "q4_prompt_variant.png")
    fig.savefig(out_png, dpi=150)
    print(f"  figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default=str(Path(__file__).resolve().parent.parent / "results"),
                    help="Directory holding the experiment groups (default: repo results/).")
    ap.add_argument("--out", default="analysis/exploration")
    ap.add_argument("--variant", action="store_true",
                    help="default vs --explore-prompt comparison (coverage + RMSE)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Folder maps are declared relative to --root; resolve them once here.
    for mapping in (MODEL_FOLDERS, EXPLORE_FOLDERS):
        for k, v in list(mapping.items()):
            mapping[k] = os.path.join(args.root, v)

    if args.variant:
        variant(args.out)
        return

    df = collect()
    # Order: reference methods first, then LLMs (large -> small to show non-monotonicity)
    order = ["random", "Optuna", "rule-based",
             "LLM — phi4:14b", "LLM — llama3:8b", "LLM — nemotron-3-nano:4b"]
    df["__o"] = df["arm"].apply(lambda a: order.index(a) if a in order else 99)
    df = df.sort_values("__o").drop(columns="__o").reset_index(drop=True)

    csv = os.path.join(args.out, "q4_exploration.csv")
    df.to_csv(csv, index=False)
    print(f"  summary -> {csv}")
    pd.set_option("display.width", 140)
    print(df.to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#6baed6" if k == "ref" else "#e6550d" for k in df["kind"]]
    x = np.arange(len(df))
    fig, (axC, axD) = plt.subplots(1, 2, figsize=(13, 4.8))

    axC.bar(x, df["coverage"], color=colors)
    axC.axhline(1.0, ls="--", color="grey", lw=1)
    axC.set_xticks(x); axC.set_xticklabels(df["arm"], rotation=30, ha="right", fontsize=8)
    axC.set_ylim(0, 1.1); axC.set_ylabel("fraction of grid covered")
    axC.set_title("Search-grid coverage in 25 attempts")
    for xi, v in zip(x, df["coverage"]):
        axC.text(xi, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)

    axD.bar(x, df["distinct"], color=colors)
    axD.set_xticks(x); axD.set_xticklabels(df["arm"], rotation=30, ha="right", fontsize=8)
    axD.set_ylim(0, 26); axD.set_ylabel("distinct settings / 25")
    axD.set_title("Distinct settings evaluated")
    for xi, v in zip(x, df["distinct"]):
        axD.text(xi, v + 0.4, f"{v:.0f}", ha="center", fontsize=8)

    if not os.environ.get("FIG_NO_TITLE"):
        fig.suptitle("Q4 — the search space is fully explorable (random/Optuna ~100%); "
                     "every LLM is capped far below it, regardless of size",
                     fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 1) if os.environ.get("FIG_NO_TITLE") else (0, 0, 1, 0.94))
    out_png = os.path.join(args.out, "q4_exploration.png")
    fig.savefig(out_png, dpi=150)
    print(f"  figure -> {out_png}")


if __name__ == "__main__":
    main()

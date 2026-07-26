"""Generate the thesis figures as vector PDFs.

Reads the aggregates written by build_thesis_results.py (plus the radar CSV for
the dataset figures) and emits print-ready PDFs into thesis/figures/.

  fig_method_comparison.pdf  -- test RMSE per method, per-seed points overlaid
  fig_budget_use.pdf         -- trained vs rejected attempts per LLM, with RMSE
  fig_dataset_split.pdf      -- the non-contiguous train/val/test index ranges
  fig_sample_trajectory.pdf  -- an example ground-truth trajectory

Run: python scripts/build_thesis_figures.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

REPO = Path(__file__).resolve().parent.parent

# Print-friendly, colour-blind-safe palette; greys for reference lines.
INK, GREY, LIGHT = "#1A1A1A", "#6B7280", "#D1D5DB"
NAVY, TEAL, CORAL, AMBER = "#243B53", "#1F7A6F", "#D9655B", "#E0A340"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": GREY,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
})

ARM_ORDER = ["fixed_reference", "rule_based", "random", "llm", "optuna"]
ARM_LABEL = {
    "fixed_reference": "Fixed\nreference",
    "rule_based": "Rule-based",
    "random": "Random",
    "llm": "LLM",
    "optuna": "Optuna",
}
ARM_COLOR = {
    "fixed_reference": INK, "rule_based": TEAL, "random": NAVY,
    "llm": CORAL, "optuna": AMBER,
}


def fig_method_comparison(res: Path, out: Path) -> None:
    """Test RMSE per method: mean bar, per-seed scatter, reference line."""
    acc = pd.read_csv(res / "per_seed_accuracy.csv")
    ref = acc[acc.arm == "fixed_reference"].rmse.mean()

    fig, ax = plt.subplots(figsize=(5.4, 3.1))
    rng = np.random.default_rng(0)
    for i, arm in enumerate(ARM_ORDER):
        g = acc[acc.arm == arm].rmse.dropna()
        if g.empty:
            continue
        # Points + mean marker rather than bars: the y-axis is zoomed to a
        # 0.05 m range, and bars rising from a clipped baseline would
        # exaggerate the differences between methods.
        ax.scatter(i + rng.uniform(-0.17, 0.17, len(g)), g, s=8,
                   color=ARM_COLOR[arm], alpha=0.55, zorder=3, linewidths=0)
        m, sd = g.mean(), g.std(ddof=1)
        ax.errorbar(i, m, yerr=(sd if np.isfinite(sd) else 0.0), fmt="_",
                    markersize=17, color=ARM_COLOR[arm], lw=1.4, capsize=4,
                    zorder=4, markeredgewidth=1.8)
        ax.text(i, ax.get_ylim()[0], "", ha="center")

    ax.axhline(ref, color=INK, lw=0.9, ls="--", zorder=1)
    ax.text(len(ARM_ORDER) - 0.42, ref, "  fixed reference", va="center",
            ha="left", fontsize=7.5, color=INK)

    ax.set_xticks(range(len(ARM_ORDER)))
    ax.set_xticklabels([ARM_LABEL[a] for a in ARM_ORDER])
    ax.set_ylabel("Test RMSE (m)")
    lo = acc.rmse.min()
    ax.set_ylim(lo - 0.006, acc.rmse.max() + 0.004)
    ax.yaxis.grid(True, color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_method_comparison.pdf")
    plt.close(fig)


def fig_budget_use(res: Path, out: Path) -> None:
    """Trained vs rejected attempts per LLM, with final RMSE annotated."""
    sm = pd.read_csv(res / "summary_by_model.csv")
    d = sm[sm.arm == "llm"].sort_values("trained_mean", ascending=True)

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    y = np.arange(len(d))
    ax.barh(y, d.trained_mean, color=TEAL, alpha=0.85, height=0.6,
            label="trained", zorder=2)
    ax.barh(y, d.rejected_mean, left=d.trained_mean, color=CORAL, alpha=0.75,
            height=0.6, label="rejected", zorder=2)

    for i, (_, r) in enumerate(d.iterrows()):
        ax.text(25.4, i, f"{r.rmse_mean:.4f} m", va="center", fontsize=7.6,
                color=INK)
        ax.text(r.trained_mean / 2, i, f"{r.trained_mean:.1f}", va="center",
                ha="center", fontsize=7.4, color="white")

    ax.axvline(25, color=GREY, lw=0.8, ls=":")
    ax.set_yticks(y)
    ax.set_yticklabels([m.replace(":", ":\\") if False else m for m in d.model])
    ax.set_xlabel("Attempts (of a 25-attempt budget)")
    ax.set_xlim(0, 31)
    ax.set_ylim(-0.6, len(d) - 0.1)
    # legend above the plot, clear of the bars
    ax.legend(loc="lower center", bbox_to_anchor=(0.42, 1.0), frameon=False,
              ncol=2, handlelength=1.2)
    ax.xaxis.grid(True, color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_budget_use.pdf")
    plt.close(fig)


def fig_dataset(repo: Path, out: Path) -> None:
    """The non-contiguous split, and an example ground-truth trajectory."""
    csv = repo / "preprocessed-RadarEXP1(in).csv"
    df = pd.read_csv(csv, header=None)
    # last two columns are the x/y position targets
    xy = df.iloc[:, -2:].to_numpy(dtype=float)
    splits = {"train": (0, 1200), "test": (1200, 1600), "val": (1600, 2000)}
    colour = {"train": NAVY, "val": TEAL, "test": CORAL}

    # --- split diagram ---
    fig, ax = plt.subplots(figsize=(5.4, 1.25))
    for name, (a, b) in splits.items():
        ax.add_patch(Rectangle((a, 0), b - a, 1, color=colour[name], alpha=0.75))
        ax.text((a + b) / 2, 0.5, f"{name}\n{a}--{b}", ha="center", va="center",
                fontsize=8, color="white")
    ax.set_xlim(0, len(df))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Sample index in the recording")
    for s in ("left", "bottom"):
        ax.spines[s].set_visible(False)
    fig.savefig(out / "fig_dataset_split.pdf")
    plt.close(fig)

    # --- sample trajectory, one panel per split ---
    # Overlaying all three splits in one axes is unreadable: the walker covers
    # the same room in every block, so the paths sit on top of each other.
    order = ["train", "test", "val"]
    fig, axes = plt.subplots(1, 3, figsize=(5.4, 2.1), sharex=True, sharey=True)
    for ax, name in zip(axes, order):
        a, b = splits[name]
        seg = xy[a:b]
        ax.plot(seg[:, 0], seg[:, 1], lw=0.5, color=colour[name], alpha=0.9)
        ax.set_title(f"{name} ({b - a} samples)", fontsize=8.5)
        ax.set_xlabel("$x$ (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.tick_params(labelsize=7)
    axes[0].set_ylabel("$y$ (m)")
    fig.savefig(out / "fig_sample_trajectory.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=str(REPO / "analysis" / "thesis_results"))
    ap.add_argument("--out", default=str(REPO / "thesis" / "figures"))
    args = ap.parse_args()
    res, out = Path(args.res), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_method_comparison(res, out)
    fig_budget_use(res, out)
    fig_dataset(REPO, out)
    fig_curve_summaries(out)
    fig_history_ablation(out)
    fig_partb_controls(out)
    fig_motion_regime(out)
    # all-arms comparison, one figure per experimental stage
    fig_all_arms_distribution(out, "history-use/history-none-llama3",
                              "fig_cdf_stage1_generic", "generic HPO, curves off")
    fig_all_arms_distribution(out, "curve-summaries/curve-llama3",
                              "fig_cdf_stage2_curves", "with curve summaries")
    fig_all_arms_distribution(out, "history-use/history-empty-llama3",
                              "fig_cdf_stage3_nohistory", "history removed")
    fig_partb_distribution(out)
    for f in sorted(out.glob("*.pdf")):
        print(f"  wrote {f.relative_to(REPO)}  ({f.stat().st_size/1024:.0f} kB)")




# =====================================================================
# Diagnostic (Part A) and motion-aware (Part B) figures.
# These read the run logs directly, so re-running the script after new
# experiments land will refresh them without any edit.
# =====================================================================
import glob as _glob
import json as _json
import os as _os

RESULTS = REPO / "results"


def _final_eval_rmse(folder: str, arm: str = "llm"):
    """Mean test RMSE per search seed, protocol-format runs."""
    out = []
    for f in sorted(_glob.glob(str(RESULTS / folder / "seed_*" / "final_evaluation_run1.json"))):
        try:
            p = _json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        v = (p.get("per_arm", {}).get(arm) or {}).get("mean_rmse")
        if v is not None:
            out.append(v)
    return np.array(out, dtype=float)


def _motion_rmse(folder: str):
    """{arm: array of per-search-seed test RMSE} for motion-format runs."""
    out: dict = {}
    for f in sorted(_glob.glob(str(RESULTS / "motion" / folder / "seed_*" / "motion_experiment_run1.json"))):
        try:
            p = _json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        for arm, v in (p.get("final_eval") or {}).items():
            r = v.get("mean_rmse") if isinstance(v, dict) else None
            if r is not None:
                out.setdefault(arm, []).append(r)
    return {k: np.array(v, dtype=float) for k, v in out.items()}


def fig_curve_summaries(out: Path) -> None:
    """Q2: adding per-epoch curve summaries, LLM vs rule-based."""
    pairs = [("llama3:8b", "history-use/history-none-llama3", "curve-summaries/curve-llama3"),
             ("nemotron-3-nano:4b", "history-use/history-none-nemotron", "curve-summaries/curve-nemotron"),
             ("phi4:14b", "history-use/history-none-phi", "curve-summaries/curve-phi")]
    ref = _final_eval_rmse(pairs[0][1], "fixed_reference")
    ref = float(ref.mean()) if ref.size else 0.2308

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(pairs))
    w = 0.19
    for k, (arm, colour, lab) in enumerate([("llm", CORAL, "LLM"),
                                            ("rule_based", TEAL, "Rule-based")]):
        off = _np_mean([_final_eval_rmse(o, arm) for _, o, _ in pairs])
        on = _np_mean([_final_eval_rmse(n, arm) for _, _, n in pairs])
        ax.bar(x + (k * 2 - 1.5) * w, off, w, color=colour, alpha=0.35,
               edgecolor=colour, label=f"{lab}, curves off")
        ax.bar(x + (k * 2 - 0.5) * w, on, w, color=colour, alpha=0.95,
               edgecolor=colour, label=f"{lab}, curves on")

    ax.axhline(ref, color=INK, lw=0.9, ls="--")
    ax.text(len(pairs) - 0.45, ref, "  fixed reference", va="bottom", ha="left",
            fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in pairs], fontsize=8)
    ax.set_ylabel("Test RMSE (m)")
    ax.set_ylim(0.220, 0.256)
    ax.legend(frameon=False, fontsize=7.3, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, 1.24))
    ax.yaxis.grid(True, color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_curve_summaries.pdf")
    plt.close(fig)


def _np_mean(arrs):
    return [float(a.mean()) if a.size else np.nan for a in arrs]


def fig_history_ablation(out: Path) -> None:
    """Q3: real vs shuffled vs empty history, LLM only."""
    conds = ["none", "shuffled", "empty"]
    label = {"none": "real history", "shuffled": "shuffled", "empty": "empty"}
    models = ["llama3", "nemotron", "phi", "gemma4"]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    x = np.arange(len(conds))
    for j, mdl in enumerate(models):
        ys = [_final_eval_rmse(f"history-use/history-{c}-{mdl}") for c in conds]
        ys = [float(a.mean()) if a.size else np.nan for a in ys]
        ax.plot(x, ys, marker="o", ms=4, lw=1.1, label=mdl,
                color=[NAVY, TEAL, CORAL, AMBER][j % 4])

    pooled = [float(np.concatenate([_final_eval_rmse(f"history-use/history-{c}-{m}")
                                    for m in models]).mean()) for c in conds]
    ax.plot(x, pooled, marker="s", ms=6, lw=2.0, color=INK, label="pooled", zorder=5)

    ref = _final_eval_rmse("history-use/history-none-llama3", "fixed_reference")
    ax.axhline(float(ref.mean()) if ref.size else 0.2308, color=INK, lw=0.9, ls="--")
    ax.set_xticks(x)
    ax.set_xticklabels([label[c] for c in conds])
    ax.set_xlabel("History shown to the LLM")
    ax.set_ylabel("Test RMSE (m)")
    ax.legend(frameon=False, fontsize=7.5, ncol=5, loc="lower center",
              bbox_to_anchor=(0.5, 1.0), handlelength=1.4, columnspacing=1.1)
    ax.yaxis.grid(True, color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_history_ablation.pdf")
    plt.close(fig)


def fig_partb_controls(out: Path) -> None:
    """Part B: does motion knowledge add anything beyond the extra knobs?"""
    series = [
        ("Plain MSE\nbaseline", _motion_rmse("motion-refs").get("baseline"), GREY),
        ("Motion rule\n(deterministic)", _motion_rmse("motion-refs").get("motion_rule"), TEAL),
        ("Random over\nsame knobs", np.concatenate([a for a in
            [_motion_rmse("motion-random").get("random"),
             _motion_rmse("motion-random-v2").get("random")] if a is not None]), NAVY),
        ("LLM, knobs only\n(no motion)", np.concatenate([a for a in
            [_motion_rmse("motion-qwen-noprofile").get("llm"),
             _motion_rmse("motion-qwen-noprofile-v2").get("llm")] if a is not None]), AMBER),
        ("LLM + motion\nprofile", np.concatenate([a for a in
            [_motion_rmse("motion-qwen3").get("llm"),
             _motion_rmse("motion-qwen-v2").get("llm")] if a is not None]), CORAL),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.1))
    rng = np.random.default_rng(1)
    for i, (name, vals, col) in enumerate(series):
        if vals is None or len(vals) == 0:
            continue
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(vals)), vals, s=8,
                   color=col, alpha=0.55, linewidths=0, zorder=3)
        sd = vals.std(ddof=1) if len(vals) > 1 else 0.0
        ax.errorbar(i, vals.mean(), yerr=sd, fmt="_", markersize=17, color=col,
                    lw=1.4, capsize=4, zorder=4, markeredgewidth=1.8)
        ax.text(i, vals.mean() + sd + 0.0010, f"{vals.mean():.4f}", ha="center",
                fontsize=7.2, color=INK)
        ax.text(i, vals.mean() + sd + 0.0016, f"$n{{=}}{len(vals)}$", ha="center",
                fontsize=6.6, color=GREY)
    ax.set_xticks(range(len(series)))
    ax.set_xticklabels([s[0] for s in series], fontsize=7.6)
    ax.set_ylabel("Test RMSE (m)")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.0012)
    ax.yaxis.grid(True, color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_partb_controls.pdf")
    plt.close(fig)


def fig_motion_regime(out: Path) -> None:
    """Per-regime error: does the motion-aware loss help where it should?"""
    def regimes(folder, arm):
        acc = {"slow": [], "medium": [], "fast": []}
        for f in sorted(_glob.glob(str(RESULTS / "motion" / folder / "seed_*" /
                                       "motion_experiment_run1.json"))):
            try:
                p = _json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            r = (p.get("arms", {}).get(arm) or {}).get("best_regime_error") or {}
            for k in acc:
                if isinstance(r.get(k), (int, float)):
                    acc[k].append(r[k])
        return [float(np.mean(v)) if v else np.nan for v in acc.values()]

    series = [("Plain MSE baseline", regimes("motion-refs", "baseline"), GREY),
              ("Motion rule", regimes("motion-refs", "motion_rule"), TEAL),
              ("LLM + motion profile", regimes("motion-qwen3", "llm"), CORAL)]
    fig, ax = plt.subplots(figsize=(5.0, 2.9))
    x = np.arange(3)
    w = 0.26
    for i, (name, vals, col) in enumerate(series):
        ax.bar(x + (i - 1) * w, vals, w, color=col, alpha=0.85, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels(["slow", "medium", "fast"])
    ax.set_xlabel("Motion regime (speed tercile)")
    ax.set_ylabel("Validation RMSE (m)")
    ax.set_ylim(0.24, 0.29)
    ax.legend(frameon=False, fontsize=7.5)
    ax.yaxis.grid(True, color=LIGHT, lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out / "fig_motion_regime.pdf")
    plt.close(fig)




def _per_seed_rmse(folder: str, arm: str) -> np.ndarray:
    """Per-evaluation-seed RMSE for one arm, de-duplicated across searches.

    A search that selects the same configuration as another search produces a
    byte-identical vector of 30 evaluation results. Pooling those would multiply
    the apparent sample size without adding information, so identical vectors
    are counted once. For the deterministic arms -- and, on this condition, for
    the LLM too -- all searches agree, leaving the true n of 30.
    """
    seen: set = set()
    out: list = []
    for f in sorted(_glob.glob(str(RESULTS / folder / "seed_*" / "final_evaluation_run1.json"))):
        try:
            p = _json.load(open(f, encoding="utf-8"))
        except Exception:
            continue
        ps = (p.get("per_arm", {}).get(arm) or {}).get("per_seed") or []
        vals = [float(e.get("rmse") if isinstance(e, dict) else e) for e in ps
                if isinstance((e.get("rmse") if isinstance(e, dict) else e), (int, float))]
        if not vals:
            continue
        key = tuple(round(v, 9) for v in vals)
        if key in seen:
            continue
        seen.add(key)
        out.extend(vals)
    return np.array(out, dtype=float)


def fig_all_arms_distribution(out: Path, folder: str, name: str, caption_stage: str,
                              arms=("fixed_reference", "rule_based", "random",
                                    "llm", "optuna")) -> None:
    """All five arms on one pair of axes: CDF plus per-seed spread.

    Left  -- cumulative distribution of test RMSE over the 30 evaluation seeds,
             which shows whether one arm dominates another across the whole
             range or only in a tail.
    Right -- the same data as points with mean and standard deviation, which
             shows the ordering and the overlap at a glance.
    """
    data = {a: _per_seed_rmse(folder, a) for a in arms}
    data = {k: v for k, v in data.items() if v.size}
    if not data:
        print(f"  [skip] {name}: no per-seed data in {folder}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.7),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    order = sorted(data, key=lambda a: data[a].mean())

    for a in order:
        v = np.sort(data[a])
        ax1.plot(v, np.arange(1, v.size + 1) / v.size, lw=1.4,
                 color=ARM_COLOR.get(a, INK),
                 label=f"{ARM_LABEL.get(a, a).replace(chr(10), ' ')} "
                       f"({data[a].mean():.4f})")
    ax1.set_xlabel("Test RMSE (m)")
    ax1.set_ylabel("Cumulative fraction")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=6.6, loc="lower right")

    rng = np.random.default_rng(3)
    for i, a in enumerate(order):
        v = data[a]
        ax2.scatter(np.full(v.size, i) + rng.uniform(-0.16, 0.16, v.size), v,
                    s=5, color=ARM_COLOR.get(a, INK), alpha=0.45, linewidths=0)
        ax2.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="_", markersize=13,
                     color=ARM_COLOR.get(a, INK), lw=1.3, capsize=3,
                     markeredgewidth=1.6, zorder=4)
    ref = data.get("fixed_reference")
    if ref is not None:
        ax1.axvline(ref.mean(), color=GREY, lw=0.8, ls=":")
        ax2.axhline(ref.mean(), color=GREY, lw=0.8, ls=":")
    ax2.set_xticks(range(len(order)))
    ax2.set_xticklabels([ARM_LABEL.get(a, a).replace("\n", " ") for a in order],
                        rotation=38, ha="right", fontsize=6.8)
    ax2.set_ylabel("Test RMSE (m)")

    for ax in (ax1, ax2):
        ax.grid(True, color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)

    n = {a: data[a].size for a in order}
    print(f"  {name} [{caption_stage}] n={sorted(set(n.values()))}: " +
          ", ".join(f"{a}={data[a].mean():.4f}" for a in order))


PARTB_ARMS = [
    ("Plain MSE baseline", [("motion-refs", "baseline")], GREY),
    ("Motion rule", [("motion-refs", "motion_rule")], TEAL),
    ("Random over knobs", [("motion-random", "random"), ("motion-random-v2", "random")], NAVY),
    ("LLM, knobs only", [("motion-qwen-noprofile", "llm"), ("motion-qwen-noprofile-v2", "llm")], AMBER),
    ("LLM + motion", [("motion-qwen3", "llm"), ("motion-qwen-v2", "llm")], CORAL),
]


def _partb_per_seed(sources) -> np.ndarray:
    """Per-evaluation-seed RMSE for a Part B arm, de-duplicated like Part A."""
    seen, out = set(), []
    for folder, arm in sources:
        for f in sorted(_glob.glob(str(RESULTS / "motion" / folder / "seed_*" /
                                       "motion_experiment_run1.json"))):
            try:
                p = _json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            v = (p.get("final_eval") or {}).get(arm)
            if not isinstance(v, dict):
                continue
            ps = v.get("per_seed") or []
            vals = [float(x.get("rmse") if isinstance(x, dict) else x) for x in ps]
            if not vals:
                continue
            key = tuple(round(x, 9) for x in vals)
            if key in seen:
                continue
            seen.add(key)
            out.extend(vals)
    return np.array(out, dtype=float)


def fig_partb_distribution(out: Path) -> None:
    """Part B: all five motion arms, CDF plus per-seed spread."""
    data = [(lab, _partb_per_seed(src), col) for lab, src, col in PARTB_ARMS]
    data = [(l, v, c) for l, v, c in data if v.size]
    if not data:
        print("  [skip] fig_cdf_partb: no per-seed data")
        return
    data.sort(key=lambda t: t[1].mean())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.8, 2.7),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    for lab, v, col in data:
        s = np.sort(v)
        ax1.plot(s, np.arange(1, s.size + 1) / s.size, lw=1.4, color=col,
                 label=f"{lab} ({v.mean():.4f})")
    ax1.set_xlabel("Test RMSE (m)")
    ax1.set_ylabel("Cumulative fraction")
    ax1.set_ylim(0, 1.02)
    ax1.legend(frameon=False, fontsize=6.6, loc="lower right")

    rng = np.random.default_rng(4)
    for i, (lab, v, col) in enumerate(data):
        ax2.scatter(np.full(v.size, i) + rng.uniform(-0.16, 0.16, v.size), v,
                    s=5, color=col, alpha=0.45, linewidths=0)
        ax2.errorbar(i, v.mean(), yerr=v.std(ddof=1), fmt="_", markersize=13,
                     color=col, lw=1.3, capsize=3, markeredgewidth=1.6, zorder=4)
    base = next((v for l, v, _ in data if l.startswith("Plain")), None)
    if base is not None:
        ax1.axvline(base.mean(), color=GREY, lw=0.8, ls=":")
        ax2.axhline(base.mean(), color=GREY, lw=0.8, ls=":")
    ax2.set_xticks(range(len(data)))
    ax2.set_xticklabels([d[0] for d in data], rotation=38, ha="right", fontsize=6.8)
    ax2.set_ylabel("Test RMSE (m)")
    for ax in (ax1, ax2):
        ax.grid(True, color=LIGHT, lw=0.6)
        ax.set_axisbelow(True)
    fig.savefig(out / "fig_cdf_partb.pdf")
    plt.close(fig)
    print("  fig_cdf_partb [motion-aware]: " +
          ", ".join(f"{l}={v.mean():.4f}(n={v.size})" for l, v, _ in data))


if __name__ == "__main__":
    main()

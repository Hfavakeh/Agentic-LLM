"""Q3 history-use ablation analysis (professor's Email-4, question #3).

Aggregates the history-ablation placebo across seeds and conditions and emits a
report-ready table (CSV) + figure (PNG). Reads the protocol-path transcripts
written by SingleAgentOptimizer.save_protocol_log (one per seed) plus the
final-eval test RMSE per arm.

The three conditions perturb ONLY the rendered history the LLM sees (engine
training + true anchor unchanged):
  - none      real history
  - shuffled  each setting's outcome reassigned to another's (misleading)
  - empty     no prior attempts shown

Per seed it computes, from <cond>/seed_<N>/protocol_log_run1.json:
  - repeats        : attempts rejected as already_tried (out of 25)
  - off_grid       : attempts rejected as value_not_in_grid / unknown_param
  - other_reject   : parse / timeout / client errors
  - distinct       : number of distinct settings actually evaluated (accepted)
and, from <cond>/seed_<N>/final_evaluation_run1.json, the per-arm mean test RMSE.

Folders are <root>/q3-<cond>-<tag> (override per condition with --none/--shuffled
/--empty). Re-run with a different --tag for the Phase-2 multi-model sweep.

Run: python analyze_q3_ablation.py --root . --tag llama10 --out analysis/q3_ablation
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

CONDITIONS = ["none", "shuffled", "empty"]

# Nicer display names for known run tags (cross-model figure legend).
TAG_LABELS = {
    "llama10": "llama3:8b",
    "nemotron": "nemotron-3-nano:4b",
    "phi": "phi4:14b",
}


def _classify_reject(reason: str) -> str:
    if not reason:
        return "other_reject"
    if reason == "already_tried":
        return "already_tried"
    if reason.startswith("value_not_in_grid") or reason.startswith("unknown_param"):
        return "off_grid"
    return "other_reject"


def _setting_sig(setting: Optional[Dict[str, Any]]) -> tuple:
    return tuple(sorted((setting or {}).items()))


def _seed_metrics(protocol_path: str) -> Dict[str, Any]:
    """Per-seed behavioural metrics from one protocol_log_run*.json."""
    with open(protocol_path) as fh:
        log = json.load(fh)
    n = len(log)
    accepted = [e for e in log if e.get("outcome") == "accepted"]
    rejected = [e for e in log if e.get("outcome") == "rejected"]
    cats = [_classify_reject(e.get("final_reason", "")) for e in rejected]
    distinct = len({_setting_sig(e.get("resolved_setting")) for e in accepted})
    # Sanity: confirm the ablation mode is consistent within the file.
    modes = {e.get("history_ablation") for e in log}
    return {
        "n_attempts":   n,
        "accepted":     len(accepted),
        "distinct":     distinct,
        "already_tried": cats.count("already_tried"),
        "off_grid":     cats.count("off_grid"),
        "other_reject": cats.count("other_reject"),
        "repeat_rate":  cats.count("already_tried") / n if n else np.nan,
        "mode_in_file": next(iter(modes)) if len(modes) == 1 else f"MIXED:{modes}",
    }


def _final_rmse(final_eval_path: str) -> Dict[str, Optional[float]]:
    """Per-arm mean test RMSE from one final_evaluation_run*.json."""
    out: Dict[str, Optional[float]] = {}
    if not os.path.exists(final_eval_path):
        return out
    with open(final_eval_path) as fh:
        fe = json.load(fh)
    per_arm = fe.get("per_arm", {})
    # The fixed-reference baseline arm is keyed "fixed_reference" here; normalise
    # to "baseline" so it lines up with the rest of the codebase.
    rename = {"fixed_reference": "baseline"}
    for arm, d in per_arm.items():
        if isinstance(d, dict):
            out[f"rmse_{rename.get(arm, arm)}"] = d.get("mean_rmse", d.get("rmse_mean"))
    return out


def collect(root: str, tag: str, overrides: Dict[str, Optional[str]]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for cond in CONDITIONS:
        folder = overrides.get(cond) or os.path.join(root, f"history-{cond}-{tag}")
        seed_dirs = sorted(glob.glob(os.path.join(folder, "seed_*")))
        if not seed_dirs:
            # also support a flat (single-seed) layout
            flat = glob.glob(os.path.join(folder, "protocol_log_run*.json"))
            seed_dirs = [folder] if flat else []
        if not seed_dirs:
            print(f"  WARNING: no seeds found for '{cond}' at {folder}")
            continue
        for sd in seed_dirs:
            plog = glob.glob(os.path.join(sd, "protocol_log_run*.json"))
            if not plog:
                continue
            seed = os.path.basename(sd).replace("seed_", "") if "seed_" in sd else "flat"
            row = {"condition": cond, "seed": seed, "folder": folder}
            row.update(_seed_metrics(plog[0]))
            felist = glob.glob(os.path.join(sd, "final_evaluation_run*.json"))
            if felist:
                row.update(_final_rmse(felist[0]))
            rows.append(row)
    return pd.DataFrame(rows)


def _agg(series: pd.Series) -> str:
    vals = series.dropna().astype(float)
    if vals.empty:
        return "n/a"
    if len(vals) == 1:
        return f"{vals.iloc[0]:.4g}"
    return f"{vals.mean():.4g} ± {vals.std(ddof=1):.2g}"


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    metrics = ["already_tried", "off_grid", "other_reject", "distinct",
               "repeat_rate", "rmse_llm", "rmse_baseline"]
    metrics = [m for m in metrics if m in df.columns]
    out = []
    for cond in CONDITIONS:
        g = df[df["condition"] == cond]
        if g.empty:
            continue
        row = {"condition": cond, "n_seeds": len(g)}
        for m in metrics:
            row[m] = _agg(g[m])
        out.append(row)
    return pd.DataFrame(out)


def make_figure(df: pd.DataFrame, out_png: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [c for c in CONDITIONS if c in set(df["condition"])]
    x = np.arange(len(present))

    def mean_std(metric):
        means, stds = [], []
        for c in present:
            if metric not in df.columns:
                means.append(np.nan); stds.append(0.0); continue
            v = df[df["condition"] == c][metric].dropna().astype(float)
            means.append(v.mean() if len(v) else np.nan)
            stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        return np.array(means), np.array(stds)

    panels = [("already_tried", "Repeats (already_tried) / 25", "#c0504d"),
              ("distinct",      "Distinct settings evaluated / 25", "#4f81bd")]
    has_rmse = "rmse_llm" in df.columns
    n = 3 if has_rmse else 2
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n, 4.2))

    for ax, (metric, title, color) in zip(axes, panels):
        m, s = mean_std(metric)
        ax.bar(x, m, yerr=s, capsize=5, color=color, alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(present)
        ax.set_title(title); ax.set_xlabel("history shown to LLM")
        for xi, mi in zip(x, m):
            if np.isfinite(mi):
                ax.text(xi, mi, f"{mi:.1f}", ha="center", va="bottom", fontsize=9)

    if has_rmse:
        ax = axes[2]
        m, s = mean_std("rmse_llm")
        ax.bar(x, m, yerr=s, capsize=5, color="#9bbb59", alpha=0.85, label="LLM arm")
        bm, _ = mean_std("rmse_baseline")
        base = np.nanmean(bm) if np.isfinite(bm).any() else np.nan
        if np.isfinite(base):
            ax.axhline(base, ls="--", color="k", lw=1, label=f"baseline {base:.3f}")
        ax.set_xticks(x); ax.set_xticklabels(present)
        ax.set_title("LLM test RMSE (m)"); ax.set_xlabel("history shown to LLM")
        lo = np.nanmin([np.nanmin(m), base]); hi = np.nanmax([np.nanmax(m), base])
        ax.set_ylim(lo * 0.97, hi * 1.02)
        ax.legend(fontsize=8)
        for xi, mi in zip(x, m):
            if np.isfinite(mi):
                ax.text(xi, mi, f"{mi:.3f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Q3 history-use ablation — does the LLM use the history it is shown?",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_png, dpi=150)
    print(f"  figure -> {out_png}")


def cross_model_figure(by_model: Dict[str, pd.DataFrame], out_png: str) -> None:
    """Grouped bars: per metric (panel), x=conditions, one bar group per model.
    Shows whether the history-use pattern holds across models/scale."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = list(by_model.keys())
    x = np.arange(len(CONDITIONS))
    w = 0.8 / max(len(models), 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(models), 3)))

    def ms(df, cond, metric):
        if metric not in df.columns:
            return np.nan, 0.0
        v = df[df["condition"] == cond][metric].dropna().astype(float)
        return (v.mean() if len(v) else np.nan,
                v.std(ddof=1) if len(v) > 1 else 0.0)

    panels = [("already_tried", "Repeats (already_tried) / 25"),
              ("distinct", "Distinct settings / 25"),
              ("rmse_llm", "LLM test RMSE (m)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    for ax, (metric, title) in zip(axes, panels):
        for mi, model in enumerate(models):
            means = [ms(by_model[model], c, metric)[0] for c in CONDITIONS]
            stds = [ms(by_model[model], c, metric)[1] for c in CONDITIONS]
            ax.bar(x + mi * w - 0.4 + w / 2, means, w, yerr=stds, capsize=4,
                   color=colors[mi], alpha=0.85, label=TAG_LABELS.get(model, model))
        ax.set_xticks(x); ax.set_xticklabels(CONDITIONS)
        ax.set_title(title); ax.set_xlabel("history shown to LLM")
        if metric == "rmse_llm":
            bvals = [ms(df, "none", "rmse_baseline")[0] for df in by_model.values()]
            base = np.nanmean(bvals) if np.isfinite(bvals).any() else np.nan
            allm = [ms(by_model[m], c, "rmse_llm")[0] for m in models for c in CONDITIONS]
            allm = [v for v in allm if np.isfinite(v)]
            if allm:
                lo = min(allm + ([base] if np.isfinite(base) else []))
                hi = max(allm + ([base] if np.isfinite(base) else []))
                ax.set_ylim(lo - (hi - lo) * 0.4, hi + (hi - lo) * 0.25)
            if np.isfinite(base):
                ax.axhline(base, ls="--", color="k", lw=1, label=f"baseline {base:.3f}")
        ax.legend(fontsize=8)
    if not os.environ.get("FIG_NO_TITLE"):
        fig.suptitle("Q3 history-use ablation across models — exploration differs hugely "
                     "by model, but accuracy is flat across history and no better than baseline",
                     fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 1) if os.environ.get("FIG_NO_TITLE") else (0, 0, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    print(f"  cross-model figure -> {out_png}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="dir holding the q3-<cond>-<tag> folders")
    ap.add_argument("--tag", default="llama10", help="folder suffix (model/run tag)")
    ap.add_argument("--none", default=None, help="explicit path for the none condition")
    ap.add_argument("--shuffled", default=None, help="explicit path for the shuffled condition")
    ap.add_argument("--empty", default=None, help="explicit path for the empty condition")
    ap.add_argument("--out", default="analysis/q3_ablation", help="output dir")
    ap.add_argument("--compare", nargs="+", default=None, metavar="TAG",
                    help="cross-model mode: tags to overlay (e.g. --compare llama10 nemotron)")
    args = ap.parse_args()

    overrides = {"none": args.none, "shuffled": args.shuffled, "empty": args.empty}
    os.makedirs(args.out, exist_ok=True)

    # ── Cross-model comparison mode ───────────────────────────────────────────
    if args.compare:
        by_model: Dict[str, pd.DataFrame] = {}
        frames = []
        for tag in args.compare:
            d = collect(args.root, tag, {})
            if d.empty:
                print(f"  WARNING: no data for tag '{tag}'")
                continue
            d["model"] = tag
            by_model[tag] = d
            frames.append(d)
        if not by_model:
            print("No data found for any --compare tag.")
            return
        combined = pd.concat(frames, ignore_index=True)
        combined_csv = os.path.join(args.out, "q3_cross_model_per_seed.csv")
        combined.to_csv(combined_csv, index=False)
        print(f"  combined per-seed -> {combined_csv}  ({len(combined)} rows)")
        agg_rows = []
        for tag, d in by_model.items():
            a = aggregate(d); a.insert(0, "model", TAG_LABELS.get(tag, tag))
            agg_rows.append(a)
        agg_all = pd.concat(agg_rows, ignore_index=True)
        agg_all.to_csv(os.path.join(args.out, "q3_cross_model_summary.csv"), index=False)
        pd.set_option("display.width", 200)
        print("\n=== Q3 cross-model summary (mean ± std across seeds) ===")
        print(agg_all.to_string(index=False))
        cross_model_figure(by_model, os.path.join(args.out, "q3_cross_model.png"))
        return

    print("Collecting per-seed Q3 metrics...")
    df = collect(args.root, args.tag, overrides)
    if df.empty:
        print("No data found. Check --root/--tag or pass explicit --none/--shuffled/--empty.")
        return

    per_seed_csv = os.path.join(args.out, "q3_per_seed.csv")
    df.sort_values(["condition", "seed"]).to_csv(per_seed_csv, index=False)
    print(f"  per-seed -> {per_seed_csv}  ({len(df)} rows)")

    agg = aggregate(df)
    agg_csv = os.path.join(args.out, "q3_summary.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"  summary -> {agg_csv}")

    pd.set_option("display.width", 170)
    print("\n=== Q3 history-use ablation — summary (mean ± std across seeds) ===")
    print(agg.to_string(index=False))

    make_figure(df, os.path.join(args.out, "q3_ablation.png"))

    # Mode sanity: every file in a condition must carry that condition's mode.
    bad = df[df.apply(lambda r: r["mode_in_file"] != r["condition"], axis=1)]
    if len(bad):
        print("\nWARNING: ablation mode mismatch in these rows:")
        print(bad[["condition", "seed", "mode_in_file"]].to_string(index=False))
    else:
        print("\nSanity: every protocol_log's recorded ablation mode matches its condition. OK")


if __name__ == "__main__":
    main()

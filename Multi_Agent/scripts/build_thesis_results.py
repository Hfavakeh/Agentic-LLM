"""Compute every number quoted in the thesis's Numerical Results section.

Reads the protocol runs (25 attempts, scoring seeds 101/102/103, best setting
re-evaluated on fresh seeds 201+) and aggregates them along the four axes
promised in main.tex, sub:metrics:

  1. Final accuracy      -- test RMSE (m) and R^2, mean +/- sd across search
                            seeds, and % change vs the fixed reference
  2. Reliability         -- win rate vs the fixed reference
  3. Sample efficiency   -- trials-to-threshold (first attempt whose validation
                            RMSE beats the fixed reference's validation RMSE)
  4. Optimization dynamics -- normalised area under the best-so-far validation
                            RMSE trajectory over the attempt budget (lower is
                            better: the loss is driven down sooner)

Writes CSVs plus a LaTeX-ready booktabs table to --out.

Run: python scripts/build_thesis_results.py --out analysis/thesis_results
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent

# numpy renamed trapz -> trapezoid in 2.0
_trapz = getattr(np, "trapezoid", None) or np.trapz

# The standard condition: real history, curves off -- the runs the thesis's
# search protocol describes. One folder per LLM; the non-LLM arms are
# model-independent but are re-run inside each folder.
CONDITION = "history-use/history-none-*"
MODEL_LABEL = {
    "history-none-llama3":   "llama3:8b",
    "history-none-phi":      "phi4:14b",
    "history-none-nemotron": "nemotron-3-nano:4b",
    "history-none-gemma4":   "gemma3-4b",
}
ARMS = ["fixed_reference", "random", "rule_based", "optuna", "llm"]
ARM_LABEL = {
    "fixed_reference": "Fixed reference",
    "random":          "Random search",
    "rule_based":      "Rule-based",
    "optuna":          "Optuna (TPE)",
    "llm":             "LLM",
}


# ---------------------------------------------------------------- accuracy
def load_final_eval(root: Path) -> pd.DataFrame:
    """One row per (model, search seed, arm) from final_evaluation_run1.json."""
    rows: List[Dict] = []
    for folder in sorted(glob.glob(str(root / CONDITION))):
        name = os.path.basename(folder)
        if name.endswith("-pilot"):
            continue
        model = MODEL_LABEL.get(name, name)
        for f in sorted(glob.glob(os.path.join(folder, "seed_*", "final_evaluation_run1.json"))):
            seed = int(os.path.basename(os.path.dirname(f)).split("_")[1])
            try:
                p = json.load(open(f, encoding="utf-8"))
            except Exception:
                continue
            per_arm = p.get("per_arm", {})
            for arm in ARMS:
                d = per_arm.get(arm)
                if not isinstance(d, dict):
                    continue
                rows.append({
                    "model": model, "search_seed": seed, "arm": arm,
                    "rmse": d.get("mean_rmse"), "r2": d.get("mean_r2"),
                    "n_eval_seeds": d.get("n_seeds"),
                })
    return pd.DataFrame(rows)


# ------------------------------------------- efficiency and dynamics
def load_trajectories(root: Path) -> pd.DataFrame:
    """One row per (model, search seed, arm) with trials-to-threshold and AUC."""
    rows: List[Dict] = []
    for folder in sorted(glob.glob(str(root / CONDITION))):
        name = os.path.basename(folder)
        if name.endswith("-pilot"):
            continue
        model = MODEL_LABEL.get(name, name)
        for f in sorted(glob.glob(os.path.join(folder, "seed_*", "per_setting_report_run1.csv"))):
            seed = int(os.path.basename(os.path.dirname(f)).split("_")[1])
            try:
                df = pd.read_csv(f)
            except Exception:
                continue
            ref = df[df.arm == "fixed_reference"]["val_rmse_mean"].dropna()
            if ref.empty:
                continue
            threshold = float(ref.iloc[0])

            for arm in ARMS:
                if arm == "fixed_reference":
                    continue
                sub = df[df.arm == arm].sort_values("attempt")
                if sub.empty:
                    continue
                budget = int(sub["attempt"].max())
                trained = sub[sub.val_rmse_mean.notna()]
                if trained.empty:
                    continue
                vals = trained["val_rmse_mean"].to_numpy(dtype=float)

                # Best-so-far indexed by ATTEMPT (1..budget), not by trained
                # setting: a rejected attempt consumes budget and yields no
                # improvement, so the curve is flat there. Without this the
                # LLM's 5 trained attempts would be compared against the
                # others' 25 on a different x-axis.
                curve = np.full(budget, np.nan)
                for a, v in zip(trained["attempt"].astype(int), vals):
                    if 1 <= a <= budget:
                        curve[a - 1] = v
                best = np.inf
                bsf = np.empty(budget)
                for i in range(budget):
                    if not np.isnan(curve[i]):
                        best = min(best, curve[i])
                    bsf[i] = best
                # attempts before the first trained setting have no value yet;
                # hold them at the fixed reference so every arm starts equal
                bsf[np.isinf(bsf)] = threshold

                final_best = float(np.nanmin(vals))
                # sample efficiency: attempt at which the arm first reached the
                # best value it would ever reach within the budget
                atb = int(np.argmax(bsf <= final_best + 1e-12)) + 1
                # trials-to-threshold: first attempt beating the fixed reference
                hit = np.where(bsf < threshold)[0]
                tt = int(hit[0]) + 1 if hit.size else np.nan

                status = sub["output_status"].astype(str)
                hp_cols = ["learning_rate", "weight_decay", "dropout", "batch_size",
                           "lstm_hidden", "lstm_layers", "window_size",
                           "optimizer_choice", "patience"]
                have = [c for c in hp_cols if c in trained.columns]
                distinct = int(trained[have].drop_duplicates().shape[0]) if have else np.nan

                rows.append({
                    "model": model, "search_seed": seed, "arm": arm,
                    "budget": budget,
                    "n_trained": int(len(vals)),
                    "n_rejected": int((status == "rejected").sum()),
                    "distinct_settings": distinct,
                    "attempts_to_best": atb,
                    "trials_to_threshold": tt,
                    "auc": (_trapz(bsf) / (budget - 1)) / threshold,
                    "best_val": final_best,
                    "threshold": threshold,
                })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(REPO / "results"))
    ap.add_argument("--out", default=str(REPO / "analysis" / "thesis_results"))
    args = ap.parse_args()
    root, out = Path(args.root), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    acc = load_final_eval(root)
    traj = load_trajectories(root)
    acc.to_csv(out / "per_seed_accuracy.csv", index=False)
    traj.to_csv(out / "per_seed_trajectory.csv", index=False)

    # --- reliability: does this arm beat the fixed reference on this seed? ---
    ref = (acc[acc.arm == "fixed_reference"]
           .set_index(["model", "search_seed"])["rmse"])
    acc["ref_rmse"] = acc.set_index(["model", "search_seed"]).index.map(ref)
    acc["beats_ref"] = acc["rmse"] < acc["ref_rmse"]

    # ---------------- per-model summary ----------------
    summary = []
    for (model, arm), g in acc.groupby(["model", "arm"]):
        t = traj[(traj.model == model) & (traj.arm == arm)]
        summary.append({
            "model": model, "arm": arm, "n_seeds": len(g),
            "rmse_mean": g.rmse.mean(), "rmse_sd": g.rmse.std(ddof=1),
            "r2_mean": g.r2.mean(),
            "pct_vs_ref": 100.0 * (g.rmse.mean() - g.ref_rmse.mean()) / g.ref_rmse.mean(),
            "win_rate": g.beats_ref.mean(),
            "trained_mean": t.n_trained.mean() if len(t) else np.nan,
            "rejected_mean": t.n_rejected.mean() if len(t) else np.nan,
            "distinct_mean": t.distinct_settings.mean() if len(t) else np.nan,
            "atb_median": t.attempts_to_best.median() if len(t) else np.nan,
            "auc_mean": t.auc.mean() if len(t) else np.nan,
        })
    sm = pd.DataFrame(summary)
    sm.to_csv(out / "summary_by_model.csv", index=False)

    # ---------------- pooled across models ----------------
    pooled = []
    for arm in ARMS:
        g = acc[acc.arm == arm]
        t = traj[traj.arm == arm]
        if g.empty:
            continue
        pooled.append({
            "arm": arm, "n": len(g),
            "rmse_mean": g.rmse.mean(), "rmse_sd": g.rmse.std(ddof=1),
            "r2_mean": g.r2.mean(),
            "pct_vs_ref": 100.0 * (g.rmse.mean() - g.ref_rmse.mean()) / g.ref_rmse.mean(),
            "win_rate": g.beats_ref.mean(),
            "trained_mean": t.n_trained.mean() if len(t) else np.nan,
            "rejected_mean": t.n_rejected.mean() if len(t) else np.nan,
            "distinct_mean": t.distinct_settings.mean() if len(t) else np.nan,
            "atb_median": t.attempts_to_best.median() if len(t) else np.nan,
            "auc_mean": t.auc.mean() if len(t) else np.nan,
        })
    pl = pd.DataFrame(pooled)
    pl.to_csv(out / "summary_pooled.csv", index=False)

    # ---------------- paired differences vs the LLM ----------------
    paired = []
    llm = acc[acc.arm == "llm"].set_index(["model", "search_seed"])["rmse"]
    for arm in ARMS:
        if arm == "llm":
            continue
        other = acc[acc.arm == arm].set_index(["model", "search_seed"])["rmse"]
        common = llm.index.intersection(other.index)
        d = (llm.loc[common] - other.loc[common]).astype(float)
        if d.empty:
            continue
        n = len(d)
        se = d.std(ddof=1) / np.sqrt(n)
        paired.append({
            "comparison": f"llm_minus_{arm}", "n": n,
            "mean_diff": d.mean(), "sd": d.std(ddof=1),
            "ci_low": d.mean() - 1.96 * se, "ci_high": d.mean() + 1.96 * se,
            "llm_better_frac": (d < 0).mean(),
        })
    pr = pd.DataFrame(paired)
    pr.to_csv(out / "paired_vs_llm.csv", index=False)

    # ---------------- LaTeX table ----------------
    tex = [
        r"\begin{table}[h]", r"\centering",
        r"\caption{Search-method comparison on the radar dataset, pooled over "
        r"the four LLM configurations and ten search seeds. Test RMSE and $R^2$ "
        r"are computed on fresh evaluation seeds; win rate is the fraction of "
        r"searches beating the fixed reference.}",
        r"\label{tab:mainresults}",
        r"\begin{tabular}{lccccc}", r"\toprule",
        r"Method & RMSE (m) & $R^2$ & vs.\ ref. (\%) & Win rate & AUC \\",
        r"\midrule",
    ]
    for _, r in pl.iterrows():
        auc = "--" if pd.isna(r.auc_mean) else f"{r.auc_mean:.3f}"
        tex.append(f"{ARM_LABEL[r.arm]} & ${r.rmse_mean:.4f} \\pm {r.rmse_sd:.4f}$ & "
                   f"${r.r2_mean:.3f}$ & ${r.pct_vs_ref:+.1f}$ & "
                   f"${r.win_rate:.2f}$ & ${auc}$ \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "table_main.tex").write_text("\n".join(tex), encoding="utf-8")

    # ---------------- console report ----------------
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== pooled across models (test RMSE in metres, lower is better) ===")
    print(pl.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== per model ===")
    print(sm.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\n=== paired differences vs LLM (negative => LLM better) ===")
    print(pr.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\n[done] wrote CSVs + table_main.tex to {out}")


if __name__ == "__main__":
    main()

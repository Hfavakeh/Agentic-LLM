"""Deeper LLM-behavior diagnostics (professor's point #3).

Adds two metrics not in analyze_llm_behavior.py and consolidates the per-model
evidence for: understanding of the task / allowed values / repeats-are-useless,
use of history, exploration, stuck-near-early, AVOIDING values random found
useful, reasonable changes after a bad result, and prompt-clarity-vs-capability.

Reads the artefacts already written by analyze_llm_behavior.py
(per_attempt_long.csv, per_run_metrics.csv) plus the per_setting_report CSVs in
the Downloads result folders.

Run: python analyze_llm_behavior_q3.py --root "C:/Users/hfava/Downloads"
"""
from __future__ import annotations
import argparse, glob, os
from collections import defaultdict
import numpy as np
import pandas as pd

HP_KEYS = ["learning_rate", "weight_decay", "dropout", "batch_size",
           "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience"]

MODEL_FOLDERS = {
    "llama38b-new": "Llama 3 8B", "llama38b-new1": "Llama 3 8B",
    "phi4-new": "Phi 4 (14B)", "phi414-new1": "Phi 4 (14B)",
    "nemotron3-new": "Nemotron 3",
    "gemma4-12b-new1": "Gemma 4 12B",   # gemma4-12b-new excluded (all 404)
    "qwen3.5-new": "Qwen 3 8B",
}


def _sig(row):
    return tuple((k, row[k]) for k in HP_KEYS)


def course_correction(long_df: pd.DataFrame):
    """After a TRAINED attempt that did NOT improve best-so-far, what does the
    LLM do next? Two sub-metrics per model:
      * next_differs  — next genuine proposal is a *different* setting (not a
        re-proposal of the just-failed one): a minimal sign of reacting.
      * reproposes_known_worse — fraction of already-tried *rejections* that
        re-propose a setting whose earlier score was WORSE than the best-so-far
        at that point (i.e. re-suggesting a known-inferior config = ignoring
        the result).
    """
    rows = []
    for model, g0 in long_df.groupby("model"):
        next_diff = next_tot = 0
        repro_worse = repro_tot = 0
        for (seed,), g in g0.groupby(["seed"]):
            g = g.sort_values("attempt").reset_index(drop=True)
            # map signature -> best (lowest) score seen so far when trained
            seen_score = {}
            best_so_far = np.inf
            for i, r in g.iterrows():
                sig = _sig(r)
                trained = bool(r["trained"])
                score = r["score"] if pd.notna(r["score"]) else None
                # course-correction: a trained, non-improving attempt
                if trained and not bool(r["improved"]):
                    # find next genuine proposal
                    for j in range(i + 1, len(g)):
                        nxt = g.iloc[j]
                        if nxt["rejection_category"] == "serving_error":
                            continue
                        next_tot += 1
                        if _sig(nxt) != sig:
                            next_diff += 1
                        break
                # re-proposes known-worse: an already_tried rejection whose
                # signature was previously trained with a score worse than the
                # current best-so-far
                if r["rejection_category"] == "already_tried":
                    repro_tot += 1
                    prev = seen_score.get(sig)
                    if prev is not None and prev > best_so_far + 1e-9:
                        repro_worse += 1
                # update memory AFTER processing
                if trained and score is not None:
                    seen_score[sig] = min(seen_score.get(sig, np.inf), score)
                    best_so_far = min(best_so_far, score)
        rows.append({
            "model": model,
            "next_differs_frac": (next_diff / next_tot) if next_tot else np.nan,
            "n_post_bad": next_tot,
            "reproposes_known_worse_frac": (repro_worse / repro_tot) if repro_tot else np.nan,
            "n_already_tried": repro_tot,
        })
    return pd.DataFrame(rows)


def avoid_random_useful(root: str, long_df: pd.DataFrame):
    """Does the LLM avoid values random search found useful?

    For each (model, seed) take random's BEST trained setting (lowest
    val_rmse_mean). Pool the winning values per HP across seeds -> the values
    random found useful. Then per model compute the fraction of those winning
    values the LLM ever proposed (genuine attempts, same model).
    """
    rand_vals = defaultdict(lambda: defaultdict(set))   # model -> hp -> set(winning vals)
    for folder, model in MODEL_FOLDERS.items():
        for csv in glob.glob(os.path.join(root, folder, "seed_*", "per_setting_report_run1.csv")):
            try:
                df = pd.read_csv(csv)
            except Exception:
                continue
            r = df[(df["arm"] == "random") & (df["trained"] == True)]  # noqa: E712
            if not len(r):
                continue
            best = r.loc[r["val_rmse_mean"].idxmin()]
            for k in HP_KEYS:
                rand_vals[model][k].add(best[k])
    # LLM proposed values per model (genuine = not serving error)
    gl = long_df[long_df["rejection_category"] != "serving_error"]
    rows = []
    for model in sorted(set(MODEL_FOLDERS.values())):
        llm_vals = {k: set(gl[gl["model"] == model][k].dropna().tolist()) for k in HP_KEYS}
        per_hp = {}
        covered = total = 0
        for k in HP_KEYS:
            win = rand_vals[model][k]
            hit = sum(1 for v in win if any(_eq(v, lv) for lv in llm_vals[k]))
            per_hp[k] = (hit, len(win))
            covered += hit; total += len(win)
        rows.append({"model": model,
                     "winning_values_covered": covered, "winning_values_total": total,
                     "coverage_frac": covered / total if total else np.nan,
                     "learning_rate": f"{per_hp['learning_rate'][0]}/{per_hp['learning_rate'][1]}",
                     "window_size": f"{per_hp['window_size'][0]}/{per_hp['window_size'][1]}",
                     "lstm_hidden": f"{per_hp['lstm_hidden'][0]}/{per_hp['lstm_hidden'][1]}"})
    return pd.DataFrame(rows)


def _eq(a, b):
    try:
        return abs(float(a) - float(b)) < 1e-9
    except (TypeError, ValueError):
        return a == b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\hfava\Downloads")
    ap.add_argument("--analysis", default="llm_behavior_analysis")
    args = ap.parse_args()

    long_df = pd.read_csv(os.path.join(args.analysis, "per_attempt_long.csv"))
    run_df = pd.read_csv(os.path.join(args.analysis, "per_run_metrics.csv"))
    run_df = run_df[~run_df["dead_run"]]

    pd.set_option("display.width", 170)

    print("\n=== Q3: understanding allowed values / repeats (per model, mean per run) ===")
    a = run_df.groupby("model").agg(
        genuine=("n_genuine_proposals", "mean"),
        off_grid=("n_off_grid", "mean"),
        already_tried=("n_already_tried", "mean"),
        distinct=("n_distinct_settings", "mean"),
        repeats=("n_repeat_proposals", "mean"),
    ).round(2)
    print(a.to_string())

    print("\n=== Q3: reasonable changes after a bad result ===")
    cc = course_correction(long_df).round(3)
    print(cc.to_string(index=False))

    print("\n=== Q3: does it avoid values random search found useful? ===")
    av = avoid_random_useful(args.root, long_df).round(3)
    print(av.to_string(index=False))

    print("\n=== Q3: prompt-clarity-vs-capability (same prompt, all models) ===")
    cap = run_df.groupby("model").agg(
        decision_reject=("decision_rejected_frac", "mean"),
        last_improve=("last_improve_attempt", "mean"),
        final_best=("final_best_score", "mean"),
    ).round(3)
    print(cap.to_string())


if __name__ == "__main__":
    main()

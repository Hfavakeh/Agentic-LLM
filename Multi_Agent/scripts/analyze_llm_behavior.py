"""LLM-optimizer behavior analysis (professor's 2026-06 email).

Ingests the multi-seed result folders produced by the protocol runner and
quantifies *how the LLM behaves as an optimizer* — not just the final tables.
It answers the email's questions with data:

  - Does the LLM keep improving over the 25 attempts, or stall? When/why?
  - Does it repeat proposals?
  - Does it avoid useful regions of the search space?
  - Is it too conservative vs random search?
  - Does it benefit from / waste its attempt budget?
  - Stronger LLM => better decisions, or only cleaner output?
  - Which proposals improved vs degraded the result? What distinguishes them?

Each LLM run = one conversation_log_run1.json (25 attempts, anchored on
best-so-far). We also read the per_setting_report CSV (all arms in one table)
and final_evaluation_run1.json (shipped test RMSE per arm) for cross-arm
comparison.

Usage:
    python analyze_llm_behavior.py --root "C:/Users/hfava/Downloads" \
        --out llm_behavior_analysis
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

# --- search space (mirror of pipeline/search_space.HP_GRID) ------------------
# window_size gets the per-dataset baseline window injected at runtime
# (Config.__post_init__). For radar the baseline window is 12, so the effective
# reachable window grid is [10, 12, 20, 30, 40, 50]. We auto-augment below with
# whatever baseline window the data actually used.
HP_GRID = {
    "learning_rate":    [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
    "weight_decay":     [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
    "dropout":          [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    "batch_size":       [32, 64, 128],
    "lstm_hidden":      [64, 128, 256],
    "lstm_layers":      [1, 2, 3],
    "window_size":      [10, 20, 30, 40, 50],
    "optimizer_choice": ["adam", "adamw"],
    "patience":         [8, 12, 16],
}
HP_KEYS = list(HP_GRID.keys())

# The 8 report model folders -> clean display names. Two physical runs per
# model family (…-new and …-new1) are treated as repeats of the same LLM.
MODEL_FOLDERS = {
    "llama38b-new":    "Llama 3 8B",
    "llama38b-new1":   "Llama 3 8B",
    "phi4-new":        "Phi 4 (14B)",
    "phi414-new1":     "Phi 4 (14B)",
    "nemotron3-new":   "Nemotron 3",
    "gemma4-12b-new":  "Gemma 4 12B",
    "gemma4-12b-new1": "Gemma 4 12B",
    "qwen3.5-new":     "Qwen 3 8B",
}


def _signature(setting: dict) -> tuple:
    """Hashable identity of a setting over the 9 searched HPs."""
    return tuple((k, setting.get(k)) for k in HP_KEYS)


def reconstruct_proposed(a: dict) -> dict:
    """The setting the LLM actually PROPOSED this attempt.

    For trained attempts ``setting`` already is the applied proposal. For
    REJECTED attempts ``setting`` is the (unchanged) anchor and the rejected
    delta lives in ``changes_from_anchor`` — so the true proposal is
    anchor + delta. This is what we must use for repeat/coverage analysis.
    """
    proposed = dict(a.get("setting", {}))
    proposed.update(a.get("changes_from_anchor") or {})
    return proposed


def rejection_category(a: dict) -> str:
    """Normalise the rejection reason into a small vocabulary.

    Two fundamentally different things get rejected:
      * DECISION failures — the model produced a parseable proposal that the
        validator refused: `already_tried` (a repeat) or `off_grid_value`
        (an out-of-grid value) or `arch_frozen`. These ARE optimizer behavior.
      * SERVING failures — the Ollama call itself failed: `client_error`
        (e.g. model tag not pulled -> 404) or `llm_call_timeout` (model too
        slow for the 300s budget). These are infrastructure, NOT optimizer
        behavior, and must be separated out / excluded.
    """
    if a.get("trained"):
        return ""
    reason = (a.get("reason") or "").lower()
    if "client_error" in reason or "timeout" in reason or "connection" in reason:
        return "serving_error"
    if "already_tried" in reason or "already tried" in reason:
        return "already_tried"
    if "not_in_grid" in reason or "not in grid" in reason:
        return "off_grid_value"
    if "arch" in reason or "frozen" in reason:
        return "arch_frozen"
    if "unknown" in reason or "key" in reason:
        return "unknown_key"
    return "other_invalid"


def is_genuine_proposal(a: dict) -> bool:
    """True if the model actually produced a proposal we can analyse.

    Trained attempts and decision-rejected attempts count; serving failures
    (no proposal ever returned) do not.
    """
    return a.get("trained") or rejection_category(a) != "serving_error"


def _grid_index(key, value):
    """Index of value in its grid (for step-distance); None if off-grid."""
    grid = HP_GRID[key]
    if value in grid:
        return grid.index(value)
    # numeric snap to nearest for distance purposes only
    try:
        diffs = [abs(float(value) - float(g)) for g in grid]
        return int(np.argmin(diffs))
    except (TypeError, ValueError):
        return None


def load_runs(root: str):
    """Yield (model_display, folder, seed, attempts_list, baseline_window)."""
    for folder, display in MODEL_FOLDERS.items():
        for conv_path in sorted(glob.glob(os.path.join(root, folder, "seed_*",
                                                        "conversation_log_run1.json"))):
            seed_dir = os.path.dirname(conv_path)
            seed = os.path.basename(seed_dir).replace("seed_", "")
            with open(conv_path, encoding="utf-8") as fh:
                attempts = json.load(fh)
            baseline_window = None
            csv_path = os.path.join(seed_dir, "per_setting_report_run1.csv")
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    ref = df[df["arm"] == "fixed_reference"]
                    if len(ref):
                        baseline_window = int(ref.iloc[0]["window_size"])
                except Exception:
                    pass
            yield display, folder, seed, attempts, baseline_window


def per_run_metrics(display, folder, seed, attempts):
    """Compute behavior metrics for ONE LLM run (25 attempts)."""
    n = len(attempts)
    scores = [a.get("score") for a in attempts]
    trained = [bool(a.get("trained")) for a in attempts]
    status = [a.get("output_status") for a in attempts]

    # best-so-far trajectory (only trained attempts move it)
    best = np.inf
    best_so_far = []
    improved_flags = []          # did THIS attempt strictly improve best-so-far?
    last_improve_attempt = 0
    n_improvements = 0
    for i, a in enumerate(attempts):
        s = a.get("score")
        improved = bool(trained[i] and s is not None and s < best - 1e-9)
        improved_flags.append(improved)
        if improved:
            best = s
            last_improve_attempt = i + 1
            n_improvements += 1
        best_so_far.append(best if np.isfinite(best) else None)

    first_score = next((a.get("score") for a in attempts
                        if a.get("trained") and a.get("score") is not None), None)
    final_best = best if np.isfinite(best) else None

    # separate serving failures (no proposal) from genuine proposals
    rej = Counter(rejection_category(a) for a in attempts if not a.get("trained"))
    n_serving = rej.get("serving_error", 0)
    genuine = [a for a in attempts if is_genuine_proposal(a)]
    n_genuine = len(genuine)

    # repetition: distinct settings + exact repeats among the RECONSTRUCTED
    # GENUINE proposals (rejected attempts store the anchor in `setting`, so we
    # rebuild anchor+delta; serving failures are dropped — they aren't proposals)
    sigs = [_signature(reconstruct_proposed(a)) for a in genuine]
    sig_counts = Counter(sigs)
    n_distinct = len(sig_counts)
    n_repeat_proposals = sum(c - 1 for c in sig_counts.values())

    # rejected attempts broken down by WHY (the budget-wasting taxonomy)
    n_rejected = sum(1 for t in trained if not t)
    n_clean = sum(1 for s in status if s == "clean")
    n_after_retry = sum(1 for s in status if s == "accepted_after_retry")
    n_already_tried = rej.get("already_tried", 0)
    n_off_grid = rej.get("off_grid_value", 0)
    n_arch_frozen = rej.get("arch_frozen", 0)
    n_other_invalid = rej.get("other_invalid", 0) + rej.get("unknown_key", 0)
    # decision-level rejection rate: of the GENUINE proposals, how many were
    # refused (repeat / off-grid / arch). Excludes serving noise.
    n_decision_rejected = n_already_tried + n_off_grid + n_arch_frozen + n_other_invalid
    decision_rejected_frac = (n_decision_rejected / n_genuine) if n_genuine else None

    # conservatism: how many HPs changed vs anchor, and grid-step distance
    n_hp_changed = []
    step_dist = []
    for a in attempts:
        ch = a.get("changes_from_anchor") or {}
        n_hp_changed.append(len(ch))
        d = 0.0
        for k, v in ch.items():
            if k in HP_GRID:
                gi_new = _grid_index(k, v)
                # anchor value not directly stored; approximate distance by
                # treating any change as >=1 grid step (categorical) — refined
                # later with full-setting distance in the long table.
                d += 1 if gi_new is not None else 1
        step_dist.append(len(ch))

    # history benefit proxy: improvements in first third vs last two thirds
    third = max(1, n // 3)
    improvements_early = sum(improved_flags[:third])
    improvements_late = sum(improved_flags[third:])

    return {
        "model": display,
        "folder": folder,
        "seed": seed,
        "n_attempts": n,
        "n_trained": sum(trained),
        "n_serving_error": n_serving,
        "n_genuine_proposals": n_genuine,
        "dead_run": n_serving >= 0.5 * n,   # serving failure dominated the run
        "n_rejected": n_rejected,
        "rejected_frac": n_rejected / n if n else None,
        "decision_rejected_frac": decision_rejected_frac,
        "n_clean": n_clean,
        "n_after_retry": n_after_retry,
        "n_distinct_settings": n_distinct,
        "n_repeat_proposals": n_repeat_proposals,
        "n_already_tried": n_already_tried,
        "n_off_grid": n_off_grid,
        "n_arch_frozen": n_arch_frozen,
        "n_other_invalid": n_other_invalid,
        "n_improvements": n_improvements,
        "last_improve_attempt": last_improve_attempt,
        "stalled_after": last_improve_attempt,   # alias
        "first_trained_score": first_score,
        "final_best_score": final_best,
        "total_gain": (first_score - final_best) if (first_score and final_best) else None,
        "mean_hp_changed_per_step": float(np.mean(n_hp_changed)) if n_hp_changed else None,
        "improvements_early_third": improvements_early,
        "improvements_late_twothirds": improvements_late,
    }


def long_attempt_table(display, folder, seed, attempts):
    """Tidy per-attempt rows across all runs (for coverage + improve/degrade)."""
    rows = []
    best = np.inf
    for i, a in enumerate(attempts):
        s = a.get("score")
        trained = bool(a.get("trained"))
        improved = bool(trained and s is not None and s < best - 1e-9)
        anchor_before = best if np.isfinite(best) else None
        # delta vs anchor (positive = worse than best-so-far)
        delta = (s - anchor_before) if (trained and s is not None and anchor_before) else None
        if improved:
            best = s
        proposed = reconstruct_proposed(a)   # true proposal (anchor+delta)
        ch = a.get("changes_from_anchor") or {}
        row = {
            "model": display, "folder": folder, "seed": seed,
            "attempt": a.get("attempt", i + 1),
            "trained": trained,
            "output_status": a.get("output_status"),
            "rejection_category": rejection_category(a),
            "diagnosis": a.get("diagnosis"),
            "strategy": a.get("strategy"),
            "confidence": a.get("confidence"),
            "score": s,
            "best_so_far": best if np.isfinite(best) else None,
            "improved": improved,
            "delta_vs_anchor": delta,
            "n_hp_changed": len(ch),
            "changed_keys": "|".join(sorted(ch.keys())) if ch else "",
        }
        for k in HP_KEYS:
            row[k] = proposed.get(k)
        rows.append(row)
    return rows


def coverage_table(long_df: pd.DataFrame, baseline_window: int | None):
    """Per-model fraction of each HP's grid values that were ever PROPOSED."""
    grid = {k: list(v) for k, v in HP_GRID.items()}
    if baseline_window and baseline_window not in grid["window_size"]:
        grid["window_size"] = sorted(grid["window_size"] + [baseline_window])
    recs = []
    for model, g in long_df.groupby("model"):
        for k in HP_KEYS:
            vals = set(g[k].dropna().tolist())
            # match against grid membership (tolerant float compare)
            covered = 0
            for gv in grid[k]:
                hit = any((v == gv) or
                          (isinstance(v, (int, float)) and isinstance(gv, (int, float))
                           and abs(float(v) - float(gv)) < 1e-9) for v in vals)
                covered += int(hit)
            recs.append({
                "model": model, "hp": k,
                "grid_size": len(grid[k]),
                "values_used": covered,
                "coverage_frac": covered / len(grid[k]),
            })
    return pd.DataFrame(recs)


def arm_comparison(root: str, dead_folders: set):
    """Cross-arm trajectories from per_setting_report CSVs (trained rows only).

    Answers 'too conservative vs random?': random/optuna/rule_based spend all
    25 budget slots on real trainings; the LLM trains far fewer (the rest are
    rejected). Returns a tidy per-(model,seed,arm) frame plus the mean
    best-so-far-vs-attempt curve per (model,arm).
    """
    rows, curves = [], {}
    for folder, display in MODEL_FOLDERS.items():
        if folder in dead_folders:
            continue
        for csv_path in sorted(glob.glob(os.path.join(root, folder, "seed_*",
                                                       "per_setting_report_run1.csv"))):
            seed = os.path.basename(os.path.dirname(csv_path)).replace("seed_", "")
            try:
                df = pd.read_csv(csv_path)
            except Exception:
                continue
            for arm in ("llm", "random", "optuna", "rule_based"):
                a = df[(df["arm"] == arm) & (df["trained"] == True)].copy()  # noqa: E712
                if not len(a):
                    continue
                a = a.sort_values("attempt")
                vals = a["val_rmse_mean"].to_numpy(dtype=float)
                # best-so-far over this arm's TRAINED settings
                bsf = np.minimum.accumulate(vals)
                # align to a 1..25 axis by attempt index (forward-fill)
                axis = np.full(25, np.nan)
                for att, b in zip(a["attempt"].to_numpy(), bsf):
                    if 1 <= int(att) <= 25:
                        axis[int(att) - 1] = b
                # forward-fill
                last = np.nan
                for i in range(25):
                    if np.isnan(axis[i]):
                        axis[i] = last
                    else:
                        last = axis[i]
                sigs = {_signature({k: r[k] for k in HP_KEYS}) for _, r in a.iterrows()}
                last_improve = int(a["attempt"].to_numpy()[int(np.argmin(vals))])
                rows.append({
                    "model": display, "arm": arm, "seed": seed,
                    "n_trained": len(a),
                    "distinct": len(sigs),
                    "final_best": float(bsf[-1]),
                    "last_improve": last_improve,
                })
                curves.setdefault((display, arm), []).append(axis)
    arm_df = pd.DataFrame(rows)
    mean_curves = {k: np.nanmean(np.vstack(v), axis=0) for k, v in curves.items()}
    return arm_df, mean_curves


def make_plots(run_df, long_df, cov_df, arm_df, mean_curves, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ---- Plot 1: best-so-far vs attempt, LLM models vs random/optuna -------
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(1, 26)
    llm_models = sorted(run_df["model"].unique())
    cmap = plt.get_cmap("tab10")
    for i, m in enumerate(llm_models):
        if (m, "llm") in mean_curves:
            ax.plot(x, mean_curves[(m, "llm")], color=cmap(i), lw=2, label=f"LLM · {m}")
    # one representative random/optuna curve (averaged over all models == same
    # random/optuna pipeline; average them all for a clean reference line)
    for arm, style in (("random", "k--"), ("optuna", "k:")):
        cs = [v for (mm, aa), v in mean_curves.items() if aa == arm]
        if cs:
            ax.plot(x, np.nanmean(np.vstack(cs), axis=0), style, lw=2.2, label=arm)
    ax.set_xlabel("optimization attempt (1–25)")
    ax.set_ylabel("best-so-far validation RMSE (m)")
    ax.set_title("Search progress: LLM arms vs random / Optuna (mean over seeds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig1_best_so_far.png"), dpi=130)
    plt.close(fig)

    # ---- Plot 2: distinct trained settings per arm ------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    piv = arm_df.groupby(["model", "arm"])["distinct"].mean().unstack("arm")
    piv = piv[[c for c in ["llm", "random", "optuna", "rule_based"] if c in piv.columns]]
    piv.plot(kind="bar", ax=ax)
    ax.axhline(25, color="grey", ls=":", lw=1)
    ax.set_ylabel("distinct settings actually trained (of 25)")
    ax.set_title("Budget use: distinct trained settings per arm")
    ax.legend(fontsize=8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_distinct_settings.png"), dpi=130)
    plt.close(fig)

    # ---- Plot 3: coverage heatmap (LLM models) ----------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    cpiv = cov_df.pivot(index="hp", values="coverage_frac", columns="model")
    im = ax.imshow(cpiv.to_numpy(), cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cpiv.columns)))
    ax.set_xticklabels(cpiv.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(cpiv.index)))
    ax.set_yticklabels(cpiv.index)
    for i in range(cpiv.shape[0]):
        for j in range(cpiv.shape[1]):
            v = cpiv.to_numpy()[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v*100:.0f}%", ha="center", va="center", fontsize=7)
    ax.set_title("Search-space coverage by HP (% of grid values ever proposed)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_coverage_heatmap.png"), dpi=130)
    plt.close(fig)

    # ---- Plot 4: improvement rate by changed-key --------------------------
    t = long_df[long_df["trained"]].copy()
    recs = []
    for keys, g in t.groupby("changed_keys"):
        if g.shape[0] >= 10 and keys:
            recs.append((keys, g.shape[0], g["improved"].mean()))
    recs.sort(key=lambda r: r[2])
    if recs:
        fig, ax = plt.subplots(figsize=(8, 5.5))
        labels = [f"{k}  (n={n})" for k, n, _ in recs]
        ax.barh(labels, [r[2] * 100 for r in recs], color="steelblue")
        ax.set_xlabel("% of trained attempts that improved best-so-far")
        ax.set_title("Which LLM changes help? (improvement rate by changed HP set)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "fig4_improvement_by_change.png"), dpi=130)
        plt.close(fig)

    print(f"Wrote 4 figures -> {out}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\Users\hfava\Downloads",
                    help="folder containing the model result dirs")
    ap.add_argument("--out", default="llm_behavior_analysis")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    run_rows, long_rows = [], []
    baseline_windows = []
    for display, folder, seed, attempts, bw in load_runs(args.root):
        run_rows.append(per_run_metrics(display, folder, seed, attempts))
        long_rows.extend(long_attempt_table(display, folder, seed, attempts))
        if bw:
            baseline_windows.append(bw)

    if not run_rows:
        raise SystemExit(f"No runs found under {args.root}. Check --root.")

    run_df = pd.DataFrame(run_rows)
    long_df = pd.DataFrame(long_rows)
    baseline_window = Counter(baseline_windows).most_common(1)[0][0] if baseline_windows else None

    run_df.to_csv(os.path.join(args.out, "per_run_metrics.csv"), index=False)
    long_df.to_csv(os.path.join(args.out, "per_attempt_long.csv"), index=False)

    # ---- console summary --------------------------------------------------
    pd.set_option("display.width", 175)
    pd.set_option("display.max_columns", 30)
    print(f"\nLoaded {len(run_df)} LLM runs across {run_df['model'].nunique()} models; "
          f"baseline_window={baseline_window}")

    print("\n=== Serving health (infrastructure, NOT optimizer behavior) ===")
    health = run_df.groupby("folder").agg(
        model=("model", "first"),
        runs=("seed", "count"),
        dead_runs=("dead_run", "sum"),
        mean_serving_err=("n_serving_error", "mean"),
        mean_genuine=("n_genuine_proposals", "mean"),
    ).round(2)
    print(health.to_string())

    # Behavior analysis uses GENUINE runs only (drop serving-dominated runs so
    # the 404/timeout folders don't masquerade as bad optimization).
    live = run_df[~run_df["dead_run"]].copy()
    dead_folders = sorted(run_df[run_df["dead_run"]]["folder"].unique())
    if dead_folders:
        print(f"\nExcluded {int(run_df['dead_run'].sum())} dead runs "
              f"(serving-failure dominated): {dead_folders}")
    live_keys = set(zip(live["folder"], live["seed"]))
    long_df = long_df[[k in live_keys for k in zip(long_df["folder"], long_df["seed"])]].copy()
    run_df = live

    genuine_long = long_df[long_df["rejection_category"] != "serving_error"].copy()
    cov_df = coverage_table(genuine_long, baseline_window)
    cov_df.to_csv(os.path.join(args.out, "coverage_by_model.csv"), index=False)

    print("\n=== Q: budget usage & stall (GENUINE runs only, mean over seeds) ===")
    agg = run_df.groupby("model").agg(
        runs=("seed", "count"),
        genuine=("n_genuine_proposals", "mean"),
        dec_rej_frac=("decision_rejected_frac", "mean"),
        distinct=("n_distinct_settings", "mean"),
        repeats=("n_repeat_proposals", "mean"),
        improvements=("n_improvements", "mean"),
        last_improve=("last_improve_attempt", "mean"),
        total_gain=("total_gain", "mean"),
        final_best=("final_best_score", "mean"),
        hp_per_step=("mean_hp_changed_per_step", "mean"),
        imp_early=("improvements_early_third", "mean"),
        imp_late=("improvements_late_twothirds", "mean"),
    ).round(3)
    print(agg.to_string())

    print("\n=== Q: rejection taxonomy (mean count per run; decision failures only) ===")
    rej = run_df.groupby("model").agg(
        already_tried=("n_already_tried", "mean"),
        off_grid=("n_off_grid", "mean"),
        arch_frozen=("n_arch_frozen", "mean"),
        other_invalid=("n_other_invalid", "mean"),
        serving_err=("n_serving_error", "mean"),
    ).round(2)
    print(rej.to_string())

    print("\n=== Q: search-space coverage (mean coverage_frac per HP, all models) ===")
    cov_piv = cov_df.pivot_table(index="hp", values="coverage_frac", aggfunc="mean").round(2)
    print(cov_piv.to_string())

    print("\n=== Q: improving vs degrading proposals (trained attempts only) ===")
    t = long_df[long_df["trained"]].copy()
    print(f"trained attempts: {len(t)}; improved best-so-far: {t['improved'].sum()} "
          f"({100*t['improved'].mean():.1f}%)")
    # which HP changes most associated with improvement
    print("\n  improvement rate by changed-key (trained attempts that changed exactly that set):")
    by_key = []
    for keys, g in t.groupby("changed_keys"):
        if g.shape[0] >= 10:
            by_key.append((keys or "(none)", g.shape[0], round(100*g["improved"].mean(), 1)))
    by_key.sort(key=lambda r: -r[2])
    for k, nn, rate in by_key[:15]:
        print(f"    {rate:5.1f}%  n={nn:4d}  {k}")

    print("\n=== Q: improvement rate by diagnosis label ===")
    dg = t.groupby("diagnosis").agg(n=("improved", "size"),
                                    improve_rate=("improved", "mean")).round(3)
    print(dg.to_string())

    # ---- cross-arm comparison (conservatism vs random) -------------------
    arm_df, mean_curves = arm_comparison(args.root, set(dead_folders))
    arm_df.to_csv(os.path.join(args.out, "arm_comparison.csv"), index=False)
    print("\n=== Q: too conservative vs random? (trained settings & final best per arm) ===")
    arm_agg = arm_df.groupby(["model", "arm"]).agg(
        n_trained=("n_trained", "mean"),
        distinct=("distinct", "mean"),
        last_improve=("last_improve", "mean"),
        final_best=("final_best", "mean"),
    ).round(3)
    print(arm_agg.to_string())

    try:
        make_plots(run_df, long_df, cov_df, arm_df, mean_curves, args.out)
    except Exception as e:  # plotting is best-effort
        print(f"[plot warning] {e}")

    print(f"\nWrote: per_run_metrics.csv, per_attempt_long.csv, coverage_by_model.csv, "
          f"arm_comparison.csv (+figures) -> {args.out}/")


if __name__ == "__main__":
    main()

"""Aggregate the motion loss-shaping experiment (`--motion-experiment`) across
seeds into the Email-6 report tables + figures.

Reads every ``<root>/seed_*/motion_experiment_run1.json`` (and the matching
``motion_protocol_log_run1.json`` when present) produced by
``arms.motion.run_motion_experiment`` / ``save_motion_results`` and emits, into
``--out``:

  * ``motion_experiment_summary.md``   — per-arm test RMSE + per-regime error +
    chosen levers + the C3 LLM's best strategy/reason (the professor's tables).
  * ``motion_experiment_summary.json`` — the same numbers, machine-readable, for
    the .docx builder.
  * ``fig_test_rmse.png``              — per-arm test RMSE (mean ± std) bars.
  * ``fig_regime_error.png``           — per-regime (slow/med/fast) error, baseline
    vs each arm's chosen levers.

Usage:
    python analyze_motion_experiment.py --root outputs-motion-<model> --out analysis/motion_<model>

The script is model-agnostic: point it at whatever the server run produced. It
tolerates a single seed (so it also runs on the local smoke output).
"""

import argparse
import json
import statistics as stats
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

ARM_LABELS = {
    "baseline":    "baseline (plain MSE)",
    "motion_rule": "C2 motion heuristic (rule)",
    "random":      "random levers",
    "llm":         "C3 LLM motion",
}
ARM_ORDER = ["baseline", "motion_rule", "random", "llm"]
LEVER_ORDER = ["v_max", "lambda_vel", "lambda_smooth",
               "bin_weight_slow", "bin_weight_medium", "bin_weight_fast"]
REGIMES = ["slow", "medium", "fast"]


def _mean_std(xs: List[float]):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return float("nan"), 0.0
    if len(xs) == 1:
        return xs[0], 0.0
    return stats.fmean(xs), stats.pstdev(xs)


def _levers_str(lv: Optional[Dict[str, Any]]) -> str:
    if not lv:
        return "plain MSE (neutral)"
    return ", ".join(f"{k}={lv[k]}" for k in LEVER_ORDER if k in lv)


def _best_attempt_reason(arm: Dict[str, Any]) -> Dict[str, str]:
    """Pull the strategy/reason/diagnosis of the LLM attempt that produced the
    arm's chosen (best) lever vector, from the arm's own attempt history."""
    best = arm.get("best_levers")
    if not best:
        return {}
    for att in arm.get("attempts", []):
        if att.get("setting") == best and att.get("strategy"):
            return {"strategy": att.get("strategy", ""),
                    "reason": att.get("reason", ""),
                    "diagnosis": att.get("diagnosis", "")}
    return {}


def load_seeds(root: Path) -> List[Dict[str, Any]]:
    seeds = []
    for jp in sorted(root.glob("seed_*/motion_experiment_run1.json")):
        try:
            data = json.loads(jp.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"  ! skip {jp}: {exc}")
            continue
        data["_seed_dir"] = jp.parent.name
        seeds.append(data)
    # Fall back to a flat (single-seed) layout: root/motion_experiment_run1.json
    if not seeds:
        jp = root / "motion_experiment_run1.json"
        if jp.exists():
            data = json.loads(jp.read_text(encoding="utf-8"))
            data["_seed_dir"] = root.name
            seeds.append(data)
    return seeds


def aggregate(seeds: List[Dict[str, Any]]) -> Dict[str, Any]:
    arms_present = set()
    for s in seeds:
        arms_present.update(s.get("arms_run", []))
    arms_present = [a for a in ARM_ORDER if a in arms_present]

    out: Dict[str, Any] = {"n_seeds": len(seeds), "arms": {},
                           "motion_profile": seeds[0].get("motion_profile", {}) if seeds else {},
                           "base_setting": seeds[0].get("base_setting", {}) if seeds else {}}

    for arm in arms_present:
        test_rmses, regime = [], {r: [] for r in REGIMES}
        spread, worst_votes = [], Counter()
        lever_votes = Counter()
        reasons = []
        for s in seeds:
            fe = s.get("final_eval", {}).get(arm, {})
            if fe.get("mean_rmse") is not None:
                test_rmses.append(fe["mean_rmse"])
            a = s.get("arms", {}).get(arm, {})
            reg = a.get("best_regime_error", {}) or {}
            for r in REGIMES:
                if reg.get(r) is not None:
                    regime[r].append(reg[r])
            if reg.get("spread_ratio") is not None:
                spread.append(reg["spread_ratio"])
            if reg.get("worst_regime"):
                worst_votes[reg["worst_regime"]] += 1
            lv = a.get("best_levers")
            if lv:
                lever_votes[tuple(lv.get(k) for k in LEVER_ORDER)] += 1
            r = _best_attempt_reason(a)
            if r.get("reason"):
                reasons.append(r)

        mean_rmse, std_rmse = _mean_std(test_rmses)
        # Most-common chosen lever vector across seeds (the "consensus" proposal).
        modal_levers = None
        if lever_votes:
            top = lever_votes.most_common(1)[0][0]
            modal_levers = {k: v for k, v in zip(LEVER_ORDER, top)}
        out["arms"][arm] = {
            "label": ARM_LABELS[arm],
            "test_rmse_mean": mean_rmse,
            "test_rmse_std": std_rmse,
            "n": len(test_rmses),
            "regime_error": {r: _mean_std(regime[r])[0] for r in REGIMES},
            "spread_ratio": _mean_std(spread)[0],
            "worst_regime": worst_votes.most_common(1)[0][0] if worst_votes else "",
            "modal_levers": modal_levers,
            "example_reasons": reasons[:5],
        }
    return out


def write_markdown(agg: Dict[str, Any], out: Path) -> None:
    L = []
    L.append("# Motion loss-shaping experiment — aggregated\n")
    L.append(f"_{agg['n_seeds']} search seed(s). HPs frozen at baseline: "
             f"`{agg.get('base_setting', {})}`._\n")

    mp = agg.get("motion_profile", {})
    if mp:
        dwell = mp.get("dwell", {}) or {}
        L.append("## Motion profile of the tracked person (radar)\n")
        L.append(f"- speed: mean {mp.get('speed_mean_mps')}, p95 {mp.get('speed_p95_mps')}, "
                 f"max {mp.get('speed_max_mps')} m/s")
        L.append(f"- turning: mean {mp.get('turn_abs_mean_deg')}°/step, "
                 f"p95 {mp.get('turn_abs_p95_deg')}°, sharp-turn share {mp.get('sharp_turn_share')}")
        L.append(f"- stop-go: stop share {dwell.get('stop_share')}, "
                 f"{dwell.get('episodes_per_min')} dwell episodes/min\n")

    L.append("## Per-arm test RMSE (metres, lower is better)\n")
    L.append("| Arm | test RMSE | ± | n | chosen levers |")
    L.append("|---|--:|--:|--:|---|")
    for arm in ARM_ORDER:
        a = agg["arms"].get(arm)
        if not a:
            continue
        L.append(f"| {a['label']} | {a['test_rmse_mean']:.4f} | {a['test_rmse_std']:.4f} | "
                 f"{a['n']} | {_levers_str(a['modal_levers'])} |")
    L.append("")

    L.append("## Per-regime validation error (metres) at each arm's chosen levers\n")
    L.append("| Arm | slow | medium | fast | worst | spread (worst/best) |")
    L.append("|---|--:|--:|--:|:--:|--:|")
    for arm in ARM_ORDER:
        a = agg["arms"].get(arm)
        if not a:
            continue
        re = a["regime_error"]
        L.append(f"| {a['label']} | {re['slow']:.4f} | {re['medium']:.4f} | {re['fast']:.4f} | "
                 f"{a['worst_regime']} | {a['spread_ratio']:.3f} |")
    L.append("")

    llm = agg["arms"].get("llm")
    if llm and llm.get("example_reasons"):
        L.append("## What the C3 LLM proposed and why (sampled)\n")
        for r in llm["example_reasons"]:
            L.append(f"- **{r.get('strategy','')}** — {r.get('reason','')} "
                     f"(diagnosis: {r.get('diagnosis','')})")
        L.append("")

    # The headline comparison the professor asked for.
    c2 = agg["arms"].get("motion_rule")
    c3 = agg["arms"].get("llm")
    base = agg["arms"].get("baseline")
    if c2 and c3:
        L.append("## Did the LLM add anything beyond the rule? (C3 vs C2)\n")
        d = c3["test_rmse_mean"] - c2["test_rmse_mean"]
        verdict = ("LLM better" if d < -1e-4 else "rule better" if d > 1e-4 else "tie")
        L.append(f"- C3 LLM test RMSE {c3['test_rmse_mean']:.4f} vs C2 rule "
                 f"{c2['test_rmse_mean']:.4f} → **{verdict}** (Δ {d:+.4f} m)")
        if base:
            L.append(f"- baseline (plain MSE) test RMSE {base['test_rmse_mean']:.4f}; "
                     f"C2 Δ {c2['test_rmse_mean']-base['test_rmse_mean']:+.4f}, "
                     f"C3 Δ {c3['test_rmse_mean']-base['test_rmse_mean']:+.4f}")
        L.append("")

    out.write_text("\n".join(L), encoding="utf-8")
    print(f"  wrote {out}")


def make_figures(agg: Dict[str, Any], out_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        print(f"  ! matplotlib unavailable ({exc}); skipping figures")
        return
    arms = [a for a in ARM_ORDER if a in agg["arms"]]
    labels = [agg["arms"][a]["label"] for a in arms]

    # Fig 1: per-arm test RMSE
    fig, ax = plt.subplots(figsize=(7, 4))
    means = [agg["arms"][a]["test_rmse_mean"] for a in arms]
    errs = [agg["arms"][a]["test_rmse_std"] for a in arms]
    ax.bar(labels, means, yerr=errs, capsize=4, color="#4C78A8")
    ax.set_ylabel("test RMSE (m)")
    ax.set_title("Motion loss-shaping — final test RMSE per arm")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_test_rmse.png", dpi=140)
    plt.close(fig)

    # Fig 2: per-regime error per arm
    fig, ax = plt.subplots(figsize=(7, 4))
    import numpy as np
    x = np.arange(len(REGIMES))
    w = 0.8 / max(len(arms), 1)
    for i, a in enumerate(arms):
        re = agg["arms"][a]["regime_error"]
        ax.bar(x + i * w, [re[r] for r in REGIMES], w, label=agg["arms"][a]["label"])
    ax.set_xticks(x + w * (len(arms) - 1) / 2)
    ax.set_xticklabels([r.capitalize() for r in REGIMES])
    ax.set_ylabel("mean position error (m)")
    ax.set_title("Per-regime validation error by arm")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_regime_error.png", dpi=140)
    plt.close(fig)
    print(f"  wrote figures -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, nargs="+",
                    help="one or more dirs each containing seed_*/motion_experiment_run1.json "
                         "(or a flat single run). Multiple roots are merged by arm, so an "
                         "LLM-only run and a baseline/C2 run combine into one comparison table.")
    ap.add_argument("--out", default=None, help="output dir (default: analysis/motion_<first root name>)")
    args = ap.parse_args()

    roots = [Path(r) for r in args.root]
    out_dir = Path(args.out) if args.out else Path("analysis") / f"motion_{roots[0].name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = []
    for root in roots:
        s = load_seeds(root)
        print(f"Loaded {len(s)} seed run(s) from {root}")
        seeds.extend(s)
    if not seeds:
        raise SystemExit(f"No motion_experiment_run1.json found under {roots}")

    agg = aggregate(seeds)
    (out_dir / "motion_experiment_summary.json").write_text(
        json.dumps(agg, indent=2, default=str), encoding="utf-8")
    write_markdown(agg, out_dir / "motion_experiment_summary.md")
    make_figures(agg, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()

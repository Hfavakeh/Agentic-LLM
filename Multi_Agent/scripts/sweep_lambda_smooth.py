"""Controlled sweep of ONE motion loss-shaping lever (Email-9).

The motion arms search all six levers jointly, so what they report is an
*argmin* — the single best-of-25 vector — which says nothing about the shape of
the response to any individual lever. The professor asked the shape question
directly: what are the allowed values, which are selected, are lower values
consistently preferred, how does performance change AS the lever increases, does
the answer differ between radar and infrared, and under what motion conditions
does the smoothness prior help or hurt.

This script answers it by holding everything else fixed:

  * the 9 conventional HPs are frozen at the ``Config`` baseline (same frozen
    setting the motion experiment uses),
  * the five other levers stay NEUTRAL (``LOSS_SHAPING_NEUTRAL``), so with
    ``lambda_vel=0`` the ``v_max`` knob is inert and the bin weights are 1.0,
  * only the swept lever moves, across the full ``LOSS_SHAPING_GRID`` list.

The ``lambda_smooth=0`` cell is therefore EXACTLY plain MSE — the baseline
control sits inside the sweep rather than being imported from another run.

Two numbers per lever value:
  * search-side  — mean validation RMSE (metres) over the protocol's 3 training
    seeds, via ``evaluate_setting``. This is the objective every arm selects on,
    so it explains what the arms were chasing.
  * headline     — mean TEST RMSE (metres) over fresh final-eval seeds, via
    ``train_and_test_setting``. Same seed list for every lever value, so the
    comparison against the lever=0 cell is PAIRED per seed.

Plus the motion-conditioned breakdown the smoothness question actually needs:
per-sample test error binned by the ground-truth trajectory's speed, |heading
change| and |acceleration|. A prior that trades turns for straights shows up as
opposite-signed deltas in the straight and sharp-turn bins; a prior that is
simply too strong degrades every bin at once.

Run (radar, then IR):

    ../venv/Scripts/python.exe scripts/sweep_lambda_smooth.py \
        --data "preprocessed-RadarEXP1(in).csv" --seeds 10 --out results/lever-sweep/radar-smooth
    ../venv/Scripts/python.exe scripts/sweep_lambda_smooth.py \
        --data "preprocessed-IR-EXP2(in).csv" --seeds 10 --out results/lever-sweep/ir-smooth
"""

import sys as _sys, pathlib as _pathlib
_REPO_ROOT = _pathlib.Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from arms.engine import (
    TRAIN_SEEDS, _coerce_setting, _default_setting, build_dataset_and_loaders,
    evaluate_setting, train_and_test_setting,
)
from experiments.final_eval import FINAL_EVAL_SEEDS_POOL
from motion_descriptors import STATIONARY_THRESHOLD_MPS, per_regime_error
from pipeline import Config, LOSS_SHAPING_GRID, LOSS_SHAPING_NEUTRAL, is_finite_number, logger

# Bin edges for the motion-conditioned error breakdown. The turn thresholds
# mirror `motion_descriptors.summarize_dynamics`, whose `sharp_turn_share` calls
# a step sharp above 45 degrees; 15 degrees separates "essentially straight"
# from a gentle course correction. Acceleration is split by its own terciles
# because its scale is dataset-dependent (3/4/5 Hz), unlike a heading angle.
TURN_BINS = ((0.0, 15.0, "straight"), (15.0, 45.0, "gentle_turn"), (45.0, 180.0, "sharp_turn"))


# ---------------------------------------------------------------------------
# Motion-conditioned error breakdown
# ---------------------------------------------------------------------------

def _target_kinematics(tgt: np.ndarray, hz: float):
    """Per-sample speed (m/s), |heading change| (deg) and |acceleration| (m/s^2)
    of the GROUND-TRUTH trajectory — the "real turns and accelerations in data"
    the prior has to be judged against. Derived exactly as in
    ``motion_descriptors.extract_motion_features`` so the bins here and the
    motion profile in the prompt describe the same quantities.
    """
    n = len(tgt)
    if n < 3:
        nan = np.full(n, np.nan)
        return nan, nan, nan
    dx = np.concatenate([[np.nan], np.diff(tgt[:, 0])])
    dy = np.concatenate([[np.nan], np.diff(tgt[:, 1])])
    speed = np.hypot(dx, dy) * hz
    accel = np.abs(np.concatenate([[np.nan], np.diff(speed)]) * hz)

    heading = np.arctan2(dy, dx)
    dheading = np.concatenate([[np.nan], np.diff(heading)])
    dheading = (dheading + np.pi) % (2.0 * np.pi) - np.pi          # wrap to (-pi, pi]
    turn = np.abs(np.degrees(dheading))
    # A heading is meaningless while the walker is standing still — the
    # direction is then pure jitter. Mask those steps out rather than let them
    # pollute the sharp-turn bin (same guard, same threshold, as
    # extract_motion_features).
    moving = speed >= STATIONARY_THRESHOLD_MPS
    turn = np.where(moving & np.roll(moving, 1), turn, np.nan)
    return speed, turn, accel


def _binned_error(preds: np.ndarray, targets: np.ndarray, hz: float) -> Dict[str, Dict[str, float]]:
    """Mean 2D Euclidean test error (metres) per motion bin, for one trained seed.

    Speed and acceleration are cut at their own terciles (scale is dataset- and
    rate-dependent); turning uses the fixed degree thresholds in ``TURN_BINS``.

    The metric is Euclidean distance, matching ``per_regime_error`` so this
    breakdown lines up with the per-regime numbers already in the run logs. It
    is NOT the same scale as ``Evaluator.compute_metrics``' headline ``rmse``,
    which averages the squared error over the x and y coordinates jointly.
    """
    idx = np.arange(len(targets) - len(preds), len(targets))
    tgt = np.asarray(targets)[idx]
    euc = np.sqrt(((np.asarray(preds) - tgt) ** 2).sum(axis=1))
    speed, turn, accel = _target_kinematics(tgt, hz)

    out: Dict[str, Dict[str, float]] = {"speed": {}, "turn": {}, "accel": {}}
    if len(euc) < 6:
        return out

    def _tercile_bins(vals: np.ndarray, names) -> Dict[str, np.ndarray]:
        finite = vals[np.isfinite(vals)]
        if len(finite) < 3:
            return {}
        lo, hi = np.quantile(finite, [1 / 3, 2 / 3])
        return {
            names[0]: np.isfinite(vals) & (vals <= lo),
            names[1]: np.isfinite(vals) & (vals > lo) & (vals <= hi),
            names[2]: np.isfinite(vals) & (vals > hi),
        }

    masks = {
        "speed": _tercile_bins(speed, ("slow", "medium", "fast")),
        "accel": _tercile_bins(accel, ("steady", "moderate", "accelerating")),
        "turn": {name: np.isfinite(turn) & (turn >= lo) & (turn < hi)
                 for lo, hi, name in TURN_BINS},
    }
    for family, fam_masks in masks.items():
        for name, m in fam_masks.items():
            if m.any():
                out[family][name] = float(euc[m].mean())
                out[family][f"{name}_n"] = int(m.sum())
    return out


def _mean_bins(per_seed: List[Dict[str, Dict[str, float]]]) -> Dict[str, Dict[str, float]]:
    """Average the per-seed bin errors. Counts are averaged too (they vary only
    with the window trim, but reporting them keeps thin bins visible)."""
    agg: Dict[str, Dict[str, float]] = {}
    for family in ("speed", "turn", "accel"):
        keys = {k for s in per_seed for k in s.get(family, {})}
        fam: Dict[str, float] = {}
        for k in sorted(keys):
            vals = [s[family][k] for s in per_seed if is_finite_number(s.get(family, {}).get(k))]
            if vals:
                fam[k] = float(np.mean(vals))
        agg[family] = fam
    return agg


# ---------------------------------------------------------------------------
# Paired statistics vs the neutral (lever = 0) cell
# ---------------------------------------------------------------------------

def _paired_vs_reference(values: List[float], reference: List[float]) -> Dict[str, Any]:
    """Per-seed paired difference (lever - reference). Both lists are the same
    seeds in the same order, so pairing removes the seed-to-seed variance that
    swamps these effects in an unpaired comparison."""
    pairs = [(v, r) for v, r in zip(values, reference)
             if is_finite_number(v) and is_finite_number(r)]
    if not pairs:
        return {"n": 0}
    diffs = np.array([v - r for v, r in pairs], dtype=float)
    n = len(diffs)
    mean = float(diffs.mean())
    sd = float(diffs.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 and sd > 0 else 0.0
    return {
        "n": n,
        "mean_delta": mean,          # negative = the lever value BEAT plain MSE
        "std_delta": sd,
        "t_stat": (mean / se) if se > 0 else float("nan"),
        "win_rate": float((diffs < 0).mean()),   # share of seeds where the lever helped
    }


# ---------------------------------------------------------------------------
# The sweep
# ---------------------------------------------------------------------------

def run_sweep(
    cfg: Config,
    lever: str,
    values: List[float],
    test_seeds: List[int],
    train_seeds: tuple = TRAIN_SEEDS,
    skip_search: bool = False,
) -> Dict[str, Any]:
    dataset, _, _, _ = build_dataset_and_loaders(cfg)
    allow_arch = bool(getattr(cfg, "allow_arch_changes", True))
    base_setting = _coerce_setting(_default_setting(cfg, allow_arch), allow_arch)

    logger.info("=" * 70)
    logger.info("LEVER SWEEP  %s over %s", lever, values)
    logger.info("  csv=%s  hz=%s  window=%s", cfg.csv_path, cfg.hz, cfg.window_size)
    logger.info("  HPs frozen at baseline: %s", base_setting)
    logger.info("  other levers NEUTRAL: %s",
                {k: v for k, v in LOSS_SHAPING_NEUTRAL.items() if k != lever})
    logger.info("  search seeds=%s | test seeds=%s", list(train_seeds), test_seeds)
    logger.info("=" * 70)

    cells: List[Dict[str, Any]] = []
    for val in values:
        levers = dict(LOSS_SHAPING_NEUTRAL)
        levers[lever] = val
        # lever at its neutral value == plain MSE; pass None so the training path
        # is byte-identical to the baseline arm rather than merely equivalent.
        shaping = None if float(val) == float(LOSS_SHAPING_NEUTRAL[lever]) else levers

        search: Dict[str, Any] = {}
        if not skip_search:
            res = evaluate_setting(cfg, dataset, base_setting,
                                   train_seeds=train_seeds, loss_shaping=shaping)
            search = {
                "val_rmse_mean": res["score"],
                "val_rmse_std": res["val_rmse_std"],
                "mean_best_epoch": res["mean_best_epoch"],
                "per_seed": [p["val_rmse"] for p in res["per_seed"]],
                "regime_error": res.get("motion_error_summary", {}),
            }
            logger.info("%s=%-5s | search val RMSE = %.4f +/- %.4f",
                        lever, val, res["score"], res["val_rmse_std"])

        per_seed_rmse: List[float] = []
        per_seed_bins: List[Dict[str, Dict[str, float]]] = []
        per_seed_regime: List[Dict[str, Any]] = []
        for seed in test_seeds:
            out = train_and_test_setting(cfg, dataset, base_setting, seed=int(seed),
                                         loss_shaping=shaping)
            rmse = out["metrics"].get("rmse")
            per_seed_rmse.append(float(rmse) if is_finite_number(rmse) else float("nan"))
            per_seed_bins.append(_binned_error(out["preds"], out["targets"], hz=cfg.hz))
            per_seed_regime.append(per_regime_error(out["preds"], out["targets"], hz=cfg.hz))
            logger.info("%s=%-5s | seed %-4s test RMSE = %s",
                        lever, val, seed,
                        f"{rmse:.4f}" if is_finite_number(rmse) else "nan")

        finite = [r for r in per_seed_rmse if is_finite_number(r)]
        cell = {
            lever: val,
            "is_neutral": shaping is None,
            "levers": levers,
            "search": search,
            "test_rmse_mean": float(np.mean(finite)) if finite else float("nan"),
            "test_rmse_std": float(np.std(finite, ddof=1)) if len(finite) > 1 else 0.0,
            "test_rmse_per_seed": per_seed_rmse,
            "n_seeds": len(finite),
            "bins": _mean_bins(per_seed_bins),
            "regime_error_per_seed": per_seed_regime,
        }
        cells.append(cell)
        logger.info("%s=%-5s | TEST RMSE = %.4f +/- %.4f (n=%d)",
                    lever, val, cell["test_rmse_mean"], cell["test_rmse_std"], cell["n_seeds"])

    # Paired deltas against the neutral cell (plain MSE).
    ref_cell = next((c for c in cells if c["is_neutral"]), None)
    if ref_cell is not None:
        ref = ref_cell["test_rmse_per_seed"]
        ref_bins = ref_cell["bins"]
        for c in cells:
            c["paired_vs_neutral"] = _paired_vs_reference(c["test_rmse_per_seed"], ref)
            c["bin_delta_vs_neutral"] = {
                fam: {k: c["bins"][fam][k] - ref_bins.get(fam, {}).get(k, float("nan"))
                      for k in c["bins"].get(fam, {}) if not k.endswith("_n")}
                for fam in ("speed", "turn", "accel")
            }

    return {
        "lever": lever,
        "values": values,
        "csv_path": cfg.csv_path,
        "hz": cfg.hz,
        "window_size": cfg.window_size,
        "base_setting": base_setting,
        "neutral_levers": dict(LOSS_SHAPING_NEUTRAL),
        "train_seeds": list(train_seeds),
        "test_seeds": list(test_seeds),
        "motion_profile": dataset.get("motion_profile", {}),
        "cells": cells,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_reports(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lever = results["lever"]
    cells = results["cells"]

    (out_dir / "lever_sweep.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8")

    with open(out_dir / "lever_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([lever, "val_rmse_mean", "val_rmse_std", "test_rmse_mean", "test_rmse_std",
                    "n_seeds", "paired_delta_vs_mse", "paired_t", "win_rate"])
        for c in cells:
            p = c.get("paired_vs_neutral", {})
            w.writerow([c[lever],
                        c["search"].get("val_rmse_mean", ""), c["search"].get("val_rmse_std", ""),
                        c["test_rmse_mean"], c["test_rmse_std"], c["n_seeds"],
                        p.get("mean_delta", ""), p.get("t_stat", ""), p.get("win_rate", "")])

    with open(out_dir / "lever_sweep_bins.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([lever, "family", "bin", "mean_error_m", "delta_vs_mse_m", "n_samples"])
        for c in cells:
            for fam in ("speed", "turn", "accel"):
                for k, v in sorted(c["bins"].get(fam, {}).items()):
                    if k.endswith("_n"):
                        continue
                    w.writerow([c[lever], fam, k, v,
                                c.get("bin_delta_vs_neutral", {}).get(fam, {}).get(k, ""),
                                c["bins"][fam].get(f"{k}_n", "")])

    # Markdown summary — the table that answers the email directly.
    L = [f"# `{lever}` sweep — {Path(results['csv_path']).name}\n",
         f"9 HPs frozen at baseline: `{results['base_setting']}`\n",
         f"All other levers neutral: "
         f"`{ {k: v for k, v in results['neutral_levers'].items() if k != lever} }`\n",
         f"Search: {len(results['train_seeds'])} trainings/value (seeds {results['train_seeds']}). "
         f"Test: {len(results['test_seeds'])} fresh seeds, paired across values.\n",
         f"| {lever} | val RMSE (search) | test RMSE | paired Δ vs plain MSE | t | seeds better |",
         "|--:|--:|--:|--:|--:|--:|"]
    for c in cells:
        p = c.get("paired_vs_neutral", {})
        s = c["search"]
        val_s = (f"{s['val_rmse_mean']:.4f} ± {s['val_rmse_std']:.4f}"
                 if s.get("val_rmse_mean") is not None else "—")
        d = p.get("mean_delta")
        t = p.get("t_stat")
        wr = p.get("win_rate", 0.0)
        label = f"{c[lever]} (plain MSE)" if c["is_neutral"] else f"{c[lever]}"
        d_s = "—" if c["is_neutral"] or d is None else f"{d:+.4f}"
        t_s = "—" if c["is_neutral"] or not is_finite_number(t) else f"{t:+.2f}"
        w_s = "—" if c["is_neutral"] else f"{wr * 100:.0f}%"
        L.append(f"| {label} | {val_s} | "
                 f"{c['test_rmse_mean']:.4f} ± {c['test_rmse_std']:.4f} | "
                 f"{d_s} | {t_s} | {w_s} |")

    # NOTE the two metrics below are on different scales and must not be read
    # against the RMSE column above. The headline `rmse` from
    # Evaluator.compute_metrics is per-COORDINATE (squared error averaged over x
    # and y jointly); the bin numbers are mean 2D EUCLIDEAN distance, matching
    # motion_descriptors.per_regime_error so the breakdown is comparable with the
    # per-regime errors already in the run logs. Euclidean ~ sqrt(2) x RMSE.
    L.append("\n## Error by motion condition\n")
    L.append("Mean 2D **Euclidean** distance in metres (same convention as the "
             "per-regime errors in the run logs), with the paired Δ vs plain MSE "
             "in brackets — negative means the prior helped that bin. This is a "
             "different scale from the per-coordinate RMSE column above; compare "
             "within this section only.\n")
    for fam, title in (("turn", "Heading change per step"),
                       ("accel", "|Acceleration| tercile"),
                       ("speed", "Speed tercile")):
        bins = [k for k in cells[0]["bins"].get(fam, {}) if not k.endswith("_n")]
        if not bins:
            continue
        L.append(f"### {title}\n")
        L.append("| " + lever + " | " + " | ".join(bins) + " |")
        L.append("|--:" * (len(bins) + 1) + "|")
        for c in cells:
            row = []
            for b in bins:
                v = c["bins"][fam].get(b, float("nan"))
                d = c.get("bin_delta_vs_neutral", {}).get(fam, {}).get(b)
                row.append(f"{v:.4f}" if c["is_neutral"] or d is None else f"{v:.4f} ({d:+.4f})")
            L.append(f"| {c[lever]} | " + " | ".join(row) + " |")
        L.append("")

    mp = results.get("motion_profile", {})
    if mp:
        L.append("## Dataset motion profile\n")
        for k in ("speed_mean_mps", "speed_p95_mps", "accel_abs_mean_mps2", "accel_abs_p95_mps2",
                  "turn_abs_mean_deg", "turn_abs_p95_deg", "sharp_turn_share", "dwell_share"):
            if k in mp:
                L.append(f"- `{k}` = {mp[k]}")
    (out_dir / "lever_sweep.md").write_text("\n".join(L), encoding="utf-8")
    logger.info("Sweep reports -> %s", out_dir)


def plot_sweep(results: Dict[str, Any], out_dir: Path) -> None:
    """RMSE vs lever value (search + test, with error bars) and the per-turn-bin
    delta that shows the straights-vs-turns trade-off."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lever, cells = results["lever"], results["cells"]
    xs = [c[lever] for c in cells]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    if any(c["search"] for c in cells):
        ax.errorbar(xs, [c["search"].get("val_rmse_mean", np.nan) for c in cells],
                    yerr=[c["search"].get("val_rmse_std", 0) for c in cells],
                    marker="o", capsize=3, label="validation (search objective)")
    ax.errorbar(xs, [c["test_rmse_mean"] for c in cells],
                yerr=[c["test_rmse_std"] for c in cells],
                marker="s", capsize=3, label="test (headline)")
    ax.set_xlabel(lever)
    ax.set_ylabel("RMSE (m)")
    ax.set_title(f"{Path(results['csv_path']).name} — response to {lever}")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    bins = [k for k in cells[0]["bins"].get("turn", {}) if not k.endswith("_n")]
    for b in bins:
        ax.plot(xs, [c.get("bin_delta_vs_neutral", {}).get("turn", {}).get(b, np.nan) for c in cells],
                marker="o", label=b)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel(lever)
    ax.set_ylabel("Δ test error vs plain MSE (m)")
    ax.set_title("Effect by heading change (negative = prior helps)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(Path(out_dir) / "lever_sweep.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lever", default="lambda_smooth", choices=sorted(LOSS_SHAPING_GRID),
                   help="which loss-shaping lever to sweep (default: lambda_smooth)")
    p.add_argument("--values", nargs="+", type=float, default=None,
                   help="values to sweep (default: the lever's full LOSS_SHAPING_GRID list)")
    p.add_argument("--data", default=None, help="dataset CSV (default: Config default = radar)")
    p.add_argument("--seeds", type=int, default=10,
                   help="number of fresh test seeds from FINAL_EVAL_SEEDS_POOL (default 10)")
    p.add_argument("--test-seeds", nargs="+", type=int, default=None,
                   help="explicit test seed list (overrides --seeds)")
    p.add_argument("--epochs", type=int, default=None, help="max epochs per training")
    p.add_argument("--skip-search", action="store_true",
                   help="test seeds only; skip the 3-training validation column")
    p.add_argument("--no-plot", action="store_true", help="skip the PNG")
    p.add_argument("--out", default=None,
                   help="output dir (default: results/lever-sweep/<lever>-<dataset>)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)

    # Pass the dataset at CONSTRUCTION so __post_init__ derives hz / window_size
    # from DATASET_SPECS. Assigning csv_path afterwards would leave them at the
    # radar default — the bug that miswired every earlier --data run.
    cfg = Config(csv_path=args.data) if args.data else Config()
    if args.epochs:
        cfg.epochs_per_round = int(args.epochs)
    # Per-regime validation error is gated on this flag in evaluate_setting; the
    # sweep always wants it, and it costs one extra forward pass.
    cfg.payload_motion = True

    tag = Path(cfg.csv_path).stem.replace("preprocessed-", "").replace("(in)", "")
    out_dir = Path(args.out) if args.out else Path("results") / "lever-sweep" / f"{args.lever}-{tag}"
    cfg.output_dir = out_dir
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    values = args.values if args.values is not None else list(LOSS_SHAPING_GRID[args.lever])
    neutral = LOSS_SHAPING_NEUTRAL[args.lever]
    if neutral not in values:
        # Without the neutral cell there is no plain-MSE control to pair against.
        values = [neutral] + list(values)
    values = sorted(dict.fromkeys(float(v) for v in values))

    test_seeds = args.test_seeds if args.test_seeds else FINAL_EVAL_SEEDS_POOL[:args.seeds]

    results = run_sweep(cfg, args.lever, values, test_seeds, skip_search=args.skip_search)
    write_reports(results, out_dir)
    if not args.no_plot:
        try:
            plot_sweep(results, out_dir)
        except Exception as exc:                     # noqa: BLE001 - plots are optional
            logger.warning("Sweep plot failed: %s", exc)

    print("\n" + "=" * 70)
    print(f"{args.lever} SWEEP COMPLETE — {cfg.csv_path}")
    print(f"{args.lever:>14} | {'val RMSE':>9} | {'test RMSE':>9} | {'Δ vs MSE':>9} | better")
    for c in results["cells"]:
        p = c.get("paired_vs_neutral", {})
        s = c["search"].get("val_rmse_mean")
        d = p.get("mean_delta")
        s_s = f"{s:.4f}" if s is not None else "—"
        d_s = "—" if c["is_neutral"] or d is None else f"{d:+.4f}"
        w_s = "—" if c["is_neutral"] else f"{p.get('win_rate', 0.0) * 100:.0f}%"
        print(f"{c[args.lever]:>14} | {s_s:>9} | {c['test_rmse_mean']:>9.4f} | "
              f"{d_s:>9} | {w_s}")
    print(f"reports -> {out_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()

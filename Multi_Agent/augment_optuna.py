"""Augment existing multi-seed run directories with the Optuna arm.

Some multi-seed sweeps (e.g. the ``point3-*`` / ``ablation-*`` runs) were
produced without the Optuna arm. This script computes the Optuna arm *only*,
on the SAME dataset and protocol as the rest of the pipeline, and merges its
numbers into each target directory's per-seed ``cross_run_metrics.json`` and
the top-level ``multi_seed_summary.json`` -- leaving every other arm untouched.

Design / fidelity notes
-----------------------
* Optuna is run through the SAME engine as the live pipeline
  (``run_optuna_search`` -> ``evaluate_setting`` on train seeds 101/102/103,
  ``train_and_test_setting`` on seed 101 for the interim test). This matches how
  the existing per-seed numbers in these runs were produced (no fresh-seed final
  eval was used -- there are no ``final_evaluation_*`` files).
* Seeding mode ``vary`` (default) ties Optuna's TPE sampler seed to each of the
  20 run-seeds, so Optuna gets a genuine across-seed distribution comparable to
  the LLM / random / rule-based arms. Mode ``fixed`` keeps the protocol's fixed
  sampler seed (1000) -> Optuna becomes a constant column (std = 0).
* Because Optuna depends only on (dataset, sampler seed) and the fixed train/
  test seeds -- never on the model directory -- the per-seed result is identical
  across directories. We therefore compute each seed's Optuna result ONCE and
  cache it (``optuna_cache_radar.json``), then inject it into every target dir.
* The injected per-seed metrics mirror exactly what ``compute_cross_run_metrics``
  emits for an ``optuna`` slot; the aggregated block is recomputed with the same
  ``_stats`` logic as ``run_multi_seed_experiment`` (existing fields untouched).

Usage
-----
    ..\\venv\\Scripts\\python.exe augment_optuna.py \\
        --dirs outputs-point3-qwen38bb outputs-ablation-repair-off-llama3.18b \\
        --seeding vary

Add ``--no-arch`` if the target runs were launched with ``--no-arch-changes``
(so Optuna's search space matches the other arms in those runs). ``--dry-run``
computes and prints without writing. Original JSONs are backed up to ``*.bak``
on first write.
"""

import argparse
import copy
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from pipeline import Config, is_finite_number, logger, safe_trapz
from arms import (
    TRAIN_SEEDS, build_dataset_and_loaders, run_optuna_search,
    train_and_test_setting,
)

CSV = "preprocessed-RadarEXP1(in).csv"
CACHE_PATH = Path("optuna_cache_radar.json")
TEST_SEED = TRAIN_SEEDS[0]  # interim single-seed test, matching the other arms


# ---------------------------------------------------------------------------
# Phase 1 -- compute (and cache) the Optuna arm per sampler seed
# ---------------------------------------------------------------------------

def _compute_optuna_for_seed(sampler_seed: int, allow_arch: bool,
                             n_attempts: int) -> Dict[str, Any]:
    """Run the Optuna arm once for a given sampler seed and return the pieces
    needed to inject an ``optuna`` slot into the cross-run metrics."""
    cfg = Config(csv_path=CSV)
    cfg.allow_arch_changes = allow_arch
    cfg.enable_final_eval = False
    scratch = Path("_optuna_scratch") / f"seed_{sampler_seed}"
    scratch.mkdir(parents=True, exist_ok=True)
    cfg.output_dir = scratch

    dataset, _, _, _ = build_dataset_and_loaders(cfg)
    search = run_optuna_search(cfg, dataset, n_attempts=n_attempts,
                               sampler_seed=sampler_seed)
    best_setting = (search["best"] or {}).get("setting", {})
    te = train_and_test_setting(cfg, dataset, best_setting, seed=TEST_SEED)
    metrics = te["metrics"]

    # Per-attempt mean validation loss (legacy scaled-space) for trials-to-threshold.
    attempts_val_loss = [
        float(a.get("val_loss")) for a in search["attempts"]
        if is_finite_number(a.get("val_loss"))
    ]
    # Best-setting final-training val-loss curve for the AUC metric.
    test_history_val_loss = [
        float(v) for v in te["history"].get("val_loss", []) if is_finite_number(v)
    ]
    return {
        "sampler_seed": sampler_seed,
        "best_setting": best_setting,
        "rmse": float(metrics["rmse"]),
        "r2": float(metrics["r2"]),
        "attempts_val_loss": attempts_val_loss,
        "test_history_val_loss": test_history_val_loss,
        "study_summary": search.get("study_summary", {}),
    }


def _load_cache() -> Dict[str, Any]:
    if CACHE_PATH.exists():
        return json.loads(CACHE_PATH.read_text())
    return {}


def _save_cache(cache: Dict[str, Any]) -> None:
    CACHE_PATH.write_text(json.dumps(cache, indent=2, default=str))


def _cache_key(sampler_seed: int, allow_arch: bool) -> str:
    return f"s{sampler_seed}_arch{int(allow_arch)}"


def ensure_optuna_results(sampler_seeds: List[int], allow_arch: bool,
                          n_attempts: int) -> Dict[int, Dict[str, Any]]:
    """Compute (or load from cache) the Optuna result for each sampler seed."""
    cache = _load_cache()
    out: Dict[int, Dict[str, Any]] = {}
    for ss in sampler_seeds:
        key = _cache_key(ss, allow_arch)
        if key in cache:
            logger.info("Optuna cache hit for sampler_seed=%d (arch=%s)", ss, allow_arch)
            out[ss] = cache[key]
            continue
        logger.info("Computing Optuna arm for sampler_seed=%d (arch=%s) ...", ss, allow_arch)
        res = _compute_optuna_for_seed(ss, allow_arch, n_attempts)
        cache[key] = res
        _save_cache(cache)  # checkpoint after each (expensive) computation
        out[ss] = res
        logger.info("  -> rmse=%.4f  r2=%.4f  best=%s", res["rmse"], res["r2"], res["best_setting"])
    return out


# ---------------------------------------------------------------------------
# Phase 2 -- inject into a directory's metrics
# ---------------------------------------------------------------------------

def _stats(vals: List[Optional[float]]) -> Dict[str, Optional[float]]:
    """Mirror of run_multi_seed_experiment._stats."""
    finite = [v for v in vals if v is not None and np.isfinite(v)]
    if not finite:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(finite, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min":  float(np.min(arr)),
        "max":  float(np.max(arr)),
        "n":    len(arr),
    }


def _trials_to_threshold(attempts_val_loss: List[float],
                         threshold: Optional[float]) -> Optional[int]:
    if threshold is None:
        return None
    for i, v in enumerate(attempts_val_loss):
        if is_finite_number(v) and v <= threshold:
            return i + 1
    return None


def _inject_optuna_into_seed_metrics(cm: Dict[str, Any],
                                     opt: Dict[str, Any]) -> Dict[str, Any]:
    """Add an ``optuna`` slot to one seed's cross_run_metrics dict, in place,
    using that seed's own baseline + threshold for the relative metrics."""
    q = cm.setdefault("optimization_quality", {})
    e = cm.setdefault("optimization_efficiency", {})

    b = q.get("mean_rmse", {}).get("baseline")
    o_rmse, o_r2 = opt["rmse"], opt["r2"]

    q.setdefault("mean_rmse", {})["optuna"] = o_rmse
    q.setdefault("rmse", {})["optuna"] = {"mean": o_rmse, "std": 0.0}
    q.setdefault("mean_r2", {})["optuna"] = o_r2
    q.setdefault("r2", {})["optuna"] = {"mean": o_r2, "std": 0.0}
    q["pct_improvement_optuna_over_baseline"] = (
        (b - o_rmse) / b * 100 if b and np.isfinite(b) else None
    )
    q["win_rate_optuna"] = (
        float(o_rmse < b) if b is not None and np.isfinite(b) else None
    )

    threshold = e.get("trials_to_threshold", {}).get("threshold")
    e.setdefault("trials_to_threshold", {})["optuna"] = _trials_to_threshold(
        opt["attempts_val_loss"], threshold
    )
    auc = safe_trapz(opt["test_history_val_loss"]) if opt["test_history_val_loss"] else None
    e.setdefault("auc_val_loss_curve", {})["optuna"] = (
        float(auc) if auc is not None and np.isfinite(auc) else None
    )
    return cm


def _recompute_aggregated_optuna(summary: Dict[str, Any]) -> None:
    """Add/refresh ONLY the optuna aggregated fields (others untouched)."""
    per_seed = summary["per_seed"]

    def collect(path: List[str]) -> List[Optional[float]]:
        vals = []
        for r in per_seed:
            obj: Any = r.get("cross_run_metrics", {})
            for k in path:
                obj = obj.get(k) if isinstance(obj, dict) else None
                if obj is None:
                    break
            vals.append(float(obj) if obj is not None and np.isfinite(float(obj)) else None)
        return vals

    agg = summary["aggregated"]
    agg.setdefault("rmse", {})["optuna"] = _stats(collect(["optimization_quality", "mean_rmse", "optuna"]))
    agg.setdefault("r2", {})["optuna"] = _stats(collect(["optimization_quality", "mean_r2", "optuna"]))
    agg["pct_improvement_optuna_over_baseline"] = _stats(
        collect(["optimization_quality", "pct_improvement_optuna_over_baseline"])
    )
    agg["win_rate_optuna"] = _stats(collect(["optimization_quality", "win_rate_optuna"]))
    agg.setdefault("trials_to_threshold", {})["optuna"] = _stats(
        collect(["optimization_efficiency", "trials_to_threshold", "optuna"])
    )
    agg.setdefault("auc_val_loss_curve", {})["optuna"] = _stats(
        collect(["optimization_efficiency", "auc_val_loss_curve", "optuna"])
    )


def _backup_once(path: Path) -> None:
    bak = path.with_suffix(path.suffix + ".bak")
    if path.exists() and not bak.exists():
        shutil.copy2(path, bak)


def augment_directory(target: Path, optuna_by_seed: Dict[int, Dict[str, Any]],
                      seeding: str, fixed_seed: int, dry_run: bool) -> None:
    summary_path = target / "multi_seed_summary.json"
    if not summary_path.exists():
        logger.warning("Skipping %s -- no multi_seed_summary.json", target)
        return
    summary = json.loads(summary_path.read_text())
    seeds: List[int] = summary["seeds"]

    for entry in summary["per_seed"]:
        seed = int(entry["seed"])
        sampler_seed = seed if seeding == "vary" else fixed_seed
        opt = optuna_by_seed[sampler_seed]
        cm = entry.get("cross_run_metrics", {})
        _inject_optuna_into_seed_metrics(cm, opt)
        entry["cross_run_metrics"] = cm

        # Mirror into the per-seed file on disk.
        seed_file = target / f"seed_{seed}" / "cross_run_metrics.json"
        if seed_file.exists():
            disk_cm = json.loads(seed_file.read_text())
            _inject_optuna_into_seed_metrics(disk_cm, opt)
            if not dry_run:
                _backup_once(seed_file)
                seed_file.write_text(json.dumps(disk_cm, indent=2, default=str))

    _recompute_aggregated_optuna(summary)

    if dry_run:
        agg = summary["aggregated"]
        logger.info("[dry-run] %s aggregated optuna rmse=%s", target.name, agg["rmse"]["optuna"])
    else:
        _backup_once(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        agg = summary["aggregated"]
        logger.info("Augmented %s | optuna rmse mean=%.4f std=%.4f (n=%s)",
                    target.name, agg["rmse"]["optuna"]["mean"] or 0,
                    agg["rmse"]["optuna"]["std"] or 0, agg["rmse"]["optuna"]["n"])


def main() -> None:
    p = argparse.ArgumentParser(description="Augment multi-seed runs with the Optuna arm.")
    p.add_argument("--dirs", nargs="+", required=True, help="Target output directories.")
    p.add_argument("--seeding", choices=["vary", "fixed"], default="vary",
                   help="vary: sampler seed = run seed; fixed: constant sampler seed.")
    p.add_argument("--fixed-seed", type=int, default=1000,
                   help="Sampler seed used when --seeding fixed.")
    p.add_argument("--no-arch", action="store_true",
                   help="Match runs launched with --no-arch-changes (freeze lstm_hidden/layers).")
    p.add_argument("--attempts", type=int, default=25, help="Optuna trial budget per seed.")
    p.add_argument("--dry-run", action="store_true", help="Compute and print, do not write.")
    args = p.parse_args()

    allow_arch = not args.no_arch
    targets = [Path(d) for d in args.dirs]

    # Gather the union of seeds across all target dirs.
    all_seeds: set = set()
    for t in targets:
        sp = t / "multi_seed_summary.json"
        if sp.exists():
            all_seeds.update(int(s) for s in json.loads(sp.read_text())["seeds"])
    if args.seeding == "vary":
        sampler_seeds = sorted(all_seeds)
    else:
        sampler_seeds = [args.fixed_seed]

    logger.info("Optuna augmentation | seeding=%s | arch=%s | sampler seeds=%s",
                args.seeding, allow_arch, sampler_seeds)
    optuna_by_seed = ensure_optuna_results(sampler_seeds, allow_arch, args.attempts)

    # When fixed, every run seed maps to the single fixed sampler seed.
    if args.seeding == "fixed":
        optuna_by_seed = {s: optuna_by_seed[args.fixed_seed] for s in all_seeds}

    for t in targets:
        augment_directory(t, optuna_by_seed, args.seeding, args.fixed_seed, args.dry_run)

    logger.info("Done.")


if __name__ == "__main__":
    main()

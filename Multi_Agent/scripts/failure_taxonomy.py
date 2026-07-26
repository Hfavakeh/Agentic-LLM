"""failure_taxonomy.py
=====================

Aggregate the LLM-arm failure-mode counters across one or more ``outputs-*``
directories and emit the comparison tables the thesis professor asked for:

* `failure_taxonomy.csv` — long-form, one row per (experiment, seed, run, arm),
  with every retry-stat counter, the per-round ``final_source`` breakdown,
  the ablation flags (`semantic_repair_enabled`, `motion_enabled`,
  `allow_arch_changes`), and test-set RMSE / R² per arm.
* `failure_taxonomy.md` — two summary tables:
    1. Failure-mode counts summed across seeds × runs, grouped by
       (experiment_dir, semantic_repair_enabled).
    2. Test RMSE per arm (baseline / llm / rule_based / random), mean ± std
       across seeds × runs, same grouping.

Usage
-----
    python failure_taxonomy.py                            # all outputs-* under repo
    python failure_taxonomy.py --dirs outputs-A outputs-B # explicit dirs
    python failure_taxonomy.py --out analysis/no_repair   # custom prefix

Designed for the "LLM with vs without semantic repair" ablation: run the
experiment twice (same seeds / model, with and without ``--no-semantic-repair``),
point this script at both output dirs, and the markdown tables will show the
paired comparison.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ARMS = ("baseline", "llm", "rule_based", "random")

RETRY_STAT_KEYS = [
    "total_attempts", "retries", "fallbacks",
    "parse_failures", "validation_failures", "empty_changes_rejections",
    "rounds_clean", "rounds_corrected", "rounds_retry_succeeded", "rounds_skipped",
    "clamps_count", "discrete_snaps_count", "unknown_keys_count",
    "diagnosis_corrections_count", "resets_model_corrections_count",
    "strict_rejections",
]

FINAL_SOURCES = ("llm_clean", "llm_corrected", "retry_clean", "retry_corrected", "skipped")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as exc:
        print(f"[warn] failed to load {path}: {exc}")
        return None


def _iter_experiments(root: Path, explicit: List[str]) -> List[Path]:
    if explicit:
        return [Path(p).resolve() for p in explicit]
    return sorted(p for p in root.glob("outputs-*") if p.is_dir())


def _iter_seed_dirs(exp_dir: Path) -> List[Tuple[Optional[int], Path]]:
    """Return (seed_int_or_None, seed_dir) pairs.

    Multi-seed runs nest results under ``seed_NNNN``; single-seed legacy runs
    write everything into the experiment dir directly. Both layouts are
    handled.
    """
    seed_dirs = sorted(exp_dir.glob("seed_*"))
    if seed_dirs:
        out: List[Tuple[Optional[int], Path]] = []
        for sd in seed_dirs:
            try:
                seed_n: Optional[int] = int(sd.name.split("_", 1)[1])
            except (IndexError, ValueError):
                seed_n = None
            out.append((seed_n, sd))
        return out
    return [(None, exp_dir)]


def _detect_run_ids(seed_dir: Path) -> List[int]:
    ids = set()
    for p in seed_dir.glob("optimization_log_run*.json"):
        stem = p.stem  # "optimization_log_run1"
        try:
            ids.add(int(stem.rsplit("run", 1)[1]))
        except (IndexError, ValueError):
            continue
    return sorted(ids) or [1]


# ---------------------------------------------------------------------------
# Per-run extraction
# ---------------------------------------------------------------------------

def _extract_rmse_r2(cross: Optional[Dict[str, Any]], arm: str
                     ) -> Tuple[Optional[float], Optional[float]]:
    if not cross:
        return None, None
    oq = cross.get("optimization_quality", {}) or {}
    rmse_blob = (oq.get("rmse") or {}).get(arm) or {}
    r2_blob   = (oq.get("r2")   or {}).get(arm) or {}
    return rmse_blob.get("mean"), r2_blob.get("mean")


def _llm_summary(log_path: Path) -> Dict[str, Any]:
    d = _load_json(log_path) or {}
    fs = d.get("final_summary", {}) or {}
    rs = fs.get("retry_stats", {}) or {}
    rounds = d.get("rounds", []) or []

    fs_counts = {k: 0 for k in FINAL_SOURCES}
    for r in rounds:
        src = r.get("final_source")
        if src in fs_counts:
            fs_counts[src] += 1

    # Old runs (predating the --no-semantic-repair flag) have no
    # `semantic_repair_enabled` field. The historical default was always to
    # silently auto-correct, so assume True and mark the inference so the
    # caller can warn the user it was applied.
    sr_raw = fs.get("semantic_repair_enabled")
    sr_inferred = sr_raw is None
    sr = True if sr_raw is None else bool(sr_raw)

    return {
        "best_val_loss":              fs.get("best_val_loss"),
        "rounds_total":               len(rounds),
        "rounds_completed":           fs.get("rounds_completed"),
        "rounds_skipped":             fs.get("rounds_skipped"),
        "semantic_repair_enabled":    sr,
        "semantic_repair_inferred":   sr_inferred,
        "motion_enabled":             fs.get("motion_enabled"),
        "allow_arch_changes":         fs.get("allow_arch_changes"),
        "retry_stats":                rs,
        "final_source_counts":        fs_counts,
    }


def _collect_run(exp_dir: Path, seed_dir: Path, seed_n: Optional[int],
                 run_id: int) -> List[Dict[str, Any]]:
    cross = _load_json(seed_dir / "cross_run_metrics.json")
    llm_log = seed_dir / f"optimization_log_run{run_id}.json"
    llm_data = _llm_summary(llm_log) if llm_log.exists() else {}

    rows: List[Dict[str, Any]] = []
    for arm in ARMS:
        rmse, r2 = _extract_rmse_r2(cross, arm)
        row: Dict[str, Any] = {
            "experiment_dir":           exp_dir.name,
            "seed":                     seed_n,
            "run_id":                   run_id,
            "arm":                      arm,
            "test_rmse":                rmse,
            "test_r2":                  r2,
            "best_val_loss":            None,
            "rounds_total":             None,
            "rounds_completed":         None,
            "rounds_skipped":           None,
            "semantic_repair_enabled":  None,
            "semantic_repair_inferred": None,
            "motion_enabled":           None,
            "allow_arch_changes":       None,
        }
        for k in RETRY_STAT_KEYS:
            row[f"rs_{k}"] = None
        for k in FINAL_SOURCES:
            row[f"final_source_{k}"] = None

        if arm == "llm" and llm_data:
            row["best_val_loss"]            = llm_data.get("best_val_loss")
            row["rounds_total"]             = llm_data.get("rounds_total")
            row["rounds_completed"]         = llm_data.get("rounds_completed")
            row["rounds_skipped"]           = llm_data.get("rounds_skipped")
            row["semantic_repair_enabled"]  = llm_data.get("semantic_repair_enabled")
            row["semantic_repair_inferred"] = llm_data.get("semantic_repair_inferred")
            row["motion_enabled"]           = llm_data.get("motion_enabled")
            row["allow_arch_changes"]       = llm_data.get("allow_arch_changes")
            for k in RETRY_STAT_KEYS:
                row[f"rs_{k}"] = llm_data["retry_stats"].get(k)
            for k in FINAL_SOURCES:
                row[f"final_source_{k}"] = llm_data["final_source_counts"].get(k, 0)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def _mean_std(vals: List[Optional[float]]) -> Tuple[float, float]:
    clean = [float(v) for v in vals
             if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not clean:
        return float("nan"), float("nan")
    m = sum(clean) / len(clean)
    if len(clean) == 1:
        return m, 0.0
    var = sum((v - m) ** 2 for v in clean) / (len(clean) - 1)
    return m, var ** 0.5


def _group_tag(exp: str, sr: Optional[bool]) -> str:
    tag = "repair=on" if sr else "repair=off" if sr is False else "repair=?"
    return f"{exp} ({tag})"


def _write_summary_markdown(rows: List[Dict[str, Any]], path: Path) -> None:
    llm_rows = [r for r in rows if r["arm"] == "llm"
                and r.get("semantic_repair_enabled") is not None]

    has_inferred = any(r.get("semantic_repair_inferred") for r in llm_rows)

    # ---- Group LLM rows by (experiment_dir, semantic_repair_enabled) ----
    groups: Dict[Tuple[str, bool], List[Dict[str, Any]]] = defaultdict(list)
    for r in llm_rows:
        groups[(r["experiment_dir"], bool(r["semantic_repair_enabled"]))].append(r)

    lines: List[str] = []
    lines.append("# LLM failure-taxonomy summary\n")
    lines.append(
        "Each group is one experiment dir split by `semantic_repair_enabled`. "
        "Counts are summed across seeds × runs in that group; metrics are "
        "mean ± std across (seed, run) pairs.\n"
    )
    if has_inferred:
        lines.append(
            "> ⚠ Some runs predate the `--no-semantic-repair` flag and have no "
            "`semantic_repair_enabled` field in their `final_summary`. They are "
            "assumed `repair=on` because that was the historical default. The "
            "`semantic_repair_inferred` column in the CSV flags them.\n"
        )

    # ---- Table 1: failure counters ----
    counter_cols = [
        ("n_runs",      "n"),
        ("rounds_total", "rnds"),
        ("clean",       "clean"),
        ("corrected",   "corr"),
        ("retry_succ",  "retry"),
        ("skipped",     "skip"),
        ("clamps",      "clmp"),
        ("snaps",       "snap"),
        ("unknown",     "unk"),
        ("diag_corr",   "diag"),
        ("reset_corr",  "rst"),
        ("strict_rej",  "strict"),
        ("parse_fail",  "parse_f"),
        ("val_fail",    "val_f"),
        ("empty_rej",   "empty"),
    ]
    lines.append("## Failure-mode counts (sums across seeds × runs)\n")
    lines.append("| group | " + " | ".join(c[1] for c in counter_cols) + " |")
    lines.append("|" + "|".join(["---"] * (len(counter_cols) + 1)) + "|")
    for (exp, sr), grp in sorted(groups.items()):
        def s(k: str) -> int:
            return int(sum((r.get(k) or 0) for r in grp))
        clean      = s("final_source_llm_clean") + s("final_source_retry_clean")
        corrected  = s("final_source_llm_corrected") + s("final_source_retry_corrected")
        values = {
            "n_runs":       len(grp),
            "rounds_total": s("rounds_total"),
            "clean":        clean,
            "corrected":    corrected,
            "retry_succ":   s("rs_rounds_retry_succeeded"),
            "skipped":      s("rs_rounds_skipped"),
            "clamps":       s("rs_clamps_count"),
            "snaps":        s("rs_discrete_snaps_count"),
            "unknown":      s("rs_unknown_keys_count"),
            "diag_corr":    s("rs_diagnosis_corrections_count"),
            "reset_corr":   s("rs_resets_model_corrections_count"),
            "strict_rej":   s("rs_strict_rejections"),
            "parse_fail":   s("rs_parse_failures"),
            "val_fail":     s("rs_validation_failures"),
            "empty_rej":    s("rs_empty_changes_rejections"),
        }
        cells = " | ".join(str(values[c[0]]) for c in counter_cols)
        lines.append(f"| {_group_tag(exp, sr)} | {cells} |")
    lines.append("")

    # ---- Table 2: rates (per-round and per-attempt percentages) ----
    lines.append("## Failure-mode rates (% of attempts / rounds)\n")
    lines.append(
        "*Rates make absolute counts comparable across runs with different "
        "round budgets. `% clean` is the share of accepted rounds where the "
        "LLM's raw output passed validation untouched.*\n"
    )
    lines.append("| group | n_runs | % clean | % corrected | % skipped | "
                 "% retries / attempt | strict rej / round |")
    lines.append("|---|---|---|---|---|---|---|")
    for (exp, sr), grp in sorted(groups.items()):
        def s(k: str) -> int:
            return int(sum((r.get(k) or 0) for r in grp))
        rounds_total = s("rounds_total") or 1
        attempts     = s("rs_total_attempts") or 1
        clean      = s("final_source_llm_clean") + s("final_source_retry_clean")
        corrected  = s("final_source_llm_corrected") + s("final_source_retry_corrected")
        skipped    = s("rs_rounds_skipped")
        retries    = s("rs_retries")
        strict_rej = s("rs_strict_rejections")
        lines.append(
            f"| {_group_tag(exp, sr)} | {len(grp)} | "
            f"{100*clean/rounds_total:.1f}% | "
            f"{100*corrected/rounds_total:.1f}% | "
            f"{100*skipped/rounds_total:.1f}% | "
            f"{100*retries/attempts:.1f}% | "
            f"{strict_rej/rounds_total:.2f} |"
        )
    lines.append("")

    # ---- Table 3: test RMSE per arm, grouped by (exp, sr) ----
    # Use the LLM row's semantic_repair_enabled to assign sibling arms (baseline,
    # rule_based, random) to the same group; they share the seed/run dir.
    sr_for: Dict[Tuple[str, Optional[int], int], Optional[bool]] = {}
    for r in rows:
        if r["arm"] == "llm" and r.get("semantic_repair_enabled") is not None:
            sr_for[(r["experiment_dir"], r.get("seed"), r["run_id"])] = bool(
                r["semantic_repair_enabled"]
            )

    arm_groups: Dict[Tuple[str, Optional[bool]], Dict[str, List[float]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for r in rows:
        sr = sr_for.get((r["experiment_dir"], r.get("seed"), r["run_id"]))
        rmse = r.get("test_rmse")
        if rmse is None:
            continue
        arm_groups[(r["experiment_dir"], sr)][r["arm"]].append(float(rmse))

    lines.append("## Test RMSE per arm (mean ± std across seeds × runs)\n")
    lines.append("| group | n | " + " | ".join(ARMS) + " |")
    lines.append("|" + "|".join(["---"] * (len(ARMS) + 2)) + "|")
    for (exp, sr), arm_map in sorted(arm_groups.items()):
        n = max((len(v) for v in arm_map.values()), default=0)
        cells: List[str] = []
        for arm in ARMS:
            m, sd = _mean_std(arm_map.get(arm, []))
            cells.append(f"{m:.4f} ± {sd:.4f}" if not math.isnan(m) else "—")
        lines.append(f"| {_group_tag(exp, sr)} | {n} | " + " | ".join(cells) + " |")
    lines.append("")

    # ---- Table 4: best validation loss for LLM arm ----
    lines.append("## LLM-arm best validation loss (mean ± std)\n")
    lines.append("| group | n | best_val_loss |")
    lines.append("|---|---|---|")
    for (exp, sr), grp in sorted(groups.items()):
        m, sd = _mean_std([r.get("best_val_loss") for r in grp])
        cell = f"{m:.4f} ± {sd:.4f}" if not math.isnan(m) else "—"
        lines.append(f"| {_group_tag(exp, sr)} | {len(grp)} | {cell} |")
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--root", default=str(Path(__file__).parent),
                    help="Directory to walk for outputs-* dirs.")
    ap.add_argument("--dirs", nargs="*", default=[],
                    help="Explicit list of experiment dirs. Overrides --root.")
    ap.add_argument("--out", default="analysis/failure_taxonomy",
                    help="Output prefix (writes .csv and .md).")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    experiments = _iter_experiments(root, args.dirs)
    if not experiments:
        print(f"[warn] no outputs-* dirs found under {root}.")
        return

    print(f"[info] root        = {root}")
    print(f"[info] experiments = {len(experiments)}")

    all_rows: List[Dict[str, Any]] = []
    for exp in experiments:
        for seed_n, sd in _iter_seed_dirs(exp):
            for run_id in _detect_run_ids(sd):
                all_rows.extend(_collect_run(exp, sd, seed_n, run_id))

    if not all_rows:
        print("[warn] no data extracted.")
        return

    inferred_runs = sorted({
        (r["experiment_dir"], r.get("seed"), r["run_id"])
        for r in all_rows
        if r["arm"] == "llm" and r.get("semantic_repair_inferred")
    })
    if inferred_runs:
        print(
            f"[warn] {len(inferred_runs)} LLM run(s) had no `semantic_repair_enabled` "
            "field in final_summary (predates --no-semantic-repair flag). Assuming "
            "repair=on for those runs since that was the historical default."
        )

    out_csv = Path(args.out)
    if not out_csv.is_absolute():
        out_csv = root / out_csv
    out_csv = out_csv.with_suffix(".csv")
    out_md  = out_csv.with_suffix(".md")
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    cols = list(all_rows[0].keys())
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"[info] wrote {out_csv}  ({len(all_rows)} rows)")

    _write_summary_markdown(all_rows, out_md)
    print(f"[info] wrote {out_md}")


if __name__ == "__main__":
    main()

"""Protocol report writers: per-setting CSV, budget table, final-eval report.

Raw numbers live here (and in the JSON logs); the LLM prompt only ever saw
the qualitative labels.
"""

import csv
import json
from typing import Any, Dict, List

import numpy as np

from arms.engine import TRAIN_SEEDS, _default_setting
from pipeline import Config, is_finite_number, logger

_SETTING_COLS = [
    "learning_rate", "weight_decay", "dropout", "batch_size",
    "lstm_hidden", "lstm_layers", "window_size", "optimizer_choice", "patience",
]
_REPORT_COLS = (
    ["arm", "attempt", "output_status", "trained",
     "val_rmse_mean", "val_rmse_std", "mean_best_epoch",
     "mean_train_loss", "mean_val_loss", "train_val_gap", "runtime_s", "diagnosis"]
    + _SETTING_COLS
)


def _attempt_to_row(arm: str, rec: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten one attempt record into a per-setting report row. Reports the
    protocol's columns: mean val RMSE, std, mean best epoch, mean train/val
    loss, train/val gap, runtime."""
    setting = rec.get("setting", {}) or {}
    row = {
        "arm":             arm,
        "attempt":         rec.get("attempt"),
        "output_status":   rec.get("output_status", ""),
        "trained":         rec.get("trained", True),
        "val_rmse_mean":   rec.get("val_rmse", rec.get("score")),
        "val_rmse_std":    rec.get("val_rmse_std"),
        "mean_best_epoch": rec.get("mean_best_epoch"),
        "mean_train_loss": rec.get("mean_train_loss"),
        "mean_val_loss":   rec.get("mean_val_loss"),
        "train_val_gap":   rec.get("mean_train_val_gap"),
        "runtime_s":       rec.get("runtime_s"),
        "diagnosis":       rec.get("diagnosis", ""),
    }
    for k in _SETTING_COLS:
        row[k] = setting.get(k)
    return row


def write_per_setting_report(config: Config, results: Dict[str, Any], run_id: int) -> None:
    """Write one CSV row per attempted setting, per arm, with the protocol's
    per-setting metrics."""
    rows: List[Dict[str, Any]] = []

    # Fixed-reference baseline: one row (final-eval-only; single training).
    b = results.get("baseline") or {}
    b_pos = [v for v in (b.get("history", {}) or {}).get("val_position_loss", []) if is_finite_number(v)]
    b_setting = _default_setting(config, getattr(config, "allow_arch_changes", True))
    base_row = {c: None for c in _REPORT_COLS}
    base_row.update({
        "arm": "fixed_reference", "attempt": 1, "output_status": "final_eval_only",
        "trained": True,
        "val_rmse_mean": float(np.sqrt(min(b_pos))) if b_pos else None,
    })
    for k in _SETTING_COLS:
        base_row[k] = b_setting.get(k)
    rows.append(base_row)

    for arm in ("llm", "random", "rule_based", "optuna"):
        slot = results.get(arm)
        if not isinstance(slot, dict):
            continue
        attempts = (slot.get("search") or {}).get("attempts", [])
        for rec in attempts:
            rows.append(_attempt_to_row(arm, rec))

    path = config.output_dir / f"per_setting_report_run{run_id}.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_REPORT_COLS)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in _REPORT_COLS})
    logger.info("Per-setting report -> %s  (%d rows)", path, len(rows))


def write_budget_table(config: Config, results: Dict[str, Any], run_id: int) -> None:
    """Write the protocol budget table (point 9) + the LLM validity accounting."""
    n = config.optimization_rounds
    e = config.epochs_per_round
    t = len(TRAIN_SEEDS)

    def _trained(arm: str) -> int:
        slot = results.get(arm) or {}
        return sum(1 for a in (slot.get("search") or {}).get("attempts", []) if a.get("trained"))

    lines = [
        f"# Budget table (run {run_id})",
        "",
        "| Method | Attempts | Trainings per valid setting | Epochs/train (max) | Val during search | Test during search | Settings actually trained |",
        "|---|---|---|---|---|---|---|",
        f"| Fixed reference | 1 | final-eval only | {e} | no | no | 1 |",
        f"| Random search | {n} | {t} | {e} | yes | no | {_trained('random')} |",
        f"| Optuna (TPE) | {n} | {t} | {e} | yes | no | {_trained('optuna')} |",
        f"| Rule-based | {n} | {t} | {e} | yes | no | {_trained('rule_based')} |",
        f"| LLM controller | {n} | {t} per valid setting | {e} | yes | no | {_trained('llm')} |",
        "",
    ]

    vs = (results.get("llm") or {}).get("validity_stats", {}) or {}
    if vs:
        lines += [
            "## LLM controller validity",
            "",
            f"- valid on first try: {vs.get('valid_first_try', 0)}",
            f"- valid after retry: {vs.get('valid_after_retry', 0)}",
            f"- rejected (invalid after retry): {vs.get('rejected', 0)}",
            f"- repeated settings proposed: {vs.get('repeats_proposed', 0)}",
            f"- invalid parameter values proposed: {vs.get('invalid_values', 0)}",
            f"- LLM timeouts: {vs.get('llm_timeouts', 0)}",
            f"- parse failures: {vs.get('parse_failures', 0)}",
            f"- actual valid settings trained: {_trained('llm')}",
            "",
        ]

    path = config.output_dir / f"budget_table_run{run_id}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Budget table -> %s", path)
    for ln in lines:
        if ln.startswith("|") or ln.startswith("- ") or ln.startswith("#"):
            logger.info("  %s", ln)


def write_final_eval_report(config: Config, final_eval: Dict[str, Any], run_id: int) -> None:
    """Write the final-evaluation table + paired differences (markdown + JSON)."""
    per_arm = final_eval.get("per_arm", {})
    paired  = final_eval.get("paired_vs_llm", {})
    order = ["fixed_reference", "llm", "random", "rule_based", "optuna"]
    label = {"fixed_reference": "Fixed reference", "llm": "LLM controller",
             "random": "Random search", "rule_based": "Rule-based", "optuna": "Optuna (TPE)"}

    lines = [
        f"# Final evaluation (run {run_id}) — best setting per method on {len(final_eval.get('seeds', []))} fresh seeds",
        "",
        "| Method | mean test RMSE | std | best seed | worst seed | mean best epoch | mean runtime (s) | #params |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for arm in order:
        a = per_arm.get(arm)
        if not a:
            continue
        lines.append(
            f"| {label[arm]} | {a['mean_rmse']:.4f} | {a['std_rmse']:.4f} | "
            f"{a['best_seed']['rmse']:.4f}@{a['best_seed']['seed']} | "
            f"{a['worst_seed']['rmse']:.4f}@{a['worst_seed']['seed']} | "
            f"{a['mean_best_epoch']:.1f} | {a['mean_runtime_s']:.1f} | {a['n_params']} |"
        )
    lines.append("")
    if paired:
        # ASCII '-' (not U+2212) so logger.info doesn't crash on Windows
        # cp1252 consoles when echoing the table back; the .md file (UTF-8) is
        # unaffected either way.
        lines += ["## Paired differences (LLM - method, per seed; negative = LLM better)", ""]
        for key, p in paired.items():
            arm = key.replace("llm_minus_", "")
            lines.append(f"- LLM - {label.get(arm, arm)}: {p['mean']:+.4f} ± {p['std']:.4f} "
                         f"(LLM better on {p['llm_better_count']}/{p['n']} seeds)")
        lines.append("")

    md_path = config.output_dir / f"final_evaluation_run{run_id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    json_path = config.output_dir / f"final_evaluation_run{run_id}.json"
    with open(json_path, "w") as f:
        json.dump(final_eval, f, indent=2, default=str)
    logger.info("Final-eval report -> %s", md_path)
    for ln in lines:
        if ln.startswith("|") or ln.startswith("- ") or ln.startswith("#"):
            logger.info("  %s", ln)

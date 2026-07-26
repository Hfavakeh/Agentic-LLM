"""Render LLM I/O conversations as clean Markdown transcripts.

For each run found under ``outputs-*/seed_*/``, emit one transcript file
per (experiment, seed) showing, for each round:
  - the metrics the LLM saw (compact bullets, not full JSON),
  - the raw LLM response (verbatim),
  - the validator's outcome (applied / rejected).

The system prompt is identical on every round, so it is rendered once at
the top of each transcript.

Usage
-----
    python build_transcripts.py                        # all experiments → analysis/<exp>/transcripts/
    python build_transcripts.py --experiment outputs-0503-llama3-8b
    python build_transcripts.py --experiment outputs-0503-llama3-8b --seed 8934
    python build_transcripts.py --interventions-only   # only seeds with ≥1 validator intervention
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

DEFAULT_ROOT = Path(__file__).resolve().parent


def fmt_num(v, decimals=4):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{decimals}f}"
    return str(v)


def render_user_payload(p: dict) -> str:
    """Compact, readable rendering of the per-round user message."""
    lines = []

    m = p.get("metrics", {}) or {}
    lines.append("**Metrics**")
    lines.append(f"- val_loss = {fmt_num(m.get('val_loss'))}, "
                 f"train_loss = {fmt_num(m.get('train_loss'))}, "
                 f"loss_ratio = {fmt_num(m.get('loss_ratio'))}")
    lines.append(f"- val_mae = {fmt_num(m.get('val_mae'))}, "
                 f"mean_euclid_m = {fmt_num(m.get('mean_euclidean_distance_m'))}")

    rs = p.get("round_summary", {}) or {}
    if any(v is not None for v in rs.values()):
        lines.append("")
        lines.append("**Round summary**")
        lines.append(f"- best_val_loss = {fmt_num(rs.get('round_best_val_loss'))}, "
                     f"avg_val_loss = {fmt_num(rs.get('round_avg_val_loss'))}, "
                     f"epochs_trained = {rs.get('round_epochs_trained', 0)}")

    t = p.get("trends", {}) or {}
    if t:
        lines.append("")
        lines.append("**Trends**")
        lines.append(f"- validation = {t.get('validation', '—')}, "
                     f"training = {t.get('training', '—')}, "
                     f"epochs_since_improvement = {t.get('epochs_since_improvement', '—')}")

    tp = p.get("training_progress", {}) or {}
    if tp:
        lines.append("")
        lines.append("**Training progress**")
        lines.append(f"- total_epochs = {tp.get('total_epochs')}, "
                     f"current_round = {tp.get('current_round')}, "
                     f"best_val_loss = {fmt_num(tp.get('best_val_loss'))}")

    tools = p.get("tool_results", {}) or {}
    if tools:
        lines.append("")
        lines.append("**Diagnostic tools**")
        for tool, result in tools.items():
            if not isinstance(result, dict):
                continue
            keys = list(result.items())
            inner = ", ".join(
                f"{k}={fmt_num(v) if isinstance(v, (int, float)) else v}"
                for k, v in keys if not isinstance(v, dict)
            )
            lines.append(f"- `{tool}`: {inner}")

    hp = p.get("current_hyperparameters", {}) or {}
    if hp:
        lines.append("")
        lines.append("**Current hyperparameters**")
        parts = [f"{k}={v}" for k, v in hp.items()]
        lines.append(f"- {', '.join(parts)}")

    oh = p.get("optimization_history", []) or []
    if oh:
        lines.append("")
        lines.append("**Optimization history (compact)**")
        for entry in oh:
            if not isinstance(entry, dict):
                lines.append(f"- {entry!r}")
                continue
            r = entry.get("round")
            vl = fmt_num(entry.get("val_loss"))
            outcome = entry.get("outcome", "—")
            ch = entry.get("changes_applied") or {}
            ch_s = ", ".join(f"{k}={v}" for k, v in ch.items()) if ch else "no change"
            lines.append(f"- round {r}: val_loss={vl}, outcome={outcome}, changes=[{ch_s}]")

    return "\n".join(lines)


def render_seed(seed_dir: Path) -> str:
    """Build the markdown for one seed."""
    conv = json.loads((seed_dir / "conversation_log_run1.json").read_text(encoding="utf-8"))
    opt  = json.loads((seed_dir / "optimization_log_run1.json").read_text(encoding="utf-8"))

    # Map round -> opt entry (incl. skipped retry rows when present)
    opt_by_round = {}
    for r in opt.get("rounds", []):
        rnum = r.get("round")
        # If a "skipped" row exists for this round, keep both; we'll render them.
        opt_by_round.setdefault(rnum, []).append(r)

    out = []
    out.append(f"# LLM conversation transcript — `{seed_dir.parent.name}` / `{seed_dir.name}`")
    out.append("")
    out.append(f"_Generated from `{seed_dir.name}/conversation_log_run1.json` and "
               f"`optimization_log_run1.json`._")
    out.append("")
    final = opt.get("final_summary", {})
    rs = final.get("retry_stats", {})
    out.append(f"**Final summary:** best_val_loss = {fmt_num(final.get('best_val_loss'))} "
               f"at round {final.get('best_round')}, "
               f"rounds_completed = {final.get('rounds_completed')}, "
               f"rounds_skipped = {final.get('rounds_skipped')}, "
               f"retries = {rs.get('retries', 0)}, "
               f"fallbacks = {rs.get('fallbacks', 0)}.")
    out.append("")

    # System prompt — once at the top.
    if conv:
        sysp = conv[0]["llm_input"]["system_prompt"]
        out.append("---")
        out.append("")
        out.append("## System prompt (identical for every round)")
        out.append("")
        out.append("```")
        out.append(sysp.strip())
        out.append("```")
        out.append("")

    # Per-round transcript.
    for entry in conv:
        rnum = entry["round"]
        out.append("---")
        out.append("")
        out.append(f"## Round {rnum}")
        out.append("")
        out.append(f"_Timestamp: {entry.get('timestamp', '—')}_")
        out.append("")

        rows = opt_by_round.get(rnum, [])
        skipped = next((r for r in rows if r.get("skipped")), None)
        applied = next((r for r in rows if not r.get("skipped")), None)

        raw_out = (entry.get("llm_raw_output") or "").strip()
        payload = entry["llm_input"].get("user_payload") or {}
        is_fallback_record = (raw_out == "" and not payload)

        if is_fallback_record and skipped:
            # The LLM call was rejected by the validator and the loop fell
            # through to a no-op fallback. The conversation log only records
            # the empty fallback; the rejected proposal lives in the
            # optimisation log's `failure_reason`.
            out.append("> **This round was rejected by the validator.** "
                       "The conversation log entry is the post-rejection fallback "
                       "(empty input + empty output by design). The rejected proposal "
                       "is captured in the optimisation log below.")
            out.append("")
        else:
            # 1) Input the LLM saw
            out.append("### LLM input (user message)")
            out.append("")
            out.append(render_user_payload(payload))
            out.append("")

            # 2) Raw LLM output
            out.append("### LLM raw output")
            out.append("")
            out.append("```json")
            out.append(raw_out)
            out.append("```")
            out.append("")

        # 3) Validator outcome (from optimization log)
        out.append("### Validator + applied result")
        out.append("")
        if skipped:
            reason = skipped.get("failure_reason", "—")
            out.append(f"- **REJECTED by validator** — `{reason}`")
            out.append("- → fallback: round skipped, no hyperparameter change applied.")
        if applied:
            ch = applied.get("changes_applied") or {}
            ch_s = ", ".join(f"{k}={v}" for k, v in ch.items()) if ch else "no change"
            out.append(f"- changes_applied: {ch_s}")
            out.append(f"- val_loss after this round: {fmt_num(applied.get('val_loss'))} "
                       f"(outcome = `{applied.get('outcome')}`, "
                       f"Δ best_val_loss = {fmt_num(applied.get('delta_best_val_loss'))})")
        out.append("")

    return "\n".join(out)


def discover_experiments(root: Path) -> List[Path]:
    """Every immediate child directory matching outputs-*."""
    return sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("outputs-")])


def discover_seeds(exp_dir: Path) -> List[Path]:
    return sorted([p for p in exp_dir.iterdir() if p.is_dir() and p.name.startswith("seed_")])


def seed_has_intervention(seed_dir: Path) -> bool:
    """True if any round in the optimisation log was rejected by the validator."""
    opt_path = seed_dir / "optimization_log_run1.json"
    if not opt_path.exists():
        return False
    try:
        opt = json.loads(opt_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return any(r.get("skipped") for r in (opt.get("rounds") or []))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT),
                    help="Project root containing outputs-* directories.")
    ap.add_argument("--out", default=None,
                    help="Output directory. Default: <root>/analysis/<experiment>/transcripts/.")
    ap.add_argument("--experiment", action="append", default=None,
                    help="Restrict to one or more experiments (e.g. outputs-0503-llama3-8b). "
                         "May be repeated; default = all.")
    ap.add_argument("--seed", action="append", default=None,
                    help="Restrict to one or more seed ids (e.g. 8934). May be repeated.")
    ap.add_argument("--interventions-only", action="store_true",
                    help="Skip seeds with no validator interventions.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    seed_filter = set(args.seed) if args.seed else None

    experiments = discover_experiments(root)
    if args.experiment:
        wanted = set(args.experiment)
        experiments = [e for e in experiments if e.name in wanted]
    if not experiments:
        print(f"[warn] no outputs-* directories matched under {root}")
        return

    n_total = 0
    n_written = 0
    for exp_dir in experiments:
        seed_dirs = discover_seeds(exp_dir)
        if seed_filter is not None:
            seed_dirs = [s for s in seed_dirs
                         if s.name.replace("seed_", "") in seed_filter]
        if not seed_dirs:
            continue

        # Default output: alongside the per-experiment report.
        out_dir = (
            Path(args.out) if args.out
            else root / "analysis" / exp_dir.name / "transcripts"
        )
        out_dir.mkdir(parents=True, exist_ok=True)

        for seed_dir in seed_dirs:
            n_total += 1
            if not (seed_dir / "conversation_log_run1.json").exists():
                continue
            if not (seed_dir / "optimization_log_run1.json").exists():
                continue
            if args.interventions_only and not seed_has_intervention(seed_dir):
                continue
            md = render_seed(seed_dir)
            out_path = out_dir / f"transcript_{seed_dir.name}.md"
            out_path.write_text(md, encoding="utf-8")
            n_written += 1
            print(f"WROTE {out_path.relative_to(root)}  ({len(md):,} chars)")

    print(f"[done] wrote {n_written} transcripts (scanned {n_total} seeds across "
          f"{len(experiments)} experiments).")


if __name__ == "__main__":
    main()

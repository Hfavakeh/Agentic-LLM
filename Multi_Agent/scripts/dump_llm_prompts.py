"""Prompt-quality and information-content audit (professor's Email-8, block 3).

Answers, from the recorded transcripts rather than from the source code, the
question "show exactly what the LLM receives at every attempt":

  1. TRANSCRIPT (markdown) — for every attempt, verbatim: the system prompt, the
     user message actually sent (including the retry variant after a rejection),
     the raw reply, the parse/validation outcome, and the resolved setting.
  2. INFORMATION AUDIT (CSV + printed summary) — per attempt, which numeric
     quantities EXISTED in the history at that moment and which of them appear
     as numbers in the rendered payload. This is the direct measurement of what
     the qualitative-label rendering discards: ordering, magnitude and the
     across-3-trainings uncertainty are each checked separately.
  3. PROMPT INVENTORY — the blocks the payload is built from (ANCHOR, BEST
     SETTINGS, LAST ATTEMPTS, ALREADY TRIED, OBSERVED PATTERNS, MOTION PROFILE),
     with their sizes in characters and tokens-ish, so "the prompt hides the
     signal in bulk" is a measurement and not an impression.

Reads `protocol_log_run*.json` written by `SingleAgentOptimizer.save_protocol_log`.
Transcripts recorded before 2026-07-26 lack the verbatim system prompt and the
per-sub-attempt user message; those fields are reported as MISSING rather than
reconstructed, so an old run cannot be mistaken for an audited one.

Run:
  python scripts/dump_llm_prompts.py --run results/history-use/history-none-qwen3
  python scripts/dump_llm_prompts.py --run <dir> --attempt 7      # one attempt
  python scripts/dump_llm_prompts.py --run <dir> --out analysis/prompt_audit
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Payload section headers, in the order `prompts.format_protocol_payload` emits
# them. Used to slice a rendered payload into blocks for the size inventory.
_BLOCKS = [
    "ANCHOR",
    "MOTION PROFILE",
    "BEST SETTINGS SO FAR",
    "LAST ATTEMPTS",
    "ALREADY TRIED",
    "OBSERVED PATTERNS",
    "TRIED LEVER VECTORS",
    "CURRENT LOSS-SHAPING LEVERS",
    "MOTION SUMMARY",
    "EVALUATED SETTINGS",
]

# The numeric quantities the driver has on hand for each past attempt. `shown_as`
# records what the qualitative payload replaces them with.
_QUANTITIES = [
    ("score",              "mean validation RMSE (m)",        "quality label + level n/10"),
    ("val_rmse_std",       "spread across the 3 trainings",   "reliability label"),
    ("per_seed",           "the 3 individual seed scores",    "not shown at all"),
    ("mean_best_epoch",    "early-stopping epoch",            "timing label"),
    ("mean_val_loss",      "validation loss",                 "gap label"),
    ("mean_train_val_gap", "train/validation gap",            "gap label"),
]


def _find_logs(run_dir: Path) -> List[Path]:
    """Every protocol log at or below `run_dir`.

    Handles all three layouts in one pass, so `--run results/history-use` sweeps
    every condition just as `--run <one-condition>` audits one:
      <run>/protocol_log_run1.json                  single flat run
      <run>/seed_<N>/protocol_log_run1.json          multi-seed run
      <run>/<condition>/seed_<N>/protocol_log_run1.json   a whole group
    """
    hits: List[str] = []
    for pat in ("protocol_log_run*.json", "motion_protocol_log_run*.json"):
        hits += sorted(glob.glob(str(run_dir / "**" / pat), recursive=True))
        hits += sorted(glob.glob(str(run_dir / pat)))
    return [Path(h) for h in dict.fromkeys(hits)]


def _condition_of(log: Path, root: Path) -> str:
    """Label a log by the run dir it belongs to (the dir above seed_<N>/)."""
    parent = log.parent
    if parent.name.startswith("seed_"):
        parent = parent.parent
    try:
        rel = parent.relative_to(root)
    except ValueError:
        return parent.name
    return str(rel) if str(rel) not in ("", ".") else root.name


def _numbers_in(text: str) -> List[float]:
    """Numbers in a rendered payload that could carry OUTCOME information.

    Values attached to a parameter name (`dropout=0.25`, `lambda_smooth=0.05`)
    are stripped first: they describe the setting, not its result. Without this
    the audit reports false positives, because validation RMSE lands in the same
    0.2-0.5 range as the dropout grid, so a score of 0.2504 "matches" the
    dropout value 0.25 at every rounding.
    """
    text = re.sub(r"[A-Za-z_]\w*\s*=\s*-?\d+\.?\d*(?:[eE][-+]?\d+)?", " ", text or "")
    out = []
    for tok in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?", text):
        try:
            out.append(float(tok))
        except ValueError:
            pass
    return out


def _appears(value: Any, payload_numbers: List[float]) -> bool:
    """Is `value` present in the payload as a number, at any sane rounding?

    A score of 0.23084 counts as shown if the payload contains 0.23, 0.231 or
    0.2308 — the check is about whether the magnitude survived, not about
    formatting. Comparison is by SIGNIFICANT digits, not decimal places:
    rounding to 2 dp would make every quantity below 0.005 collide with 0.0 and
    report spurious matches for small numbers such as val_rmse_std.
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return False
    if v != v or v in (float("inf"), float("-inf")) or v == 0.0:
        return False
    for n in payload_numbers:
        if n == 0.0:
            continue
        for nd in (2, 3, 4):
            if f"{v:.{nd}g}" == f"{n:.{nd}g}":
                return True
    return False


def _block_sizes(payload: str) -> Dict[str, int]:
    """Character count per payload section (in emission order)."""
    if not payload:
        return {}
    positions: List[Tuple[int, str]] = []
    for name in _BLOCKS:
        m = re.search(r"==\s*" + re.escape(name), payload)
        if m:
            positions.append((m.start(), name))
    positions.sort()
    sizes: Dict[str, int] = {}
    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(payload)
        sizes[name] = end - start
    return sizes


def audit_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
    """One attempt → one audit row."""
    payload = entry.get("rendered_payload") or ""
    system  = entry.get("system_prompt")
    subs    = entry.get("sub_attempts") or []
    hist    = entry.get("history_snapshot") or []
    nums    = _numbers_in(payload)

    row: Dict[str, Any] = {
        "attempt":         entry.get("attempt_index"),
        "prompt_variant":  entry.get("prompt_variant", "unrecorded"),
        "history_ablation": entry.get("history_ablation", "n/a"),
        "n_history":       entry.get("n_history", len(hist)),
        "outcome":         entry.get("outcome"),
        "final_reason":    entry.get("final_reason", ""),
        "n_sub_attempts":  len(subs),
        "system_prompt_chars": len(system) if system else 0,
        "payload_chars":   len(payload),
        "total_chars":     (len(system) if system else 0) + len(payload),
        "payload_numbers": len(nums),
        "verbatim_system_prompt": bool(system),
        "verbatim_user_messages": all("user_message" in s for s in subs) if subs else False,
    }

    # Per-quantity retention: of the past attempts that HAD this number, in how
    # many did the number itself reach the prompt?
    for key, _desc, _shown in _QUANTITIES:
        have = shown = 0
        for h in hist:
            val = h.get(key)
            if isinstance(val, list):
                if not val:
                    continue
                have += 1
                # per-seed list: shown only if every element appears
                seed_vals = [v.get("score") if isinstance(v, dict) else v for v in val]
                if seed_vals and all(_appears(v, nums) for v in seed_vals):
                    shown += 1
                continue
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval != fval or fval in (float("inf"), float("-inf")):
                continue
            have += 1
            if _appears(fval, nums):
                shown += 1
        row[f"have_{key}"] = have
        row[f"shown_{key}"] = shown

    row.update({f"block_{k}_chars": v for k, v in _block_sizes(payload).items()})
    return row


def write_transcript(entries: List[Dict[str, Any]], out_path: Path,
                     source: Path, only: Optional[int] = None) -> None:
    """Verbatim markdown transcript — exactly what was sent and received."""
    lines: List[str] = [
        f"# LLM prompt transcript — {source}",
        "",
        "Verbatim record of every proposer call: the system prompt, the user",
        "message actually sent, and the raw reply. Nothing here is",
        "reconstructed or reformatted.",
        "",
    ]
    for e in entries:
        idx = e.get("attempt_index")
        if only is not None and idx != only:
            continue
        subs = e.get("sub_attempts") or []
        lines += [
            "---",
            "",
            f"## Attempt {idx} — {e.get('outcome', '?')}"
            + (f" ({e.get('final_reason')})" if e.get("final_reason") else ""),
            "",
            f"- prompt variant: `{e.get('prompt_variant', 'unrecorded')}`",
            f"- history_ablation: `{e.get('history_ablation', 'n/a')}`",
            f"- past attempts available: {e.get('n_history', 0)}",
            f"- sub-attempts (retries): {len(subs)}",
            "",
        ]
        system = e.get("system_prompt")
        lines += ["### System prompt", ""]
        if system:
            lines += ["```text", system.rstrip(), "```", ""]
        else:
            lines += ["> MISSING — this transcript predates the Email-8 "
                      "instrumentation (2026-07-26), which records the system "
                      "prompt verbatim.", ""]

        if not subs:
            lines += ["### User message", "", "```text",
                      (e.get("rendered_payload") or "").rstrip(), "```", ""]
        for s in subs:
            n = s.get("attempt", 0)
            label = "User message" if n == 0 else f"User message (retry {n})"
            msg = s.get("user_message")
            lines += [f"### {label}", ""]
            if msg is None:
                base = e.get("rendered_payload") or ""
                lines += ["> Verbatim message MISSING (pre-2026-07-26 "
                          "transcript). The base payload below was sent, plus "
                          "the rejection feedback for retries.", "",
                          "```text", base.rstrip(), "```", ""]
            else:
                lines += ["```text", msg.rstrip(), "```", ""]
            lines += ["### Raw reply", "",
                      "```text", (s.get("raw") or "(no reply / call failed)").rstrip(),
                      "```", "",
                      f"- status: `{s.get('status')}`",
                      f"- parsed changes: `{s.get('parsed_changes')}`",
                      ""]
        if e.get("resolved_setting"):
            lines += [f"- resolved setting: `{e.get('resolved_setting')}`", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")


def summarise(df: pd.DataFrame) -> str:
    """Printed summary of the information audit."""
    out: List[str] = []
    out.append(f"attempts audited            : {len(df)}")
    out.append(f"prompt variant(s)           : {', '.join(sorted(set(df['prompt_variant'])))}")
    out.append(f"verbatim system prompt      : {int(df['verbatim_system_prompt'].sum())}/{len(df)} attempts")
    out.append(f"verbatim user messages      : {int(df['verbatim_user_messages'].sum())}/{len(df)} attempts")
    out.append(f"prompt size (system+user)   : {df['total_chars'].mean():.0f} chars mean, "
               f"{df['total_chars'].max()} max")
    out.append(f"numbers present in payload  : {df['payload_numbers'].mean():.1f} mean per attempt")
    legacy = int((~df["verbatim_system_prompt"]).sum())
    if legacy:
        out.append(f"NOTE: {legacy}/{len(df)} attempts come from pre-2026-07-26 "
                   "transcripts, whose history snapshot stored only `score`. For "
                   "those, an 'available 0' below means the SNAPSHOT lacks the "
                   "field, not that the run lacked the measurement — the driver "
                   "always computes val_rmse_std and per_seed. Re-run to audit "
                   "those quantities.")
        out.append("")
    out.append("Numeric information reaching the prompt (over all past attempts referenced):")
    for key, desc, shown_as in _QUANTITIES:
        have, shown = int(df[f"have_{key}"].sum()), int(df[f"shown_{key}"].sum())
        pct = (100.0 * shown / have) if have else float("nan")
        out.append(f"  {desc:<34s} available {have:5d}  as a number {shown:5d} "
                   f"({pct:5.1f}%)   -> shown as: {shown_as}")
    if "condition" in df.columns and df["condition"].nunique() > 1:
        out.append("")
        out.append("Per condition (attempts | mean prompt chars | scores shown as numbers):")
        for cond, g in df.groupby("condition"):
            have, shown = int(g["have_score"].sum()), int(g["shown_score"].sum())
            out.append(f"  {cond:<34s} {len(g):5d} | {g['total_chars'].mean():8.0f} | "
                       f"{shown}/{have}")
    blocks = [c for c in df.columns if c.startswith("block_")]
    if blocks:
        out.append("")
        out.append("Payload composition (mean chars per block):")
        for c in sorted(blocks, key=lambda c: -df[c].fillna(0).mean()):
            name = c[len("block_"):-len("_chars")]
            out.append(f"  {name:<30s} {df[c].fillna(0).mean():8.0f}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True,
                    help="run dir containing protocol_log_run*.json (flat or seed_*/)")
    ap.add_argument("--out", default="analysis/prompt_audit",
                    help="output dir for the transcript + audit CSV")
    ap.add_argument("--attempt", type=int, default=None,
                    help="restrict the transcript to one attempt index")
    ap.add_argument("--seed", default=None,
                    help="restrict to one seed_<N> subdir")
    args = ap.parse_args()

    run_dir = Path(args.run)
    if not run_dir.is_absolute():
        cand = _REPO_ROOT / run_dir
        run_dir = cand if cand.exists() else run_dir
    if not run_dir.exists():
        raise SystemExit(f"run dir not found: {run_dir}")

    logs = _find_logs(run_dir)
    if args.seed:
        logs = [p for p in logs if f"seed_{args.seed}" in str(p)]
    if not logs:
        raise SystemExit(f"no protocol_log_run*.json under {run_dir}")

    out_dir = Path(args.out)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    for log in logs:
        entries = json.loads(log.read_text(encoding="utf-8"))
        if not isinstance(entries, list):
            print(f"  skipped (unexpected format): {log}")
            continue
        seed = next((p for p in log.parts if p.startswith("seed_")), "flat")
        cond = _condition_of(log, run_dir)
        stem = f"{cond.replace(os.sep, '-')}__{seed}__{log.stem}"
        t_path = out_dir / f"transcript__{stem}.md"
        write_transcript(entries, t_path, log, only=args.attempt)
        rows = [audit_entry(e) for e in entries]
        for r in rows:
            r["source"] = str(log.relative_to(run_dir.parent)) if run_dir.parent in log.parents else str(log)
            r["condition"] = cond
            r["seed"] = seed
        all_rows.extend(rows)
        print(f"  {cond} / {seed}  ->  {t_path.name}  ({len(entries)} attempts)")

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / f"prompt_information_audit__{run_dir.name}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nAudit CSV -> {csv_path}\n")
    print(summarise(df))


if __name__ == "__main__":
    main()

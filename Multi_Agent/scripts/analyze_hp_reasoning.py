"""Depth-of-reasoning analysis for the Part A generic-HP LLM arm (Email-7 point 4).

The Part B counterpart is `analyze_motion_reasoning.py`; this reports the SAME
five things for the 9-hyperparameter search so Part A and Part B are directly
comparable:
  * distribution of diagnoses,
  * distribution of proposed actions (which HP is moved, and in which direction),
  * whether different diagnoses map to different actions,
  * whether the stated reason matches the proposed change (keyword grounding),
  * whether the proposed change improves the running best score.

This separates "follows the protocol" (clean, in-grid, no repeats) from "reasons
well" (diagnosis varies, drives the action, reason matches change, change helps).

Usage:
    python scripts/analyze_hp_reasoning.py \
        --llm results/history-use/history-none-qwen3 \
        --label qwen3-real --out analysis/reasoning_partA_qwen3
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

HPS = ["learning_rate", "weight_decay", "dropout", "batch_size", "window_size",
       "optimizer_choice", "patience", "lstm_hidden", "lstm_layers"]

# Keywords that count as "the reason talks about this hyperparameter".
REASON_KW = {
    "learning_rate":    ["learning rate", "learning-rate", "lr "],
    "weight_decay":     ["weight decay", "weight-decay", "regulari", "l2"],
    "dropout":          ["dropout"],
    "batch_size":       ["batch"],
    "window_size":      ["window", "sequence length", "seq len", "context length",
                         "temporal context"],
    "optimizer_choice": ["optimizer", "adamw", "adam"],
    "patience":         ["patience", "early stop", "early-stop"],
    "lstm_hidden":      ["hidden", "capacity", "units", "width", "size of the lstm"],
    "lstm_layers":      ["layer", "depth", "deeper", "shallower"],
}


def parse_reply(raw: str) -> dict:
    """Pull the protocol's 5 labelled lines out of a raw LLM reply."""
    out = {}
    for ln in (raw or "").splitlines():
        m = re.match(r"\s*(diagnosis|strategy|changes|reason|confidence)\s*:\s*(.*)",
                     ln, re.IGNORECASE)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def actions_from_changes(changes: dict, anchor: dict) -> list:
    """Describe a proposal as <hp>_up / <hp>_down (or _switch for categoricals)."""
    acts = []
    for k, v in (changes or {}).items():
        if k not in HPS:
            continue
        base = (anchor or {}).get(k)
        nv, bv = _num(v), _num(base)
        if nv is not None and bv is not None:
            if nv > bv:
                acts.append(f"{k}_up")
            elif nv < bv:
                acts.append(f"{k}_down")
        else:
            acts.append(f"{k}_switch")
    return acts or ["no_change"]


def reason_mentions_changed_hp(reason: str, changes: dict) -> bool:
    """True if the stated reason names at least one HP it actually moved."""
    r = (reason or "").lower()
    touched = [k for k in (changes or {}) if k in HPS]
    if not touched:
        return False
    for k in touched:
        if any(kw in r for kw in REASON_KW.get(k, [k])):
            return True
    return False


def collect(llm_dir: Path):
    diag_dist = Counter()
    action_dist = Counter()
    diag_to_actions = defaultdict(Counter)
    reason_match = {"match": 0, "mismatch": 0, "no_hp": 0}
    improve = {"improved": 0, "not_improved": 0}
    n_accepted = 0
    n_seeds = 0

    for plog_f in sorted(glob.glob(str(llm_dir / "seed_*" / "protocol_log_run*.json"))):
        seed_dir = Path(plog_f).parent
        plog = json.loads(Path(plog_f).read_text(encoding="utf-8"))
        n_seeds += 1

        # Scores per applied round -> "did it improve the running best?"
        olog_f = next(iter(sorted(glob.glob(str(seed_dir / "optimization_log_run*.json")))), None)
        rounds = []
        if olog_f:
            o = json.loads(Path(olog_f).read_text(encoding="utf-8"))
            rounds = o.get("rounds", []) if isinstance(o, dict) else []
        score_by_round = {r.get("round"): r.get("val_rmse") for r in rounds}

        best = None
        for entry in plog:
            if entry.get("outcome") != "accepted":
                continue
            subs = entry.get("sub_attempts") or []
            if not subs:
                continue
            sub = subs[-1]                      # the accepted attempt
            changes = sub.get("parsed_changes") or {}
            anchor = entry.get("anchor") or {}
            parsed = parse_reply(sub.get("raw", ""))
            diag = (sub.get("diagnosis") or parsed.get("diagnosis") or "unknown").strip().lower()

            n_accepted += 1
            diag_dist[diag] += 1
            acts = actions_from_changes(changes, anchor)
            for a in acts:
                action_dist[a] += 1
                diag_to_actions[diag][a] += 1

            if not [k for k in changes if k in HPS]:
                reason_match["no_hp"] += 1
            elif reason_mentions_changed_hp(parsed.get("reason", ""), changes):
                reason_match["match"] += 1
            else:
                reason_match["mismatch"] += 1

            sc = score_by_round.get(entry.get("attempt_index"))
            if sc is not None:
                if best is None or sc < best:
                    improve["improved"] += 1
                    best = sc
                else:
                    improve["not_improved"] += 1

    return dict(diag_dist=diag_dist, action_dist=action_dist,
                diag_to_actions=diag_to_actions, reason_match=reason_match,
                improve=improve, n_accepted=n_accepted, n_seeds=n_seeds)


def render(label: str, r: dict) -> str:
    n = r["n_accepted"]
    pct = lambda k, d: f"{100.0 * k / d:.0f}%" if d else "n/a"
    L = [f"# Depth-of-reasoning analysis — {label}, Part A generic HP search (Email-7 point 4)\n",
         f"_{n} accepted proposals over {r['n_seeds']} search seeds._\n",
         "## 1. Diagnosis distribution\n"]
    for k, v in r["diag_dist"].most_common():
        L.append(f"- {k}: {v} ({pct(v, n)})")
    if r["diag_dist"]:
        top, tv = r["diag_dist"].most_common(1)[0]
        verdict = ("near-constant label (little diagnostic variation)"
                   if tv / n >= 0.75 else "some variation")
        L.append(f"\n**{top}** covers {pct(tv, n)} of accepted proposals → {verdict}.")

    L.append("\n## 2. Proposed-action distribution\n")
    for k, v in r["action_dist"].most_common():
        L.append(f"- {k}: {v}")

    L.append("\n## 3. Do different diagnoses lead to different actions?\n")
    L.append("| diagnosis | top actions |")
    L.append("|---|---|")
    for d, cnt in sorted(r["diag_to_actions"].items(),
                         key=lambda kv: -r["diag_dist"][kv[0]]):
        tops = ", ".join(f"{a} {c}" for a, c in cnt.most_common(3))
        L.append(f"| {d} ({r['diag_dist'][d]}) | {tops} |")

    rm = r["reason_match"]
    tot = rm["match"] + rm["mismatch"]
    L.append("\n## 4. Does the stated reason match the proposed change?\n")
    L.append(f"- reason names an HP it actually moved: {rm['match']}/{tot} ({pct(rm['match'], tot)})")
    L.append(f"- no HP change to check: {rm['no_hp']}")

    im = r["improve"]
    tot2 = im["improved"] + im["not_improved"]
    L.append("\n## 5. Does the proposed change improve the running best?\n")
    L.append(f"- improved best-so-far: {im['improved']}/{tot2} ({pct(im['improved'], tot2)})")
    L.append(f"- did not improve: {im['not_improved']}/{tot2}")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True, help="Part A run dir (seed_*/protocol_log_run*.json)")
    ap.add_argument("--label", default="LLM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    r = collect(Path(args.llm))
    if not r["n_accepted"]:
        print(f"No accepted proposals found under {args.llm}")
        return
    text = render(args.label, r)
    out_dir = Path(args.out) if args.out else Path("analysis") / f"reasoning_partA_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reasoning_depth_partA.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"wrote {out_dir / 'reasoning_depth_partA.md'}")


if __name__ == "__main__":
    main()

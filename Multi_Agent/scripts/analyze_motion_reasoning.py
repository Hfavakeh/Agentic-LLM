"""Depth-of-reasoning analysis for the motion LLM arm (Email-7 point 4).

For a model's motion run it reports, across all accepted proposals:
  * distribution of diagnoses,
  * distribution of proposed actions (which loss knobs are engaged),
  * whether different diagnoses map to different actions,
  * whether the stated reason matches the proposed change (keyword grounding),
  * whether the proposed change improves the running best score.

This separates "follows the protocol" (clean, in-grid, no repeats) from "reasons
well" (diagnosis varies, drives the action, reason matches change, change helps).

Usage:
    python analyze_motion_reasoning.py --llm motion-qwen3 --label qwen3 --out analysis/motion_qwen3_full
"""

import argparse
import glob
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

LEVERS = ["v_max", "lambda_vel", "lambda_smooth",
          "bin_weight_slow", "bin_weight_medium", "bin_weight_fast"]

REASON_KW = {
    "fast":   ["fast"],
    "medium": ["medium", "mid-speed", "mid speed", "moderate"],
    "slow":   ["slow", "stationary", "low-speed", "low speed"],
    "vel":    ["speed", "velocity", "v_max", "plausib", "outlier", "implausib"],
    "smooth": ["smooth", "jerk", "accel"],
}


def parse_reply(raw: str) -> dict:
    out = {}
    for ln in (raw or "").splitlines():
        m = re.match(r"\s*(diagnosis|strategy|changes|reason|confidence)\s*:\s*(.*)",
                     ln, re.IGNORECASE)
        if m:
            out[m.group(1).lower()] = m.group(2).strip()
    return out


def actions_from_levers(lv: dict) -> list:
    """Salient 'actions' a lever vector represents (relative to neutral MSE)."""
    acts = []
    try:
        s, m, f = float(lv["bin_weight_slow"]), float(lv["bin_weight_medium"]), float(lv["bin_weight_fast"])
        if f > 1 and f >= m and f >= s: acts.append("upweight_fast")
        if m > 1 and m >= f and m >= s: acts.append("upweight_medium")
        if s > 1 and s >= f and s >= m: acts.append("upweight_slow")
        if float(lv.get("lambda_vel", 0)) > 0: acts.append("velocity_penalty")
        if float(lv.get("lambda_smooth", 0)) > 0: acts.append("smoothness_penalty")
    except (KeyError, ValueError, TypeError):
        pass
    return acts or ["near_neutral"]


def primary_regime(lv: dict):
    try:
        w = {"fast": float(lv["bin_weight_fast"]), "medium": float(lv["bin_weight_medium"]),
             "slow": float(lv["bin_weight_slow"])}
        top = max(w, key=w.get)
        return top if w[top] > 1 else None
    except (KeyError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    ap.add_argument("--label", default="LLM")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    llm_dir = Path(args.llm)
    diag_dist = Counter()
    action_dist = Counter()
    diag_to_actions = defaultdict(Counter)
    reason_match = {"match": 0, "mismatch": 0, "no_regime": 0}
    improve = {"improved": 0, "not_improved": 0}
    n_accepted = 0
    n_seeds = 0

    for run_f in sorted(glob.glob(str(llm_dir / "seed_*" / "motion_experiment_run1.json"))):
        seed_dir = Path(run_f).parent
        run = json.loads(Path(run_f).read_text(encoding="utf-8"))
        plog_f = seed_dir / "motion_protocol_log_run1.json"
        plog = json.loads(plog_f.read_text(encoding="utf-8")) if plog_f.exists() else []
        n_seeds += 1

        # attempt index -> parsed reply (diagnosis/reason) from the protocol log
        parsed_by_attempt = {}
        for e in plog:
            if e.get("outcome") != "accepted":
                continue
            raw = ""
            for sa in e.get("sub_attempts", []):
                if sa.get("status") == "ok" and sa.get("raw"):
                    raw = sa["raw"]
            if not raw and e.get("sub_attempts"):
                raw = e["sub_attempts"][-1].get("raw", "")
            parsed_by_attempt[e.get("attempt_index")] = parse_reply(raw)

        # running-best over the accepted attempts (scores live in the run json)
        best = float("inf")
        attempts = run["arms"]["llm"]["attempts"]
        acc_i = 0
        for att in attempts:
            if att.get("output_status") == "rejected":
                continue
            acc_i += 1
            n_accepted += 1
            lv = att.get("setting", {})
            score = att.get("score", float("inf"))
            parsed = parsed_by_attempt.get(acc_i, {})
            diag = (parsed.get("diagnosis") or "unknown").lower()
            diag_dist[diag] += 1
            acts = actions_from_levers(lv)
            for a in acts:
                action_dist[a] += 1
                diag_to_actions[diag][a] += 1

            # reason grounding: does the reason mention the up-weighted regime?
            reg = primary_regime(lv)
            reason = (parsed.get("reason") or "").lower()
            if reg is None:
                reason_match["no_regime"] += 1
            elif any(kw in reason for kw in REASON_KW[reg]):
                reason_match["match"] += 1
            else:
                reason_match["mismatch"] += 1

            # does this change improve the running best?
            if isinstance(score, (int, float)) and score == score:
                if score < best:
                    improve["improved"] += 1
                    best = score
                else:
                    improve["not_improved"] += 1

    # ---- report ----
    L = [f"# Depth-of-reasoning analysis — {args.label} (Email-7 point 4)\n",
         f"_{n_accepted} accepted proposals over {n_seeds} search seeds._\n"]

    top_diag, top_n = diag_dist.most_common(1)[0]
    L.append("## 1. Diagnosis distribution\n")
    for d, c in diag_dist.most_common():
        L.append(f"- {d}: {c} ({100*c/n_accepted:.0f}%)")
    L.append(f"\n**{top_diag}** covers {100*top_n/n_accepted:.0f}% of accepted proposals "
             f"→ {'near-constant label (little diagnostic variation)' if top_n/n_accepted > 0.8 else 'some variation'}.\n")

    L.append("## 2. Proposed-action distribution\n")
    for a, c in action_dist.most_common():
        L.append(f"- {a}: {c}")
    L.append("")

    L.append("## 3. Do different diagnoses lead to different actions?\n")
    if len([d for d in diag_dist if diag_dist[d] >= 3]) <= 1:
        L.append("Only one diagnosis is used with any frequency, so actions cannot be "
                 "conditioned on the diagnosis — the diagnosis does **not** drive the action.\n")
    else:
        L.append("| diagnosis | top actions |")
        L.append("|---|---|")
        for d, c in diag_dist.most_common():
            if c < 3:
                continue
            tops = ", ".join(f"{a} {n}" for a, n in diag_to_actions[d].most_common(3))
            L.append(f"| {d} ({c}) | {tops} |")
        L.append("")

    L.append("## 4. Does the stated reason match the proposed change?\n")
    graded = reason_match["match"] + reason_match["mismatch"]
    rate = (100 * reason_match["match"] / graded) if graded else 0
    L.append(f"- reason mentions the up-weighted regime: {reason_match['match']}/{graded} "
             f"({rate:.0f}%)")
    L.append(f"- no clear regime up-weight to check: {reason_match['no_regime']}\n")

    L.append("## 5. Does the proposed change improve the running best?\n")
    tot = improve["improved"] + improve["not_improved"]
    L.append(f"- improved best-so-far: {improve['improved']}/{tot} "
             f"({100*improve['improved']/tot:.0f}%)")
    L.append(f"- did not improve: {improve['not_improved']}/{tot}\n")

    out_dir = Path(args.out) if args.out else Path("analysis") / f"motion_{args.label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reasoning_depth.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out_dir/'reasoning_depth.md'}")


if __name__ == "__main__":
    main()

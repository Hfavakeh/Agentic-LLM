# Experiment Plan — Information Sufficiency for the LLM Optimizer

**Context:** Response to Prof. Mihai Lazarescu's reply (Email-4) to the last report. The professor accepted the cause-level analysis and the baseline explanation (manual tuning beat rule-based because manual tuning used *per-epoch curve information* while rule-based saw only post-hoc aggregate labels), but raised one pivotal challenge:

> *If rule-based lacked the right information, does the LLM really receive the right information?*

A valid, well-formatted proposal is **not** the same as a good optimization decision. Passing output validation only proves a stronger model can follow instructions — it does not prove the prompt is optimal, the task unambiguous, or that the LLM understood the problem. This work is a **prerequisite gate** before starting the main thesis direction (human-motion information: speed, acceleration, turning, stop-go, trajectory roughness, errors per motion regime).

The professor's five questions:

1. What info does the LLM actually need to improve its decisions?
2. Do per-epoch curve summaries improve the LLM — or the rule-based controller?
3. Does the LLM actually use history, or mostly ignore it?
4. Why does the LLM explore poorly — prompt, model, or search space?
5. Do motion-aware summaries change anything? What, and how?

---

## 1. The reframe: a payload-richness ablation

All five questions reduce to one axis the current code holds **fixed** — *how much information the proposer receives*. Both the LLM and rule-based arms currently get an identical, deliberately information-poor payload:

- `arms/prompts.py` → `format_protocol_payload` renders **only qualitative labels** (best/good/poor, low/med/high, small/med/large); raw numbers are stripped by design.
- `arms/engine.py` → `evaluate_setting` collapses the full `trainer.history` per-epoch curves to **scalars** (best-epoch value, mean over 3 seeds). The validation/gap *trajectory* is computed and then discarded.
- `format_protocol_payload` takes no `motion_profile`, so motion is **absent** from the protocol prompt even though `engine.py` (`build_dataset_and_loaders`) already computes `dataset["motion_profile"]`.

**Design:** hold the search machinery fixed; vary what the proposer sees; measure whether decision quality changes — for *both* the LLM and rule-based arms (the professor explicitly wants that comparison in Q2). This single factorial answers all five questions.

### Payload variants

A `Config` flag set (independent booleans, not one ordinal level, so a clean factorial is possible). Each gates a branch in a refactored `format_protocol_payload`:

| Variant | Flag | Added vs P0 | Targets |
|---|---|---|---|
| **P0** baseline | (default) | current qualitative labels only | control |
| **P1** +curves | `payload_curves` | per-epoch val-RMSE/gap **trajectory summary** per setting (shape: monotone↓ / plateau@k / diverging / noisy; epochs-to-best) | **Q2** |
| **P2** +raw | `payload_raw_numbers` | actual RMSE values + std, not just labels | **Q1** |
| **P3** +motion | `payload_motion` | the `motion_profile` block (speed distribution, dwell / stop-go) | **Q5** |

P1 requires surfacing curves: `evaluate_setting` must add a `val_curve` / `gap_curve` summary to each `per_seed` record and aggregate it into the result; the driver `rec` (`arms/driver.py`) must carry it into `history`.

---

## 2. Step 0 — Prerequisite instrumentation (blocks Q1, Q3, Q4)

A logging gap makes the questions unanswerable today: the **protocol path** (`SingleAgentOptimizer.propose_setting`, `arms/llm.py`) returns `"raw"` but **never appends to `conversation_log`** — only the deprecated warm-loop path does. We cannot ask "does the LLM use history" without storing what history it saw and what it replied.

**Change:** in `propose_setting`, append a record per attempt capturing:
- the **exact rendered payload** (`base_user`),
- the raw reply (`raw_last`),
- the parsed proposal (`last_parsed`),
- the anchor setting,
- a snapshot of the current `history` (settings + scores).

Add a `save_protocol_log()` mirror of `save_conversation_log`. ~30 lines; foundation for everything below.

---

## 3. Per-question specs

### Q3 — Does the LLM use history, or ignore it?  *(FIRST experiment)*

Cheapest test, no new payload required, and it settles the most fundamental question — whether the model reads its input at all — before we invest in richer payloads.

**A causal placebo probe.** Feed the *same* model three history conditions:
- **(a) real** history,
- **(b) shuffled** — settings ↔ scores permuted so the "best so far" / ANCHOR and the `OBSERVED PATTERNS` block are wrong,
- **(c) empty** history.

If proposals do **not** degrade under (b)/(c), the LLM is ignoring history.

**Implementation:** a `history_ablation ∈ {none, shuffled, empty}` flag, applied as a transform on `history` just before `format_protocol_payload` in `propose_setting`. (Engine training is unaffected; only the *rendered context* is perturbed.)

**Metrics** (from Step-0 logs):
- proposal-vs-history consistency: does the proposal avoid values flagged bad in `OBSERVED PATTERNS`?
- diagnosis fidelity: does the LLM's `diagnosis` line match the rendered `behavior label`?
- repeat-rate (`repeats_proposed`, already tracked in `retry_stats`).
- final outcome: best val-RMSE under each condition.

**Expected discriminating result:** real ≫ shuffled in consistency/quality ⇒ history is used; real ≈ shuffled ≈ empty ⇒ history ignored (the professor's suspicion confirmed, and a finding in its own right).

### Q2 — Do per-epoch curve summaries help the LLM or the rule-based controller?

P0 vs P1 for **both** arms (the professor's exact phrasing). The rule-based controller's `_diagnose_protocol` currently reads aggregate labels; add a curve-aware diagnosis branch so it can exploit the same trajectory summary. **2×2 design** (arm × curves).

**Metrics:** best val-RMSE during search + final test RMSE (fresh seeds). Directly tests his hypothesis that manual tuning beat rule-based *because* of per-epoch curves: if P1 lifts rule-based toward manual-tuning quality, the hypothesis holds.

### Q1 — What info does the LLM actually need?

Answered by the **whole factorial**, not one run. Conditions: P0, P2 (+raw), P1 (+curves), all-on. Metric = best final-eval test RMSE per condition (existing `run_final_evaluation`, fresh seeds 201+). The variant that moves the headline metric names the information that mattered. Secondary placebo: **raw numbers but scrambled labels**, to separate *content* from merely *more tokens*.

### Q4 — Why does the LLM explore poorly: prompt, model, or search space?

Decompose with three manipulations:
- **Search space:** compare LLM grid-coverage to the random / Optuna arms over the same 25 attempts (per-HP value coverage, unique settings / 25).
- **Model:** re-run the best payload across the model ladder already in the sweeps (qwen3 4b / 8b / …).
- **Prompt:** P0 vs an explicit "explore an untried region" instruction.

**Metrics:** unique-settings count, per-HP value coverage, mean edit-distance of proposed deltas, fraction of attempts touching only the anchor neighborhood.

### Q5 — Do motion-aware summaries change anything? What, how?

P0 vs P3.

**Metrics:** headline test RMSE **plus** per-motion-regime error breakdown (slow / medium / fast bins from `motion_descriptors.py`) — does motion info specifically reduce error in the hard regimes? Report *what* changed and *how*, since this bridges to the main thesis direction.

---

## 4. Unifying design & budget

**Core factorial:** `{P0, P1, P2, P3, all-on} × {LLM, rule-based} × {radar, cap, IR}`, each 25 attempts × 3 train seeds, then final eval on 30 fresh seeds. Plus the Q3 placebo (3 history conditions, LLM-only) and the Q4 model-ladder (LLM-only). Random + Optuna run once per dataset as fixed reference lines.

**Sequencing:**
1. **Step 0** — instrumentation (protocol-path payload/reply logging).
2. **Q3 placebo** — cheapest; settles "does it read the input at all" first.
3. **Q2 curves** — the professor's headline hypothesis.
4. **Q1 / Q4** — fall out of the factorial.
5. **Q5 motion** — last; bridges to the main thesis.

---

## 5. Code touch-points (summary)

| Area | File / function | Change |
|---|---|---|
| Protocol logging | `arms/llm.py` `propose_setting` | append per-attempt record (payload, raw, parsed, anchor, history snapshot); add `save_protocol_log()` |
| History placebo | `arms/llm.py` `propose_setting` | `history_ablation ∈ {none, shuffled, empty}` transform before `format_protocol_payload` |
| Payload variants | `arms/prompts.py` `format_protocol_payload`, `arms/labels.py` | branch on `payload_curves` / `payload_raw_numbers` / `payload_motion` |
| Surface curves | `arms/engine.py` `evaluate_setting` | add `val_curve` / `gap_curve` summary to `per_seed` + aggregate |
| Carry curves | `arms/driver.py` `run_proposer_search` | include curve summary in `history` `rec` |
| Curve-aware rule-based | `arms/rule_based.py` `_diagnose_protocol` | optional curve branch (Q2) |
| Config flags | `pipeline/config.py` `Config` | `payload_curves`, `payload_raw_numbers`, `payload_motion`, `history_ablation` |
| Exploration metrics | reporting / `analyze_llm_behavior.py` | unique-settings, per-HP coverage, edit-distance, anchor-locality |

---

*Generated as the planning response to Email-4; first experiment to build is the Q3 history-use placebo (Section 3, Q3).*

# Q2 — Do per-epoch training-curve summaries improve the LLM, or the rule-based controller?

*Information-sufficiency study (Email-4 gate), question 2. Dataset: radar
(`preprocessed-RadarEXP1`). Search budget: 25 attempts/run, each setting trained
3× from scratch (seeds 101/102/103); final test RMSE re-evaluated on 30 fresh
seeds. 10 search seeds per condition.*

## 1. The question

On the previous report we argued the rule-based controller could not reproduce
manual tuning because it only saw **post-hoc aggregate labels** (final score,
gap, best-epoch timing), whereas a human read the **per-epoch validation
curve**. Q2 tests that claim directly: if we add a per-epoch curve summary to the
information each method sees, does it improve the LLM, the rule-based controller,
or both?

## 2. Method — a per-epoch curve-shape signal, A/B tested

Each tried setting already reports aggregate qualitative labels (validation
quality, reliability, train/val gap, best-epoch timing). We add **one more
qualitative label: the shape of the validation curve over epochs**, derived from
the metres validation trajectory (the same metric the score uses):

| Curve shape | Meaning | Suggested action |
|---|---|---|
| **still improving** | hit the epoch budget with the best epoch at the tail (curve still descending) | more capacity / higher learning rate |
| **overfitting upturn** | validation rose again after reaching its minimum | add regularization |
| **unstable / noisy** | the curve oscillated epoch-to-epoch | lower LR / raise batch size |
| **clean plateau** | converged and flattened | (no extra signal beyond the aggregates) |

This shape captures what the best-epoch scalars cannot: the *trajectory* —
whether training was truncated mid-descent, whether validation turned back up
over epochs, and whether it oscillated.

We A/B it with a single lever (`--payload-curves`):
- **P0 (curves OFF):** the existing aggregate-only payload — these are the
  Q3 `none` runs (real history, curves off = the defaults).
- **P1 (curves ON):** the same, plus the curve-shape label per setting. The LLM
  payload shows the label and the system prompt explains how to use it; the
  **rule-based controller diagnoses from the curve shape** (still improving →
  underfitting, upturn → overfitting, noisy → unstable). The curve-aware
  diagnosis is the *only* change to the rule-based arm, so any difference in its
  result is attributable to the curve information alone.

Both arms run in every experiment, so one P0 run and one P1 run per model yield
the full 2×2 (arm × curves). Models: nemotron-3-nano (4B), llama3.1 (8B), phi4
(14B); 10 seeds each.

## 3. Results

![Q2 curves OFF vs ON, per arm](../analysis/q2_curves/q2_curves.png)

| Arm | curves OFF → ON | beats baseline (0.231)? |
|---|---|---|
| **Rule-based** (all 3 models, identical) | 0.2371 → **0.2264** (−0.011) | OFF: no → **ON: yes** |
| LLM — llama3:8b | 0.2453 → 0.2356 (−0.010) | no → no |
| LLM — nemotron-3-nano:4b | 0.2487 → 0.2427 (−0.006) | no → no |
| LLM — phi4:14b | 0.2440 → 0.2364 (−0.008) | no → no |

All three LLM pairs are clean A/Bs (same model on both sides).

LLM exploration (distinct settings evaluated / 25) **shrank** with curves on in
every model: llama 9.1 → 3.7, phi4 13.9 → 4.0, nemotron 24.2 → 21.8.

## 4. Interpretation

**(a) Per-epoch curves clearly help the rule-based controller — and it becomes
the only arm that beats the baseline.** The rule-based RMSE is identical across
all three pairs because the controller is deterministic and does not depend on
the LLM model; adding the curve signal moves it 0.2371 → 0.2264, crossing from
*below* to *above* the fixed baseline (0.231). With curves on, the deterministic
rule-based controller is the single best arm, beating every LLM condition. This
is the direct confirmation of the hypothesis from the last report: the per-epoch
curve was the missing ingredient: give it to the controller and the gap to
manual-style tuning closes.

**(b) Curves give the LLM a smaller, consistent improvement — by making it
explore less.** All three LLMs improve by ~0.006–0.010, and all three sharply
reduce exploration (distinct settings collapse, most strongly for llama and
phi4). The LLM treats the curve diagnosis as a strong instruction and converges
quickly onto a few settings — fewer, but better targeted. That the largest model
(phi4 14B) shows the same pattern as the others indicates this is a general
effect of the information, not a model-specific quirk.

**(c) The LLM extracts no more value from the curve than a fixed rule does.** The
deep result, tying Q2 back to Q3: with curves on, the LLM (0.236–0.243) is still
worse than the baseline *and* worse than the curve-aware rule-based controller
(0.226). Feeding the same per-epoch curve signal to a simple shape→action lookup
beats feeding it to any of the three LLMs. Richer information helps — but routing
it through the LLM's reasoning adds nothing over a deterministic mapping. This is
consistent with Q3: the LLM follows instructions and reacts to the information it
is given, but does not convert that information into better optimization
decisions than a trivial controller using the same information.

## 5. Honest caveats

- **Rule-based RMSE is deterministic** (fixed training seeds + a deterministic
  controller), so its 0.2371 / 0.2264 values have little seed variance — read the
  −0.011 improvement as a tight point comparison. The LLM values carry genuine
  10-seed variance.
- **All LLM pairs are clean A/Bs** — each uses the same model on the curves-off
  and curves-on side (llama runs are all llama3:8b, nemotron all 4B, phi4 all
  14B) — so the OFF→ON deltas are attributable to the curve information.
- **Single dataset (radar).** Confirm on capacitive and IR before claiming full
  generality.
- The curve-shape labels are derived from heuristic thresholds on the metres
  validation curve (upturn ratio, oscillation, still-improving-at-budget); the
  qualitative shape, not raw numbers, is what each method sees.

## 6. Conclusion

Per-epoch curve summaries **do** improve both methods — but they help the
**rule-based controller** most, and only the curve-aware rule-based controller
beats the fixed baseline. The LLM improves modestly and becomes more decisive
(less exploration), yet still trails both the baseline and the curve-aware rule.
The information was the bottleneck (consistent with Q3), but the LLM is not the
best consumer of it: a deterministic rule using the same curve signal does
better. Next: Q4 (why the LLM explores poorly) is now largely answered across
Q2–Q3 — it is not model capability, and even with the right information the LLM
adds no value over a simple rule — leaving the prompt / search-space framing and
Q5 (motion-aware summaries) as the remaining levers.

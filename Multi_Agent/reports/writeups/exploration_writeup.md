# Q4 — Why does the LLM explore poorly: prompt, model, or search space?

*Information-sufficiency study (Email-4 gate), question 4. Dataset: radar
(`preprocessed-RadarEXP1`). 25 attempts/run, 10 seeds. This question is answered
from the existing curves-off, real-history runs — no new training was needed.*

## 1. The question

Across Q3 we saw the LLM evaluate few distinct settings and waste much of its
budget re-proposing already-tried ones. Q4 asks *why*: is the limitation the
**search space** (too large / awkward to navigate), the **model** (not capable
enough), or the **prompt** (how we ask)?

## 2. Method

Every run already contains four search arms (LLM, random, Optuna, rule-based),
all drawing from the **same discrete grid**. We measure, per arm, how much of
that grid it actually explores in its 25 attempts:

- **grid coverage** — for each of the 9 hyperparameters, the fraction of its
  allowed grid values that the arm tried, averaged over the 9 (1.0 = touched
  every value of every HP).
- **distinct settings** — how many different configurations it evaluated (out of
  25).

random, Optuna and the rule-based controller do not depend on the LLM, so they
are averaged across all runs and act as reference points. The LLM is shown per
model (llama3:8b, nemotron-3-nano 4B, phi4 14B). If random/Optuna cover the grid
easily, the space is not the limiter; if every LLM falls short regardless of
size, the cause is shared — i.e. the prompt.

## 3. Results

![Q4 grid coverage and distinct settings per arm](../analysis/q4_exploration/q4_exploration.png)

| Arm | distinct / 25 | grid coverage |
|---|---|---|
| random | 25 | **0.99** |
| Optuna | 19 | **1.00** |
| rule-based | 25 | 0.61 |
| LLM — phi4:14b | 13.9 | 0.51 |
| LLM — llama3:8b | 9.1 | 0.41 |
| LLM — nemotron-3-nano:4b | 24.2 | 0.61 |

## 4. Interpretation — the three causes, resolved

**Search space — ruled out.** Random and Optuna cover **~100% of the grid in 25
attempts**. The space is small and easily navigable; it is not what holds the LLM
back.

**Model — partial, and non-monotonic.** LLM exploration breadth depends on the
model but does **not** increase with size: nemotron (4B) reaches 0.61 coverage —
matching the rule-based controller — while llama (8B) manages only 0.41 and phi4
(14B) 0.51. So a bigger / stronger model is not a broader searcher. And from Q3,
even the best LLM explorer (nemotron) does not beat the baseline — so the model
governs *how much* gets explored, not whether the decisions are good. Capability
is not the binding constraint.

**Prompt — the primary driver.** Every LLM, regardless of size, is capped well
below random/Optuna (best LLM 0.61 vs ~1.0). The one thing they all share is the
instruction: *"propose a SMALL change relative to the best setting so far
(ANCHOR)."* That anchoring confines proposals to the neighbourhood of best-so-far,
whereas random/Optuna have no anchor and sweep the whole grid. Q2 reinforces this
— adding per-epoch curve information made every LLM explore **even less**
(converging harder onto the anchor). The prompt's "small delta vs anchor" framing
is the main cause of the LLM's narrow exploration.

**The deeper synthesis (Q2 + Q3 + Q4).** Poor exploration is a *symptom*, not the
root limit. Even when exploration is broad (nemotron's 0.61 ≈ the rule-based
controller's) or the information is rich (curves on), the LLM still does not beat
a deterministic rule or the fixed baseline. The binding constraint is not prompt,
model, or space taken alone — it is that **the LLM does not convert information or
exploration into better *selection*** than a trivial controller using the same
inputs. The prompt limits how widely it looks; but widening the search would not,
on this evidence, make it choose better.

## 5. Confirmation — the prompt-variant experiment

To move the prompt diagnosis from *inferred* to *tested*, we re-ran the LLM arm
with a variant system prompt (`--explore-prompt`) that drops the "small change vs
the anchor" instruction and instead asks the model to propose settings in
*untried* regions, changing several hyperparameters and making large moves. Same
models, same 10 seeds, everything else identical.

![Q4 prompt variant — coverage rises, RMSE stays flat](../analysis/q4_exploration/q4_prompt_variant.png)

| Model | grid coverage (default → explore) | test RMSE (default → explore) |
|---|---|---|
| llama3:8b | 0.41 → **0.68** (+0.27) | 0.245 → 0.242 (−0.003) |
| nemotron-3-nano:4b | 0.61 → **0.91** (+0.30) | 0.249 → 0.247 (−0.002) |
| phi4:14b | 0.51 → **0.92** (+0.41) | 0.244 → 0.245 (+0.001) |

Two results, the second the important one:

- **The prompt diagnosis is confirmed.** Dropping the anchor instruction lifts
  grid coverage sharply toward random's ~1.0 (nemotron and phi4 reach ~0.91–0.92;
  distinct settings rise to ~20–25). The "small change vs anchor" instruction was
  indeed what capped the LLM's exploration — not the model, not the search space.
- **But fixing exploration changed nothing about the outcome.** Despite exploring
  2–3× more of the grid and nearly matching random, test RMSE stays flat (within
  ±0.003) and still sits above the baseline (0.231) and well above the curve-aware
  rule-based controller (0.226). Letting the LLM look everywhere did not make it
  choose better.

This is the capstone of the study: **exploration was never the real bottleneck.**
The prompt genuinely constrained *how widely* the LLM searched, but removing that
constraint did not improve its decisions — consistent with Q2 and Q3, where richer
information and real history likewise failed to translate into better selection.

## 6. Honest caveats

- Coverage is averaged over the 9 HPs with per-HP coverage capped at 1.0; it
  measures breadth of values tried, not whether the *combinations* were
  well-chosen.
- random / Optuna explore broadly **by construction** — the comparison shows the
  space is navigable, not that broad search is the goal (broad but undirected
  search is exactly the random arm, which also does not dominate).
- The prompt-variant confirmation (§5) is solid on coverage but the "RMSE flat"
  half rests on fewer seeds for two models — the explore runs completed 10 seeds
  for phi4 but only 7–8 for llama3 / nemotron (and slightly fewer for the RMSE
  metric). The coverage jump is large and consistent; completing those runs to 10
  seeds would make the RMSE comparison airtight.
- Single dataset (radar).

## 7. Conclusion

The LLM explores poorly **because of the prompt, not the search space or model
capacity** — confirmed directly (§5): the "small change vs the anchor"
instruction keeps every model (4B–14B) confined to ≤0.61 grid coverage while
random/Optuna reach ~1.0, and swapping it for an exploration instruction lifts
coverage to 0.68–0.92. But the same experiment sharpens the overall finding:
exploration is not the real bottleneck — the extra coverage bought no better test
RMSE, and across Q2–Q4 the LLM fails to turn more information *or* more
exploration into better decisions than a simple rule. The binding constraint is
selection quality, not information, prompt, model, or search breadth. With the
prompt lever now tested, Q5 (motion-aware summaries) is the remaining question and
the bridge to the main thesis direction.

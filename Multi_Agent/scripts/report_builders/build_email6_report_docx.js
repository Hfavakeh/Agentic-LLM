// Email-6 preview report: (A) stronger-model conclusion + (B) first motion-aware
// experiment, hypothesis-structured. Content mirrors docs/email6_preview_report.md.
// Reuses the house docx style from build_email5_report_docx.js.
const fs = require('fs');
const path = require('path');

const NODE_MODULES = 'C:/Users/hfava/AppData/Roaming/npm/node_modules';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType, ShadingType,
} = require(path.join(NODE_MODULES, 'docx'));

const FONT = 'Calibri', MONO = 'Consolas';
const HEAD_FILL = '1F3864', ALT_FILL = 'F2F2F2', TODO = 'B25000';
const border = { style: BorderStyle.SINGLE, size: 8, color: '808080' };
const cellBorders = { top: border, bottom: border, left: border, right: border };

function P(text, opts = {}) {
  return new Paragraph({ spacing: { before: 60, after: 60 }, ...opts,
    children: [new TextRun({ text, font: FONT, size: 22, ...(opts.run || {}) })] });
}
function PR(runs, opts = {}) {
  return new Paragraph({ spacing: { before: 60, after: 60 }, ...opts,
    children: runs.map(r => new TextRun({ text: r.text, font: r.mono ? MONO : FONT,
      size: r.mono ? 20 : 22, bold: !!r.bold, italics: !!r.italic, color: r.color || '000000' })) });
}
function H1(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_1, spacing: { before: 240, after: 100 },
    children: [new TextRun({ text, font: FONT, size: 28, bold: true, color: HEAD_FILL })] });
}
function H2(text) {
  return new Paragraph({ heading: HeadingLevel.HEADING_2, spacing: { before: 160, after: 60 },
    children: [new TextRun({ text, font: FONT, size: 24, bold: true, color: HEAD_FILL })] });
}
function cap(text) {
  return new Paragraph({ spacing: { before: 120, after: 40 },
    children: [new TextRun({ text, font: FONT, size: 20, italics: true, color: '595959' })] });
}
function bulletR(runs) {
  return new Paragraph({ numbering: { reference: 'bullets', level: 0 }, spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({ text: r.text, font: r.mono ? MONO : FONT,
      size: r.mono ? 20 : 22, bold: !!r.bold, italics: !!r.italic, color: r.color || '000000' })) });
}
function numItem(runs) {
  return new Paragraph({ numbering: { reference: 'nums', level: 0 }, spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun({ text: r.text, font: FONT, size: 22, bold: !!r.bold, italics: !!r.italic })) });
}
function code(lines) {
  return new Paragraph({ spacing: { before: 40, after: 40 }, shading: { fill: 'F2F2F2', type: ShadingType.CLEAR },
    children: lines.flatMap((ln, i) => {
      const run = new TextRun({ text: ln, font: MONO, size: 18 });
      return i === 0 ? [run] : [new TextRun({ text: ln, font: MONO, size: 18, break: 1 })];
    }) });
}
function cell(content, opts = {}) {
  const para = new Paragraph({ alignment: opts.align || AlignmentType.LEFT, spacing: { before: 20, after: 20 },
    children: (typeof content === 'string' ? [{ text: content }] : content).map(r => new TextRun({
      text: r.text, font: r.mono || opts.mono ? MONO : FONT, size: opts.mono ? 18 : 20,
      bold: r.bold !== undefined ? r.bold : !!opts.bold, italics: !!r.italic,
      color: r.color || opts.color || '000000' })) });
  return new TableCell({ borders: cellBorders, width: { size: opts.width, type: WidthType.DXA },
    shading: opts.fill ? { fill: opts.fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 50, bottom: 50, left: 90, right: 90 }, children: [para] });
}
function mkTable({ widths, header, rows }) {
  const total = widths.reduce((a, b) => a + b, 0);
  const head = new TableRow({ tableHeader: true,
    children: header.map((h, i) => cell(h, { width: widths[i], fill: HEAD_FILL, bold: true, color: 'FFFFFF', align: AlignmentType.CENTER })) });
  const body = rows.map((r, ri) => new TableRow({
    children: r.map((c, i) => {
      const fill = ri % 2 === 0 ? undefined : ALT_FILL;
      if (typeof c === 'string') return cell(c, { width: widths[i], fill });
      return cell([c], { width: widths[i], fill, bold: c.bold, align: c.align });
    }) }));
  return new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: [head, ...body] });
}
const R = (a) => a === 'r' ? AlignmentType.RIGHT : a === 'c' ? AlignmentType.CENTER : AlignmentType.LEFT;
const FR = { text: '— (from run)', align: R('c'), italic: true, color: TODO }; // placeholder cell

// content ────────────────────────────────────────────────────────────────────
const content = [];
content.push(H1('Revised preview report — from LLM hyperparameter tuning to motion-aware guidance'));
content.push(cap('Prepared for Prof. Mihai Lazarescu — revision of the motion preview, addressing the eight points raised in your reply. Dataset: radar (preprocessed-RadarEXP1). Structure unchanged: (A) conclusion from the stronger-model diagnostic, and (B) the motion-aware guidance experiment, structured around hypotheses.'));
content.push(H2('What changed since the version you saw'));
content.push(PR([{ text: 'The main Part B claim has been withdrawn. ', bold: true }, { text: 'The earlier report stated that the motion-aware LLM “reaches test RMSE 0.2285, below both the plain-MSE baseline (0.2319) and the rule-based (0.2301), the first LLM in this project to beat the baseline”. Your point 2 was right: that ordering was inside the seed noise. On the paired test the LLM does beat plain MSE (p = 0.009), but it ' }, { text: 'does not beat the motion rule', bold: true }, { text: ' (p = 0.29) — and, tested for the first time here, ' }, { text: 'the motion rule does not beat plain MSE either', bold: true }, { text: ' (p = 0.35). The controls in B.6 then show why: undirected random search over the same six knobs matches the motion-aware LLM (p = 0.64), and removing the motion summary costs nothing (p = 0.31). The gain over plain MSE comes from the extra loss flexibility, not from motion knowledge.' }]));
content.push(bulletR([{ text: 'Other substantive changes: ', bold: true }, { text: 'a grid-floor bug was found and fixed (v_max could not go below 1.0 m/s, above every walker’s p95 speed), so all Part B numbers were re-run on the corrected grid; Qwen3-14B now has the symmetric history cells you asked for, and they ' }, { text: 'invert', bold: true }, { text: ' Gemma-4’s history result; the reasoning analysis was extended to Part A, where it contradicts the shallow-reasoning finding from Part B.' }]));
content.push(P('Where each of your eight points is addressed:', { run: { bold: true } }));
content.push(mkTable({
  widths: [700, 5150, 3160],
  header: ['#', 'Your point', 'Where / status'],
  rows: [
    [{ text: '1', align: R('c') }, 'Part A Gemma-4 / Qwen3 asymmetry — test symmetrically or motivate', { text: 'Part A table — Qwen3-14B real/shuffled/empty added (10 seeds each). DONE', align: R('l') }],
    [{ text: '2', align: R('c') }, 'Report paired seed-level results (per-seed, mean Δ, CI, wins/losses)', { text: 'B.4 + Appendix A. DONE', align: R('l') }],
    [{ text: '3', align: R('c') }, 'Why only Qwen3 in Part B? Report Gemma-4; do not cherry-pick', { text: 'B.8 cross-model, both on the same fixed grid. DONE', align: R('l') }],
    [{ text: '4', align: R('c') }, 'How substantial is the LLM reasoning — in BOTH Part A and Part B', { text: '“Depth of LLM reasoning in Part A” + B.5. DONE', align: R('l') }],
    [{ text: '5', align: R('c') }, 'Separate motion knowledge from the extra tunable knobs (controls)', { text: 'B.6 — five arms over the same six knobs. DONE', align: R('l') }],
    [{ text: '6', align: R('c') }, 'No test leakage — name the split for every motion quantity', { text: 'B.9. DONE', align: R('l') }],
    [{ text: '7', align: R('c') }, 'Exact loss formula, alignment, shuffle-safety', { text: 'B.9. DONE', align: R('l') }],
    [{ text: '8', align: R('c') }, 'Add ≥1 more subject / segment / motion profile', { text: 'B.10 — full replication on IR (2nd profile). DONE', align: R('l') }],
  ],
}));

// ---- Part A ----
content.push(H1('Part A — Stronger-model diagnostic: conclusion'));
content.push(PR([{ text: 'Setup (unchanged). ', bold: true }, { text: 'The protocol was frozen and only the LLM was swapped: same dataset, same 9-hyperparameter discrete grid, same 25 attempts, same 3 trainings per setting (seeds 101/102/103), same fresh-seed final evaluation, same five comparison arms (baseline, LLM, random, Optuna, curve-aware rule-based). We ran two stronger models — Gemma-4 and Qwen3-14B — and, on Qwen3-14B, the SoTA optimizer-prompting recipe OPRO.' }]));
content.push(PR([{ text: 'Headline result — final test RMSE (metres, lower is better). ', bold: true }, { text: 'Reference arms are model-invariant: baseline 0.2308, rule-based 0.2371, random ≈ 0.242, Optuna 0.2614.' }]));
content.push(mkTable({
  widths: [3050, 700, 1500, 1400, 1600, 1660],
  header: ['Cell', 'n', 'baseline', 'LLM', 'rule-based', 'seeds LLM < rule'],
  rows: [
    ['Gemma-4 — real history', { text: '9', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2437', align: R('r'), bold: true }, { text: '0.2371', align: R('r') }, { text: '0 / 9', align: R('c'), bold: true }],
    ['Gemma-4 — shuffled history', { text: '8', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2465', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0 / 8', align: R('c') }],
    ['Gemma-4 — empty history', { text: '8', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2486', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0 / 8', align: R('c') }],
    ['Gemma-4 — explore prompt', { text: '10', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2504', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0 / 10', align: R('c') }],
    ['Qwen3-14B — real history', { text: '10', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2512', align: R('r'), bold: true }, { text: '0.2371', align: R('r') }, { text: '0 / 10', align: R('c') }],
    ['Qwen3-14B — shuffled history', { text: '10', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2406', align: R('r'), bold: true }, { text: '0.2371', align: R('r') }, { text: '2 / 10', align: R('c'), bold: true }],
    ['Qwen3-14B — empty history', { text: '10', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2476', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0 / 10', align: R('c') }],
    ['Qwen3-14B — OPRO prompt', { text: '8', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2451', align: R('r'), bold: true }, { text: '0.2371', align: R('r') }, { text: '0 / 8', align: R('c'), bold: true }],
    ['Qwen3-14B — explore prompt', { text: '9', align: R('c') }, { text: '0.2308', align: R('r') }, { text: '0.2500', align: R('r') }, { text: '0.2371', align: R('r') }, { text: '0 / 9', align: R('c') }],
  ],
}));
content.push(cap('Reading the last column: the rule-based controller is deterministic, so its final RMSE is the SAME 0.2371 on every seed. The column is therefore a per-seed tally of how often the LLM landed below that fixed line — NOT a verdict that the LLM won. The only non-zero entry (Qwen3-14B shuffled, 2/10) still has a WORSE mean than the rule (0.2406 vs 0.2371): those two seeds cleared the line by 1.2 mm and 0.6 mm, while the other eight fell above it, one by 16 mm. Compare cell means, not this tally.'));
content.push(P('Answers to the six diagnostic questions:', { spacing: { before: 120, after: 40 }, children: undefined, run: { bold: true } }));
content.push(numItem([{ text: 'What improved with Gemma-4 and Qwen3-14B? Proposal quality.', bold: true }]));
content.push(P('Qwen3-14B proposes ~25/25 distinct settings, essentially zero repeats, zero invalid outputs, no timeouts, and reaches 0.85–0.95 grid coverage — matching random/Optuna and far above the small models (≤ 0.61). The small-model hallmark (constant repetition) is gone. OPRO gave the single cleanest run of all.'));
content.push(numItem([{ text: 'What did not improve? The thing that matters — the final model.', bold: true }]));
content.push(P('No LLM cell mean beats the rule-based controller (0 of 9 cells), and at seed level only 2 of 82 seed-cells fall below it — while 0 of 82 beat the fixed baseline. The best LLM result anywhere is Qwen3-14B with SHUFFLED history (0.2406), i.e. a placebo cell; the best faithful-history result (Gemma-4 real history 0.2437) is worse. Both still sit above the baseline (0.2308) and the rule (0.2371).'));
content.push(numItem([{ text: 'Did the stronger models use history better? Both DEPEND on it, but only Gemma-4 profits from it — Qwen3-14B is actively hurt by truthful history.', bold: true }]));
content.push(P('Both models were run symmetrically across real / shuffled / empty history (10 seeds each for Qwen3-14B; 9/8/8 for Gemma-4). Behaviourally both clearly read the history: degrading it collapses proposal diversity in the same direction. For Qwen3-14B, distinct settings fall 24.3 → 19.7 → 15.3 and the repeat rate rises 0.028 → 0.084 → 0.388 (a 14× increase with no history at all); Gemma-4 shows the same collapse (19.4 → 15.4 → 8.9 distinct, repeats 3 → 7 → 57).'));
content.push(P('But the two models diverge on what that history buys them. Gemma-4 degrades monotonically, as expected if history is genuinely useful: 0.2437 real → 0.2465 shuffled → 0.2486 empty. Qwen3-14B inverts it — real history is its WORST cell (0.2512), shuffled its BEST (0.2406), empty in between (0.2476). The shuffled-vs-real gap is significant and paired over the same 10 seeds (Δ = −0.0105, t p = 0.0057, Wilcoxon p = 0.0098, shuffled better on 8/10 seeds); empty-vs-real is not (p = 0.24).'));
content.push(PR([{ text: 'Read: history use and history benefit are different things. ', bold: true }, { text: 'Qwen3-14B demonstrably consumes the history — remove it and its proposal machinery falls apart — yet feeding it a truthful history produces a worse-generalizing model than feeding it a corrupted one. This is question 5 in miniature: the search overfits the small validation set, so the model that follows a faithful history chases the validation minimum hardest and pays for it on test. Corrupting the history breaks that chase. It also means the earlier “stronger models use history better” claim was a Gemma-4-specific finding, not a general one — which is exactly why the symmetric cells were needed.' }]));
content.push(numItem([{ text: 'Did they explore better? Yes — Qwen3-14B reaches near-random grid coverage.', bold: true }]));
content.push(P('Exploration breadth is no longer a limitation at this scale.'));
content.push(numItem([{ text: 'Did they select configs that improved validation but failed on test? Exactly — this is the crux.', bold: true }]));
content.push(P('On the metric the search optimizes, the LLM and the controller are near-tied (val RMSE ≈ 0.193 vs 0.1927) and both crush the baseline (≈ 0.29). But those low-validation settings overfit the small validation set: the same settings land at 0.244–0.250 on held-out test, worse than the baseline’s 0.231. The LLM’s proposals are not poor on their own metric — the whole search overfits validation, and the LLM overfits it slightly more.'));
content.push(numItem([{ text: 'Why does the curve-aware rule-based controller stay competitive / better?', bold: true }]));
content.push(P('It uses the same signals but converts them into a stable, low-variance selection. The LLM chases the validation minimum more aggressively and pays for it on test. Capability, prompt style, exploration breadth and history use are now all ruled out as the binding constraint — what remains is selection under a noisy validation signal, where a conservative deterministic rule wins.'));
content.push(H2('Depth of LLM reasoning in Part A — and how it differs from Part B'));
content.push(P('Clean, in-grid, non-repeating proposals show only that the LLM follows the protocol. Applying the identical five-part reasoning analysis used in Part B (B.5) to the Part A protocol logs gives a sharply different — and more favourable — picture:'));
content.push(mkTable({
  widths: [3400, 2900, 2710],
  header: ['Reasoning metric', 'Part A (generic HP)', 'Part B (motion loss)'],
  rows: [
    ['accepted proposals (qwen3 / gemma4)', { text: '243 / 175', align: R('c') }, { text: '173 / 209', align: R('c') }],
    ['most common diagnosis', { text: 'underfit 41% / plateau 42%', align: R('c'), bold: true }, { text: 'underfit 79% / plateau 80%', align: R('c') }],
    ['distinct diagnosis labels used', { text: '6 / 5', align: R('c'), bold: true }, { text: 'near-single label', align: R('c') }],
    ['reason names the knob it moved', { text: '96% / 93%', align: R('c'), bold: true }, { text: '46% / 61%', align: R('c') }],
    ['proposals improving running best', { text: '26% / 31%', align: R('c') }, { text: '24% / 16%', align: R('c') }],
  ],
}));
content.push(PR([{ text: 'The professor’s suspicion holds for Part B, but NOT for Part A. ', bold: true }, { text: 'In the motion task the diagnosis is a near-constant generic label (79–80%) that does not drive the action. In generic HP tuning the same models use the full label set, and — for qwen3 — the diagnosis genuinely selects the action in the textbook direction: ' }, { text: 'possible_overfitting_tendency', mono: true }, { text: ' → raise dropout (29/67), ' }, { text: 'possible_underfitting_tendency', mono: true }, { text: ' → raise LSTM capacity (lstm_hidden_up, 18), ' }, { text: 'plateau', mono: true }, { text: ' → raise patience (3/5). The stated reason names the hyperparameter actually moved in 96% of proposals, versus 46% in Part B.' }]));
content.push(PR([{ text: 'Honest caveat on the cross-model side: ', bold: true }, { text: 'gemma4’s coupling is weaker than qwen3’s — ' }, { text: 'dropout_up', mono: true }, { text: ' is its top action under 4 of its 5 diagnoses, so it leans on one default move regardless of what it diagnosed. Diagnosis variety and reason-grounding are nonetheless high for both models.' }]));
content.push(PR([{ text: 'Why this matters for the conclusion. ', bold: true }, { text: 'It rules out shallow reasoning as the explanation for Part A’s failure. The LLM diagnoses sensibly, acts consistently with its diagnosis, and explains itself accurately — and still never beats the baseline (0 of 82 seed-cells) or the rule. So Part A is not a reasoning failure; it is a ' }, { text: 'selection failure under a noisy validation signal', bold: true }, { text: ' (question 5). The shallow-reasoning finding is specific to Part B, where the motion task is a far less familiar operationalisation.' }]));
content.push(cap('Reproduce: python scripts/analyze_hp_reasoning.py --llm results/history-use/history-none-<TAG> --label <TAG>-real --out analysis/reasoning_partA_<TAG>  (Part B counterpart: analyze_motion_reasoning.py).'));

content.push(PR([{ text: 'Conclusion (not “the LLM failed”). ', bold: true }, { text: 'In generic hyperparameter tuning on this small, well-behaved search space, a stronger LLM improves proposal validity, diversity and history use — and reasons substantively (it diagnoses with a varied, action-driving label set and explains its changes accurately) — but none of this translates into better generalization. Better proposals and sound reasoning did not produce a better optimum. This closes both the “it’s just a weak model” and the “it never really reasoned” loopholes, and satisfies the stop condition for generic LLM HP tuning, motivating the pivot to the thesis core in Part B.' }]));
content.push(cap('Full numbers, behaviour tables, best-so-far curve and caveats: docs/email5_stronger_models_report.md (and Email5_Stronger_Models_Report.docx).'));

// ---- Part B ----
content.push(H1('Part B — Motion-aware guidance: first experiment'));
content.push(PR([{ text: 'The question that defines the thesis: ', }, { text: 'can an LLM use human-motion knowledge to improve a small localization network?', bold: true }, { text: ' Instead of tuning generic hyperparameters, we now freeze the network and let the proposer reshape the training objective (the loss) from interpretable summaries of how the tracked person moves and where the model’s error concentrates by motion regime.' }]));
content.push(PR([{ text: 'Headline result (radar, paired over 30 seeds, with controls). ', bold: true }, { text: 'Searching the six loss knobs beats plain MSE (0.2319): every arm that touches them lands in a tight 0.2286–0.2303 band. But the controls show that ', }, { text: 'motion knowledge is not what does the work', bold: true }, { text: '. Undirected RANDOM search over the same knobs — with zero motion knowledge — reaches 0.2289, statistically indistinguishable from the motion-aware LLM (0.2286; Δ = −0.0003, p = 0.64), and stripping the motion summary from the LLM changes nothing (0.2294; p = 0.31). The LLM also does not beat the deterministic motion rule (0.2303; p = 0.29). A behavioural analysis explains it: the LLM emits a near-fixed recipe (a velocity-plausibility penalty on 100% of proposals) under a single dominant diagnosis label. ', }, { text: 'Net on radar: the gain over plain MSE comes from the extra loss flexibility, not from the LLM’s motion knowledge or reasoning. ', bold: true }, { text: 'That flexibility claim is itself radar-specific — on the second motion profile (IR, B.10) every knob-touching arm is significantly WORSE than plain MSE, while the LLM-vs-random null replicates. The result that holds on both datasets is that motion knowledge adds nothing over undirected search.' }]));

content.push(H2('B.1  Experimental design'));
content.push(P('The 9 conventional hyperparameters are frozen at the baseline setting. The only thing that varies is a six-lever loss-shaping vector:'));
content.push(mkTable({
  widths: [2600, 3400, 3010],
  header: ['Lever', 'Meaning', 'Human-motion rationale'],
  rows: [
    [{ text: 'v_max', mono: true }, 'plausible top walking speed (m/s); faster predicted steps penalised', 'set just above the observed p95 speed'],
    [{ text: 'lambda_vel', mono: true }, 'strength of the speed-plausibility penalty (0 = off)', 'raise when predictions look noisy / motion is smooth'],
    [{ text: 'lambda_smooth', mono: true }, 'penalty on implausible acceleration / jerk (0 = off)', 'raise when trajectory is smooth with frequent dwells'],
    [{ text: 'bin_weight_slow / medium / fast', mono: true }, 'per-speed-regime error weights (1.0 = neutral)', 'up-weight the regime the model fits worst'],
  ],
}));
content.push(P('Neutral levers (lambda_vel=0, lambda_smooth=0, all weights 1.0) = plain MSE. Same protocol structure as the HP bake-off: each lever vector is trained 3× from scratch, scored by mean validation RMSE in metres; the winner is evaluated on fresh seeds. Four arms, mapping directly onto your request:'));
content.push(bulletR([{ text: 'baseline', bold: true }, { text: ' — plain MSE (the floor).' }]));
content.push(bulletR([{ text: 'C2 — motion heuristic (the “simple rule”)', bold: true }, { text: ' — a fixed, deterministic motion-to-lever mapping (v_max ≈ 1.1× p95 speed, gentle penalties, fast-regime up-weight).' }]));
content.push(bulletR([{ text: 'C3 — LLM motion', bold: true }, { text: ' — the LLM reads the motion summaries + per-regime error and proposes lever vectors.' }]));
content.push(bulletR([{ text: 'random', bold: true }, { text: ' — 25 random lever vectors (undirected reference).' }]));
content.push(PR([{ text: 'The central comparison is C3 vs C2: does the LLM’s motion interpretation beat a fixed motion rule?', bold: true }]));

content.push(H2('B.2  Evidence — motion profile of the tracked person (radar, real)'));
content.push(P('Computed from the target trajectory only (no training), converted to physical units at 4 Hz:'));
content.push(mkTable({
  widths: [3400, 2000, 3610],
  header: ['Feature', 'Value', 'Reading'],
  rows: [
    ['speed mean / p95 / max', { text: '0.31 / 0.59 / 3.34 m/s', align: R('c'), bold: true }, 'slow walker with occasional fast bursts'],
    ['acceleration mean / p95', { text: '0.58 / 1.86 m/s²', align: R('c') }, 'moderate start-stop dynamics'],
    ['turning mean / p95', { text: '26° / 107° per step', align: R('c') }, 'frequent direction changes'],
    ['sharp-turn share (|turn| > 45°)', { text: '19.1 %', align: R('c'), bold: true }, '~1 in 5 moving steps is a sharp turn'],
    ['stop share / dwell episodes', { text: '2.6 % / 5.8 per min', align: R('c') }, 'rarely fully stationary; brief pauses'],
  ],
}));
content.push(PR([{ text: 'Motion reading: ', bold: true }, { text: 'the person mostly walks slowly and turns a lot, with short fast bursts. Prior HP-tuning results already showed the model’s error is worst in the fast regime (per-regime error spread ≈ 1.1×, worst = fast). This is the qualitative summary the LLM is given, e.g. “the person walks slowly with frequent turns and occasional fast bursts; the model’s error concentrates in the fast regime.”' }]));

content.push(H2('B.3  Hypothesis-structured worked example'));
content.push(PR([{ text: 'Hypothesis. ', bold: true }, { text: 'The model fails mainly during fast movement (and, relatedly, turning). Up-weighting the fast regime and penalising implausibly fast / jerky predicted steps should reduce fast-regime error without harming the rest.', italic: true }]));
content.push(PR([{ text: 'Evidence — per-regime error table (baseline). ', bold: true }, { text: 'Mean Euclidean position error by target-speed tercile, at the baseline model’s best epoch:' }]));
content.push(mkTable({
  widths: [4000, 2500, 2510],
  header: ['Regime', 'Baseline error (m)', 'note'],
  rows: [
    ['slow', { text: '0.2650', align: R('c') }, ''],
    ['medium', { text: '0.2765', align: R('c') }, ''],
    [{ text: 'fast', bold: true }, { text: '0.2781', align: R('c'), bold: true }, { text: '← worst regime, as hypothesised', align: R('l'), italic: true }],
    ['spread (worst / best)', { text: '1.049', align: R('c') }, ''],
  ],
}));
content.push(PR([{ text: 'LLM interpretation (C3). ', bold: true }, { text: 'Shown the profile + this table, qwen3 consistently diagnosed “possible underfitting tendency” (334 of 339 accepted proposals) and reasoned in motion terms — setting v_max just above the p95 speed and up-weighting the harder regimes. Verbatim: ' }, { text: '“raising v_max to match p95 (0.6)… aligns with the motion’s high jerk and outlier speeds”', italic: true }, { text: ' and ', }, { text: '“medium-speed regime has highest error (0.2737), so increasing its weight while lowering lambda_smooth to accommodate jerky motion”.', italic: true }]));
content.push(PR([{ text: 'Proposed change — C3 (LLM) vs C2 (rule):', bold: true }]));
content.push(mkTable({
  widths: [3600, 2700, 2710],
  header: ['Lever', 'C2 rule (deterministic)', 'C3 LLM (qwen3, modal over seeds)'],
  rows: [
    [{ text: 'v_max', mono: true }, { text: '0.75', align: R('c') }, { text: '0.75', align: R('c') }],
    [{ text: 'lambda_vel', mono: true }, { text: '0.1', align: R('c') }, { text: '0.3', align: R('c'), bold: true }],
    [{ text: 'lambda_smooth', mono: true }, { text: '0.1', align: R('c') }, { text: '0.05', align: R('c'), bold: true }],
    [{ text: 'bin_weight_slow', mono: true }, { text: '1.0', align: R('c') }, { text: '1.0', align: R('c') }],
    [{ text: 'bin_weight_medium', mono: true }, { text: '1.0', align: R('c') }, { text: '3.0', align: R('c'), bold: true }],
    [{ text: 'bin_weight_fast', mono: true }, { text: '1.5', align: R('c'), bold: true }, { text: '2.0', align: R('c'), bold: true }],
  ],
}));
content.push(cap('Both key off the same p95 speed for v_max (=0.75). The difference: the rule applies a single gentle fast-regime up-weight (×1.5); the LLM up-weights BOTH medium (×3) and fast (×2) and triples the velocity-plausibility penalty (0.1→0.3) while easing the smoothness penalty (0.1→0.05) — a more aggressive, motion-reasoned loss.'));
content.push(PR([{ text: 'Result. ', bold: true }, { text: 'Whether each proposal (i) reduced the targeted fast-regime error and (ii) improved or harmed the overall test RMSE:' }]));
content.push(mkTable({
  widths: [3300, 2200, 1800, 1710],
  header: ['Arm', 'test RMSE (m)', 'Δ vs baseline', 'fast-regime err'],
  rows: [
    ['baseline (plain MSE)', { text: '0.2319 ± 0.0085', align: R('c') }, { text: '—', align: R('c') }, { text: '0.2781', align: R('c') }],
    ['C2 motion rule', { text: '0.2303 ± 0.0088', align: R('c') }, { text: '−0.0016', align: R('c') }, { text: '0.2616', align: R('c') }],
    [{ text: 'C3 LLM motion', bold: true }, { text: '0.2286', align: R('c'), bold: true }, { text: '−0.0033', align: R('c'), bold: true }, { text: '0.2637', align: R('c') }],
  ],
}));
content.push(cap('Baseline / C2 are single deterministic vectors, ± over the 30 final-eval seeds. C3 is the mean over 10 LLM search seeds (each final-eval’d on the same 30 seeds); 8 of those 10 fell below the C2 rule (0.2303), the other two (0.2307, 0.2310) just above it (search-seed range 0.2273–0.2310).'));
content.push(PR([{ text: 'Note — Part A and Part B have DIFFERENT baseline numbers, and they must not be compared across parts. ', bold: true }, { text: 'Part A’s baseline is 0.2308; Part B’s is 0.2319. The nine hyperparameters are identical and both are evaluated on the same 30 fresh seeds (201–230), but every one of the 30 per-seed values differs (mean offset 1.1 mm, max 18 mm): Part B’s plain-MSE baseline is trained through the motion pipeline, which carries the previous two target positions as per-sample fields, so the training stream is not bit-identical even with neutral levers. ' }, { text: 'Within Part B this is harmless', bold: true }, { text: ' — baseline, C2, LLM and random all run through that same pipeline, so every Part B comparison is like-for-like. But the 1.1 mm offset is the same order as Part B’s headline effects (−3.3 mm vs baseline, −1.8 mm vs rule), so quoting Part A’s 0.2308 against a Part B arm would be misleading. Each part is internally consistent; only cross-part arithmetic is invalid.' }]));

content.push(H2('B.4  Paired seed-level significance (qwen3)'));
content.push(P('All arms are final-eval’d on the same 30 seeds (201–230), so the comparison is paired per seed. LLM error per seed is the mean over the 10 search-seed lever vectors; a win = LLM error below the comparator on that seed.'));
content.push(mkTable({
  widths: [2500, 1500, 2400, 1400, 1200],
  header: ['Comparison', 'mean Δ (m)', '95% CI', 'paired t p', 'LLM wins'],
  rows: [
    [{ text: 'LLM − baseline', bold: true }, { text: '−0.0033', align: R('c'), bold: true }, { text: '[−0.0057, −0.0009]', align: R('c') }, { text: '0.009', align: R('c'), bold: true }, { text: '22 / 30', align: R('c') }],
    [{ text: 'LLM − rule', bold: true }, { text: '−0.0018', align: R('c') }, { text: '[−0.0051, +0.0016]', align: R('c') }, { text: '0.29', align: R('c') }, { text: '17 / 30', align: R('c') }],
    [{ text: 'C2 rule − baseline', bold: true }, { text: '−0.0015', align: R('c') }, { text: '[−0.0048, +0.0018]', align: R('c') }, { text: '0.35', align: R('c') }, { text: '15 / 30', align: R('c'), bold: true }],
  ],
}));
content.push(PR([{ text: '“Beats” here means STATISTICALLY beats — not simply a lower number. ', bold: true }, { text: 'In absolute terms the LLM IS lower than the C2 rule: 0.2286 vs 0.2303, better by 1.8 mm. But that gap is not reproducible. The per-seed differences run from −18.4 mm to +21.9 mm with a standard deviation of 8.9 mm — roughly ' }, { text: 'five times larger than the 1.8 mm mean gap', bold: true }, { text: ' — so re-running with a different seed pool could flip the sign. Throughout Part B, “does not beat” means the difference is inside that seed noise; it never means the raw number was higher. (Part A’s per-seed tally column is a different, purely descriptive count — see the note under that table.)' }]));
content.push(PR([{ text: 'Read: ', bold: true }, { text: 'the LLM ' }, { text: 'significantly beats the plain-MSE baseline', bold: true }, { text: ' (CI excludes 0, 22/30 seed wins), but ' }, { text: 'does not beat the C2 motion rule', bold: true }, { text: ' — that difference’s CI spans 0, p = 0.29, and the LLM wins on only 17/30 seeds (a coin flip). Wilcoxon agrees (0.007 vs 0.33). gemma4 gives the same pattern (vs baseline p = 0.026, 20/30; vs rule p = 0.38, 17/30). The full per-seed table is in analysis/motion_qwen3_full/paired_seed_analysis.md.' }]));
content.push(PR([{ text: 'Important — the C2 rule’s own “gain” is also noise. ', bold: true }, { text: 'The third row paired-tests the comparator itself. C2 sits 1.5 mm below plain MSE, but that difference is ' }, { text: 'not significant', bold: true }, { text: ' (CI spans 0, p = 0.35, rule lower on just 15/30 seeds — an exact coin flip). So the −0.0016 shown for C2 in B.3 must not be read as the motion rule beating the baseline. This matters for the whole of Part B: the LLM is being measured against a comparator that does not itself reliably improve on plain MSE.' }]));

content.push(H2('B.5  Depth of LLM reasoning (protocol logs)'));
content.push(P('Clean, in-grid, non-repeating proposals show the LLM follows the protocol; they do not show it reasons well. Across the accepted proposals (173 qwen3 / 209 gemma4):'));
content.push(bulletR([{ text: 'Diagnosis is a dominant generic label. ', bold: true }, { text: 'qwen3 says “possible underfitting tendency” on 79% of proposals; gemma4 says “plateau” on 80%. (This confirms the professor’s suspicion.)' }]));
content.push(bulletR([{ text: 'The diagnosis does not drive the action. ', bold: true }, { text: 'Whatever the diagnosis, both models emit the same recipe — add a velocity-plausibility penalty (173/173 qwen3, i.e. every proposal), add a smoothness penalty, and up-weight the fast/medium regime. Actions are essentially constant across diagnoses.' }]));
content.push(bulletR([{ text: 'The reason only partly matches the change. ', bold: true }, { text: 'The stated reason mentions the actually up-weighted regime in 46% (qwen3) / 61% (gemma4) of proposals.' }]));
content.push(bulletR([{ text: 'Most proposals don’t improve the result. ', bold: true }, { text: 'Only 24% (qwen3) / 16% (gemma4) of accepted proposals beat the running-best score; the search works by keeping the best of many near-identical tries, not by reasoning toward it.' }]));
content.push(PR([{ text: 'So the LLM applies a sensible but near-fixed motion prior under a dominant label — which is essentially what the C2 rule encodes, and explains B.4: it does not beat the rule because it is, in effect, reproducing it.' }]));

content.push(H2('B.6  Controls — is it motion knowledge, or just the six extra knobs?'));
content.push(P('The six-knob loss is strictly more flexible than plain MSE, so some of the gain could come from the extra tunable parameters rather than motion knowledge. Two controls isolate this, both over the SAME six knobs and the same protocol: (a) undirected RANDOM search (zero motion knowledge), and (b) the LLM given the per-regime error only, with the motion-summary block removed.'));
content.push(mkTable({
  widths: [3300, 2000, 2200, 1860],
  header: ['Arm', 'test RMSE (m)', 'motion knowledge', 'vs baseline p'],
  rows: [
    ['baseline (plain MSE)', { text: '0.2319', align: R('c') }, { text: '—', align: R('c') }, { text: '—', align: R('c') }],
    ['C2 motion rule', { text: '0.2303', align: R('c') }, { text: 'yes (fixed)', align: R('c') }, { text: '—', align: R('c') }],
    [{ text: 'random over 6 knobs', bold: true }, { text: '0.2289', align: R('c'), bold: true }, { text: 'NONE', align: R('c'), bold: true }, { text: '0.017', align: R('c') }],
    [{ text: 'qwen3 — per-regime error only', bold: true }, { text: '0.2294', align: R('c'), bold: true }, { text: 'no summary', align: R('c') }, { text: '0.044', align: R('c') }],
    ['gemma4 — full motion', { text: '0.2289', align: R('c') }, { text: 'yes', align: R('c') }, { text: '0.026', align: R('c') }],
    ['qwen3 — full motion', { text: '0.2286', align: R('c') }, { text: 'yes', align: R('c') }, { text: '0.009', align: R('c') }],
  ],
}));
content.push(P('Direct paired contrasts (30 seeds):'));
content.push(bulletR([{ text: 'qwen3-full vs random: ', bold: true }, { text: 'Δ = −0.0003, 95% CI [−0.0015, +0.0010], p = 0.641, 17/30 wins — ' }, { text: 'not significant', bold: true }, { text: '. An undirected search knowing nothing about motion matches the motion-aware LLM.' }]));
content.push(bulletR([{ text: 'qwen3-full vs no-profile: ', bold: true }, { text: 'Δ = −0.0008, CI [−0.0025, +0.0008], p = 0.311, 18/30 wins — ' }, { text: 'not significant', bold: true }, { text: '. Removing the motion interpretation changes nothing; the per-regime error signal alone is enough.' }]));
content.push(bulletR([{ text: 'random vs baseline: ', bold: true }, { text: 'Δ = −0.0030, p = 0.017 — random alone already beats plain MSE.' }]));
content.push(bulletR([{ text: 'random vs rule: ', bold: true }, { text: 'p = 0.351 — not significant.' }]));
content.push(PR([{ text: 'Conclusion: the gain over plain MSE is attributable to the six extra loss knobs, not to motion knowledge or LLM reasoning.', bold: true }]));

content.push(H2('B.7  Interpretation — did the LLM add anything beyond the rule?'));
content.push(bulletR([{ text: 'Over plain MSE: on radar yes, but not because of motion — and it does not replicate. ', bold: true }, { text: 'On radar every arm that tunes the six knobs beats MSE by ~2–3 mm, including random search with no motion knowledge, so the flexibility rather than the knowledge was doing the work. On IR (B.10) that same flexibility makes every arm significantly WORSE than plain MSE, so it is not a dependable gain at all.' }]));
content.push(bulletR([{ text: 'Over the motion rule: no. ', bold: true }, { text: 'The LLM’s edge over C2 is within seed noise (p = 0.29).' }]));
content.push(bulletR([{ text: 'Over random search: no. ', bold: true }, { text: 'The LLM is statistically indistinguishable from undirected random search over the same knobs (p = 0.64), and removing its motion summary entirely costs nothing (p = 0.31).' }]));
content.push(bulletR([{ text: 'Why: the reasoning is shallow. ', bold: true }, { text: 'A near-constant diagnosis label, an action that does not depend on the diagnosis, a reason that matches the change only ~half to two-thirds of the time, and only 18% of proposals improving anything (B.5).' }]));
content.push(bulletR([{ text: 'Net. ', bold: true }, { text: 'On this dataset, as operationalised through these six loss knobs, the LLM’s human-motion knowledge does not measurably improve the localization network beyond a one-line rule or random search. This is a negative result of the same kind as Part A — and, like Part A, it is informative: it localises the failure to the operationalisation, not to the model.' }]));
content.push(bulletR([{ text: 'Honest caveats. ', bold: true }, { text: 'With the full 10 search seeds the two borderline results have firmed up in the same direction: random-vs-baseline now clears significance (p = 0.017), and qwen3-vs-random shows no LLM advantage at all (p = 0.64, 17/30) — so the "extra knobs, not motion" reading is if anything cleaner than in the preview. The main remaining caveat is that this rests on a single trajectory (see B.10).' }]));

content.push(H2('B.8  Cross-model check (gemma4)'));
content.push(P('To guard against reporting only the better-looking model, a second strong model (gemma4) was run under the identical protocol and is reported in full:'));
content.push(mkTable({
  widths: [2900, 2200, 3910],
  header: ['Arm', 'test RMSE (m)', 'chosen loss (modal)'],
  rows: [
    ['baseline (plain MSE)', { text: '0.2319', align: R('c') }, '—'],
    ['C2 motion rule', { text: '0.2303', align: R('c') }, 'fast ×1.5'],
    [{ text: 'C3 LLM — qwen3', bold: true }, { text: '0.2286', align: R('c'), bold: true }, 'medium ×3 + fast ×2, λ_vel 0.3'],
    [{ text: 'C3 LLM — gemma4', bold: true }, { text: '0.2289', align: R('c'), bold: true }, 'fast ×3 + slow ×2, λ_vel 0.1, λ_smooth 0.1'],
  ],
}));
content.push(PR([{ text: 'Both models tell the same story', bold: true }, { text: ': each significantly beats the plain-MSE baseline but neither beats the C2 rule (gemma4 vs baseline p = 0.026, 20/30; vs rule p = 0.38, 17/30). They reached it via different recipes (qwen3 spread weight across medium + fast; gemma4 leaned on fast ×3 plus slow ×2) yet landed within ~0.3 mm of each other. gemma4’s decisions were clean (2 repeats in 250) but it is a slow model — 39 of its 41 rejections were 300 s serving timeouts. So the conclusion is not a single-model artefact and does not depend on which model we picked.' }]));

content.push(H2('B.9  Validity checks (leakage & loss)'));
content.push(bulletR([{ text: 'No test leakage. ', bold: true }, { text: 'Every motion quantity is computed from train or val, never test: the motion profile (speed/accel/turning/stop-go/roughness) from the TRAIN targets; the loss’s speed-regime edges from TRAIN terciles (the controller sets only the per-bin weights, never the edges); the per-regime error shown to the LLM from VAL. Selection is on val RMSE; the test set is touched only at the final evaluation.' }]));
content.push(bulletR([{ text: 'Loss is temporally sound. ', bold: true }, { text: 'total = base + λ_vel·mean(relu(pred_speed − v_max)²) + λ_smooth·mean(‖accel‖²), with pred_speed = ‖pred − prev_y‖·scale·hz and accel = (pred − prev_y) − (prev_y − prev_prev_y). It uses both predicted and (previous) target positions; consecutive positions are carried as per-sample fields (t−1, t−2), so shuffling the training batches does NOT break the velocity/smoothness terms.' }]));

content.push(H2('B.10  Cross-dataset replication — a second motion profile (IR)'));
content.push(PR([{ text: 'Your point 8. ', bold: true }, { text: 'Everything above rests on one trajectory. We repeated the full experiment on a second sensor and motion profile: the IR dataset (5 Hz), whose walker is markedly different from radar — smoother, straighter paths with 3× fewer sharp turns (sharp-turn share 0.061 vs 0.201) and a lower top speed. This is the profile where a speed-plausibility prior ought to help most, so it is a demanding test rather than a convenient one. Same protocol throughout: HPs frozen, six levers searched, 10 search seeds, the same 30 fresh final-eval seeds, same corrected grid.' }]));
content.push(mkTable({
  widths: [3100, 1700, 1700, 1500, 1010],
  header: ['Arm', 'radar RMSE', 'IR RMSE', 'IR Δ vs base', 'IR p'],
  rows: [
    [{ text: 'baseline (plain MSE)', bold: true }, { text: '0.2319', align: R('c') }, { text: '0.2139', align: R('c'), bold: true }, { text: '—', align: R('c') }, { text: '—', align: R('c') }],
    ['C2 motion rule', { text: '0.2303', align: R('c') }, { text: '0.2160', align: R('c') }, { text: '+0.0021', align: R('c') }, { text: '0.045', align: R('c') }],
    ['random over 6 knobs', { text: '0.2289', align: R('c') }, { text: '0.2178', align: R('c') }, { text: '+0.0039', align: R('c') }, { text: '<0.0001', align: R('c') }],
    [{ text: 'C3 LLM motion', bold: true }, { text: '0.2286', align: R('c') }, { text: '0.2194', align: R('c'), bold: true }, { text: '+0.0055', align: R('c'), bold: true }, { text: '0.0002', align: R('c'), bold: true }],
  ],
}));
content.push(cap('On radar the ordering is LLM < random < rule < baseline (loss-shaping helps). On IR it is exactly inverted: baseline < rule < random < LLM (loss-shaping hurts). All IR differences vs baseline are significant, and all 10 of 10 LLM search seeds landed above the IR baseline — this is a consistent reversal, not an outlier.'));
content.push(P('Two results replicate across both datasets, and one does not:'));
content.push(bulletR([{ text: 'REPLICATES — the LLM never separates from random search. ', bold: true }, { text: 'radar Δ = −0.0003, p = 0.64; IR Δ = +0.0016, p = 0.17. On both profiles, undirected search over the same six levers matches the motion-aware LLM. The motion knowledge adds nothing, twice.' }]));
content.push(bulletR([{ text: 'REPLICATES — no motion arm beats the deterministic rule by a meaningful margin. ', bold: true }, { text: 'radar p = 0.29 (ns); IR the LLM is significantly WORSE than the rule (p = 0.027).' }]));
content.push(bulletR([{ text: 'DOES NOT REPLICATE — the gain over plain MSE. ', bold: true }, { text: 'On radar every knob-touching arm beat plain MSE by ~2–3 mm; on IR every one of them is significantly worse (LLM by 5.5 mm, p = 0.0002). The extra loss flexibility is not reliably beneficial — it helped on one profile and hurt on the other.' }]));
content.push(PR([{ text: 'Revised conclusion. ', bold: true }, { text: 'The claim that "the six extra knobs are what produce the gain" was itself radar-specific and does not survive replication. What survives both datasets is the narrower and stronger negative result: ' }, { text: 'the LLM’s human-motion knowledge is statistically indistinguishable from random search over the same levers, on two contrasting motion profiles', bold: true }, { text: ' — and on the profile where the motion prior should have mattered most, motion-aware loss shaping actively degrades the model. We report this as a replication failure of our own earlier positive finding, which is why the second profile was worth running.' }]));
content.push(cap('IR refs (baseline + C2) and the random control were run locally; the LLM arm on the server. Folders: results/motion/motion-refs-ir-v2, motion-random-ir-v2, motion-qwen-ir-v2. The IR motion profile is computed from the TRAIN split only, as on radar (B.9).'));

content.push(H2('B.11  Status & next steps'));
content.push(PR([{ text: 'Complete on radar: qwen3 + gemma4 full motion, the C2 rule, the plain-MSE baseline, and both point-5 controls (random over the six knobs; LLM with per-regime error only). ', bold: true }, { text: 'All eight points are now closed. The symmetric Part A cells for Qwen3-14B (real / shuffled / empty history, 10 seeds each) are complete, resolving the Gemma-4-vs-Qwen3 asymmetry; and the single-trajectory threat is retired by the full IR replication in B.10, which reversed our own earlier positive claim. Two datasets now agree that motion knowledge adds nothing over undirected search. Remaining work, in order of value: (1) a third profile — capacitive (3 Hz) refs are already regenerated on the corrected grid, so only the LLM + random arms are outstanding — to establish whether the radar gain or the IR harm is the anomaly; (2) a mostly-stationary or stop-go profile, which requires SEGMENTING a trajectory rather than a new dataset (none of the three datasets has a stop share above ~3%); (3) if the negative result holds on a third profile, the operationalisation itself (six loss knobs) should be reconsidered rather than the model — the LLM reasons well in Part A, so the constraint is where motion knowledge is being injected, not the LLM. No further generic-HP experiments.' }]));

content.push(H1('Appendix A — per-seed paired errors (qwen3)'));
content.push(P('All arms final-eval’d on the same 30 seeds (201–230). Δ negative = LLM error lower (better) on that seed.'));
let perSeed = [];
try { perSeed = JSON.parse(fs.readFileSync('analysis/motion_qwen3_full/paired_per_seed.json', 'utf8')); } catch (e) { /* table omitted if missing */ }
if (perSeed.length) {
  content.push(mkTable({
    widths: [1400, 1900, 1900, 1900, 1130, 1130],
    header: ['seed', 'baseline', 'rule', 'LLM', 'LLM−base', 'LLM−rule'],
    rows: perSeed.map(r => [
      { text: String(r.seed), align: R('c') },
      { text: r.base.toFixed(4), align: R('c') },
      { text: r.rule.toFixed(4), align: R('c') },
      { text: r.llm.toFixed(4), align: R('c') },
      { text: (r.lb >= 0 ? '+' : '') + r.lb.toFixed(4), align: R('c') },
      { text: (r.lr >= 0 ? '+' : '') + r.lr.toFixed(4), align: R('c') },
    ]),
  }));
}

content.push(H1('Appendix B — how to reproduce / extend'));
content.push(P('The motion profile is consumed only by the LLM arm. Per model, run the LLM arm (server, GPU):'));
content.push(code([
  'docker compose run --rm app --motion-experiment \\',
  '  --motion-arms llm --model <MODEL_TAG> \\',
  '  --seeds 17 42 73 128 256 314 451 512 666 777 \\',
  '  --rounds 25 --final-eval-seeds 30 \\',
  '  --output outputs/motion-<MODEL_TAG>',
]));
content.push(P('The baseline and C2 rule are deterministic and model-independent, so they are computed ONCE (no LLM needed) and reused across models:'));
content.push(code([
  'python main.py --motion-experiment --motion-arms baseline motion_rule \\',
  '  --seeds 42 --final-eval-seeds 30 --output outputs-motion-refs',
]));
content.push(P('Then aggregate the LLM run together with the shared refs into one comparison table + figures:'));
content.push(code([
  'python analyze_motion_experiment.py \\',
  '  --root outputs/motion-<MODEL_TAG> outputs-motion-refs \\',
  '  --out analysis/motion_<MODEL_TAG>',
]));
content.push(P('Notes:'));
content.push(bulletR([{ text: 'The aggregator merges multiple --root dirs by arm, so the LLM-only run and the shared baseline/C2 run combine automatically. It handles any seed count.' }]));
content.push(bulletR([{ text: 'baseline / C2 need only one search seed (they are deterministic); the 30 final-eval seeds give the number. Re-run only if the dataset or split changes.' }]));
content.push(bulletR([{ text: 'Each seed_*/motion_protocol_log_run1.json holds the LLM’s rendered payload, raw reply and parsed levers — the source for B.3’s interpretation line.' }]));

// document ───────────────────────────────────────────────────────────────────
const doc = new Document({
  creator: 'H. Favakeh',
  title: 'Revised preview report — motion-aware guidance (addressing the eight points)',
  styles: {
    default: { document: { run: { font: FONT, size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 28, bold: true, font: FONT, color: HEAD_FILL }, paragraph: { spacing: { before: 240, after: 100 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: FONT, color: HEAD_FILL }, paragraph: { spacing: { before: 160, after: 60 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [
      { reference: 'bullets', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] },
      { reference: 'nums', levels: [{ level: 0, format: LevelFormat.DECIMAL, text: '%1.', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 620, hanging: 320 } } } }] },
    ],
  },
  sections: [{
    properties: { page: { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } } },
    children: content,
  }],
});

Packer.toBuffer(doc).then(buf => {
  const target = process.argv[2] || path.resolve('reports/Email7_Revised_Preview_Report.docx');
  try {
    fs.writeFileSync(target, buf);
    console.log('Wrote:', target, '(' + buf.length + ' bytes)');
  } catch (err) {
    if (err.code === 'EBUSY') {
      const stamp = new Date().toISOString().replace(/[-:T]/g, '').slice(0, 13);
      const ext = path.extname(target), stem = target.slice(0, -ext.length);
      const fallback = `${stem}_${stamp}${ext}`;
      fs.writeFileSync(fallback, buf);
      console.log('Original locked; wrote fallback:', fallback);
    } else { throw err; }
  }
});

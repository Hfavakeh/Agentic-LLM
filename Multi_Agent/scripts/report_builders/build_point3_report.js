const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
        ShadingType } = require("docx");
const fs = require("fs");

const CW = 9360; // content width, US Letter, 1" margins

const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, w, opts = {}) {
  return new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: opts.fill || "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 60, bottom: 60, left: 110, right: 110 },
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!opts.bold, size: 18,
                               color: opts.color || "000000" })],
    })],
  });
}

function row(cells, w, opts = {}) {
  return new TableRow({
    children: cells.map((c, i) =>
      cell(c, w[i], { bold: opts.bold, fill: opts.fill,
                      align: (opts.leftAll || i === 0) ? AlignmentType.LEFT : AlignmentType.CENTER,
                      color: opts.colors ? opts.colors[i] : undefined })),
  });
}

const W = [3360, 2400, 1800, 1800];
const resultRows = [
  ["Controller", "Test RMSE (mean ± std)", "vs baseline", "n seeds"],
  ["Baseline (no optimisation)", "0.2460 ± 0.012", "—", "20"],
  ["C1 — metric-only rule-based", "0.2601 ± 0.007", "worse", "20"],
  ["C2 — motion-aware rule-based", "0.2666 ± 0.010", "worse", "20"],
  ["C3 — LLM qwen3:8b, motion-on", "0.2891 ± 0.024", "worse", "20"],
  ["C3 — LLM qwen3:8b, motion-off", "0.2899 ± 0.033", "worse", "20"],
  ["C3 — LLM llama3.1:8b", "0.2466 ± 0.006", "≈ baseline", "20"],
  ["Random search", "~0.37", "much worse", "20"],
];

// Code-change table — file by file
const CCW = [1640, 2780, 4940];
const ccHead = ["File", "Symbol / entry point", "What changed"];
const ccRows = [
  ["model_pipeline.py", "HP_BOUNDS + LOSS_SHAPING_KEYS",
   "Six motion loss-shaping levers added to the search space — v_max, "
   + "lambda_vel, lambda_smooth and bin_weight_slow / medium / fast. "
   + "LOSS_SHAPING_KEYS marks which HP keys route to loss shaping."],
  ["model_pipeline.py", "DEFAULT_LOSS_SHAPING",
   "Default lever values (lambda_vel = 0, lambda_smooth = 0, "
   + "bin_weights = None) reproduce plain MSE exactly — motion shaping is "
   + "fully opt-in."],
  ["model_pipeline.py", "Trainer._compute_total_loss / apply_loss_shaping_update",
   "New composite objective: speed-bin-weighted MSE + lambda_vel · "
   + "velocity-plausibility penalty + lambda_smooth · smoothness/jerk "
   + "penalty. apply_loss_shaping_update hot-swaps the levers on the running "
   + "trainer. Test RMSE is computed from prior-free position error, never "
   + "the shaped loss."],
  ["model_pipeline.py", "Config.enable_motion / Config.motion_rule",
   "Two new dataclass fields — the master switch for the motion feature, "
   + "and the C1 ↔ C2 rule-arm swap."],
  ["Agent.py", "MotionAwareRuleBasedOptimizer  (new class)",
   "Controller arm C2 — subclasses RuleBasedOptimizer; _motion_loss_shaping "
   + "sets the six levers from fixed heuristics on the motion profile "
   + "(v_max = p95 speed × 1.1, λ = 0.1, fast-bin weight 1.5)."],
  ["Agent.py", "_validate_proposal + system prompt",
   "The six levers are added to the proposal schema and clamped to "
   + "HP_BOUNDS; the system prompt gains motion-aware sections and a soft "
   + "check that the reasoning cites a motion feature (skipped under "
   + "--no-motion)."],
  ["main.py", "--no-motion / --motion-rule CLI flags",
   "--no-motion sets cfg.enable_motion = False (plain MSE, no diagnostics, "
   + "motion-agnostic prompt); --motion-rule sets cfg.motion_rule = True "
   + "(rule arm = C2). motion_enabled is written to final_summary."],
  ["main.py", "motion-profile & diagnostics wiring",
   "The dataset motion profile is computed once (extract_motion_features / "
   + "summarize_motion) and supplied every round; a per-round per-bin error "
   + "breakdown is computed via motion_error_payload; loss-shaping levers "
   + "are routed from proposals into Trainer.loss_shaping; "
   + "sample_random_hparams gains the same six levers."],
  ["motion_descriptors.py", "extract_motion_features, summarize_motion, "
   + "llm_payload, error_payload",
   "Motion-feature module — speed in m/s, dwell / stop-go, p95 etc., and "
   + "the per-bin error breakdown that feeds the LLM payload and the "
   + "motion-weighted MSE."],
];

// Motion-profile metric tables (radar training split)
const MTW = [1740, 1620, 6000];
const mtHead = ["Metric", "Radar value", "What it means"];
const speedRows = [
  ["Speed mean", "0.302 m/s",
   "Average speed across all training-split samples."],
  ["Speed std", "0.191 m/s",
   "Spread of speeds around the mean — large relative to the mean, so "
   + "speed varies considerably."],
  ["Speed median", "0.284 m/s",
   "The middle value. It sits below the mean, so the distribution is "
   + "right-skewed: mostly slow walking with a few fast bursts."],
  ["Speed max", "3.344 m/s",
   "The single fastest sample — far above the median; an outlier / sensor "
   + "spike rather than real human walking."],
  ["Speed p95", "0.600 m/s",
   "95% of samples are slower than this. A robust speed ceiling that "
   + "captures realistic fast walking while ignoring the noisy top 5%."],
];
const dwellRows = [
  ["Stop share", "3%",
   "Fraction of samples labelled 'stop' — the subject is almost always "
   + "moving."],
  ["Dwell episodes", "33",
   "Number of separate pauses (contiguous runs of stop samples) in the "
   + "trajectory."],
  ["Dwell duration", "mean 0.27 s · p95 0.50 s",
   "Pause length = stop-run length ÷ sampling rate. Pauses are very short "
   + "— about one sample at 4 Hz."],
  ["Episodes / minute", "6.6",
   "Frequency of hesitation — how often the subject briefly pauses."],
];

// LLM controller behaviour tables (radar, 20 seeds x 5 rounds = 100 rounds/model)
const BW  = [2380, 1745, 1745, 1745, 1745];
const behHead = ["Property", "qwen3:8b", "llama3:8b", "gemma3:4b", "phi4-mini:3.8b"];
const costRows = [
  ["Prompt tokens / round",          "2011 ± 250", "1816 ± 199", "1375 ± 244", "1883 ± 215"],
  ["Completion tokens / round",      "1686 ± 645", "107 ± 7",    "124 ± 9",    "114 ± 13"],
  ["Response time / round",          "75 ± 29 s",  "5 ± 1 s",    "3 ± 1 s",    "3 ± 0 s"],
  ["Levers changed / round",         "3 ± 1",      "2 ± 0",      "2 ± 0",      "3 ± 2"],
  ["Clean output (no validator fix)","68%",        "77%",        "55%",        "35%"],
  ["Model resets",                   "3%",         "0%",         "0%",         "38%"],
];
const tendHead = ["Field", "qwen3:8b", "llama3:8b", "gemma3:4b", "phi4-mini:3.8b"];
const tendRows = [
  ["Most frequent diagnosis",   "overfitting 49%", "overfitting 34%", "plateau 62%",   "plateau 83%"],
  ["Most frequent strategy",    "regularise 69%",  "regularise 90%",  "regularise 97%","explore 66%"],
  ["Self-reported confidence",  "high 61%",        "medium 100%",     "medium 77%",    "high 70%"],
];

function P(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after === undefined ? 120 : opts.after },
    alignment: opts.align,
    children: [new TextRun({ text, size: opts.size || 21, bold: !!opts.bold,
                             italics: !!opts.italics, color: opts.color })],
  });
}

function bullet(runs) {
  return new Paragraph({
    numbering: { reference: "bul", level: 0 },
    spacing: { after: 70 },
    children: runs,
  });
}
function T(text, opts = {}) {
  return new TextRun({ text, size: 21, bold: !!opts.bold, italics: !!opts.italics });
}
function numbered(runs) {
  return new Paragraph({ numbering: { reference: "num", level: 0 },
    spacing: { after: 70 }, children: runs });
}

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 24, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 0 } },
    ],
  },
  numbering: {
    config: [
      { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET,
        text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 220 } } } }] },
      { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL,
        text: "%1.", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 360, hanging: 240 } } } }] },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1440, bottom: 1080, left: 1440 },
      },
    },
    children: [
      new Paragraph({
        spacing: { after: 40 },
        children: [new TextRun({
          text: "Scientific-Core Experiment: Can an LLM Use Human-Motion "
              + "Knowledge to Improve NN Training?",
          bold: true, size: 26, color: "1F3864" })],
      }),
      new Paragraph({
        spacing: { after: 160 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: "1F3864", space: 2 } },
        children: [new TextRun({
          text: "Thesis progress report — indoor localization (radar dataset) — "
              + "models: qwen3:8b, llama3.1:8b, phi4-mini:3.8b",
          italics: true, size: 18, color: "595959" })],
      }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Objective")] }),
      P("The previous round of work was assessed as conventional hyperparameter "
      + "optimization rather than the core of the thesis. This phase therefore "
      + "tests a stronger question: can an LLM analyse interpretable motion "
      + "features and shape the training objective itself — loss-function "
      + "weights and human-motion priors — and does that contribute scientific "
      + "value beyond a deterministic rule-based controller?"),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. What was built")] }),
      P("The plain-MSE objective was extended into a composite, motion-aware "
      + "training loss:", { after: 60 }),
      new Paragraph({
        spacing: { after: 100 }, alignment: AlignmentType.CENTER,
        children: [new TextRun({
          text: "L  =  speed-bin-weighted MSE  +  λ_vel · velocity-plausibility "
              + "penalty  +  λ_smooth · smoothness/jerk penalty",
          italics: true, size: 20 })],
      }),
      bullet([T("Velocity-plausibility penalty", { bold: true }),
              T(" — penalises predicted speeds above a physical human ceiling "
              + "v_max (m/s).")]),
      bullet([T("Smoothness penalty", { bold: true }),
              T(" — penalises implausible acceleration across three "
              + "consecutive trajectory points.")]),
      bullet([T("Speed-bin weighting", { bold: true }),
              T(" — re-weights position error by the sample’s motion "
              + "regime (slow / medium / fast speed terciles).")]),
      P("Velocity and acceleration are evaluated in real metres, so v_max is a "
      + "genuine physical quantity the controller can reason about. The "
      + "controller proposes six new levers (v_max, λ_vel, λ_smooth, three "
      + "bin weights). The dataset’s real motion profile — speed "
      + "mean/std/p95 and dwell/stop-go behaviour — is computed once and "
      + "supplied in every round. Defaults reproduce plain MSE exactly, so the "
      + "feature is fully opt-in.", { after: 80 }),
      P("Three controllers were compared over identical levers:", { after: 60 }),
      bullet([T("C1 — metric-only rule-based", { bold: true }),
              T(": deterministic, no motion access (baseline controller).")]),
      bullet([T("C2 — motion-aware rule-based", { bold: true }),
              T(": sets the levers from fixed heuristics on the motion profile "
              + "(e.g. v_max = p95 speed × 1.1).")]),
      bullet([T("C3 — the LLM", { bold: true }),
              T(": sets the same levers by reasoning over the motion profile.")]),

      new Paragraph({ spacing: { before: 150, after: 80 },
        children: [new TextRun({ text: "Code changes — file by file",
          bold: true, size: 22, color: "1F7A6F" })] }),
      P("Each item above maps to the following concrete edits, so a claim "
      + "can be traced to a specific file and symbol rather than a general "
      + "description.", { after: 80, size: 18 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: CCW,
        rows: [row(ccHead, CCW, { bold: true, fill: "1F3864",
          colors: Array(3).fill("FFFFFF") })].concat(
          ccRows.map((r, i) => row(r, CCW,
            { fill: i % 2 ? "FFFFFF" : "EEF1F6", leftAll: true }))) }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. How the motion profile is computed")] }),
      P("The six loss-shaping levers are grounded in a motion profile derived "
      + "from the trajectory targets — the (x, y) position columns of the "
      + "dataset. It is computed once, on the training split only (radar: rows "
      + "0–1199, before windowing), so no test or validation data leaks into "
      + "the controller's view."),
      P("Per-sample speed.", { bold: true, after: 50 }),
      P("Consecutive target points are differenced and the step length "
      + "√(Δx² + Δy²) is multiplied by the dataset sampling rate (radar = "
      + "4 Hz) to give a speed in m/s. The rate conversion makes speed a true "
      + "physical quantity, comparable across the cap (3 Hz), radar (4 Hz) and "
      + "IR (5 Hz) datasets. That per-sample speed series is then summarised "
      + "by the statistics below."),
      P("Speed distribution — radar training split.", { bold: true, after: 60 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: MTW,
        rows: [row(mtHead, MTW, { bold: true, fill: "1F3864",
          colors: Array(3).fill("FFFFFF") })].concat(
          speedRows.map((r, i) => row(r, MTW,
            { fill: i % 2 ? "FFFFFF" : "EEF1F6", leftAll: true }))) }),
      P("v_max is set from p95, not max: p95 (0.60 m/s) captures realistic "
      + "fast walking while ignoring the glitchy top 5%. Both the rule-based "
      + "recipe and the LLM land near v_max = p95 × 1.1 ≈ 0.66 m/s.",
        { before: 80, after: 100 }),
      P("Dwell / stop-go.", { bold: true, after: 50 }),
      P("Each sample is labelled “stop” when its speed is below "
      + "0.05 m/s, otherwise “go”. A dwell episode is a contiguous "
      + "run of stop samples; its run length ÷ sampling rate gives a duration "
      + "in seconds. This characterises the rhythm of movement — smooth "
      + "continuous motion versus a hesitant start-stop gait."),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: MTW,
        rows: [row(mtHead, MTW, { bold: true, fill: "1F3864",
          colors: Array(3).fill("FFFFFF") })].concat(
          dwellRows.map((r, i) => row(r, MTW,
            { fill: i % 2 ? "FFFFFF" : "EEF1F6", leftAll: true }))) }),
      P("Together the radar subject walks slowly (~0.3 m/s), rarely exceeds "
      + "0.6 m/s, and pauses briefly but often — a stop-and-go gait. Speed "
      + "tells the model how fast the target moves; dwell tells it the rhythm. "
      + "Both are the human-behaviour context the LLM is meant to exploit when "
      + "shaping the loss.", { before: 80 }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Experimental design")] }),
      P("Radar dataset, 20 paired random seeds. Each run trains a fixed "
      + "baseline, the LLM arm, random search, and a rule-based arm (C1 or C2). "
      + "Ablations: motion-on vs motion-off (LLM), C1 vs C2, across qwen3:8b, "
      + "llama3.1:8b and phi4-mini:3.8b. Final models are selected on "
      + "prior-free real-metre position error, so reported RMSE is never "
      + "computed from the shaped loss."),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. Results")] }),
      P("Engagement — whether the LLM actually uses the motion levers — is "
      + "strongly model-dependent: phi4-mini 0/100 rounds, llama3.1 negligible, "
      + "qwen3:8b 85/100 rounds. qwen3 reads the observed p95 speed and sets "
      + "v_max from it (proposed v_max mean 0.634; observed p95 0.60, recipe "
      + "value 0.66) — genuine data-grounded reasoning.", { after: 100 }),
      new Table({
        width: { size: CW, type: WidthType.DXA },
        columnWidths: W,
        rows: resultRows.map((r, i) =>
          row(r, W, i === 0
              ? { bold: true, fill: "1F3864", colors: ["FFFFFF","FFFFFF","FFFFFF","FFFFFF"] }
              : { fill: i % 2 ? "FFFFFF" : "EEF1F6" })),
      }),
      P("Test RMSE on the radar test set (lower is better).",
        { italics: true, size: 16, color: "595959", after: 140 }),
      P("Four findings follow from the isolated ablations:", { after: 60 }),
      new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 70 },
        children: [T("Motion loss-shaping is inert on accuracy", { bold: true }),
          T(": the LLM arm scores 0.2891 with shaping on and 0.2899 with it off "
          + "— a 0.0008 difference, far inside the run-to-run noise.")] }),
      new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 70 },
        children: [T("A fixed motion heuristic does not help either", { bold: true }),
          T(": C2 (0.2666) is marginally worse than C1 (0.2601). Adding motion "
          + "loss-shaping to the rule-based controller did not improve it.")] }),
      new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 70 },
        children: [T("The LLM’s motion interpretation does not beat the fixed "
          + "heuristic", { bold: true }),
          T(": C2 ≤ C3 (qwen3, motion-on).")] }),
      new Paragraph({ numbering: { reference: "num", level: 0 }, spacing: { after: 120 },
        children: [T("No controller beats the no-optimisation baseline (0.246)",
          { bold: true }),
          T("; the strongest LLM (llama3.1:8b) merely matches it.")] }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("6. LLM controller behaviour")] }),
      P("Beyond accuracy, we characterise the controller as a stochastic "
      + "decision policy: each round the LLM emits a structured proposal, and "
      + "we measure the properties of that proposal-generation process across "
      + "four models on radar (20 seeds × 5 rounds = 100 rounds per model). "
      + "The gemma3:4b run is the C3 LLM arm of the semantic-repair ablation "
      + "(repair-on), whose system prompt is motion-aware but whose user "
      + "payload omits the static motion_profile; its numbers describe output "
      + "discipline rather than motion engagement."),
      P("Per-round output schema.", { bold: true, after: 50 }),
      P("Every proposal carries a diagnosis (primary_problem ∈ {healthy, "
      + "overfitting, underfitting, plateau, no_data}, a severity, and a "
      + "free-text situation); a strategy label; a free-text reasoning "
      + "justification; a self-reported confidence; an expected_improvement "
      + "prediction; the proposed_changes themselves; and a resets_model flag. "
      + "diagnosis, strategy and confidence are enum-valued fields the "
      + "pipeline parses, logs and depends on."),
      P("Generation cost and action profile.", { bold: true, after: 60 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: BW,
        rows: [row(behHead, BW, { bold: true, fill: "1F3864",
          colors: Array(5).fill("FFFFFF") })].concat(
          costRows.map((r, i) => row(r, BW,
            { fill: i % 2 ? "FFFFFF" : "EEF1F6" }))) }),
      P("Decision tendencies.", { bold: true, before: 100, after: 60 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: BW,
        rows: [row(tendHead, BW, { bold: true, fill: "1F3864",
          colors: Array(5).fill("FFFFFF") })].concat(
          tendRows.map((r, i) => row(r, BW,
            { fill: i % 2 ? "FFFFFF" : "EEF1F6" }))) }),
      P("Findings:", { bold: true, before: 100, after: 60 }),
      numbered([T("Output discipline is capability-ordered.", { bold: true }),
        T(" Share of rounds whose output passes validation untouched: "
        + "llama3:8b 77%, qwen3:8b 68%, gemma3:4b 55%, phi4-mini 35% — "
        + "monotone in model size. phi4-mini also leaks free text into the "
        + "enum-valued strategy and confidence fields (whole sentences stored "
        + "as the 'strategy'), so the smallest model does not reliably hold "
        + "the output structure.")]),
      numbered([T("Compute is dominated by chain-of-thought, not the answer.",
        { bold: true }),
        T(" qwen3:8b — a reasoning model — spends ~1686 completion tokens and "
        + "75 s per round, versus ~110 tokens and 3–5 s for llama3:8b and "
        + "phi4-mini. The parsed reasoning field is short for all three "
        + "(~100–160 characters), so qwen3's tokens go to internal "
        + "deliberation, not a longer proposal.")]),
      numbered([T("Self-reported confidence is uninformative.", { bold: true }),
        T(" llama3:8b reports 'medium' on all 100 rounds; gemma3:4b reports "
        + "'medium' on 77%; qwen3:8b reports 'high' 61% of the time; "
        + "phi4-mini's confidence field is partly malformed. None of it "
        + "tracks the uniformly negative outcome — the field carries no "
        + "usable signal.")]),
      numbered([T("Diagnosis can collapse to a fixed label.", { bold: true }),
        T(" phi4-mini labels the training state 'plateau' in 83% of rounds "
        + "regardless of the metrics — a near-degenerate policy; gemma3:4b "
        + "shows the same lean (plateau 62%); qwen3:8b leans 'overfitting' "
        + "(49%), llama3:8b is more evenly spread.")]),
      numbered([T("Strategy is regularisation-dominated.", { bold: true }),
        T(" qwen3:8b 69%, llama3:8b 90%, gemma3:4b 97% pick 'regularise' "
        + "— consistent with the over-regularised, mildly underfitting models "
        + "of Section 5; the controllers rarely 'exploit'. phi4-mini instead "
        + "defaults to 'explore' (66%).")]),
      numbered([T("Action is small, but reset behaviour differs.",
        { bold: true }),
        T(" All four change only 2–3 levers per round, but phi4-mini "
        + "re-initialises the model in 38% of rounds (discarding learned "
        + "weights), whereas qwen3:8b, llama3:8b and gemma3:4b almost never "
        + "do.")]),
      P("Taken together, the controllers produce fluent, well-structured, "
      + "plausible-looking control output, and the larger models hold the "
      + "format better — but the structured fields the pipeline depends on "
      + "(diagnosis, strategy, confidence) are only weakly informative, and "
      + "confidence is not calibrated at all. This is consistent with the "
      + "headline result of Section 5: none of this behaviour translates into "
      + "an accuracy gain.", { before: 90 }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("7. Conclusion")] }),
      P("On the radar dataset the LLM can analyse motion features and propose "
      + "data-grounded loss shaping — qwen3:8b demonstrably reads the speed "
      + "distribution and sets a physically correct v_max. However, neither the "
      + "LLM’s motion reasoning nor an equivalent fixed motion heuristic "
      + "improves localization accuracy, and the LLM as an optimiser does not "
      + "beat the fixed baseline. The contribution of this phase is a "
      + "well-isolated, ablation-backed negative result rather than a positive "
      + "one: the full C1 / C2 / C3 × motion-on/off × baseline / random "
      + "comparison cleanly separates capability (the LLM can use motion data) "
      + "from effect (it does not change the outcome)."),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("8. Caveats and next steps")] }),
      bullet([T("Radar only.", { bold: true }),
              T(" The cap and IR datasets are pending. The cross-dataset "
              + "adaptivity test — does the LLM shape the loss differently for "
              + "different motion profiles — remains open.")]),
      bullet([T("Structural insulation.", { bold: true }),
              T(" Checkpoint selection uses a prior-free position metric, so "
              + "loss-shaping can only help indirectly via training dynamics; "
              + "this bounds its possible upside.")]),
      bullet([T("Next.", { bold: true }),
              T(" Run cap/IR for the adaptivity test, and assess whether "
              + "selecting checkpoints on the shaped objective changes the "
              + "picture.")]),
    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("Point3_Report_v2.docx", b);
  console.log("wrote Point3_Report_v2.docx");
});

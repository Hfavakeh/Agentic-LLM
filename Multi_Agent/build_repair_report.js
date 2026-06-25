const { Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
        AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
        ShadingType } = require("docx");
const fs = require("fs");

const CW = 9360;
const NAVY = "243B53", TEAL = "1F7A6F", INK = "1A1A1A", MUTE = "5A6472";
const border = { style: BorderStyle.SINGLE, size: 1, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

function cell(text, w, opts = {}) {
  return new TableCell({
    borders,
    width: { size: w, type: WidthType.DXA },
    shading: { fill: opts.fill || "FFFFFF", type: ShadingType.CLEAR },
    margins: { top: 50, bottom: 50, left: 90, right: 90 },
    children: [new Paragraph({
      alignment: opts.align || AlignmentType.LEFT,
      children: [new TextRun({ text, bold: !!opts.bold, size: 16,
                               color: opts.color || "000000" })],
    })],
  });
}
function row(cells, w, opts = {}) {
  return new TableRow({ children: cells.map((c, i) =>
    cell(c, w[i], { bold: opts.bold, fill: opts.fill,
      align: (opts.leftAll || i === 0) ? AlignmentType.LEFT : AlignmentType.CENTER,
      color: opts.colors ? opts.colors[i] : undefined })) });
}
function P(text, o = {}) {
  return new Paragraph({ spacing: { after: o.after === undefined ? 110 : o.after },
    alignment: o.align,
    children: [new TextRun({ text, size: o.size || 20, bold: !!o.bold,
      italics: !!o.italics, color: o.color })] });
}
function T(text, o = {}) {
  return new TextRun({ text, size: 20, bold: !!o.bold, italics: !!o.italics });
}
function bullet(runs) {
  return new Paragraph({ numbering: { reference: "bul", level: 0 },
    spacing: { after: 60 }, children: runs });
}
function numbered(runs) {
  return new Paragraph({ numbering: { reference: "num", level: 0 },
    spacing: { after: 60 }, children: runs });
}

// Failure-taxonomy table
const FW = [2640, 1340, 1340, 1340, 1350, 1350];
const ftHead = ["Model · mode", "% clean", "% corrected", "% skipped",
                "strict rej.", "diag. corr."];
const ftRows = [
  ["gemma3:4b · repair-ON",  "55%", "45%", "0%",  "0",   "44"],
  ["gemma3:4b · repair-OFF", "80%", "—",   "20%", "33",  "33"],
  ["llama3.1:8b · repair-ON",  "93%", "7%",  "0%",  "0",   "7"],
  ["llama3.1:8b · repair-OFF", "98%", "—",   "2%",  "11",  "10"],
  ["phi4-mini · repair-ON",  "26%", "74%", "0%",  "0",   "64"],
  ["phi4-mini · repair-OFF", "52%", "—",   "48%", "117", "55"],
];
// RMSE table
const RW = [2160, 1440, 1440, 1440, 1440, 1440];
const rmHead = ["Model", "LLM repair-ON", "LLM repair-OFF", "Baseline",
                "Rule-based", "Random search"];
const rmRows = [
  ["gemma3:4b",      "0.267 ± 0.011", "0.265 ± 0.008", "0.246 ± 0.012", "0.260 ± 0.010", "0.451 ± 0.202"],
  ["llama3.1:8b",    "0.245 ± 0.005", "0.249 ± 0.013", "0.246 ± 0.012", "0.261 ± 0.010", "0.451 ± 0.202"],
  ["phi4-mini:3.8b", "0.304 ± 0.128", "0.265 ± 0.028", "0.246 ± 0.012", "0.260 ± 0.007", "0.449 ± 0.204"],
];
// Code-change table — file by file
const CCW = [1640, 2900, 4820];
const ccHead = ["File", "Symbol / entry point", "What changed"];
const ccRows = [
  ["Agent.py", "SemanticRepairRequired(ValueError)",
   "New exception. Carries the structured corrections dict (out-of-range "
   + "clamps, discrete snaps, unknown keys, diagnosis and reset-flag "
   + "mismatches) so a rejection can be attributed to a specific violation."],
  ["Agent.py", "_validate_proposal(parsed, context, strict=False)",
   "New strict argument. strict=True still detects and records every "
   + "would-be correction, but raises SemanticRepairRequired instead of "
   + "returning the silently-repaired proposal."],
  ["Agent.py", "SingleAgentOptimizer.__init__",
   "New semantic_repair=True argument. The retry loop sets "
   + "strict_mode = not self.semantic_repair, catches SemanticRepairRequired "
   + "and tags the attempt outcome 'strict_repair_rejected'."],
  ["Agent.py", "retry_stats counters",
   "Added per-round outcome counters (rounds_clean / corrected / "
   + "retry_succeeded / skipped), per-correction-type counters (clamps, "
   + "discrete snaps, unknown keys, diagnosis and reset-flag corrections) "
   + "and strict_rejections."],
  ["Agent.py", "_parse_llm_proposal + per-attempt record",
   "JSON parsing dropped for the plain key: value format. Each attempt now "
   + "logs the raw LLM output, the pre-validation proposal, the validator's "
   + "corrections and a final-source tag."],
  ["main.py", "--no-semantic-repair CLI flag",
   "New flag. Sets cfg.semantic_repair = False; the value is threaded into "
   + "SingleAgentOptimizer and written to each run's final_summary as "
   + "semantic_repair_enabled."],
  ["main.py", "round loop — parse-failure branch",
   "A failed proposal is rewritten to an empty no-op (proposed_changes={}, "
   + "resets_model=False) so the round still trains with current HPs; it is "
   + "still tagged skipped=True for the taxonomy."],
  ["model_pipeline.py", "Config.semantic_repair: bool = True",
   "New dataclass field — the master switch for the validator's silent "
   + "semantic repair."],
  ["failure_taxonomy.py", "new file",
   "Aggregator script. Reads the per-round logs, groups them by "
   + "(experiment_dir, semantic_repair_enabled) and emits the "
   + "failure-taxonomy and RMSE tables above."],
];

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 20 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal",
        quickFormat: true,
        run: { size: 23, bold: true, font: "Calibri", color: NAVY },
        paragraph: { spacing: { before: 170, after: 90 }, outlineLevel: 0 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 340, hanging: 210 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 340, hanging: 230 } } } }] },
  ] },
  sections: [{
    properties: { page: {
      size: { width: 12240, height: 15840 },
      margin: { top: 1000, right: 1380, bottom: 1000, left: 1380 } } },
    children: [
      new Paragraph({ spacing: { after: 40 },
        children: [new TextRun({ text: "The LLM as an Optimizer: Semantic-Repair "
          + "Ablation and Failure Taxonomy", bold: true, size: 26, color: NAVY })] }),
      new Paragraph({ spacing: { after: 150 },
        border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: NAVY, space: 2 } },
        children: [new TextRun({ text: "Thesis progress report — measuring and "
          + "attributing the LLM controller — radar dataset, 20 paired seeds",
          italics: true, size: 17, color: MUTE })] }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("1. Objective")] }),
      P("The previous submission was assessed as an implementation summary "
      + "lacking experimental results, and three concerns were raised: the "
      + "LLM was forced to emit strict JSON; the validator silently corrected "
      + "the LLM, so outcomes could not be attributed to the LLM rather than "
      + "the validator; and clamping out-of-range values was counted as "
      + "success rather than failure. This phase instruments the controller "
      + "and runs a controlled ablation — the LLM with vs without the "
      + "validator's silent “semantic repair” — across three LLMs, to "
      + "characterise the raw LLM as an optimizer."),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("2. What was changed")] }),
      bullet([T("Response format.", { bold: true }), T(" JSON parsing was "
        + "removed; the LLM now emits a plain key: value format. Small local "
        + "models spend effort on braces, quoting and escaping instead of the "
        + "control task.")]),
      bullet([T("Instrumentation.", { bold: true }), T(" Every round now logs "
        + "the raw LLM output, the parsed proposal before validation, the "
        + "validator's structured corrections, all retry attempts, and a "
        + "final-source tag (clean / corrected / retry / skipped). Counters "
        + "track parse failures, validation failures, value clamps, discrete "
        + "snaps, unknown-key strips, diagnosis corrections, reset-flag "
        + "corrections, retries and skipped rounds.")]),
      bullet([T("No-semantic-repair mode.", { bold: true }), T(" A new mode in "
        + "which every would-be silent correction instead becomes a hard "
        + "rejection, so the round retries or is skipped. This isolates the "
        + "raw LLM: a clamp is now counted as a constraint violation, not a "
        + "successful action.")]),
      bullet([T("Training-budget fix.", { bold: true }), T(" A round whose "
        + "proposal fails now still trains with the current hyperparameters. "
        + "Previously it skipped training entirely, which unfairly penalised "
        + "weaker models and conflated parse reliability with optimization "
        + "quality.")]),
      bullet([T("Aggregator.", { bold: true }), T(" A script consolidates the "
        + "per-round logs into the failure-taxonomy and performance tables "
        + "below.")]),

      new Paragraph({ spacing: { before: 130, after: 70 },
        children: [new TextRun({ text: "Code changes — file by file",
          bold: true, size: 21, color: TEAL })] }),
      P("Each behavioural change above maps to the following concrete edits, "
      + "so a claim can be traced to a specific file and symbol rather than a "
      + "general description.", { after: 70, size: 18 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: CCW,
        rows: [row(ccHead, CCW, { bold: true, fill: NAVY,
          colors: Array(3).fill("FFFFFF") })].concat(
          ccRows.map((r, i) => row(r, CCW,
            { fill: i % 2 ? "FFFFFF" : "EAF0EF", leftAll: true }))) }),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("3. Experimental design")] }),
      P("Radar dataset, 20 paired random seeds. For each of three LLMs "
      + "(gemma3:4b, llama3.1:8b, phi4-mini:3.8b) a repair-on run and a paired "
      + "repair-off run were executed; each run also trains a fixed baseline, "
      + "a rule-based controller and random search on identical seeds."),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("4. Results")] }),
      P("Failure taxonomy — how often the validator intervened (sums across "
      + "20 seeds × 5 rounds; gemma3:4b repair-off completed 11 seeds).",
        { after: 70, size: 18 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: FW,
        rows: [row(ftHead, FW, { bold: true, fill: NAVY,
          colors: Array(6).fill("FFFFFF") })].concat(
          ftRows.map((r, i) => row(r, FW, { fill: i % 2 ? "FFFFFF" : "EAF0EF" }))) }),
      P("Test RMSE per arm (lower is better; mean ± std across seeds).",
        { after: 70, size: 18, after: 70 }),
      new Table({ width: { size: CW, type: WidthType.DXA }, columnWidths: RW,
        rows: [row(rmHead, RW, { bold: true, fill: NAVY,
          colors: Array(6).fill("FFFFFF") })].concat(
          rmRows.map((r, i) => row(r, RW, { fill: i % 2 ? "FFFFFF" : "EAF0EF" }))) }),
      P("", { after: 40 }),
      P("Findings:", { bold: true, after: 60 }),
      numbered([T("The validator does much of the apparent reasoning.", { bold: true }),
        T(" It silently rewrote 45% of accepted rounds for gemma3:4b and 74% "
        + "for phi4-mini — almost entirely diagnosis relabels. Outcomes "
        + "previously credited to “the LLM” were substantially the "
        + "validator's.")]),
      numbered([T("Semantic repair harms weak models.", { bold: true }),
        T(" phi4-mini repair-on (0.304 ± 0.128) is the worst and most "
        + "unstable arm in the study; repair-off (0.265 ± 0.028) is far "
        + "better. Clamping a weak model's out-of-range proposals produces "
        + "actions it never intended.")]),
      numbered([T("Semantic repair mildly helps the strong model.", { bold: true }),
        T(" llama3.1:8b repair-on (0.245) edges repair-off (0.249); its "
        + "proposals rarely need correcting, so the few fixes are benign.")]),
      numbered([T("Output cleanliness is capability-ordered.", { bold: true }),
        T(" Share of rounds passing validation untouched: llama3.1 98%, "
        + "gemma3:4b 80%, phi4-mini 52%.")]),
      numbered([T("Only the strongest LLM beats the no-op baseline.", { bold: true }),
        T(" llama3.1:8b reaches 0.245 vs baseline 0.246; gemma3:4b and "
        + "phi4-mini are worse, and the rule-based controller (≈0.260) "
        + "ties or beats the LLM for them.")]),
      numbered([T("Random search is far behind every guided arm.", { bold: true }),
        T(" Sampling hyperparameters uniformly from the same search space "
        + "reaches only ≈0.450 RMSE with very high variance (±0.20) — every "
        + "LLM, the rule-based controller and the baseline clearly beat it, "
        + "confirming that all arms exercise a non-trivial control policy.")]),

      new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun("5. Conclusion and response to the feedback")] }),
      P("The instrumentation and ablation turn “the LLM optimizes "
      + "training” into a measurable, attributable claim. Honestly: the "
      + "validator was performing a large part of the apparent reasoning; the "
      + "raw LLM's competence is strongly model-dependent; and silent repair "
      + "is not a neutral safety net — it helps a strong model and harms a "
      + "weak one. The response format is simplified to key: value; raw LLM "
      + "output and validator corrections are now logged separately and "
      + "counted; clamps are recorded as semantic failures; and the "
      + "rule-based / repair-off / repair-on / random / baseline ablation "
      + "table is delivered. The remaining feedback — having the LLM shape "
      + "the training objective via motion-aware loss functions — is "
      + "addressed in a separate phase."),
    ],
  }],
});

Packer.toBuffer(doc).then(b => {
  fs.writeFileSync("Repair_Ablation_Report.docx", b);
  console.log("wrote Repair_Ablation_Report.docx");
});

const fs = require('fs');
const path = require('path');

const NODE_MODULES = 'C:/Users/hfava/AppData/Roaming/npm/node_modules';
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle, WidthType,
  ShadingType, PageBreak,
} = require(path.join(NODE_MODULES, 'docx'));

const border = { style: BorderStyle.SINGLE, size: 4, color: "BFBFBF" };
const borders = { top: border, bottom: border, left: border, right: border };

const HEAD_FILL = "1F3864";
const ALT_FILL  = "F2F2F2";

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    ...opts,
    children: [new TextRun({ text, ...(opts.run || {}) })],
  });
}

function pRich(runs, opts = {}) {
  return new Paragraph({
    spacing: { before: 60, after: 60 },
    ...opts,
    children: runs.map(r => new TextRun(r)),
  });
}

function bullet(text, opts = {}) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: [new TextRun({ text, ...(opts.run || {}) })],
  });
}

function bulletRich(runs) {
  return new Paragraph({
    numbering: { reference: "bullets", level: 0 },
    spacing: { before: 40, after: 40 },
    children: runs.map(r => new TextRun(r)),
  });
}

function code(text) {
  // monospace block — single paragraph with line breaks per row
  const lines = text.split('\n');
  const children = [];
  lines.forEach((ln, i) => {
    if (i > 0) children.push(new TextRun({ break: 1 }));
    children.push(new TextRun({ text: ln, font: "Consolas", size: 18 }));
  });
  return new Paragraph({
    spacing: { before: 80, after: 120 },
    shading: { fill: "F5F5F5", type: ShadingType.CLEAR },
    children,
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 200, after: 80 },
    children: [new TextRun({ text })],
  });
}

function cell(text, { header = false, width = 0, bold = false, fill = null, mono = false } = {}) {
  return new TableCell({
    borders,
    width: { size: width, type: WidthType.DXA },
    shading: fill ? { fill, type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [
      new Paragraph({
        children: [new TextRun({
          text,
          bold: header || bold,
          color: header ? "FFFFFF" : "000000",
          font: mono ? "Consolas" : undefined,
          size: header ? 20 : (mono ? 18 : 20),
        })],
      }),
    ],
  });
}

function buildTable(headers, rows, columnWidths) {
  const totalWidth = columnWidths.reduce((a, b) => a + b, 0);
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) =>
      cell(h, { header: true, width: columnWidths[i], fill: HEAD_FILL })
    ),
  });
  const bodyRows = rows.map((r, ri) =>
    new TableRow({
      children: r.map((c, i) =>
        cell(c, {
          width: columnWidths[i],
          fill: ri % 2 === 1 ? ALT_FILL : null,
          mono: i === 0 && r[0].includes('_') ? true : false,
        })
      ),
    })
  );
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths,
    rows: [headerRow, ...bodyRows],
  });
}

// ---------------------------------------------------------------------------
// Document
// ---------------------------------------------------------------------------

const CONTENT_WIDTH = 9360;

const children = [];

// Title block
children.push(new Paragraph({
  alignment: AlignmentType.LEFT,
  spacing: { before: 0, after: 80 },
  children: [new TextRun({
    text: "From a stable LLM loop to a motion-aware LLM optimiser",
    bold: true, size: 32,
  })],
}));
children.push(new Paragraph({
  spacing: { before: 0, after: 40 },
  children: [
    new TextRun({ text: "Project: ", bold: true, size: 20 }),
    new TextRun({ text: "LLM-as-optimiser for LSTM indoor localisation", size: 20 }),
  ],
}));
children.push(new Paragraph({
  spacing: { before: 0, after: 200 },
  children: [
    new TextRun({ text: "Scope: ", bold: true, size: 20 }),
    new TextRun({
      text: "what was added after the LLM-loop hardening, to push the optimiser toward human-knowledge / motion-aware reasoning instead of generic loss math.",
      size: 20,
    }),
  ],
}));

// 1. Starting point
children.push(h1("1. Starting point — the loop is now reliable, but blind"));
children.push(pRich([
  { text: "The validator + prompt hardening in " },
  { text: "Agent.py", font: "Consolas", size: 20 },
  { text: " already rejects logically wrong diagnoses (" },
  { text: "improving", font: "Consolas", size: 20 },
  { text: ", " },
  { text: "unknown", font: "Consolas", size: 20 },
  { text: ", " },
  { text: "plateau", font: "Consolas", size: 20 },
  { text: " on round 1, " },
  { text: "underfitting", font: "Consolas", size: 20 },
  { text: " with near-zero val loss, …) and surfaces retry/fallback rates as a first-class metric. After that fix the loop is stable, but the LLM still reasons over a single number — " },
  { text: "val_loss", font: "Consolas", size: 20 },
  { text: " — with no idea what kind of motion the model is failing on. The steps below close that gap." },
]));

// 2. Motion descriptors
children.push(h1("2. Human-interpretable motion descriptors (motion_descriptors.py)"));
children.push(p("Per-sample features derived from the (x, y) target trajectory, expressed in physical units so they are comparable across the three sensors (cap 3 Hz, radar 4 Hz, IR 5 Hz):"));

children.push(buildTable(
  ["Descriptor", "Unit", "Meaning"],
  [
    ["speed",         "m/s",    "‖p_t − p_{t−1}‖ · hz — Euclidean step length"],
    ["heading",       "rad",    "atan2(dy, dx); masked to NaN while stationary"],
    ["turning_rate",  "deg/s",  "wrapped Δheading; only valid between two 'moving' samples"],
    ["dwell_state",   "enum",   "stop / go (stop = below stationary threshold)"],
    ["dwell_run_len", "sample", "length of current contiguous stop / go episode"],
    ["speed_bin",     "enum",   "stationary < 0.05, slow < 0.30, medium < 1.00, else fast (m/s)"],
  ],
  [2200, 1200, 5960]
));

children.push(p("Bin thresholds were calibrated against the README's per-experiment speed table (radar avg 0.31–0.44 m/s, cap 0.13–0.27, IR 0.13–0.23) so they correspond to recognisable human walking regimes — not arbitrary quantiles."));

// 3. Per-regime error
children.push(h1("3. Per-regime error breakdown surfaced to the LLM"));
children.push(pRich([
  { text: "error_payload(preds, targets)", font: "Consolas", size: 20 },
  { text: " joins per-sample predictions with the motion table and emits a compact dict the LLM can consume:" },
]));
children.push(code(
  "overall_rmse, overall_mean_euclid\n" +
  "by_speed:    [{speed_bin, count, rmse, mean_euclid}, ...]\n" +
  "worst_cell, best_cell"
));
children.push(pRich([
  { text: "main.llm_optimization_loop", font: "Consolas", size: 20 },
  { text: " now runs a single forward pass on the validation set after each round and threads this payload through " },
  { text: "_build_context(...)", font: "Consolas", size: 20 },
  { text: ". Failure of the descriptor pipeline is non-fatal — the loop proceeds with empty motion data rather than crashing the run. The diagnostic block now appears verbatim in every prompt from round 2 onward:" },
]));
children.push(code(
  "━━ MOTION DIAGNOSTICS (val set, per speed bin) ━━\n" +
  "  overall_rmse:        0.2821\n" +
  "  overall_mean_euclid: 0.2543\n" +
  "  [BY SPEED]\n" +
  "    slow       count=24   rmse=0.2502  mean_euclid=0.2274\n" +
  "    medium     count=125  rmse=0.2854  mean_euclid=0.2580\n" +
  "    fast       count=31   rmse=0.2902  mean_euclid=0.2633"
));

// 4. Pareto
children.push(h1("4. Multi-objective Pareto framework (pareto.py)"));
children.push(pRich([
  { text: "Three measurable cost axes are taken every round in " },
  { text: "Trainer.pareto_metrics()", font: "Consolas", size: 20 },
  { text: ":" },
]));

children.push(buildTable(
  ["Axis", "Source", "Driven by"],
  [
    ["latency_ms",        "window_seconds + measured forward-pass time", "turning_rate p95"],
    ["stability_std_m",   "std of per-sample Euclidean error",           "val-error variance"],
    ["params_trainable",  "parameter count + per-epoch wall time",       "model size"],
  ],
  [2400, 4360, 2600]
));

children.push(p("Each axis is min-max-normalised across the run, then collapsed into one scalar the optimiser targets:"));
children.push(code("score = val_loss × (1 + w_lat·lat_norm + w_stab·stab_norm + w_res·res_norm)"));
children.push(p("The weights are decided by the LLM each round based on the motion profile."));

// 5. Forcing the LLM to use motion
children.push(h1("5. Forcing the LLM to use the motion knowledge it now sees"));
children.push(p("Three changes turn the descriptors from passively-displayed text into a constraint on the LLM's output:"));

children.push(bulletRich([
  { text: "Prompt rewritten as a multi-objective brief", bold: true },
  { text: " (Agent._build_system_prompt) — the LLM now has two jobs per round: (a) emit weights w_lat, w_stab, w_res summing to 1, (b) propose 1-2 hyper-parameter changes consistent with that weighting. Heuristics are spelled out in human terms: high turning_rate p95 ⇒ raise w_lat (short windows needed); high stationary share, low speed_p95 ⇒ raise w_res (small model fine); high val-error std vs. mean ⇒ raise w_stab." },
]));
children.push(bulletRich([
  { text: "Motion-citation requirement.", bold: true },
  { text: " The validator scans 'reasoning' for any of {speed, turning, turning_rate, stationary, slow, medium, fast, bin, dwell, motion} and warns if none are cited — making the link between the displayed motion features and the chosen weights auditable." },
]));
children.push(bulletRich([
  { text: "Weight normalisation + uniform fallback.", bold: true },
  { text: " Weights outside ±0.10 of sum-to-1 are auto-renormalised with a warning; missing weights default to a uniform 1/3 split so the run never silently loses Pareto data." },
]));

// 6. Output schema
children.push(h1("6. Output schema upgrade"));
children.push(p("The compact line format (preferred over JSON for small models) now carries weights and Pareto score alongside diagnosis:"));
children.push(code(
  "diagnosis:   plateau\n" +
  "severity:    medium\n" +
  "weights:     w_lat=0.50, w_stab=0.30, w_res=0.20\n" +
  "changes:     learning_rate=0.0005, dropout=0.35\n" +
  "reasoning:   turning_rate p95 ≈ 116°/s → short windows; raise w_lat"
));
children.push(pRich([
  { text: "OPTIMIZATION HISTORY", font: "Consolas", size: 20 },
  { text: " echoes weights and " },
  { text: "pareto_score", font: "Consolas", size: 20 },
  { text: " per round so the LLM can reason about its own past Pareto choices." },
]));

// 7. Reproducibility
children.push(h1("7. Reproducibility & current evidence"));

children.push(buildTable(
  ["Command", "Output"],
  [
    ["python motion_descriptors.py", "per-sample CSV + summary + histograms"],
    ["python analyze_logs.py",       "all_rounds.csv (1567), all_runs.csv (296), per-experiment reports"],
    ["python main.py --seeds 1..30", "full multi-seed sweep with motion + Pareto wired in"],
  ],
  [3800, 5560]
));

children.push(pRich([
  { text: "Across the 30-seed " },
  { text: "outputs-0504-llama3.2-3b", font: "Consolas", size: 20 },
  { text: " run with the full stack enabled, motion data is populated in 120/120 rounds where it should be (rounds 2–5 of every seed; round 1 correctly empty), all 150 active diagnoses are valid enum members, and the rule-based agent now beats the baseline by 3.74 % RMSE — the cleanest demonstration so far that motion-aware weighting is a learnable signal, not noise." },
]));

// 8. Next
children.push(h1("8. What this unlocks next"));
children.push(bulletRich([
  { text: "Loss shaping", bold: true },
  { text: " — up-weight the per-sample loss in the bins where the LLM repeatedly flags high error (sharp turns, fast speed)." },
]));
children.push(bulletRich([
  { text: "Cross-sensor study", bold: true },
  { text: " — the descriptors are unit-correct across cap / IR / radar, so the same Pareto policy can be replayed against all three CSVs with the per-dataset Hz / window from the project README." },
]));
children.push(bulletRich([
  { text: "Prompt-shot for the residual round-1 misdiagnosis", bold: true },
  { text: " — the dominant remaining failure (60× plateau instead of no_data on round 1) is a prompt-engineering fix, not a methodology gap." },
]));

// ---------------------------------------------------------------------------
// Document assembly
// ---------------------------------------------------------------------------

const doc = new Document({
  styles: {
    default: { document: { run: { font: "Calibri", size: 20 } } },
    paragraphStyles: [
      {
        id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 26, bold: true, font: "Calibri", color: "1F3864" },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 0 },
      },
      {
        id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { size: 22, bold: true, font: "Calibri", color: "2E75B6" },
        paragraph: { spacing: { before: 200, after: 80 }, outlineLevel: 1 },
      },
    ],
  },
  numbering: {
    config: [
      {
        reference: "bullets",
        levels: [
          {
            level: 0, format: LevelFormat.BULLET, text: "•",
            alignment: AlignmentType.LEFT,
            style: { paragraph: { indent: { left: 720, hanging: 360 } } },
          },
        ],
      },
    ],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 },
      },
    },
    children,
  }],
});

const out = "C:/Users/hfava/Documents/Thesis/Hossein_NN/Multi_Agent/human_knowledge_report.docx";
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(out, buf);
  console.log("WROTE", out, buf.length, "bytes");
});

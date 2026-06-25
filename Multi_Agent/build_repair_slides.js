const pptxgen = require("pptxgenjs");
const p = new pptxgen();
p.defineLayout({ name: "W", width: 13.333, height: 7.5 });
p.layout = "W";

const NAVY = "243B53", TEAL = "1F7A6F", INK = "1F2937", MUTE = "6B7280";
const CARD = "EAF0EF", WHITE = "FFFFFF", ICE = "CFE3DF";
const RED = "B5524B", GOOD = "2C7A6B", AMBER = "C08A2E";
const HF = "Georgia", BF = "Calibri";

function title(s, t) {
  s.addText(t, { x: 0.6, y: 0.42, w: 12.13, h: 0.85, fontFace: HF, fontSize: 29,
    bold: true, color: NAVY, valign: "middle" });
}

// ---- Slide 1 — Title -----------------------------------------------------
let s = p.addSlide();
s.background = { color: NAVY };
s.addText("THESIS PROGRESS REPORT  ·  CONTROLLER MEASUREMENT", {
  x: 0.75, y: 1.3, w: 11, h: 0.4, fontFace: BF, fontSize: 14, bold: true,
  color: TEAL, charSpacing: 3 });
s.addText("The LLM as an optimizer:\nsemantic-repair ablation & failure taxonomy", {
  x: 0.75, y: 1.9, w: 11.8, h: 2.3, fontFace: HF, fontSize: 36, bold: true,
  color: WHITE, lineSpacing: 44 });
s.addText("Instrumenting the controller and running the LLM with vs without "
  + "the validator's silent corrections — to measure, and correctly "
  + "attribute, the raw LLM as an optimizer.", {
  x: 0.75, y: 4.45, w: 10.8, h: 1.1, fontFace: BF, fontSize: 18, color: ICE,
  lineSpacing: 26 });
s.addText("Radar dataset  ·  20 paired seeds  ·  gemma3:4b · llama3.1:8b · "
  + "phi4-mini:3.8b", {
  x: 0.75, y: 6.45, w: 12, h: 0.4, fontFace: BF, fontSize: 13, color: "8FA6B6" });

// ---- Slide 2 — What was changed ------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "What was changed");
const ch = [
  ["Simpler response format", "Dropped JSON — the LLM now emits plain "
    + "key: value lines. Small models no longer spend effort on braces, "
    + "quoting and escaping."],
  ["Full instrumentation", "Each round logs raw LLM output, the pre-validation "
    + "proposal, every validator correction, all retries, and a final-source "
    + "tag (clean / corrected / skipped)."],
  ["No-semantic-repair mode", "Every would-be silent fix becomes a hard "
    + "rejection. A clamp is now counted as a constraint violation, not a "
    + "successful action."],
  ["Training-budget fix", "A failed round still trains with current HPs — "
    + "previously it skipped training, conflating parse reliability with "
    + "optimization quality."],
];
ch.forEach((c, i) => {
  const col = i % 2, rowi = Math.floor(i / 2);
  const x = 0.6 + col * 6.18, y = 1.5 + rowi * 2.55;
  s.addShape(p.ShapeType.rect, { x, y, w: 5.95, h: 2.3, fill: { color: CARD },
    line: { color: CARD } });
  s.addShape(p.ShapeType.rect, { x, y, w: 0.12, h: 2.3, fill: { color: TEAL },
    line: { color: TEAL } });
  s.addText(c[0], { x: x + 0.3, y: y + 0.2, w: 5.4, h: 0.55, fontFace: HF,
    fontSize: 17, bold: true, color: NAVY });
  s.addText(c[1], { x: x + 0.3, y: y + 0.78, w: 5.45, h: 1.4, fontFace: BF,
    fontSize: 13, color: INK, lineSpacing: 18 });
});

// ---- Slide 3 — Failure taxonomy ------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "The failure taxonomy: how often the validator intervened");
s.addText("Per model × mode, over 20 seeds × 5 rounds. “corrected” = the "
  + "validator silently rewrote the proposal; “skipped” = no valid "
  + "proposal, round trains unchanged.", {
  x: 0.6, y: 1.3, w: 12.13, h: 0.55, fontFace: BF, fontSize: 13, color: MUTE });
const fhead = ["Model · mode", "% clean", "% corrected", "% skipped",
               "strict rej.", "diag. corr."];
const frows = [
  ["gemma3:4b · repair-ON",  "55%", "45%", "0%",  "0",   "44"],
  ["gemma3:4b · repair-OFF", "80%", "—", "20%", "33", "33"],
  ["llama3.1:8b · repair-ON",  "93%", "7%",  "0%",  "0",  "7"],
  ["llama3.1:8b · repair-OFF", "98%", "—", "2%", "11", "10"],
  ["phi4-mini · repair-ON",  "26%", "74%", "0%",  "0",   "64"],
  ["phi4-mini · repair-OFF", "52%", "—", "48%", "117", "55"],
];
const ftbl = [fhead.map(h => ({ text: h, options: { bold: true, color: WHITE,
  fill: { color: NAVY }, align: "center", fontFace: BF } }))];
ftbl[0][0].options.align = "left";
frows.forEach((r, i) => {
  const bg = i % 2 ? "FFFFFF" : "EAF0EF";
  ftbl.push(r.map((v, j) => ({ text: v, options: { color: INK,
    fill: { color: bg }, align: j === 0 ? "left" : "center", fontFace: BF,
    bold: j === 0 } })));
});
s.addTable(ftbl, { x: 0.6, y: 2.0, w: 12.13, colW: [3.13, 1.8, 1.8, 1.8, 1.8, 1.8],
  rowH: 0.5, fontSize: 13, valign: "middle",
  border: { type: "solid", color: "CBD9D6", pt: 1 } });
s.addShape(p.ShapeType.rect, { x: 0.6, y: 5.45, w: 12.13, h: 1.05, fill: { color: NAVY },
  line: { color: NAVY } });
s.addText([
  { text: "The attribution problem, quantified:  ", options: { bold: true, color: TEAL } },
  { text: "with repair on, the validator silently rewrote 45% of gemma3:4b "
    + "rounds and 74% of phi4-mini rounds — almost all diagnosis relabels. "
    + "Results credited to “the LLM” were substantially the validator's.",
    options: { color: ICE } },
], { x: 0.9, y: 5.55, w: 11.5, h: 0.85, fontFace: BF, fontSize: 13,
  valign: "middle", lineSpacing: 17 });

// ---- Slide 4 — Results ---------------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "Results: test RMSE, repair-ON vs repair-OFF");
const rhead = ["Model", "LLM repair-ON", "LLM repair-OFF", "Baseline", "Rule-based"];
const rrows = [
  ["gemma3:4b",      "0.267 ± 0.011", "0.265 ± 0.008", "0.246", "0.260"],
  ["llama3.1:8b",    "0.245 ± 0.005", "0.249 ± 0.013", "0.246", "0.261"],
  ["phi4-mini:3.8b", "0.304 ± 0.128", "0.265 ± 0.028", "0.246", "0.260"],
];
const rtbl = [rhead.map(h => ({ text: h, options: { bold: true, color: WHITE,
  fill: { color: NAVY }, align: "center", fontFace: BF } }))];
rtbl[0][0].options.align = "left";
rrows.forEach((r, i) => {
  const bg = i % 2 ? "FFFFFF" : "EAF0EF";
  rtbl.push(r.map((v, j) => ({ text: v, options: { color: INK,
    fill: { color: bg }, align: j === 0 ? "left" : "center", fontFace: BF,
    bold: j === 0 } })));
});
s.addTable(rtbl, { x: 0.6, y: 1.5, w: 12.13, colW: [2.93, 2.55, 2.55, 2.05, 2.05],
  rowH: 0.62, fontSize: 14, valign: "middle",
  border: { type: "solid", color: "CBD9D6", pt: 1 } });
const cards = [
  ["Weak model, repair HURTS", "phi4-mini repair-ON 0.304 ± 0.128 — the "
    + "worst, most unstable arm. Clamped proposals are not what the LLM "
    + "intended.", RED],
  ["Strong model, repair helps a little", "llama3.1:8b repair-ON 0.245 edges "
    + "repair-OFF 0.249 — the few corrections it needs are benign.", GOOD],
  ["Only llama3.1 beats baseline", "0.245 vs baseline 0.246. gemma3:4b and "
    + "phi4-mini stay worse; rule-based (~0.260) ties or beats them.", AMBER],
];
cards.forEach((c, i) => {
  const x = 0.6 + i * 4.12;
  s.addShape(p.ShapeType.rect, { x, y: 3.85, w: 3.89, h: 2.65, fill: { color: CARD },
    line: { color: CARD } });
  s.addShape(p.ShapeType.rect, { x, y: 3.85, w: 3.89, h: 0.14, fill: { color: c[2] },
    line: { color: c[2] } });
  s.addText(c[0], { x: x + 0.22, y: 4.05, w: 3.5, h: 0.85, fontFace: HF,
    fontSize: 14.5, bold: true, color: NAVY });
  s.addText(c[1], { x: x + 0.22, y: 4.9, w: 3.5, h: 1.5, fontFace: BF,
    fontSize: 12, color: INK, lineSpacing: 16 });
});

// ---- Slide 5 — Findings --------------------------------------------------
s = p.addSlide();
s.background = { color: WHITE };
title(s, "What this tells us about the LLM as an optimizer");
const finds = [
  ["1", "The validator did much of the reasoning", "Up to 74% of accepted "
    + "rounds were silently rewritten — mostly diagnosis relabels."],
  ["2", "Repair is not a neutral safety net", "It helps a strong model and "
    + "harms a weak one — its sign depends on model capability."],
  ["3", "Competence is capability-ordered", "Untouched-output rate: "
    + "llama3.1 98%, gemma3:4b 80%, phi4-mini 52%."],
  ["4", "The raw LLM rarely beats simple baselines", "Only llama3.1:8b "
    + "matches the no-op baseline; rule-based ties or beats the rest."],
];
finds.forEach((f, i) => {
  const col = i % 2, rowi = Math.floor(i / 2);
  const x = 0.6 + col * 6.18, y = 1.5 + rowi * 2.5;
  s.addShape(p.ShapeType.rect, { x, y, w: 5.95, h: 2.25, fill: { color: CARD },
    line: { color: CARD } });
  s.addShape(p.ShapeType.ellipse, { x: x + 0.28, y: y + 0.3, w: 0.68, h: 0.68,
    fill: { color: TEAL }, line: { color: TEAL } });
  s.addText(f[0], { x: x + 0.28, y: y + 0.3, w: 0.68, h: 0.68, fontFace: HF,
    fontSize: 22, bold: true, color: WHITE, align: "center", valign: "middle" });
  s.addText(f[1], { x: x + 1.18, y: y + 0.3, w: 4.5, h: 0.7, fontFace: HF,
    fontSize: 15.5, bold: true, color: NAVY, valign: "middle" });
  s.addText(f[2], { x: x + 0.34, y: y + 1.12, w: 5.3, h: 1.0, fontFace: BF,
    fontSize: 12.5, color: INK, lineSpacing: 16 });
});

// ---- Slide 6 — Conclusion (dark) -----------------------------------------
s = p.addSlide();
s.background = { color: NAVY };
s.addText("Conclusion: a measurable, attributable claim", {
  x: 0.75, y: 0.6, w: 11.8, h: 0.8, fontFace: HF, fontSize: 29, bold: true,
  color: WHITE });
s.addText("The instrumentation and ablation turn “the LLM optimizes "
  + "training” into something measurable. Honestly: the validator was "
  + "performing a large part of the apparent reasoning, the raw LLM's "
  + "competence is strongly model-dependent, and silent repair helps a "
  + "strong model while harming a weak one.", {
  x: 0.75, y: 1.5, w: 11.8, h: 1.55, fontFace: BF, fontSize: 15, color: ICE,
  lineSpacing: 23 });
s.addText("Addresses the feedback:", {
  x: 0.75, y: 3.15, w: 11, h: 0.4, fontFace: HF, fontSize: 16, bold: true,
  color: TEAL });
const resp = [
  ["Format", "JSON dropped for a plain key: value format."],
  ["Attribution", "Raw LLM output and validator corrections logged separately and counted."],
  ["Honest accounting", "Clamps recorded as semantic failures, not successes."],
  ["Ablations", "Rule-based / repair-off / repair-on / random / baseline table delivered."],
  ["Open", "Motion-aware loss shaping — the scientific core — is a separate phase."],
];
resp.forEach((r, i) => {
  const y = 3.62 + i * 0.72;
  s.addShape(p.ShapeType.rect, { x: 0.75, y: y + 0.05, w: 0.12, h: 0.5,
    fill: { color: TEAL }, line: { color: TEAL } });
  s.addText(r[0], { x: 1.05, y, w: 2.7, h: 0.6, fontFace: HF, fontSize: 14,
    bold: true, color: TEAL, valign: "middle" });
  s.addText(r[1], { x: 3.8, y, w: 8.75, h: 0.6, fontFace: BF, fontSize: 13,
    color: ICE, valign: "middle" });
});

p.writeFile({ fileName: "Repair_Ablation_Presentation.pptx" })
  .then(() => console.log("wrote Repair_Ablation_Presentation.pptx"));

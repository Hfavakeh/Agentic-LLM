"""
generate_report.py
==================
Reads one or more conversation_log_run*.json files produced by the
experiment and generates a single self-contained HTML report that
shows the professor exactly what the LLM received and responded with
each round — presented in a clean, navigable format.

Usage
-----
# Auto-discover all logs in an output directory:
python generate_report.py --dir outputs-0416-qwen2.5-coder

# Specific log file(s):
python generate_report.py --logs outputs/conversation_log_run1.json

# Multi-seed run (searches subdirectories):
python generate_report.py --dir outputs-0416-qwen2.5-coder --recursive

# Custom output path:
python generate_report.py --dir outputs --out report.html
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_logs(paths: List[Path]) -> List[Dict[str, Any]]:
    """Load and label conversation log files."""
    runs = []
    for p in sorted(paths):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            runs.append({"file": str(p), "rounds": data if isinstance(data, list) else []})
        except Exception as exc:
            print(f"  WARNING: could not load {p}: {exc}", file=sys.stderr)
    return runs


def _discover_logs(directory: Path, recursive: bool) -> List[Path]:
    pattern = "conversation_log_run*.json"
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))


def _colour_for_diagnosis(problem: str) -> str:
    return {
        "overfitting":  "#e74c3c",
        "underfitting": "#e67e22",
        "plateau":      "#8e44ad",
        "healthy":      "#27ae60",
        "no_data":      "#7f8c8d",
    }.get(problem, "#2c3e50")


def _badge(text: str, colour: str) -> str:
    return (
        f'<span style="background:{colour};color:#fff;padding:2px 9px;'
        f'border-radius:12px;font-size:0.78em;font-weight:600;'
        f'letter-spacing:0.5px">{text}</span>'
    )


def _json_block(obj: Any, label: str = "", collapsed: bool = True) -> str:
    """Collapsible <details> block with syntax-highlighted JSON."""
    raw = json.dumps(obj, indent=2, default=str) if not isinstance(obj, str) else obj
    # Minimal syntax colouring via regex replacement
    raw_esc = (
        raw.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    def _colourise(s: str) -> str:
        s = re.sub(r'"(.*?)"(?=\s*:)', r'<span class="jk">"\1"</span>', s)
        s = re.sub(r':\s*"(.*?)"', lambda m: ': <span class="jv">"' + m.group(1) + '"</span>', s)
        s = re.sub(r':\s*(-?\d+\.?\d*(?:e[+-]?\d+)?)', r': <span class="jn">\1</span>', s)
        s = re.sub(r':\s*(true|false|null)\b', r': <span class="jb">\1</span>', s)
        return s

    body = _colourise(raw_esc)
    summary = label or "JSON"
    state   = "" if collapsed else " open"
    return (
        f'<details{state} class="jblock">'
        f'<summary>{summary} <span class="toggle-hint">click to expand</span></summary>'
        f'<pre class="jcode">{body}</pre>'
        f'</details>'
    )


# ---------------------------------------------------------------------------
# HTML sections
# ---------------------------------------------------------------------------

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #f0f2f5;
       color: #1a1a2e; line-height: 1.5; }

/* ── Top nav ── */
.topbar { background: #1a1a2e; color: #fff; padding: 14px 32px;
          display: flex; align-items: center; gap: 16px; position: sticky;
          top: 0; z-index: 100; box-shadow: 0 2px 8px rgba(0,0,0,.35); }
.topbar h1 { font-size: 1.15em; font-weight: 700; letter-spacing: .3px; }
.topbar .subtitle { font-size: .85em; opacity: .65; }

/* ── Layout ── */
.container { max-width: 1200px; margin: 0 auto; padding: 28px 20px 60px; }
.run-header { background: #fff; border-radius: 10px; padding: 18px 24px;
              margin-bottom: 28px; border-left: 5px solid #2980b9;
              box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.run-header h2 { font-size: 1.05em; color: #2980b9; margin-bottom: 4px; }
.run-header .meta { font-size: .82em; color: #666; }

/* ── Round card ── */
.round-card { background: #fff; border-radius: 10px; margin-bottom: 22px;
              box-shadow: 0 1px 5px rgba(0,0,0,.09); overflow: hidden; }
.round-card-header { display: flex; align-items: center; gap: 12px;
                     padding: 13px 22px; border-bottom: 1px solid #eee;
                     cursor: pointer; user-select: none; }
.round-card-header:hover { background: #fafbfc; }
.round-num { font-weight: 700; font-size: 1em; color: #555; min-width: 68px; }
.round-card-body { padding: 0 22px 20px; }

/* ── Section titles ── */
.section { margin-top: 18px; }
.section-title { font-size: .78em; font-weight: 700; text-transform: uppercase;
                 letter-spacing: .9px; color: #888; margin-bottom: 8px;
                 border-bottom: 1px solid #eee; padding-bottom: 4px; }

/* ── Metric grid ── */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fill,minmax(180px,1fr));
               gap: 10px; margin-top: 4px; }
.metric-box { background: #f8f9fc; border-radius: 7px; padding: 10px 14px;
              border: 1px solid #e8eaf0; }
.metric-box .label { font-size: .72em; color: #888; text-transform: uppercase;
                     letter-spacing: .5px; }
.metric-box .value { font-size: 1.05em; font-weight: 700; color: #1a1a2e;
                     margin-top: 2px; word-break: break-all; }

/* ── HP table ── */
.hp-table { width: 100%; border-collapse: collapse; font-size: .85em; }
.hp-table th { text-align: left; padding: 6px 10px; background: #f0f2f5;
               font-weight: 600; color: #555; }
.hp-table td { padding: 5px 10px; border-top: 1px solid #f0f2f5; }
.hp-table tr:hover td { background: #fafbfc; }
.hp-changed { color: #e67e22; font-weight: 700; }

/* ── Proposal box ── */
.proposal-box { background: #f0f9f4; border: 1px solid #b7dfca;
                border-radius: 8px; padding: 14px 18px; margin-top: 6px; }
.proposal-box .reasoning { font-style: italic; color: #2c7a50; margin-bottom: 8px; }
.prop-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }
.prop-item { font-size: .85em; }
.prop-item .pk { color: #888; }
.prop-item .pv { font-weight: 600; }

/* ── Raw output box ── */
.raw-output { background: #1e1e2e; color: #cdd6f4; border-radius: 8px;
              padding: 14px 16px; font-family: 'Consolas','Courier New',monospace;
              font-size: .8em; white-space: pre-wrap; word-break: break-all;
              max-height: 340px; overflow-y: auto; margin-top: 6px; }

/* ── JSON blocks ── */
details.jblock { margin-top: 4px; }
details.jblock summary { cursor: pointer; font-size: .82em; font-weight: 600;
                          color: #555; padding: 5px 2px; list-style: none; }
details.jblock summary::before { content: "▶ "; font-size: .75em; }
details[open].jblock summary::before { content: "▼ "; }
.toggle-hint { font-weight: 400; font-size: .9em; color: #aaa; }
pre.jcode { background: #f5f6fa; border: 1px solid #e0e3ec; border-radius: 6px;
            padding: 10px 14px; font-size: .8em; font-family: 'Consolas',
            'Courier New', monospace; white-space: pre-wrap;
            word-break: break-all; max-height: 400px; overflow-y: auto;
            margin-top: 6px; }
.jk  { color: #0969da; }
.jv  { color: #0a3069; }
.jn  { color: #e36209; }
.jb  { color: #8250df; }

/* ── System prompt ── */
.sysprompt { background: #fff9e6; border: 1px solid #ffe58f; border-radius: 8px;
             padding: 14px 18px; font-family: 'Consolas','Courier New',monospace;
             font-size: .8em; white-space: pre-wrap; word-break: break-all;
             max-height: 300px; overflow-y: auto; margin-top: 6px; }

/* ── Fallback banner ── */
.fallback-banner { background: #fff3cd; border: 1px solid #ffc107;
                   border-radius: 6px; padding: 8px 14px; font-size: .83em;
                   color: #856404; margin-top: 8px; }

/* ── TOC ── */
.toc { background: #fff; border-radius: 10px; padding: 18px 24px;
       margin-bottom: 28px; box-shadow: 0 1px 4px rgba(0,0,0,.08); }
.toc h3 { font-size: .9em; color: #555; margin-bottom: 10px; }
.toc ul { list-style: none; display: flex; flex-wrap: wrap; gap: 8px; }
.toc a { color: #2980b9; font-size: .85em; text-decoration: none;
         padding: 3px 10px; border: 1px solid #d0e4f7; border-radius: 20px; }
.toc a:hover { background: #eaf3fb; }
"""


def _render_metrics(metrics: Dict, round_summary: Dict) -> str:
    items = []
    key_labels = [
        ("val_loss",                  "Val Loss"),
        ("train_loss",                "Train Loss"),
        ("val_mae",                   "Val MAE"),
        ("mean_euclidean_distance_m", "Euclidean Dist (m)"),
    ]
    for k, label in key_labels:
        v = metrics.get(k)
        if v is not None:
            items.append(f'<div class="metric-box"><div class="label">{label}</div>'
                         f'<div class="value">{float(v):.5f}</div></div>')

    rs_labels = [
        ("round_best_val_loss", "Round Best ValLoss"),
        ("round_avg_val_loss",  "Round Avg ValLoss"),
        ("round_best_rmse",     "Round Best RMSE"),
        ("round_avg_rmse",      "Round Avg RMSE"),
        ("round_epochs_trained","Epochs This Round"),
    ]
    for k, label in rs_labels:
        v = round_summary.get(k)
        if v is not None:
            fmt = f"{float(v):.5f}" if isinstance(v, float) else str(v)
            items.append(f'<div class="metric-box"><div class="label">{label}</div>'
                         f'<div class="value">{fmt}</div></div>')

    return '<div class="metric-grid">' + "".join(items) + "</div>"


def _render_hyperparams(current_hps: Dict, proposed_changes: Dict) -> str:
    rows = []
    for k, v in sorted(current_hps.items()):
        changed = k in (proposed_changes or {})
        cls     = ' class="hp-changed"' if changed else ""
        new_val = f" → <strong>{proposed_changes[k]}</strong>" if changed else ""
        rows.append(f"<tr><td>{k}</td>"
                    f"<td{cls}>{v}{new_val}</td></tr>")
    return ('<table class="hp-table"><tr><th>Parameter</th><th>Value</th></tr>'
            + "".join(rows) + "</table>")


def _render_proposal(proposal: Dict) -> str:
    changes = proposal.get("proposed_changes") or {}
    fb = proposal.get("fallback_used", False)

    items = []
    for field, label in [
        ("strategy",             "Strategy"),
        ("confidence",           "Confidence"),
        ("expected_improvement", "Expected improvement"),
    ]:
        v = proposal.get(field)
        if v:
            items.append(f'<div class="prop-item"><span class="pk">{label}: </span>'
                         f'<span class="pv">{v}</span></div>')

    changes_html = ", ".join(
        f"<strong>{k}</strong> = {v}" for k, v in changes.items()
    ) or "<em>no changes</em>"

    fallback_banner = (
        f'<div class="fallback-banner">⚠ Rule-based fallback used'
        f'{": " + proposal.get("fallback_reason","") if proposal.get("fallback_reason") else ""}'
        f'</div>'
    ) if fb else ""

    return (
        f'{fallback_banner}'
        f'<div class="proposal-box">'
        f'<div class="reasoning">"{proposal.get("reasoning", "—")}"</div>'
        f'<div><strong>Changes proposed:</strong> {changes_html}</div>'
        f'<div class="prop-grid" style="margin-top:8px">{"".join(items)}</div>'
        f'</div>'
    )


def _render_round(entry: Dict, run_idx: int) -> str:
    r        = entry.get("round", "?")
    diag     = entry.get("diagnosis", {})
    problem  = diag.get("primary_problem", "unknown")
    severity = diag.get("severity", "unknown")
    situation = diag.get("situation", "")
    colour   = _colour_for_diagnosis(problem)
    elapsed  = entry.get("total_time_s", 0.0)

    badge_prob = _badge(problem, colour)
    badge_sev  = _badge(severity, "#555")
    card_id    = f"r{run_idx}_round{r}"

    llm_input    = entry.get("llm_input", {})
    raw_output   = entry.get("llm_raw_output", "")
    metrics      = llm_input.get("user_payload", {}).get("metrics", {})
    round_summary = llm_input.get("user_payload", {}).get("round_summary", {})
    current_hps  = llm_input.get("user_payload", {}).get("current_hyperparameters", {})
    opt_hist     = llm_input.get("user_payload", {}).get("optimization_history", [])
    tool_results = llm_input.get("user_payload", {}).get("tool_results", {})
    proposal     = entry.get("proposal", {})

    body = f"""
<div class="round-card" id="{card_id}">
  <div class="round-card-header" onclick="toggle('{card_id}_body')">
    <span class="round-num">Round {r}</span>
    {badge_prob} {badge_sev}
    <span style="flex:1;font-size:.85em;color:#666;margin-left:6px">{situation}</span>
    <span style="font-size:.78em;color:#aaa">{elapsed:.1f}s</span>
  </div>
  <div id="{card_id}_body" style="display:none">
    <div class="round-card-body">

      <!-- ── INPUT SECTION ── -->
      <div class="section">
        <div class="section-title">📥 Input — what the LLM received</div>

        <!-- Metrics -->
        <div style="margin-bottom:10px">
          <div style="font-size:.82em;font-weight:600;color:#555;margin-bottom:5px">
            Training metrics &amp; round summary</div>
          {_render_metrics(metrics, round_summary)}
        </div>

        <!-- Current HPs -->
        <div style="margin-bottom:10px">
          <div style="font-size:.82em;font-weight:600;color:#555;margin-bottom:5px">
            Current hyperparameters
            <span style="font-weight:400;color:#e67e22;font-size:.9em">
              (orange = proposed changes)</span></div>
          {_render_hyperparams(current_hps, proposal.get("proposed_changes") or {})}
        </div>

        <!-- Optimization history -->
        {_json_block(opt_hist, f"Optimization history ({len(opt_hist)} rounds)", collapsed=True)}

        <!-- Tool results -->
        {_json_block(tool_results, "Tool diagnostics (training curve, gradients, capacity…)", collapsed=True)}

        <!-- Full payload -->
        {_json_block(llm_input.get("user_payload", {}), "Full user payload (complete JSON sent to LLM)", collapsed=True)}
      </div>

      <!-- ── OUTPUT SECTION ── -->
      <div class="section">
        <div class="section-title">📤 Output — what the LLM returned</div>

        <!-- Raw text -->
        <div style="font-size:.82em;font-weight:600;color:#555;margin-bottom:5px">
          Raw LLM text output</div>
        <div class="raw-output">{raw_output.replace("<","&lt;").replace(">","&gt;") or "<em style='color:#888'>not recorded (pre-patch run)</em>"}</div>

        <!-- Parsed decision -->
        <div style="margin-top:14px">
          <div style="font-size:.82em;font-weight:600;color:#555;margin-bottom:5px">
            Parsed decision &amp; proposed changes</div>
          {_render_proposal(proposal)}
        </div>
      </div>

    </div>
  </div>
</div>"""
    return body


def _render_system_prompt(system_prompt: str) -> str:
    if not system_prompt:
        return ""
    return f"""
<div class="section" style="margin-bottom:24px">
  <div class="section-title">⚙️ System prompt (same for all rounds)</div>
  <details class="jblock">
    <summary>Show system prompt <span class="toggle-hint">click to expand</span></summary>
    <div class="sysprompt">{system_prompt.replace("<","&lt;").replace(">","&gt;")}</div>
  </details>
</div>"""


# ---------------------------------------------------------------------------
# Full HTML page
# ---------------------------------------------------------------------------

def generate_html(runs: List[Dict], title: str = "LLM Agent — Conversation Report") -> str:

    # Build TOC entries
    toc_links = []
    for i, run in enumerate(runs):
        fname = Path(run["file"]).name
        run_num = re.search(r"run(\d+)", fname)
        label   = f"Run {run_num.group(1)}" if run_num else fname
        n_rounds = len(run["rounds"])
        toc_links.append(f'<li><a href="#run_{i}">{label} ({n_rounds} rounds)</a></li>')

    toc_html = f"""
<div class="toc">
  <h3>📑 Contents</h3>
  <ul>{"".join(toc_links)}</ul>
</div>""" if len(runs) > 1 else ""

    # Build run sections
    run_sections = []
    for i, run in enumerate(runs):
        fname    = Path(run["file"]).name
        rounds   = run["rounds"]
        n_rounds = len(rounds)

        # Extract system prompt from first round that has it
        sys_prompt = ""
        for entry in rounds:
            sp = entry.get("llm_input", {}).get("system_prompt", "")
            if sp:
                sys_prompt = sp
                break

        run_html  = f'<div id="run_{i}" class="run-header">'
        run_html += f'<h2>📁 {fname}</h2>'
        run_html += f'<div class="meta">{n_rounds} rounds &nbsp;|&nbsp; {run["file"]}</div>'
        run_html += '</div>'
        run_html += _render_system_prompt(sys_prompt)

        for entry in rounds:
            run_html += _render_round(entry, run_idx=i)

        run_sections.append(run_html)

    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>

<div class="topbar">
  <div>
    <div class="h1" style="font-size:1.15em;font-weight:700">🤖 {title}</div>
    <div class="subtitle">Generated {now} &nbsp;·&nbsp; {len(runs)} run(s)</div>
  </div>
</div>

<div class="container">
  {toc_html}
  {"".join(run_sections)}
</div>

<script>
function toggle(id) {{
  var el = document.getElementById(id);
  el.style.display = el.style.display === 'none' ? 'block' : 'none';
}}
// Open round 1 of first run by default so the professor sees something immediately
window.addEventListener('DOMContentLoaded', function() {{
  var first = document.getElementById('r0_round1_body');
  if (first) first.style.display = 'block';
}});
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Generate an HTML report from conversation_log_run*.json files"
    )
    p.add_argument("--dir",  default=".",
                   help="Output directory to search for conversation logs (default: .)")
    p.add_argument("--logs", nargs="+", metavar="FILE",
                   help="Explicit log file path(s); overrides --dir")
    p.add_argument("--recursive", action="store_true",
                   help="Search --dir recursively (for multi-seed runs)")
    p.add_argument("--out", default="llm_report.html",
                   help="Output HTML filename (default: llm_report.html)")
    p.add_argument("--title", default="LLM Agent — Conversation Report",
                   help="Page title")
    args = p.parse_args()

    if args.logs:
        paths = [Path(f) for f in args.logs]
    else:
        d = Path(args.dir)
        if not d.exists():
            sys.exit(f"Directory not found: {d}")
        paths = _discover_logs(d, recursive=args.recursive)
        if not paths:
            sys.exit(f"No conversation_log_run*.json files found in {d}"
                     + (" (try --recursive)" if not args.recursive else ""))

    print(f"Found {len(paths)} log file(s):")
    for p in sorted(paths):
        print(f"  {p}")

    runs = _load_logs(sorted(paths))
    if not runs:
        sys.exit("No logs could be loaded.")

    html = generate_html(runs, title=args.title)

    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    total_rounds = sum(len(r["rounds"]) for r in runs)
    print(f"\n✓ Report written → {out.resolve()}")
    print(f"  {len(runs)} run(s), {total_rounds} round(s) total")
    print(f"  Open in any browser — no internet connection needed.")


if __name__ == "__main__":
    main()
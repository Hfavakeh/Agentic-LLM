"""Shared logging setup for the whole experiment.

Importing this module (directly or via the ``pipeline`` package) configures
the root logger exactly once: a UTF-8 file handler writing
``Agent_optimization_<timestamp>.log`` in the repo root, plus a console
handler. Every other module imports the shared ``logger`` from here so the
whole run logs into one file.
"""

import logging
import sys
from datetime import datetime

# Reconfigure the console streams to UTF-8 with replacement on Windows so the
# StreamHandler below doesn't crash when a log line contains a glyph absent
# from the default cp1252 codepage (e.g. U+2212 '-' in the final-eval report,
# arrows in section headers). This was responsible for run-aborts mid-experiment
# in seed_777 of the post-fix llama38b run — the report wrote fine to .md
# (UTF-8) but the echo-to-console call raised UnicodeEncodeError and killed the
# whole run before final_eval / cross_run_metrics could be saved.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.FileHandler(
            f"Agent_optimization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
            encoding="utf-8",
        ),
        logging.StreamHandler(),
    ],
)

# Single shared logger for the whole experiment (training, agents, search,
# reporting). Import this instead of creating per-module loggers so the log
# file stays one coherent transcript.
logger = logging.getLogger("multi_agent")

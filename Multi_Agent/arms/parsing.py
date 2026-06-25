"""Parsers for the line-oriented `key: value` LLM response formats.

JSON is intentionally not supported. Small local LLMs (3B-8B) spend too much
effort on braces, commas, quoting, escaping, and schema details when forced
into JSON, and that effort is taken away from the actual control task. The
system prompts ask for plain `key: value` lines; any non-conforming response
is treated as a parse failure and triggers the standard retry / skip logic.
"""

import re
from typing import Any, Dict, Optional

from pipeline import is_finite_number

from .labels import _normalize_diagnosis
from .validation import ALLOWED_DIAGNOSES, ALLOWED_HP_KEYS


def _parse_scalar(raw_value: Any) -> Any:
    """Parse one scalar value from an LLM response."""
    if isinstance(raw_value, dict):
        for key in ("value", "new_value", "new", "set_to"):
            if key in raw_value:
                return _parse_scalar(raw_value[key])
        return raw_value
    if not isinstance(raw_value, str):
        return raw_value

    value = raw_value.strip().strip("`").strip()
    value = re.split(r"\s+#|\s+//|\s+\(", value, maxsplit=1)[0].strip()
    value = value.strip("\"'")
    lower = value.lower()
    if lower in {"true", "yes", "y"}:
        return True
    if lower in {"false", "no", "n"}:
        return False
    if lower in {"none", "null", "n/a", "na"}:
        return None
    try:
        if re.fullmatch(r"[-+]?\d+", value):
            return int(value)
        if re.fullmatch(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:e[-+]?\d+)?", value, flags=re.IGNORECASE):
            return float(value)
    except Exception:
        pass
    return value


def _round_metric(value: Any, digits: int = 4) -> Optional[float]:
    return round(float(value), digits) if is_finite_number(value) else None


def _parse_changes_text(text: Any) -> Dict[str, Any]:
    """Parse a compact changes field such as 'learning_rate=0.0005, dropout=0.35'."""
    if isinstance(text, dict):
        return {k: _parse_scalar(v) for k, v in text.items() if k in ALLOWED_HP_KEYS}
    if text is None:
        return {}
    s = str(text).strip()
    if not s or s.lower() in {"none", "no changes", "no change", "{}", "[]", "null", "n/a", "na"}:
        return {}

    changes: Dict[str, Any] = {}
    hp_names = "|".join(re.escape(k) for k in sorted(ALLOWED_HP_KEYS, key=len, reverse=True))
    for key in ALLOWED_HP_KEYS:
        pattern = (
            rf"\b{re.escape(key)}\b\s*(?:=|:|->|to)\s*"
            rf"(.+?)(?=(?:[,;]\s*|\s+)({hp_names})\b\s*(?:=|:|->|to)|[,;\n]|$)"
        )
        match = re.search(pattern, s, flags=re.IGNORECASE)
        if match:
            changes[key] = _parse_scalar(match.group(1))
    return changes


def _parse_text_proposal(raw: str) -> Dict[str, Any]:
    """Parse the preferred line-oriented LLM response format (warm-loop shape).

    Expected shape:
        diagnosis: healthy
        severity: low
        changes: learning_rate=0.0005

    The parser is intentionally forgiving about markdown bullets, bold labels,
    and '=' versus ':' separators.
    """
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:text|markdown)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    fields: Dict[str, Any] = {}
    direct_changes: Dict[str, Any] = {}
    aliases = {
        "diagnosis": "diagnosis",
        "primary_problem": "diagnosis",
        "problem": "diagnosis",
        "severity": "severity",
        "situation": "situation",
        "changes": "changes",
        "proposed_changes": "changes",
        "hyperparameters": "changes",
        "resets_model": "resets_model",
        "reset_model": "resets_model",
        "reset": "resets_model",
        "strategy": "strategy",
        "reasoning": "reasoning",
        "confidence": "confidence",
        "expected_improvement": "expected_improvement",
        # Multi-objective Pareto cost weights.
        "w_lat": "w_lat",
        "w_stab": "w_stab",
        "w_res": "w_res",
    }

    for line in text.splitlines():
        clean = line.strip()
        clean = re.sub(r"^\s*[-*+]\s+", "", clean)
        clean = clean.replace("**", "").strip()
        if not clean:
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\s*(?:=|:)\s*(.+)$", clean)
        if not match:
            continue
        key = match.group(1).strip().lower().replace(" ", "_")
        value = match.group(2).strip()
        if key in ALLOWED_HP_KEYS:
            direct_changes[key] = _parse_scalar(value)
            continue
        canonical = aliases.get(key)
        if canonical:
            fields[canonical] = value

    raw_diagnosis = str(fields.get("diagnosis", "unknown")).strip().lower()
    # Small LLMs (e.g. qwen2.5-coder:3b) often emit `diagnosis: healthy(low)`,
    # collapsing severity into the diagnosis line. Pull the leading token out
    # and reuse any parenthetical as a severity fallback.
    diag_match = re.match(r"([a-z_][a-z0-9_]*)\s*\(?\s*([a-z]+)?\s*\)?", raw_diagnosis)
    if diag_match:
        diagnosis = diag_match.group(1).replace(" ", "_")
        embedded_severity = (diag_match.group(2) or "").strip()
    else:
        diagnosis = raw_diagnosis.replace(" ", "_")
        embedded_severity = ""
    severity = str(fields.get("severity", "")).strip().lower() or embedded_severity or "unknown"
    changes = _parse_changes_text(fields.get("changes"))
    changes.update(direct_changes)

    if diagnosis not in ALLOWED_DIAGNOSES:
        raise ValueError("No parseable text proposal found in response")

    # Multi-objective Pareto cost weights — collected only when the LLM
    # provides them; None lets the validator fall back to a uniform split.
    weight_fields = {
        wk: _parse_scalar(fields[wk])
        for wk in ("w_lat", "w_stab", "w_res")
        if wk in fields
    }

    return {
        "diagnosis": {
            "primary_problem": diagnosis,
            "severity": severity,
            "situation": str(fields.get("situation", "")).strip(),
        },
        "proposed_changes": changes,
        "resets_model": bool(_parse_scalar(fields.get("resets_model", False))),
        "strategy": str(fields.get("strategy", "exploit")).strip().lower(),
        "reasoning": str(fields.get("reasoning", "")).strip(),
        "confidence": str(fields.get("confidence", "medium")).strip().lower(),
        "expected_improvement": str(fields.get("expected_improvement", "")).strip(),
        "pareto_weights": weight_fields or None,
    }


def _parse_llm_proposal(raw: str) -> Dict[str, Any]:
    """Parse the warm-loop `key: value` response format (deprecated path)."""
    return _parse_text_proposal(raw)


def _parse_protocol_proposal(raw: str) -> Dict[str, Any]:
    """Parse the protocol output (diagnosis / strategy / changes / reason /
    confidence). Reuses `_parse_changes_text` for the delta (which already
    filters to allowed grid keys). Raises ValueError if no usable changes line
    and no recognisable diagnosis are present."""
    text = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    text = re.sub(r"^```(?:\w+)?\s*|\s*```$", "", text, flags=re.IGNORECASE | re.MULTILINE).strip()

    fields: Dict[str, str] = {}
    direct_changes: Dict[str, Any] = {}
    for line in text.splitlines():
        clean = re.sub(r"^\s*[-*+]\s+", "", line.strip()).replace("**", "").strip()
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_ ]*)\s*[:=]\s*(.+)$", clean)
        if not m:
            continue
        key = m.group(1).strip().lower().replace(" ", "_")
        val = m.group(2).strip()
        if key in ALLOWED_HP_KEYS:
            direct_changes[key] = _parse_scalar(val)
        elif key in ("diagnosis", "strategy", "changes", "reason", "reasoning", "confidence"):
            fields[key] = val

    changes = _parse_changes_text(fields.get("changes"))
    changes.update(direct_changes)
    return {
        "diagnosis": _normalize_diagnosis(fields.get("diagnosis")),
        "strategy":  (fields.get("strategy") or "exploit").strip().lower(),
        "proposed_changes": changes,
        "reason":    (fields.get("reason") or fields.get("reasoning") or "").strip(),
        "confidence": (fields.get("confidence") or "medium").strip().lower(),
    }

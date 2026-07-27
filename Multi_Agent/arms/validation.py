"""Hard validation of proposed settings against the protocol search grid.

This module owns the shared validation constants (allowed keys, discrete
value sets, architecture keys) plus the protocol path's hard validator:
grid membership + arch-frozen + not-already-tried, with NO silent repair.
The legacy warm-loop soft validator (clamp/snap/auto-correct) lives in
``agents.legacy``.
"""

from typing import Any, Dict

from pipeline import HP_GRID, LOSS_SHAPING_GRID, OPTIMIZER_CHOICES

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

ALLOWED_DIAGNOSES = {"overfitting", "underfitting", "plateau", "healthy", "no_data"}

# The 9 conventional hyperparameters of the protocol search space. Motion
# loss-shaping levers are deferred from the search space for now (see memory:
# professor-protocol), so they are NOT accepted here — a proposal naming one
# is treated as an unknown key.
ALLOWED_HP_KEYS = set(HP_GRID.keys())

# Parameters that must be integers.
INTEGER_HP_KEYS = {"lstm_hidden", "lstm_layers", "batch_size", "window_size", "patience"}

# Integer / categorical params validated by exact grid membership (the
# existing snap branch int-coerces these safely). The float-valued grid
# params (learning_rate, weight_decay, dropout) are validated against
# HP_GRID with on-grid membership in the hard-validation rework (Step 2/3);
# for now they go through the range-clamp path so the samplers, which only
# ever emit on-grid values, work end-to-end.
DISCRETE_HP_VALUES: Dict[str, set] = {
    "lstm_hidden":      set(HP_GRID["lstm_hidden"]),
    "lstm_layers":      set(HP_GRID["lstm_layers"]),
    "batch_size":       set(HP_GRID["batch_size"]),
    "window_size":      set(HP_GRID["window_size"]),
    "patience":         set(HP_GRID["patience"]),
    "optimizer_choice": set(OPTIMIZER_CHOICES),
}

# Architecture-changing parameters (trigger model rebuild).
ARCH_CHANGE_KEYS = {"lstm_hidden", "lstm_layers"}


# ---------------------------------------------------------------------------
# Protocol-path hard validation (no silent repair)
# ---------------------------------------------------------------------------

def _value_in_grid(key: str, val: Any) -> bool:
    """Hard membership test against HP_GRID (the deferred Step-1 check, landed
    here for the protocol path). Categorical exact; numeric within tolerance."""
    grid = HP_GRID.get(key)
    if grid is None:
        return False
    if key == "optimizer_choice":
        return val in grid
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return False
    return any(abs(fv - float(g)) <= 1e-9 + 1e-6 * abs(float(g)) for g in grid)


def _snap_to_grid(key: str, val: Any) -> Any:
    """Return the grid value the (possibly typed) val corresponds to, so the
    stored setting uses canonical grid values (e.g. int batch_size)."""
    grid = HP_GRID.get(key, [])
    if key == "optimizer_choice":
        return val
    fv = float(val)
    return min(grid, key=lambda g: abs(float(g) - fv))


def validate_protocol_changes(
    changes: Dict[str, Any],
    anchor: Dict[str, Any],
    allow_arch_changes: bool,
    is_tried,
    require_complete: bool = False,
) -> tuple:
    """Hard-validate a proposed delta against the grid and the already-tried set.

    Returns (resolved_setting, ok, reason). `reason` is "" on success, else a
    short machine tag the retry feedback is built from. No silent repair: an
    out-of-grid value, an unknown key, an arch change while frozen, or a repeat
    all fail (the protocol's hard-validation main run).

    `require_complete=True` is the Email-8 anchor-removed arm: the payload does
    not show the best-so-far setting, so the proposal must name EVERY searchable
    hyperparameter. Without this check the merge below would quietly fill the
    unnamed ones from the true anchor, and the arm would still be anchored —
    exactly the thing the ablation is meant to remove.
    """
    if not isinstance(changes, dict):
        return None, False, "no_changes_parsed"
    if require_complete:
        needed = ALLOWED_HP_KEYS - (set() if allow_arch_changes else ARCH_CHANGE_KEYS)
        missing = sorted(needed - set(changes))
        if missing:
            return None, False, f"incomplete_setting:{','.join(missing)}"
    for k, v in changes.items():
        if k not in ALLOWED_HP_KEYS:
            return None, False, f"unknown_param:{k}"
        if not allow_arch_changes and k in ARCH_CHANGE_KEYS:
            return None, False, f"arch_frozen:{k}"
        if not _value_in_grid(k, v):
            return None, False, f"value_not_in_grid:{k}={v}"
    resolved = {**anchor, **{k: _snap_to_grid(k, v) for k, v in changes.items()}}
    if is_tried(resolved):
        return None, False, "already_tried"
    return resolved, True, ""


# ---------------------------------------------------------------------------
# Motion experiment — loss-shaping lever validation (same hard-validation
# contract as the HP path: grid membership + not-already-tried, no repair).
# ---------------------------------------------------------------------------

LOSS_SHAPING_KEYS_ORDERED = [
    "v_max", "lambda_vel", "lambda_smooth",
    "bin_weight_slow", "bin_weight_medium", "bin_weight_fast",
]


def _lever_in_grid(key: str, val: Any) -> bool:
    grid = LOSS_SHAPING_GRID.get(key)
    if grid is None:
        return False
    try:
        fv = float(val)
    except (TypeError, ValueError):
        return False
    return any(abs(fv - float(g)) <= 1e-9 + 1e-6 * abs(float(g)) for g in grid)


def _snap_lever(key: str, val: Any) -> Any:
    grid = LOSS_SHAPING_GRID.get(key, [])
    fv = float(val)
    return min(grid, key=lambda g: abs(float(g) - fv))


def validate_loss_shaping_changes(
    changes: Dict[str, Any],
    anchor: Dict[str, Any],
    is_tried,
) -> tuple:
    """Hard-validate a proposed loss-shaping lever vector against
    LOSS_SHAPING_GRID and the already-tried set (motion experiment).

    Returns (resolved_vector, ok, reason). The resolved vector is the anchor
    (neutral / best-so-far levers) overlaid with the snapped proposed levers,
    so a partial proposal still yields a full 6-lever vector. No silent repair:
    an out-of-grid value, an unknown lever, or a repeat all fail.
    """
    if not isinstance(changes, dict):
        return None, False, "no_changes_parsed"
    for k, v in changes.items():
        if k not in LOSS_SHAPING_GRID:
            return None, False, f"unknown_lever:{k}"
        if not _lever_in_grid(k, v):
            return None, False, f"value_not_in_grid:{k}={v}"
    resolved = {**anchor, **{k: _snap_lever(k, v) for k, v in changes.items()}}
    if is_tried(resolved):
        return None, False, "already_tried"
    return resolved, True, ""


def _human_reason(reason: str) -> str:
    """Turn a validate_protocol_changes tag into one line of LLM feedback."""
    if reason.startswith("unknown_param:"):
        return f"'{reason.split(':',1)[1]}' is not a tunable parameter."
    if reason.startswith("unknown_lever:"):
        return f"'{reason.split(':',1)[1]}' is not a loss-shaping lever."
    if reason.startswith("arch_frozen:"):
        return f"{reason.split(':',1)[1]} is fixed this run; do not change it."
    if reason.startswith("value_not_in_grid:"):
        return f"{reason.split(':',1)[1]} is not an allowed value; pick one from the list."
    if reason == "already_tried":
        return "that setting was already tried; propose a different change."
    if reason == "no_changes_parsed":
        return "no valid 'changes:' line was found."
    if reason.startswith("incomplete_setting:"):
        missing = reason.split(":", 1)[1]
        return (f"give a COMPLETE setting — these are missing: {missing}.")
    return reason


def _grid_neighbor(key: str, current: Any, direction: int) -> Any:
    """Step one position up (+1) or down (-1) the sorted grid for `key`.
    Returns the current value (no-op) if already at the end."""
    grid = HP_GRID.get(key, [])
    if not grid or key == "optimizer_choice":
        return current
    ordered = sorted(grid, key=float)
    # Find the closest index to current.
    try:
        cf = float(current)
        idx = min(range(len(ordered)), key=lambda i: abs(float(ordered[i]) - cf))
    except (TypeError, ValueError):
        idx = 0
    new_idx = max(0, min(len(ordered) - 1, idx + direction))
    return ordered[new_idx]

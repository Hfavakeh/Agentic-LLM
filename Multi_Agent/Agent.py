"""Back-compat shim — the proposer agents now live in the ``arms`` package.

Import from ``arms`` (or its submodules) in new code:

    from arms import SingleAgentOptimizer, RuleBasedOptimizer

This module only re-exports the old public names so historical imports
(`from Agent import ...`) keep working. Safe to delete once nothing imports
`Agent` anymore.
"""

from arms import (  # noqa: F401
    MotionAwareRuleBasedOptimizer,
    OptimizerTools,
    PROTOCOL_DIAGNOSES,
    RuleBasedOptimizer,
    SemanticRepairRequired,
    SingleAgentOptimizer,
    format_protocol_payload,
    protocol_system_prompt,
    validate_protocol_changes,
)

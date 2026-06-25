"""The five comparison arms plus the shared machinery that drives them.

Each run of the bake-off compares five methods, all gathered here:
  - baseline.py       — fixed-reference defaults, trained once
  - random_search.py  — uniform draws from the shared HP grid
  - optuna_search.py  — Bayesian (TPE) sampler over the same grid
  - llm.py            — Ollama-backed proposer (SingleAgentOptimizer)
  - rule_based.py     — deterministic proposer (+ motion-aware C2 variant)

Shared infrastructure every arm leans on lives alongside them:
  - engine.py         — `evaluate_setting` (3 trainings, mean val RMSE) +
                        `train_and_test_setting`, the from-scratch core
  - driver.py         — `run_proposer_search`, the 25-attempt loop the LLM
                        and rule-based arms share (proposer passed in)
  - parsing / validation / prompts / labels — LLM + rule-based support
  - legacy.py         — DEPRECATED warm-loop machinery, slated for removal
"""

from .baseline import run_baseline
from .driver import run_proposer_search
from .engine import (
    TRAIN_SEEDS, build_dataset_and_loaders, evaluate_setting, make_loaders,
    setting_signature, train_and_test_setting,
)
from .labels import PROTOCOL_DIAGNOSES
from .legacy import OptimizerTools, SemanticRepairRequired
from .llm import SingleAgentOptimizer
from .optuna_search import (
    OPTUNA_N_STARTUP_TRIALS, OPTUNA_SAMPLER_SEED, run_optuna_search,
)
from .prompts import format_protocol_payload, protocol_system_prompt
from .random_search import run_random_search, sample_random_hparams
from .rule_based import MotionAwareRuleBasedOptimizer, RuleBasedOptimizer
from .validation import (
    ALLOWED_DIAGNOSES, ALLOWED_HP_KEYS, ARCH_CHANGE_KEYS, validate_protocol_changes,
)

__all__ = [
    # arms
    "run_baseline",
    "run_random_search", "sample_random_hparams",
    "run_optuna_search", "OPTUNA_N_STARTUP_TRIALS", "OPTUNA_SAMPLER_SEED",
    "SingleAgentOptimizer",
    "RuleBasedOptimizer", "MotionAwareRuleBasedOptimizer",
    # shared engine + driver
    "TRAIN_SEEDS", "build_dataset_and_loaders", "evaluate_setting",
    "make_loaders", "setting_signature", "train_and_test_setting",
    "run_proposer_search",
    # proposer support
    "PROTOCOL_DIAGNOSES",
    "format_protocol_payload", "protocol_system_prompt",
    "ALLOWED_DIAGNOSES", "ALLOWED_HP_KEYS", "ARCH_CHANGE_KEYS",
    "validate_protocol_changes",
    "OptimizerTools", "SemanticRepairRequired",
]

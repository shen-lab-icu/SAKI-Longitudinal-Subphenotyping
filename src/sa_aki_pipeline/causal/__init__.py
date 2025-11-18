"""Causal inference utilities (PSM, balancing diagnostics)."""

from .config import PSMConfig
from .psm import propensity_score_matching, compute_standardized_mean_difference

__all__ = [
    "PSMConfig",
    "propensity_score_matching",
    "compute_standardized_mean_difference",
]

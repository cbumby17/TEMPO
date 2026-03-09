"""
tempo — Trajectory Enrichment via Matrix Profile Outcomes

Top-level package exposing the TEMPO public API.
"""

from tempo import simulate
from tempo.preprocess import preprocess, clr_transform, bray_curtis_trajectory, check_baseline
from tempo.harbinger import harbinger, compute_matrix_profile
from tempo.stats import (
    permutation_test, enrichment_score, survival_test,
    compute_resistance, compute_resilience, compare_recovery,
)
from tempo.viz import plot_motifs, plot_enrichment, plot_survival, plot_resistance_resilience
from tempo.datasets import load_example_data

__version__ = "0.2.0"
__all__ = [
    "simulate",
    "preprocess",
    "clr_transform",
    "bray_curtis_trajectory",
    "check_baseline",
    "harbinger",
    "compute_matrix_profile",
    "permutation_test",
    "enrichment_score",
    "survival_test",
    "compute_resistance",
    "compute_resilience",
    "compare_recovery",
    "plot_motifs",
    "plot_enrichment",
    "plot_survival",
    "plot_resistance_resilience",
    "load_example_data",
]

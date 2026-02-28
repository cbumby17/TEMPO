"""
tempo — Trajectory Enrichment via Matrix Profile Outcomes

Top-level package exposing the TEMPO public API.
"""

from tempo import simulate
from tempo.preprocess import preprocess, clr_transform, bray_curtis_trajectory
from tempo.harbinger import harbinger, compute_matrix_profile
from tempo.stats import permutation_test, enrichment_score, survival_test
from tempo.viz import plot_motifs, plot_enrichment
from tempo.datasets import load_example_data

__version__ = "0.1.0"
__all__ = [
    "simulate",
    "preprocess",
    "clr_transform",
    "bray_curtis_trajectory",
    "harbinger",
    "compute_matrix_profile",
    "permutation_test",
    "enrichment_score",
    "survival_test",
    "plot_motifs",
    "plot_enrichment",
    "load_example_data",
]

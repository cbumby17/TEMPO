"""
tempo/stats.py

Statistical testing for Harbinger analysis results.

Provides permutation-based significance testing, GSEA-style enrichment
scoring, and survival-integrated analysis for trajectory motifs identified
by the Harbinger algorithm.
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


def permutation_test(
    df: pd.DataFrame,
    feature: str,
    motif_window: tuple,
    n_permutations: int = 1000,
    outcome_col: str = "outcome",
    seed: Optional[int] = 42,
) -> dict:
    """
    Permutation test for enrichment of a trajectory motif in case subjects.

    Computes the observed enrichment score for a given feature/window, then
    estimates the null distribution by randomly permuting outcome labels
    and recomputing the score n_permutations times.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value, outcome.
    feature : str
        The feature (e.g., "feature_000") to test.
    motif_window : tuple of (int, int)
        (start, end) timepoint range defining the candidate motif window.
    n_permutations : int
        Number of permutation iterations for null distribution estimation.
    outcome_col : str
        Column containing outcome labels (binary 0/1).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        Keys: observed_score, p_value, null_mean, null_sd, n_permutations.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Notes
    -----
    The permutation is subject-level (outcome labels are permuted across
    subjects, not timepoints), preserving within-subject temporal structure.
    """
    raise NotImplementedError("permutation_test() is not yet implemented.")


def enrichment_score(
    df: pd.DataFrame,
    feature: str,
    motif_window: tuple,
    outcome_col: str = "outcome",
    method: str = "mean_difference",
) -> float:
    """
    Compute the enrichment score for a trajectory motif.

    Quantifies how much more prominent the motif is in case subjects
    compared to controls during the specified time window.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value, outcome.
    feature : str
        Feature to score.
    motif_window : tuple of (int, int)
        (start, end) timepoints defining the motif window.
    outcome_col : str
        Column containing outcome labels.
    method : str
        Scoring method:
        "mean_difference" → mean(case values) - mean(control values) in window
        "gsea"            → GSEA-style running enrichment score
        "auc"             → AUC of ROC curve separating cases from controls

    Returns
    -------
    float
        Enrichment score. Higher values indicate stronger case enrichment.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError(
        f"enrichment_score() is not yet implemented. "
        f"Planned methods: 'mean_difference', 'gsea', 'auc'."
    )


def survival_test(
    df: pd.DataFrame,
    feature: str,
    motif_window: tuple,
    time_col: str = "time_to_event",
    event_col: str = "outcome",
    method: str = "logrank",
) -> dict:
    """
    Test association between trajectory motif and survival outcome.

    Stratifies subjects by motif presence/absence and tests for differences
    in time-to-event using log-rank test or Cox proportional hazards regression.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns including time_to_event and outcome.
        Typically output of simulate_longitudinal(outcome_type="survival").
    feature : str
        Feature defining the trajectory motif.
    motif_window : tuple of (int, int)
        (start, end) timepoints defining the motif window.
    time_col : str
        Column containing time-to-event values.
    event_col : str
        Column containing event indicator (1=event, 0=censored).
    method : str
        Statistical test:
        "logrank" → log-rank test (requires scipy or lifelines)
        "cox"     → Cox proportional hazards (requires lifelines)

    Returns
    -------
    dict
        Keys: statistic, p_value, method, feature, motif_window.
        For "cox": also includes hazard_ratio, confidence_interval.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.
    """
    raise NotImplementedError(
        "survival_test() is not yet implemented. "
        "Planned: log-rank test via scipy.stats, Cox via lifelines."
    )

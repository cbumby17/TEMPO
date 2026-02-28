"""
tempo/harbinger.py

Harbinger analysis: matrix profile-based trajectory motif discovery.

Implements the core TEMPO algorithm: for each feature, compute the matrix
profile across subjects to identify conserved trajectory patterns (motifs)
enriched in one outcome group. Uses STUMPY for efficient matrix profile
computation.

Core algorithm:
    1. For each feature, extract per-subject time series
    2. Compute the pan-matrix profile across case and control subjects
    3. Identify motifs (low matrix profile regions = conserved patterns)
    4. Score enrichment of each motif in cases vs controls
    5. Return ranked features with motif windows and enrichment scores
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


def harbinger(
    df: pd.DataFrame,
    window_size: int = 3,
    top_k: int = 10,
    outcome_col: str = "outcome",
    n_permutations: int = 1000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Run Harbinger analysis on a preprocessed longitudinal dataframe.

    Identifies features with trajectory motifs enriched in the case outcome
    group using matrix profile-based motif discovery (via STUMPY).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed long-format dataframe with columns:
        subject_id, timepoint, feature, value, outcome.
        Typically the output of preprocess() or simulate_longitudinal().
    window_size : int
        Subsequence length (in timepoints) for matrix profile computation.
        Should be < n_timepoints / 2. Larger windows detect broader motifs.
    top_k : int
        Number of top enriched features to return in the results.
    outcome_col : str
        Column name containing the outcome labels (binary: 0/1).
    n_permutations : int
        Number of permutations for significance testing of enrichment scores.
    seed : int, optional
        Random seed for permutation testing.

    Returns
    -------
    pd.DataFrame
        Results dataframe with columns:
        - feature: feature name
        - motif_window: (start, end) timepoint of the top motif
        - enrichment_score: degree of case enrichment for this motif
        - p_value: permutation-based p-value
        - matrix_profile_min: minimum matrix profile value (motif strength)
        Sorted by enrichment_score descending.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Examples
    --------
    >>> from tempo import simulate, harbinger
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> results = harbinger(df, window_size=3, top_k=5)
    >>> results.head()
    """
    raise NotImplementedError(
        "harbinger() is not yet implemented. "
        "Planned: STUMPY-based matrix profile computation + permutation enrichment test."
    )


def compute_matrix_profile(
    time_series: np.ndarray,
    window_size: int,
    cross_series: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute the matrix profile for a time series or pair of time series.

    Wraps STUMPY to compute either the self-join (motif discovery within
    one set of subjects) or ab-join (comparing case vs control trajectories).

    Parameters
    ----------
    time_series : np.ndarray
        1D array of shape (n_timepoints,) or 2D array of shape
        (n_subjects, n_timepoints) for pan-matrix profile computation.
    window_size : int
        Subsequence length for matrix profile computation.
    cross_series : np.ndarray, optional
        If provided, compute the AB-join matrix profile between
        time_series (query) and cross_series (reference).
        Shape must match time_series.

    Returns
    -------
    np.ndarray
        Matrix profile array of shape (n_timepoints - window_size + 1,).
        Lower values indicate more conserved (motif-like) subsequences.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Notes
    -----
    Uses stumpy.stump for 1D self-join, stumpy.stumped for distributed
    computation, and stumpy.mstump for multidimensional matrix profiles.
    See: https://stumpy.readthedocs.io/
    """
    raise NotImplementedError(
        "compute_matrix_profile() is not yet implemented. "
        "Planned: wraps stumpy.stump / stumpy.stumped."
    )

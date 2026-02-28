"""
tempo/harbinger.py

Harbinger analysis: matrix profile-based trajectory motif discovery.

Implements the core TEMPO algorithm: for each feature, compute the
pan-matrix profile across case subjects to identify the most conserved
trajectory subsequence, then score how enriched that pattern is in cases
vs controls. Uses STUMPY for efficient matrix profile computation.

Core algorithm (per feature):
    1. Pivot long-format data to a subjects × timepoints matrix.
    2. Compute the pan-matrix profile across case subjects with mstump —
       this finds the window where ALL cases show the most conserved pattern.
    3. Score enrichment at the best motif window: mean(case) − mean(ctrl).
    4. Permute outcome labels to build a null distribution and derive p-values.
    5. Return results ranked by enrichment score.
"""

import numpy as np
import pandas as pd
import stumpy
from typing import Optional


def harbinger(
    df: pd.DataFrame,
    window_size: int = 3,
    top_k: int = 10,
    outcome_col: str = "outcome",
    n_permutations: int = 1000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Run Harbinger analysis on a longitudinal dataframe.

    Identifies features with trajectory motifs enriched in the case outcome
    group (outcome == 1) using matrix profile-based motif discovery via STUMPY.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns:
        subject_id, timepoint, feature, value, outcome.
        Typically the output of simulate_longitudinal() or preprocess().
    window_size : int
        Subsequence length (in timepoints) for matrix profile computation.
        Must be < n_timepoints. Larger windows detect broader motifs.
    top_k : int
        Maximum number of top-ranked features to return.
    outcome_col : str
        Column containing binary outcome labels (1 = case, 0 = control).
    n_permutations : int
        Number of label-permutation iterations for p-value estimation.
    seed : int, optional
        Random seed for reproducibility of permutation tests.

    Returns
    -------
    pd.DataFrame
        Results with columns:
          feature            — feature name
          motif_window       — (start, end) timepoint tuple of the best motif
          enrichment_score   — mean(case values) − mean(ctrl values) in window
          p_value            — fraction of permutations with score ≥ observed
          matrix_profile_min — minimum pan-matrix-profile value (motif strength)
        Sorted by enrichment_score descending, at most top_k rows.

    Raises
    ------
    ValueError
        If window_size >= n_timepoints.

    Examples
    --------
    >>> from tempo import simulate
    >>> from tempo.harbinger import harbinger
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> results = harbinger(df, window_size=3, top_k=5)
    >>> results.head()
    """
    rng = np.random.default_rng(seed)

    n_timepoints = df["timepoint"].nunique()
    if window_size >= n_timepoints:
        raise ValueError(
            f"window_size ({window_size}) must be less than "
            f"n_timepoints ({n_timepoints})."
        )

    records = []

    for feature, feat_df in df.groupby("feature"):
        wide = (
            feat_df.pivot(index="subject_id", columns="timepoint", values="value")
            .reindex(sorted(feat_df["timepoint"].unique()), axis=1)
        )
        timepoints = wide.columns.tolist()  # actual timepoint values, sorted

        outcome_map = feat_df.groupby("subject_id")[outcome_col].first()
        case_subj = outcome_map[outcome_map == 1].index.tolist()
        ctrl_subj = outcome_map[outcome_map == 0].index.tolist()

        if len(case_subj) < 2:
            continue

        # Pan-matrix profile: most conserved window across ALL case subjects.
        # mstump expects shape (d, n) where d = subjects, n = timepoints.
        T_case = wide.loc[case_subj].values.astype(float)
        pan_mp = compute_matrix_profile(T_case, window_size)

        if not np.any(np.isfinite(pan_mp)):
            continue

        # Best motif: index of minimum pan-MP value.
        motif_idx = int(np.nanargmin(pan_mp))
        motif_start = timepoints[motif_idx]
        motif_end = timepoints[motif_idx + window_size - 1]
        window_tps = timepoints[motif_idx: motif_idx + window_size]

        # Enrichment score: mean case value − mean ctrl value in motif window.
        obs_score = _window_enrichment(wide, case_subj, ctrl_subj, window_tps)

        # Permutation test: shuffle outcome labels, keep window fixed.
        all_subj = np.array(case_subj + ctrl_subj)
        n_cases = len(case_subj)
        perm_scores = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(all_subj)
            perm_scores[i] = _window_enrichment(
                wide, perm[:n_cases].tolist(), perm[n_cases:].tolist(), window_tps
            )

        p_value = float(np.mean(perm_scores >= obs_score))

        records.append({
            "feature": feature,
            "motif_window": (motif_start, motif_end),
            "enrichment_score": round(obs_score, 6),
            "p_value": round(p_value, 4),
            "matrix_profile_min": round(float(np.nanmin(pan_mp)), 6),
        })

    if not records:
        return pd.DataFrame(
            columns=["feature", "motif_window", "enrichment_score", "p_value",
                     "matrix_profile_min"]
        )

    return (
        pd.DataFrame(records)
        .sort_values("enrichment_score", ascending=False)
        .reset_index(drop=True)
        .head(top_k)
    )


def compute_matrix_profile(
    time_series: np.ndarray,
    window_size: int,
    cross_series: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute a matrix profile for a time series or collection of time series.

    Wraps STUMPY to return a 1D distance profile of length
    (n_timepoints − window_size + 1).

    Parameters
    ----------
    time_series : np.ndarray
        Shape (n_timepoints,) — 1D self-join or AB-join query.
        Shape (n_subjects, n_timepoints) — 2D pan-matrix profile across subjects.
    window_size : int
        Subsequence length for the matrix profile.
    cross_series : np.ndarray, optional
        Reference series for AB-join. Only valid when time_series is 1D.
        Shape must be (n_timepoints,).

    Returns
    -------
    np.ndarray
        1D array of shape (n_timepoints − window_size + 1,).
        Lower values indicate more conserved (motif-like) subsequences.

    Notes
    -----
    Dispatch:
      1D, no cross_series → stumpy.stump self-join       → mp[:, 0]
      1D, with cross_series → stumpy.stump AB-join        → mp[:, 0]
      2D (subjects × timepoints) → stumpy.mstump          → mp[-1] (pan-MP)
    """
    time_series = np.asarray(time_series, dtype=float)

    if time_series.ndim == 1:
        if cross_series is None:
            mp = stumpy.stump(time_series, m=window_size)
        else:
            mp = stumpy.stump(
                time_series, m=window_size, T_B=np.asarray(cross_series, dtype=float)
            )
        return np.array(mp[:, 0], dtype=float)

    elif time_series.ndim == 2:
        # rows = subjects (dimensions), cols = timepoints
        mp, _ = stumpy.mstump(time_series, m=window_size)
        return mp[-1]  # pan-matrix profile across all subjects

    else:
        raise ValueError(
            f"time_series must be 1D or 2D, got shape {time_series.shape}."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _window_enrichment(
    wide: pd.DataFrame,
    case_subj: list,
    ctrl_subj: list,
    window_tps: list,
) -> float:
    """Mean case value minus mean control value over window_tps columns."""
    case_mean = wide.loc[case_subj, window_tps].values.mean() if case_subj else 0.0
    ctrl_mean = wide.loc[ctrl_subj, window_tps].values.mean() if ctrl_subj else 0.0
    return float(case_mean - ctrl_mean)

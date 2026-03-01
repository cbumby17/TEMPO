"""
tempo/harbinger.py

Harbinger analysis: matrix profile-based trajectory motif discovery.

Implements the core TEMPO algorithm: for each feature, compute the
pan-matrix profile across case subjects to identify the most conserved
trajectory subsequence, then score how enriched that pattern is in cases
vs controls. Uses STUMPY for efficient matrix profile computation.

Core algorithm (per feature):
    1. Pivot long-format data to a subjects × timepoints matrix.
    2. For each candidate window size, compute the pan-matrix profile
       across case subjects with mstump — this finds the window where
       ALL cases show the most conserved pattern.
    3. Select the (size, position) pair with the highest enrichment score
       across all candidate sizes.
    4. Permute outcome labels once (for the winning window) to derive p-values.
    5. Return results ranked by enrichment score.

Note on multiple window sizes: scanning multiple sizes and picking the best
enrichment score is slightly anti-conservative (selection bias inflates the
observed score relative to the null). The permutation test accounts for the
observed score's magnitude but not the size-selection step itself. Results
should be interpreted with this in mind, especially when the range of
candidate sizes is large.
"""

import numpy as np
import pandas as pd
import stumpy
from typing import Optional


def _resolve_window_sizes(
    window_size: Optional[int],
    window_sizes: Optional[list],
    window_size_range: Optional[tuple],
) -> list:
    """Resolve the three size-specifying params into a single list of ints.

    Exactly one of the three may be non-None. If all three are None, defaults
    to [3] (backward-compatible with the original single-size default).

    Parameters
    ----------
    window_size : int, optional
        Single window size — resolved to [window_size].
    window_sizes : list, optional
        Explicit list of sizes — returned as-is.
    window_size_range : tuple, optional
        (min_size, max_size) inclusive — expanded to list(range(min, max+1)).

    Returns
    -------
    list of int

    Raises
    ------
    ValueError
        If more than one of the three parameters is non-None.
    """
    n_given = sum(x is not None for x in [window_size, window_sizes, window_size_range])
    if n_given > 1:
        raise ValueError(
            "Provide at most one of window_size, window_sizes, or window_size_range."
        )
    if window_size is not None:
        return [window_size]
    if window_sizes is not None:
        return list(window_sizes)
    if window_size_range is not None:
        lo, hi = window_size_range
        return list(range(lo, hi + 1))
    return [3]  # default


def harbinger(
    df: pd.DataFrame,
    window_size: Optional[int] = None,
    window_sizes: Optional[list] = None,
    window_size_range: Optional[tuple] = None,
    top_k: int = 10,
    outcome_col: str = "outcome",
    n_permutations: int = 1000,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Run Harbinger analysis on a longitudinal dataframe.

    Identifies features with trajectory motifs enriched in the case outcome
    group (outcome == 1) using matrix profile-based motif discovery via STUMPY.

    Window size can be specified in three ways (at most one may be given):
    - ``window_size=k`` — single fixed size (backward-compatible default = 3).
    - ``window_sizes=[3, 5, 7]`` — explicit list of sizes to try.
    - ``window_size_range=(3, 8)`` — try every integer from 3 to 8 inclusive.

    When multiple sizes are given, each feature independently selects the size
    that maximises its enrichment score. The permutation test runs once per
    feature on the winning (size, window) pair.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns:
        subject_id, timepoint, feature, value, outcome.
        Typically the output of simulate_longitudinal() or preprocess().
    window_size : int, optional
        Single subsequence length. Raises ValueError if >= n_timepoints.
        Mutually exclusive with window_sizes and window_size_range.
    window_sizes : list of int, optional
        Explicit list of subsequence lengths to scan per feature.
        Sizes >= n_timepoints for a given feature are silently skipped.
        Mutually exclusive with window_size and window_size_range.
    window_size_range : tuple of (int, int), optional
        (min_size, max_size) inclusive. Every integer in this range is tried.
        Sizes >= n_timepoints for a given feature are silently skipped.
        Mutually exclusive with window_size and window_sizes.
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
          window_size        — the winning window size (int) for this feature
          motif_window       — (start, end) timepoint tuple of the best motif
          enrichment_score   — mean(case values) − mean(ctrl values) in window
          p_value            — fraction of permutations with score ≥ observed
          matrix_profile_min — minimum pan-matrix-profile value (motif strength)
        Sorted by enrichment_score descending, at most top_k rows.

    Raises
    ------
    ValueError
        If window_size >= n_timepoints (single-size backward-compat check).
        If more than one of window_size, window_sizes, window_size_range is given.

    Examples
    --------
    >>> from tempo import simulate
    >>> from tempo.harbinger import harbinger
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> results = harbinger(df, window_size=3, top_k=5)
    >>> results.head()

    >>> # Multi-window scan
    >>> results = harbinger(df, window_sizes=[2, 3, 4], top_k=5)
    >>> results = harbinger(df, window_size_range=(2, 5), top_k=5)
    """
    sizes = _resolve_window_sizes(window_size, window_sizes, window_size_range)

    rng = np.random.default_rng(seed)

    # Backward-compat: single window_size still raises on invalid input.
    n_timepoints = df["timepoint"].nunique()
    if window_size is not None and window_size >= n_timepoints:
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
        n_tp = len(timepoints)

        outcome_map = feat_df.groupby("subject_id")[outcome_col].first()
        case_subj = outcome_map[outcome_map == 1].index.tolist()
        ctrl_subj = outcome_map[outcome_map == 0].index.tolist()

        if len(case_subj) < 2:
            continue

        T_case = wide.loc[case_subj].values.astype(float)

        # Scan all candidate sizes; pick the winner by a sum-based criterion
        # (mean_enrichment * window_size) so that sustained enrichment over
        # a longer window beats a sharp peak in a very short window.
        # best_score retains the *mean* enrichment for reporting and the
        # permutation test (keeping the public API unchanged).
        best_selection = -np.inf   # sum-based, internal selection only
        best_score = None          # mean-based, for output + permutation test
        best_ws = None
        best_window_tps = None
        best_pan_mp = None

        for ws in sizes:
            if ws >= n_tp or ws < 3:  # STUMPY requires m >= 3
                continue
            pan_mp = compute_matrix_profile(T_case, ws)
            if not np.any(np.isfinite(pan_mp)):
                continue
            motif_idx = int(np.nanargmin(pan_mp))
            window_tps = timepoints[motif_idx: motif_idx + ws]
            score = _window_enrichment(wide, case_subj, ctrl_subj, window_tps)
            selection = score * ws   # rewards sustained enrichment
            if selection > best_selection:
                best_selection = selection
                best_score = score
                best_ws = ws
                best_window_tps = window_tps
                best_pan_mp = pan_mp

        if best_ws is None:
            continue

        # Permutation test: run once for the winning (ws, window_tps).
        all_subj = np.array(case_subj + ctrl_subj)
        n_cases = len(case_subj)
        perm_scores = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng.permutation(all_subj)
            perm_scores[i] = _window_enrichment(
                wide, perm[:n_cases].tolist(), perm[n_cases:].tolist(), best_window_tps
            )

        p_value = float(np.mean(perm_scores >= best_score))
        motif_start = best_window_tps[0]
        motif_end = best_window_tps[-1]

        records.append({
            "feature": feature,
            "window_size": best_ws,
            "motif_window": (motif_start, motif_end),
            "enrichment_score": round(best_score, 6),
            "p_value": round(p_value, 4),
            "matrix_profile_min": round(float(np.nanmin(best_pan_mp)), 6),
        })

    if not records:
        return pd.DataFrame(
            columns=["feature", "window_size", "motif_window", "enrichment_score",
                     "p_value", "matrix_profile_min"]
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

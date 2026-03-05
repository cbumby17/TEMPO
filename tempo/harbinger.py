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

Note on multiple window sizes: when multiple sizes are scanned, each
permutation takes the max enrichment across all candidate (size, window)
pairs, mirroring the real-data selection step. This "max-over-candidates"
approach corrects the anti-conservative bias that arises from picking the
best window size by looking at the data.
"""

import warnings
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
    enrichment_method: str = "mean_difference",
    direction: str = "up",
    covariate_cols: Optional[list] = None,
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
    that maximises its enrichment score. To correct for the selection bias
    this introduces, each permutation takes the max enrichment across every
    valid candidate window, mirroring the real-data selection step.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns:
        subject_id, timepoint, feature, value, outcome.
        Typically the output of simulate_longitudinal() or preprocess().
        Subjects with sporadic missing timepoints are handled via
        within-subject linear interpolation before matrix profile
        computation (boundary NaNs are filled by nearest known value).
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
    enrichment_method : str
        Method used to score case/control separation in the motif window.

        "mean_difference" (default)
            mean(case values) − mean(ctrl values). Simple and interpretable.

        "template_correlation"
            Each subject is scored by their Pearson correlation to the mean
            case trajectory (the "template"). The enrichment score is
            mean(case correlations) − mean(ctrl correlations). Captures shape,
            direction, and oscillation — patterns missed by mean_difference.

    direction : str
        Which enrichment direction to surface in top-k results.

        "up" (default)
            Cases elevated above controls. Backward-compatible behaviour.
            Ranks by enrichment_score descending.

        "down"
            Cases depressed below controls (e.g. immune suppression, tolerance).
            Ranks by enrichment_score ascending (most negative first).
            The permutation p-value tests whether the observed score is
            significantly more negative than the null distribution.

        "both"
            Surface the strongest hits in either direction. Ranks by
            abs(enrichment_score) descending. Adds a ``direction`` column
            ('up' or 'down') to label each feature. The permutation p-value
            is two-sided: fraction of |null scores| ≥ |observed score|.

    covariate_cols : list of str, optional
        Column names in ``df`` to use for stratified permutation testing.
        Labels are shuffled only within strata defined by the unique
        combinations of these covariate values, preserving the covariate
        distribution under the null hypothesis.

        Use this when a covariate (e.g. sex, age group, batch) is imbalanced
        between cases and controls — without stratification, a harbinger hit
        could reflect the covariate rather than the group trajectory.

        Covariate columns must be present in ``df`` and should be categorical
        or pre-binned.  Columns with more than 10 unique values trigger a
        warning — consider binning continuous variables first (e.g.
        ``pd.qcut(df['age'], q=4, labels=['Q1','Q2','Q3','Q4'])``).

        Subjects in a stratum where all members share the same outcome label
        (entirely cases or entirely controls) are left unchanged — there is
        nothing to permute in a homogeneous stratum.

    Returns
    -------
    pd.DataFrame
        Results with columns:
          feature            — feature name
          window_size        — the winning window size (int) for this feature
          motif_window       — (start, end) timepoint tuple of the best motif
          enrichment_score   — signed raw score: mean(case) − mean(ctrl)
          p_value            — permutation p-value (one- or two-sided per direction)
          q_value            — Benjamini-Hochberg FDR-adjusted p-value across
                               all features tested (computed before top_k filter)
          matrix_profile_min — minimum pan-matrix-profile value (motif strength)
          direction          — 'up' or 'down' (only present when direction='both')
        Sorted by enrichment_score descending ('up'), ascending ('down'), or
        abs(enrichment_score) descending ('both'). At most top_k rows.

    Raises
    ------
    ValueError
        If window_size >= n_timepoints (single-size backward-compat check).
        If more than one of window_size, window_sizes, window_size_range is given.
        If direction is not 'up', 'down', or 'both'.
        If any column in covariate_cols is not present in df.

    Examples
    --------
    >>> from tempo import simulate
    >>> from tempo.harbinger import harbinger
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> results = harbinger(df, window_size=3, top_k=5)
    >>> results.head()

    >>> # Downward enrichment (cases suppressed below controls)
    >>> results = harbinger(df, window_size=3, direction='down')

    >>> # Both directions
    >>> results = harbinger(df, window_size=3, direction='both')

    >>> # Multi-window scan
    >>> results = harbinger(df, window_sizes=[2, 3, 4], top_k=5)
    >>> results = harbinger(df, window_size_range=(2, 5), top_k=5)
    """
    if direction not in ("up", "down", "both"):
        raise ValueError(
            f"direction must be 'up', 'down', or 'both', got '{direction}'."
        )

    if covariate_cols is not None:
        missing = [c for c in covariate_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"covariate_cols not found in df: {missing}. "
                f"Available columns: {df.columns.tolist()}"
            )
        for col in covariate_cols:
            n_unique = df[col].nunique()
            if n_unique > 10:
                warnings.warn(
                    f"Covariate '{col}' has {n_unique} unique values and may be "
                    f"continuous. Consider binning it before passing to harbinger() "
                    f"(e.g. pd.qcut(df['{col}'], q=4, labels=['Q1','Q2','Q3','Q4'])).",
                    UserWarning,
                    stacklevel=2,
                )
        # Extract per-subject covariate values once from the full df.
        # Values are constant per subject; use first occurrence per subject.
        _subj_covars = (
            df.groupby("subject_id")[covariate_cols]
            .first()
            .fillna("__missing__")
        )
    else:
        _subj_covars = None

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

        # Impute sporadic missing timepoints via within-subject linear
        # interpolation. bfill/ffill catch NaNs at the series boundaries.
        # STUMPY returns all-inf for any series containing NaN, so this
        # step is required when subjects have irregular sampling.
        if wide.isna().any().any():
            wide = (
                wide
                .interpolate(axis=1, method="linear")
                .bfill(axis=1)
                .ffill(axis=1)
            )

        outcome_map = feat_df.groupby("subject_id")[outcome_col].first()
        case_subj = outcome_map[outcome_map == 1].index.tolist()
        ctrl_subj = outcome_map[outcome_map == 0].index.tolist()

        if len(case_subj) < 2:
            continue

        T_case = wide.loc[case_subj].values.astype(float)

        enrich_fn = _make_enrichment_fn(enrichment_method)

        # Scan all candidate sizes; pick best by direction-adjusted score.
        # Also collect every valid (ws, window_tps) pair for the permutation
        # test — needed to correct for the selection bias introduced when
        # multiple window sizes are tried (issue #4).
        # _apply_direction maps raw scores into a comparable signed value:
        #   'up'   → score as-is    (higher = better)
        #   'down' → negated score  (more negative raw = higher signed = better)
        #   'both' → abs(score)     (larger magnitude = better)
        best_signed = -np.inf
        best_raw = None
        best_ws = None
        best_window_tps = None
        best_pan_mp = None
        candidate_windows = []  # list of (ws, window_tps) for valid candidates

        for ws in sizes:
            if ws >= n_tp or ws < 3:  # STUMPY requires m >= 3
                continue
            pan_mp = compute_matrix_profile(T_case, ws)
            if not np.any(np.isfinite(pan_mp)):
                continue
            motif_idx = int(np.nanargmin(pan_mp))

            # Evaluate the matrix-profile argmin and its immediate neighbours
            # (±1 position).  The pan-matrix profile uses z-normalised Euclidean
            # distance, which finds the most shape-similar window across cases —
            # not necessarily the window with the best case-control enrichment.
            # For motifs with noisy onset timing, the argmin can land one position
            # away from the true start, causing a larger window size to win when
            # it accidentally covers the correct start.  Checking argmin ± 1 lets
            # the enrichment score resolve the tie at essentially no extra cost
            # (3× per window size; permutation test applies the same max-over-
            # candidates correction so p-values remain calibrated).
            for pos in (motif_idx - 1, motif_idx, motif_idx + 1):
                if pos < 0 or pos + ws > n_tp:
                    continue
                window_tps = timepoints[pos: pos + ws]
                score = enrich_fn(wide, case_subj, ctrl_subj, window_tps)
                signed = _apply_direction(score, direction)
                candidate_windows.append((ws, window_tps))
                if signed > best_signed:
                    best_signed = signed
                    best_raw = score
                    best_ws = ws
                    best_window_tps = window_tps
                    best_pan_mp = pan_mp

        if best_ws is None:
            continue

        # Permutation test.
        # Multiple candidates → "max-over-candidates" test: each permutation
        # takes the max direction-adjusted score across all candidate windows,
        # mirroring the real-data selection step (corrects selection bias).
        # Stratified permutation: if covariate_cols were given, labels are
        # shuffled within strata rather than globally.
        all_subj = np.array(case_subj + ctrl_subj)
        strata = (
            _subj_covars.loc[all_subj]
            .astype(str)
            .apply(lambda row: "|".join(row), axis=1)
            .values
            if _subj_covars is not None else None
        )
        perm_signed = np.empty(n_permutations)
        for i in range(n_permutations):
            perm_case, perm_ctrl = _permute_labels(all_subj, case_subj, strata, rng)
            perm_signed[i] = max(
                _apply_direction(enrich_fn(wide, perm_case, perm_ctrl, win_tps), direction)
                for _, win_tps in candidate_windows
            )

        p_value = float(np.mean(perm_signed >= best_signed))
        motif_start = best_window_tps[0]
        motif_end = best_window_tps[-1]

        records.append({
            "feature": feature,
            "window_size": best_ws,
            "motif_window": (motif_start, motif_end),
            "enrichment_score": round(best_raw, 6),
            "p_value": round(p_value, 4),
            "matrix_profile_min": round(float(np.nanmin(best_pan_mp)), 6),
        })

    base_cols = ["feature", "window_size", "motif_window", "enrichment_score",
                 "p_value", "q_value", "matrix_profile_min"]
    if not records:
        cols = base_cols + (["direction"] if direction == "both" else [])
        return pd.DataFrame(columns=cols)

    out = pd.DataFrame(records)
    out["q_value"] = _bh_correct(out["p_value"].values)
    out["q_value"] = out["q_value"].round(4)

    if direction == "up":
        out = out.sort_values("enrichment_score", ascending=False)
    elif direction == "down":
        out = out.sort_values("enrichment_score", ascending=True)
    else:  # "both"
        out["direction"] = out["enrichment_score"].apply(
            lambda x: "up" if x >= 0 else "down"
        )
        out = out.sort_values("enrichment_score", key=lambda s: s.abs(), ascending=False)

    return out.reset_index(drop=True).head(top_k)


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

def _permute_labels(
    all_subj: np.ndarray,
    case_subj: list,
    strata: Optional[np.ndarray],
    rng: np.random.Generator,
) -> tuple:
    """Permute case/control labels, optionally within covariate strata.

    Without strata (strata=None): globally shuffle all subject labels —
    the existing behaviour.

    With strata: shuffle labels *within* each stratum, preserving the number
    of cases per stratum.  Subjects in a stratum where every member shares
    the same outcome label (all-case or all-control) are left unchanged —
    there is nothing to permute in a homogeneous stratum.

    Parameters
    ----------
    all_subj : np.ndarray
        1D array of all subject IDs in the order (cases, controls).
    case_subj : list
        Subject IDs that are cases in the observed data.
    strata : np.ndarray or None
        1D array of stratum labels, aligned with all_subj.
        None triggers the global (unstratified) path.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns
    -------
    (perm_case, perm_ctrl) — lists of subject IDs for one permutation.
    """
    if strata is None:
        n_cases = len(case_subj)
        perm = rng.permutation(all_subj)
        return perm[:n_cases].tolist(), perm[n_cases:].tolist()

    case_set = set(case_subj)
    is_case = np.array([s in case_set for s in all_subj])
    perm_is_case = is_case.copy()

    for stratum in np.unique(strata):
        idx = np.where(strata == stratum)[0]
        if len(idx) < 2:
            continue  # singleton — nothing to shuffle
        n_cases_s = int(is_case[idx].sum())
        if n_cases_s == 0 or n_cases_s == len(idx):
            continue  # homogeneous stratum — nothing to shuffle
        shuffled = rng.permutation(len(idx))
        perm_is_case[idx] = is_case[idx[shuffled]]

    return all_subj[perm_is_case].tolist(), all_subj[~perm_is_case].tolist()


def _apply_direction(score: float, direction: str) -> float:
    """Map a raw enrichment score to a direction-adjusted comparable value.

    'up'   → score as-is  (higher raw score = better)
    'down' → -score        (more negative raw score = higher signed = better)
    'both' → abs(score)    (larger magnitude in either direction = better)

    Used consistently in window selection, permutation test, and p-value
    computation so that all three steps respect the requested direction.
    """
    if direction == "up":
        return score
    elif direction == "down":
        return -score
    else:
        return abs(score)


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


def _window_enrichment_template_corr(
    wide: pd.DataFrame,
    case_subj: list,
    ctrl_subj: list,
    window_tps: list,
) -> float:
    """Template-correlation enrichment: mean(case Pearson r) − mean(ctrl Pearson r).

    Case subjects are scored using a leave-one-out (LOO) template — each case
    is correlated against the mean of all OTHER cases. This removes the in-sample
    circularity that occurs when a subject contributes to the template it is then
    scored against, which inflates case correlations and produces anti-conservative
    p-values at small n_cases.

    Control subjects are scored against the full case template (mean of all cases)
    since they have no in-sample contribution.

    Called fresh per permutation iteration so the template is always recomputed
    from the current (permuted) case group.
    """
    if not case_subj:
        return 0.0
    case_vals = wide.loc[case_subj, window_tps].values.astype(float)

    def _r(row, tmpl):
        if row.std() < 1e-8 or tmpl.std() < 1e-8:
            return 0.0
        c = np.corrcoef(row, tmpl)[0, 1]
        return float(0.0 if np.isnan(c) else c)

    # LOO case correlations: each case scored against the template built from
    # all other cases, eliminating the in-sample upward bias.
    n_cases = len(case_subj)
    case_corrs = np.empty(n_cases)
    for i in range(n_cases):
        loo_template = np.delete(case_vals, i, axis=0).mean(axis=0)
        case_corrs[i] = _r(case_vals[i], loo_template)

    # Control correlations: scored against the full case template.
    full_template = case_vals.mean(axis=0)
    ctrl_corrs = np.zeros(1)
    if ctrl_subj:
        ctrl_vals = wide.loc[ctrl_subj, window_tps].values.astype(float)
        ctrl_corrs = np.array([_r(ctrl_vals[i], full_template) for i in range(len(ctrl_subj))])
    return float(case_corrs.mean() - ctrl_corrs.mean())


def _make_enrichment_fn(method: str):
    """Return the appropriate per-window enrichment function for the given method."""
    if method == "mean_difference":
        return _window_enrichment
    elif method == "template_correlation":
        return _window_enrichment_template_corr
    else:
        raise ValueError(
            f"Unknown enrichment_method '{method}'. "
            "Choose from: 'mean_difference', 'template_correlation'."
        )


def _bh_correct(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : np.ndarray
        1D array of raw p-values.

    Returns
    -------
    np.ndarray
        FDR-adjusted q-values, clipped to [0, 1].
    """
    n = len(p_values)
    if n == 0:
        return np.array([])
    order = np.argsort(p_values)
    ranks = np.empty(n)
    ranks[order] = np.arange(1, n + 1)
    q = p_values * n / ranks
    # Enforce monotonicity from largest rank downward
    q_sorted = q[order]
    for i in range(n - 2, -1, -1):
        q_sorted[i] = min(q_sorted[i], q_sorted[i + 1])
    q[order] = q_sorted
    return np.clip(q, 0.0, 1.0)

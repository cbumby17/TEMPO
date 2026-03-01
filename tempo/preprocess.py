"""
tempo/preprocess.py

Preprocessing functions for longitudinal biological data.

Transforms raw count or compositional data into trajectory-distance
representations suitable for matrix profile analysis. Handles
compositional data (16S microbiome) and continuous data (flow cytometry,
gene expression) via separate normalization strategies.
"""

import pandas as pd
import numpy as np
from typing import Optional


def preprocess(
    df: pd.DataFrame,
    method: str = "bray_curtis",
    clr: bool = False,
    pseudo_count: float = 1e-6,
) -> pd.DataFrame:
    """
    Preprocess a longitudinal dataframe for Harbinger analysis.

    Converts a long-format feature-by-timepoint dataframe into per-subject
    trajectory distance matrices ready for matrix profile computation.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value, outcome.
        Typically the output of simulate_longitudinal or simulate_continuous.
    method : str
        Distance/transformation method to apply:
        "bray_curtis" → pairwise Bray-Curtis dissimilarity between adjacent timepoints
        "clr"         → centered log-ratio transform (for compositional data)
        "none"        → no transformation; use raw values
    clr : bool
        If True, apply CLR transform before computing Bray-Curtis distances.
        Only relevant when method="bray_curtis". Ignored for other methods.
    pseudo_count : float
        Small value added before log transforms to avoid log(0).

    Returns
    -------
    pd.DataFrame
        Preprocessed long-format dataframe.
        - method="none": same schema as input, values unchanged
        - method="clr": same schema as input, values are CLR-transformed
        - method="bray_curtis": columns subject_id, timepoint, distance, outcome;
          one row per subject per timepoint transition (n_timepoints - 1 rows per subject)

    Raises
    ------
    ValueError
        If method is not one of "bray_curtis", "clr", "none".

    Examples
    --------
    >>> from tempo import simulate
    >>> from tempo.preprocess import preprocess
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> processed = preprocess(df, method="bray_curtis")
    >>> processed.head()
    """
    if method == "none":
        return df.copy()
    elif method == "clr":
        return clr_transform(df, pseudo_count=pseudo_count)
    elif method == "bray_curtis":
        working = clr_transform(df, pseudo_count=pseudo_count) if clr else df
        return bray_curtis_trajectory(working)
    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'bray_curtis', 'clr', 'none'."
        )


def clr_transform(
    df: pd.DataFrame,
    pseudo_count: float = 1e-6,
) -> pd.DataFrame:
    """
    Apply centered log-ratio (CLR) transformation to compositional data.

    CLR is the standard transformation for Aitchison-geometry analysis of
    compositional data. Each value is log-transformed and then centered by
    subtracting the geometric mean across features within a sample.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value.
        Values should be non-negative compositional data (e.g., relative abundances).
    pseudo_count : float
        Added to all values before taking the log to handle zeros.
        Typical values: 1e-6 (preserve near-zero structure) or 0.5 (conservative).

    Returns
    -------
    pd.DataFrame
        Same structure as input with CLR-transformed values. Preserves all
        non-value columns (subject_id, timepoint, feature, outcome, etc.)
        and df.attrs metadata.

    Notes
    -----
    CLR(x_i) = log(x_i / geometric_mean(x)) for each feature i in a sample.
    Equivalently: log(x_i + pseudo_count) - mean(log(x + pseudo_count)).
    After CLR, the data lies in real space and standard Euclidean methods apply.
    """
    result = df.copy()
    log_vals = np.log(result["value"].values + pseudo_count)
    group_means = (
        pd.Series(log_vals, index=result.index)
        .groupby([result["subject_id"], result["timepoint"]])
        .transform("mean")
    )
    result["value"] = log_vals - group_means.values
    result.attrs = df.attrs.copy()
    return result


def bray_curtis_trajectory(
    df: pd.DataFrame,
    subjects: Optional[list] = None,
) -> pd.DataFrame:
    """
    Compute Bray-Curtis dissimilarity between adjacent timepoints per subject.

    Converts a compositional trajectory into a scalar dissimilarity time series.
    Each value represents how much the community composition changed between
    consecutive timepoints, producing a (n_timepoints - 1)-length series per subject.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value.
        An outcome column is preserved in the output if present.
    subjects : list of str, optional
        Subset of subject IDs to process. If None, processes all subjects.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, distance, outcome.
        timepoint refers to the later of the two adjacent timepoints (t), so the
        first timepoint per subject is absent (there is no t-1 for t=0).
        Preserves df.attrs metadata.

    Notes
    -----
    Bray-Curtis dissimilarity between two compositions x and y:
        BC(x, y) = 1 - 2 * sum(min(x_i, y_i)) / (sum(x) + sum(y))
    Range [0, 1]: 0 = identical, 1 = completely dissimilar.
    """
    if subjects is not None:
        df = df[df["subject_id"].isin(subjects)]

    has_outcome = "outcome" in df.columns
    records = []

    for subj, subj_df in df.groupby("subject_id"):
        outcome = subj_df["outcome"].iloc[0] if has_outcome else None

        wide = (
            subj_df.pivot(index="timepoint", columns="feature", values="value")
            .sort_index()
        )
        timepoints = wide.index.tolist()

        for i in range(1, len(timepoints)):
            x = wide.iloc[i - 1].values
            y = wide.iloc[i].values
            denom = x.sum() + y.sum()
            bc = 0.0 if denom == 0 else 1.0 - 2.0 * np.minimum(x, y).sum() / denom

            record = {"subject_id": subj, "timepoint": timepoints[i], "distance": bc}
            if has_outcome:
                record["outcome"] = outcome
            records.append(record)

    result = pd.DataFrame(records)
    result.attrs = df.attrs.copy()
    return result


def check_baseline(
    df: pd.DataFrame,
    timepoint=None,
    outcome_col: str = "outcome",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test whether cases and controls differ at baseline.

    For each feature, compares case and control values at the specified
    baseline timepoint using a two-sided Mann-Whitney U test. Features
    with a significant difference at baseline may confound Harbinger
    analysis — the detected motif could reflect a pre-existing difference
    rather than a post-perturbation trajectory divergence.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature,
        value, outcome. The same format accepted by harbinger().
    timepoint : int or float, optional
        The timepoint to use as baseline. Defaults to the earliest timepoint
        in the data.
    outcome_col : str
        Column containing binary outcome labels (1 = case, 0 = control).
    alpha : float
        Significance threshold for the ``significant`` flag column.
        Default 0.05.

    Returns
    -------
    pd.DataFrame
        One row per feature with columns:
          feature        — feature name
          baseline_tp    — the timepoint tested
          case_mean      — mean value in cases at baseline
          ctrl_mean      — mean value in controls at baseline
          mean_diff      — case_mean − ctrl_mean
          p_value        — two-sided Mann-Whitney U p-value
          significant    — True if p_value < alpha

        Sorted by p_value ascending so the most concerning features appear
        first.

    Examples
    --------
    >>> df = tempo.load_example_data()
    >>> report = tempo.check_baseline(df)
    >>> report[report['significant']]   # features with baseline imbalance
    """
    from scipy.stats import mannwhitneyu

    if timepoint is None:
        timepoint = df["timepoint"].min()

    baseline = df[df["timepoint"] == timepoint]

    records = []
    for feat, feat_df in baseline.groupby("feature"):
        cases = feat_df[feat_df[outcome_col] == 1]["value"].values
        ctrls = feat_df[feat_df[outcome_col] == 0]["value"].values

        if len(cases) < 2 or len(ctrls) < 2:
            continue

        stat, p = mannwhitneyu(cases, ctrls, alternative="two-sided")
        records.append({
            "feature": feat,
            "baseline_tp": timepoint,
            "case_mean": float(cases.mean()),
            "ctrl_mean": float(ctrls.mean()),
            "mean_diff": float(cases.mean() - ctrls.mean()),
            "p_value": round(float(p), 4),
            "significant": bool(p < alpha),
        })

    return (
        pd.DataFrame(records)
        .sort_values("p_value")
        .reset_index(drop=True)
    )

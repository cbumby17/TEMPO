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
        If True, apply CLR transform before computing distances (compositional data).
        Ignored when method="clr".
    pseudo_count : float
        Small value added before log transforms to avoid log(0).

    Returns
    -------
    pd.DataFrame
        Preprocessed long-format dataframe, same schema as input but with
        transformed values. Additional column `distance` may be present
        for trajectory-distance methods.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Examples
    --------
    >>> from tempo import simulate, preprocess
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> processed = preprocess(df, method="bray_curtis")
    """
    raise NotImplementedError(
        "preprocess() is not yet implemented. "
        "Planned methods: 'bray_curtis', 'clr', 'none'."
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
        Same structure as input with CLR-transformed values.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Notes
    -----
    CLR(x_i) = log(x_i / geometric_mean(x)) for each feature i in a sample.
    After CLR, the data lies in real space and standard Euclidean methods apply.
    """
    raise NotImplementedError("clr_transform() is not yet implemented.")


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
    subjects : list of str, optional
        Subset of subject IDs to process. If None, processes all subjects.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, distance, outcome.
        The `distance` column contains Bray-Curtis dissimilarity between
        timepoint t and timepoint t-1.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Notes
    -----
    Bray-Curtis dissimilarity between two compositions x and y:
        BC(x, y) = 1 - 2 * sum(min(x_i, y_i)) / (sum(x) + sum(y))
    Range [0, 1]: 0 = identical, 1 = completely dissimilar.
    """
    raise NotImplementedError("bray_curtis_trajectory() is not yet implemented.")

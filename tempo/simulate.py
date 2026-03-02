"""
tempo/simulate.py

Simulation framework for TEMPO validation.

Generates synthetic longitudinal compositional time series with known
embedded trajectory motifs, allowing ground-truth evaluation of
Harbinger analysis sensitivity and specificity.

Core design:
    - Subjects are assigned to outcome groups
    - Case subjects receive an embedded motif at a specified (or random) window
    - Control subjects receive only noise trajectories
    - Compositional constraints are respected via Dirichlet sampling
    - Zeros are introduced at a user-specified rate to mimic real 16S data
"""

import numpy as np
import pandas as pd
from typing import Optional, Union


# ---------------------------------------------------------------------------
# Primary simulation entry point
# ---------------------------------------------------------------------------

def simulate_longitudinal(
    n_subjects: int = 30,
    n_timepoints: int = 10,
    n_features: int = 20,
    n_cases: int = 10,
    motif_features: Optional[list] = None,
    motif_window: Optional[tuple] = None,
    motif_strength: float = 1.5,
    noise_sd: float = 0.3,
    zero_inflation: float = 0.1,
    outcome_type: str = "binary",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Simulate longitudinal compositional data with embedded trajectory motifs.

    Parameters
    ----------
    n_subjects : int
        Total number of subjects (cases + controls).
    n_timepoints : int
        Number of evenly spaced timepoints (timepoint 0 = baseline).
    n_features : int
        Number of features (taxa, genes, cell populations).
    n_cases : int
        Number of subjects assigned to the case outcome group.
        Controls = n_subjects - n_cases.
    motif_features : list of int, optional
        Indices of features that carry the discriminative motif signal.
        If None, defaults to the first 3 features.
    motif_window : tuple of (int, int), optional
        (start, end) timepoint indices defining the motif window.
        If None, defaults to the middle third of the trajectory.
    motif_strength : float
        Effect size of the motif (multiplicative shift in case subjects).
        1.0 = no effect, 2.0 = strong effect. Default 1.5 is moderate.
    noise_sd : float
        Standard deviation of Gaussian noise added to all trajectories.
    zero_inflation : float
        Proportion of values to replace with zero (mimics sparse 16S data).
        0.0 = no zeros, 0.3 = highly sparse.
    outcome_type : str
        "binary"     → outcome column is 0/1
        "continuous" → outcome is a continuous score (useful for enrichment testing)
        "survival"   → adds a time_to_event column (useful for survival testing)
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        subject_id, timepoint, feature, value, outcome
        (and time_to_event if outcome_type="survival")

    Examples
    --------
    >>> df = simulate_longitudinal(n_subjects=40, n_cases=15, motif_strength=2.0, seed=0)
    >>> df.head()
    """
    rng = np.random.default_rng(seed)

    if motif_features is None:
        motif_features = [0, 1, 2]

    if motif_window is None:
        third = n_timepoints // 3
        motif_window = (third, 2 * third)

    n_controls = n_subjects - n_cases
    subjects = [f"case_{i:03d}" for i in range(n_cases)] + \
               [f"ctrl_{i:03d}" for i in range(n_controls)]
    outcomes = [1] * n_cases + [0] * n_controls

    records = []

    for subj, outcome in zip(subjects, outcomes):
        # Baseline composition: Dirichlet sample (sums to 1)
        baseline = rng.dirichlet(alpha=np.ones(n_features) * 2)

        for t in range(n_timepoints):
            # Random walk from baseline (in log space to respect positivity)
            noise = rng.normal(0, noise_sd, size=n_features)
            composition = baseline * np.exp(noise * t * 0.1)

            # Embed motif in case subjects during the motif window
            if outcome == 1 and motif_window[0] <= t <= motif_window[1]:
                for feat_idx in motif_features:
                    composition[feat_idx] *= motif_strength

            # Re-normalize to compositional (sums to 1)
            composition = np.abs(composition)
            composition /= composition.sum()

            # Zero inflation
            zero_mask = rng.random(n_features) < zero_inflation
            composition[zero_mask] = 0.0

            # Re-normalize after zeroing (if anything remains)
            total = composition.sum()
            if total > 0:
                composition /= total

            for feat_idx in range(n_features):
                records.append({
                    "subject_id": subj,
                    "timepoint": t,
                    "feature": f"feature_{feat_idx:03d}",
                    "value": composition[feat_idx],
                    "outcome": outcome,
                })

    df = pd.DataFrame(records)

    # Add survival-style outcome if requested
    if outcome_type == "survival":
        df = _add_survival_outcome(df, n_timepoints, rng)
    elif outcome_type == "continuous":
        df = _add_continuous_outcome(df, rng)

    # Attach simulation metadata as attributes for downstream validation
    df.attrs["motif_features"] = [f"feature_{i:03d}" for i in motif_features]
    df.attrs["motif_window"] = motif_window
    df.attrs["n_cases"] = n_cases
    df.attrs["n_controls"] = n_controls
    df.attrs["motif_strength"] = motif_strength

    return df


# ---------------------------------------------------------------------------
# Non-compositional simulation (flow cytometry, gene expression)
# ---------------------------------------------------------------------------

def simulate_continuous(
    n_subjects: int = 30,
    n_timepoints: int = 10,
    n_features: int = 10,
    n_cases: int = 10,
    motif_features: Optional[list] = None,
    motif_window: Optional[tuple] = None,
    motif_type: str = "step",
    motif_strength: float = 2.0,
    noise_sd: float = 0.5,
    baseline_mean: float = 0.0,
    outcome_type: str = "binary",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Simulate longitudinal continuous data (flow cytometry, gene expression).

    Unlike simulate_longitudinal, values are not compositionally constrained.
    Motif shapes can be step, ramp, or pulse, reflecting common biological
    response patterns.

    Parameters
    ----------
    motif_type : str
        Shape of the embedded motif:
        "step"        → sustained increase during motif window (e.g., activation)
        "ramp"        → linear increase during motif window (e.g., gradual expansion)
        "pulse"       → transient spike at motif window midpoint (e.g., acute response)
        "oscillating" → alternating +strength / −strength pattern (e.g., oscillatory response)
    motif_strength : float
        Amplitude of the motif signal in units of noise_sd.
    baseline_mean : float
        Mean of the baseline distribution across features.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
        subject_id, timepoint, feature, value, outcome
    """
    rng = np.random.default_rng(seed)

    if motif_features is None:
        motif_features = [0, 1, 2]

    if motif_window is None:
        third = n_timepoints // 3
        motif_window = (third, 2 * third)

    n_controls = n_subjects - n_cases
    subjects = [f"case_{i:03d}" for i in range(n_cases)] + \
               [f"ctrl_{i:03d}" for i in range(n_controls)]
    outcomes = [1] * n_cases + [0] * n_controls

    motif_signal = _build_motif_signal(motif_type, motif_window, n_timepoints, motif_strength)

    records = []
    for subj, outcome in zip(subjects, outcomes):
        subject_baseline = rng.normal(baseline_mean, 1.0, size=n_features)

        for t in range(n_timepoints):
            values = subject_baseline + rng.normal(0, noise_sd, size=n_features)

            if outcome == 1:
                for feat_idx in motif_features:
                    values[feat_idx] += motif_signal[t]

            for feat_idx in range(n_features):
                records.append({
                    "subject_id": subj,
                    "timepoint": t,
                    "feature": f"feature_{feat_idx:03d}",
                    "value": values[feat_idx],
                    "outcome": outcome,
                })

    df = pd.DataFrame(records)

    if outcome_type == "survival":
        df = _add_survival_outcome(df, n_timepoints, rng)
    elif outcome_type == "continuous":
        df = _add_continuous_outcome(df, rng)

    df.attrs["motif_features"] = [f"feature_{i:03d}" for i in motif_features]
    df.attrs["motif_window"] = motif_window
    df.attrs["motif_type"] = motif_type
    df.attrs["motif_strength"] = motif_strength

    return df


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def get_ground_truth(df: pd.DataFrame) -> dict:
    """
    Extract ground truth metadata from a simulated dataframe.

    Returns a dict with motif_features, motif_window, case_subjects, etc.
    Used to evaluate whether Harbinger analysis recovers the known signal.

    Parameters
    ----------
    df : pd.DataFrame
        Output of simulate_longitudinal or simulate_continuous.

    Returns
    -------
    dict
        Ground truth metadata.
    """
    if not df.attrs:
        raise ValueError(
            "This dataframe does not have simulation metadata. "
            "Make sure it was generated by simulate_longitudinal or simulate_continuous."
        )
    return {
        "motif_features": df.attrs.get("motif_features"),
        "motif_window": df.attrs.get("motif_window"),
        "n_cases": df.attrs.get("n_cases"),
        "n_controls": df.attrs.get("n_controls"),
        "motif_strength": df.attrs.get("motif_strength"),
        "case_subjects": df[df["outcome"] == 1]["subject_id"].unique().tolist(),
        "control_subjects": df[df["outcome"] == 0]["subject_id"].unique().tolist(),
    }


def evaluation_report(detected_features: list, detected_window: tuple, df: pd.DataFrame) -> dict:
    """
    Compare Harbinger analysis output against known ground truth.

    Parameters
    ----------
    detected_features : list of str
        Features flagged as motif-bearing by Harbinger analysis.
    detected_window : tuple of (int, int)
        Window identified by Harbinger analysis.
    df : pd.DataFrame
        Simulated dataframe with ground truth in attrs.

    Returns
    -------
    dict
        feature_recall, feature_precision, window_overlap (Jaccard)
    """
    truth = get_ground_truth(df)
    true_features = set(truth["motif_features"])
    detected_features = set(detected_features)

    recall = len(true_features & detected_features) / len(true_features) if true_features else 0
    precision = len(true_features & detected_features) / len(detected_features) if detected_features else 0

    # Window overlap as Jaccard index
    true_window = set(range(truth["motif_window"][0], truth["motif_window"][1] + 1))
    det_window = set(range(detected_window[0], detected_window[1] + 1))
    jaccard = len(true_window & det_window) / len(true_window | det_window) if (true_window | det_window) else 0

    return {
        "feature_recall": round(recall, 3),
        "feature_precision": round(precision, 3),
        "window_jaccard": round(jaccard, 3),
        "true_features": list(true_features),
        "detected_features": list(detected_features),
        "true_window": truth["motif_window"],
        "detected_window": detected_window,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_motif_signal(motif_type, motif_window, n_timepoints, strength):
    """Build a 1D motif signal vector of length n_timepoints."""
    signal = np.zeros(n_timepoints)
    start, end = motif_window
    window_len = end - start + 1

    if motif_type == "step":
        signal[start:end + 1] = strength

    elif motif_type == "ramp":
        signal[start:end + 1] = np.linspace(0, strength, window_len)

    elif motif_type == "pulse":
        mid = (start + end) // 2
        signal[mid] = strength
        if mid - 1 >= start:
            signal[mid - 1] = strength * 0.5
        if mid + 1 <= end:
            signal[mid + 1] = strength * 0.5

    elif motif_type == "oscillating":
        window_len = end - start + 1
        signal[start:end + 1] = [
            strength if i % 2 == 0 else -strength for i in range(window_len)
        ]

    else:
        raise ValueError(
            f"Unknown motif_type '{motif_type}'. Choose from: step, ramp, pulse, oscillating."
        )

    return signal


def _add_survival_outcome(df, n_timepoints, rng):
    """Add time_to_event column: cases get earlier events, controls get censored."""
    subject_outcomes = df[["subject_id", "outcome"]].drop_duplicates()
    tte = {}
    for _, row in subject_outcomes.iterrows():
        if row["outcome"] == 1:
            tte[row["subject_id"]] = int(rng.integers(n_timepoints // 2, n_timepoints))
        else:
            tte[row["subject_id"]] = int(rng.integers(n_timepoints, n_timepoints * 2))
    df["time_to_event"] = df["subject_id"].map(tte)
    return df


def _add_continuous_outcome(df, rng):
    """Replace binary outcome with continuous score correlated with case status."""
    subject_outcomes = df[["subject_id", "outcome"]].drop_duplicates()
    scores = {}
    for _, row in subject_outcomes.iterrows():
        base = 1.0 if row["outcome"] == 1 else 0.0
        scores[row["subject_id"]] = base + rng.normal(0, 0.3)
    df["outcome"] = df["subject_id"].map(scores)
    return df

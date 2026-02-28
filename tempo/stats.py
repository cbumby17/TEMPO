"""
tempo/stats.py

Statistical testing for Harbinger analysis results.

Three functions covering the main study designs:

    permutation_test  — label-shuffling null distribution for a specific
                        (feature, window) pair; returns observed score + p-value.

    enrichment_score  — standalone scorer with three methods:
                        mean_difference, GSEA-style running score, and AUC.

    survival_test     — stratify subjects by motif presence and test for
                        time-to-event differences (log-rank or Cox PH).
"""

import numpy as np
import pandas as pd
from scipy.stats import CensoredData, logrank as _scipy_logrank
from typing import Optional


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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

    For a given (feature, window) pair, the test statistic is the mean
    difference in per-subject window scores between cases and controls.
    The null distribution is built by randomly permuting outcome labels
    across subjects (preserving each subject's temporal structure) and
    recomputing the statistic n_permutations times.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature,
        value, outcome.
    feature : str
        Feature name to test (e.g. "feature_000").
    motif_window : tuple of (int, int)
        (start, end) timepoints defining the candidate motif window,
        inclusive on both ends.
    n_permutations : int
        Number of label-permutation iterations.
    outcome_col : str
        Column containing binary outcome labels (1 = case, 0 = control).
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    dict
        observed_score  — mean(case window scores) − mean(ctrl window scores)
        p_value         — fraction of null scores ≥ observed score
        null_mean       — mean of the permutation null distribution
        null_sd         — SD of the permutation null distribution
        n_permutations  — number of permutations run
    """
    rng = np.random.default_rng(seed)

    subject_scores = _subject_window_scores(df, feature, motif_window, outcome_col)
    scores = subject_scores["score"].values
    outcomes = subject_scores[outcome_col].values

    case_mask = outcomes == 1
    obs_score = float(scores[case_mask].mean() - scores[~case_mask].mean())

    # Build null distribution: permute labels, keep scores fixed.
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(outcomes)
        case_m = perm == 1
        null_scores[i] = scores[case_m].mean() - scores[~case_m].mean()

    p_value = float(np.mean(null_scores >= obs_score))

    return {
        "observed_score": round(obs_score, 6),
        "p_value": round(p_value, 4),
        "null_mean": round(float(null_scores.mean()), 6),
        "null_sd": round(float(null_scores.std()), 6),
        "n_permutations": n_permutations,
    }


def enrichment_score(
    df: pd.DataFrame,
    feature: str,
    motif_window: tuple,
    outcome_col: str = "outcome",
    method: str = "mean_difference",
) -> float:
    """
    Compute the enrichment score for a trajectory motif.

    All three methods reduce each subject to a single scalar — their mean
    feature value over the motif window — then measure how well that
    scalar separates cases from controls.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature,
        value, outcome.
    feature : str
        Feature name to score.
    motif_window : tuple of (int, int)
        (start, end) timepoints defining the motif window, inclusive.
    outcome_col : str
        Column containing binary outcome labels (1 = case, 0 = control).
    method : str
        Scoring method:

        "mean_difference"
            mean(case window scores) − mean(ctrl window scores).
            Simple and interpretable; the same statistic used inside
            harbinger(). Positive values = cases are elevated.

        "auc"
            Area under the ROC curve treating the window score as a
            classifier. 0.5 = no separation, 1.0 = perfect separation.
            Threshold-free and invariant to monotone transformations of
            the score.

        "gsea"
            GSEA-style weighted running enrichment score. Subjects are
            ranked by window score (descending). Walking down the ranked
            list, each case hit is weighted proportionally to its score;
            each control miss decrements by 1/n_ctrl. The enrichment
            score is the maximum of the running sum. Captures whether
            cases cluster at the top of the ranking, analogous to
            gene-set enrichment analysis applied to subjects.

    Returns
    -------
    float
        Enrichment score. Higher = stronger case enrichment.

    Raises
    ------
    ValueError
        If method is not one of "mean_difference", "auc", "gsea".
    """
    subject_scores = _subject_window_scores(df, feature, motif_window, outcome_col)
    scores = subject_scores["score"].values
    outcomes = subject_scores[outcome_col].values

    case_mask = outcomes == 1
    ctrl_mask = ~case_mask

    if method == "mean_difference":
        return float(scores[case_mask].mean() - scores[ctrl_mask].mean())

    elif method == "auc":
        from sklearn.metrics import roc_auc_score
        if len(np.unique(outcomes)) < 2:
            return 0.5  # undefined when only one class present
        return float(roc_auc_score(outcomes, scores))

    elif method == "gsea":
        n_cases = int(case_mask.sum())
        n_ctrl = int(ctrl_mask.sum())
        if n_cases == 0 or n_ctrl == 0:
            return 0.0

        # Rank subjects by window score, descending.
        order = np.argsort(scores)[::-1]
        ranked_outcomes = outcomes[order]
        ranked_scores = scores[order]

        # Weight each case hit by its absolute score (weighted GSEA, p=1).
        case_score_sum = np.abs(ranked_scores[ranked_outcomes == 1]).sum()

        running = 0.0
        max_running = 0.0
        for i, outcome in enumerate(ranked_outcomes):
            if outcome == 1:
                w = abs(ranked_scores[i]) / case_score_sum if case_score_sum > 0 else 1.0 / n_cases
                running += w
            else:
                running -= 1.0 / n_ctrl
            if running > max_running:
                max_running = running

        return float(max_running)

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'mean_difference', 'auc', 'gsea'."
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
    Test whether trajectory motif presence predicts time-to-event outcome.

    Subjects are stratified into "motif-positive" (window score ≥ median)
    and "motif-negative" (< median) groups. For log-rank, survival curves
    of the two groups are compared. For Cox, the continuous window score
    is used directly as a covariate.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe including time_to_event and outcome columns.
        Typically output of simulate_longitudinal(outcome_type="survival").
    feature : str
        Feature defining the trajectory motif.
    motif_window : tuple of (int, int)
        (start, end) timepoints defining the motif window, inclusive.
    time_col : str
        Column containing time-to-event values (numeric, > 0).
    event_col : str
        Column containing the event indicator (1 = event occurred,
        0 = censored). This doubles as the case/control label in
        simulated data.
    method : str
        "logrank" — log-rank test via scipy.stats.logrank.
                    Subjects stratified at median motif score.
                    Works without optional dependencies.

        "cox"     — Cox proportional hazards regression via lifelines.
                    Uses motif score as a continuous covariate (no
                    median stratification). Install lifelines with:
                    pip install tempo-bio[survival]

    Returns
    -------
    dict
        Common keys: statistic, p_value, method, feature, motif_window.

        log-rank additional keys:
            n_motif_positive, n_motif_negative, score_threshold

        cox additional keys:
            hazard_ratio, confidence_interval (95% CI tuple)

    Raises
    ------
    ValueError
        If method is not "logrank" or "cox".
    ImportError
        If method="cox" and lifelines is not installed.
    """
    # Per-subject motif score (mean feature value in window).
    subject_scores = _subject_window_scores(df, feature, motif_window, outcome_col=None)

    # Survival metadata: one row per subject.
    survival = (
        df[["subject_id", time_col, event_col]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    data = subject_scores.set_index("subject_id").join(survival).reset_index()

    if method == "logrank":
        # Stratify at median window score.
        threshold = float(data["score"].median())
        pos = data[data["score"] >= threshold]
        neg = data[data["score"] < threshold]

        def _censored(grp):
            uncensored = grp[time_col].values[grp[event_col].values == 1]
            right_censored = grp[time_col].values[grp[event_col].values == 0]
            return CensoredData(uncensored=uncensored, right=right_censored)

        result = _scipy_logrank(_censored(pos), _censored(neg))

        return {
            "statistic": round(float(result.statistic), 6),
            "p_value": round(float(result.pvalue), 4),
            "method": "logrank",
            "feature": feature,
            "motif_window": motif_window,
            "n_motif_positive": len(pos),
            "n_motif_negative": len(neg),
            "score_threshold": round(threshold, 6),
        }

    elif method == "cox":
        try:
            from lifelines import CoxPHFitter
        except ImportError:
            raise ImportError(
                "Cox proportional hazards requires lifelines. "
                "Install it with: pip install tempo-bio[survival]"
            )

        cph = CoxPHFitter()
        cph.fit(
            data[["score", time_col, event_col]],
            duration_col=time_col,
            event_col=event_col,
        )
        row = cph.summary.loc["score"]
        return {
            "statistic": round(float(row["z"]), 6),
            "p_value": round(float(row["p"]), 4),
            "method": "cox",
            "feature": feature,
            "motif_window": motif_window,
            "hazard_ratio": round(float(np.exp(row["coef"])), 4),
            "confidence_interval": (
                round(float(np.exp(row["coef lower 95%"])), 4),
                round(float(np.exp(row["coef upper 95%"])), 4),
            ),
        }

    else:
        raise ValueError(
            f"Unknown method '{method}'. Choose from: 'logrank', 'cox'."
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _subject_window_scores(
    df: pd.DataFrame,
    feature: str,
    motif_window: tuple,
    outcome_col: Optional[str] = "outcome",
) -> pd.DataFrame:
    """
    Compute per-subject mean feature value in the motif window.

    Returns a DataFrame with columns: subject_id, score[, outcome_col].
    """
    feat_df = df[df["feature"] == feature]
    window_df = feat_df[
        feat_df["timepoint"].between(motif_window[0], motif_window[1])
    ]

    agg = {"score": ("value", "mean")}
    if outcome_col is not None and outcome_col in window_df.columns:
        agg[outcome_col] = (outcome_col, "first")

    return window_df.groupby("subject_id").agg(**agg).reset_index()

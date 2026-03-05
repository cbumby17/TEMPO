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
from scipy.stats import CensoredData, logrank as _scipy_logrank, mannwhitneyu
from typing import Optional
from tempo.harbinger import _permute_labels


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
    covariate_cols: Optional[list] = None,
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
    covariate_cols : list of str, optional
        Column names in ``df`` to use for stratified permutation.  Labels
        are shuffled only within strata defined by combinations of these
        covariate values, preserving the covariate distribution under the
        null.  Mirrors the same parameter on ``harbinger()``.

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
    all_subj = subject_scores["subject_id"].values
    scores = subject_scores["score"].values
    outcomes = subject_scores[outcome_col].values

    case_subj = all_subj[outcomes == 1].tolist()
    case_mask = outcomes == 1
    obs_score = float(scores[case_mask].mean() - scores[~case_mask].mean())

    # Build per-subject strata if covariate_cols provided.
    if covariate_cols is not None:
        subj_covars = (
            df.groupby("subject_id")[covariate_cols]
            .first()
            .fillna("__missing__")
        )
        strata = (
            subj_covars.loc[all_subj]
            .astype(str)
            .apply(lambda row: "|".join(row), axis=1)
            .values
        )
    else:
        strata = None

    # Build null distribution: permute labels (stratified or global).
    score_map = dict(zip(all_subj, scores))
    null_scores = np.empty(n_permutations)
    for i in range(n_permutations):
        perm_case, perm_ctrl = _permute_labels(all_subj, case_subj, strata, rng)
        c_scores = np.array([score_map[s] for s in perm_case])
        k_scores = np.array([score_map[s] for s in perm_ctrl])
        null_scores[i] = c_scores.mean() - k_scores.mean()

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

        "template_correlation"
            Each subject is scored by their Pearson correlation to the mean
            case trajectory (the "template") in the motif window. The
            enrichment score is mean(case correlations) − mean(ctrl
            correlations). Captures shape, direction, and oscillation —
            patterns that are invisible to mean_difference. Range: [−2, 2].

    Returns
    -------
    float
        Enrichment score. Higher = stronger case enrichment.

    Raises
    ------
    ValueError
        If method is not one of "mean_difference", "auc", "gsea",
        "template_correlation".
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

    elif method == "template_correlation":
        # Build subjects × timepoints matrix for the motif window.
        feat_df = df[df["feature"] == feature]
        window_df = feat_df[
            feat_df["timepoint"].between(motif_window[0], motif_window[1])
        ]
        wide = (
            window_df.pivot(index="subject_id", columns="timepoint", values="value")
            .reindex(sorted(window_df["timepoint"].unique()), axis=1)
        )
        if wide.shape[1] < 2:
            return 0.0

        outcome_map = window_df.groupby("subject_id")[outcome_col].first()
        case_subj = outcome_map[outcome_map == 1].index.tolist()
        ctrl_subj = outcome_map[outcome_map == 0].index.tolist()

        if not case_subj:
            return 0.0

        case_vals = wide.loc[case_subj].values.astype(float)
        template = case_vals.mean(axis=0)
        if template.std() < 1e-8:
            return 0.0

        def _r(row):
            if row.std() < 1e-8:
                return 0.0
            c = float(np.corrcoef(row, template)[0, 1])
            return 0.0 if np.isnan(c) else c

        case_corrs = np.array([_r(case_vals[i]) for i in range(len(case_subj))])
        if ctrl_subj:
            ctrl_vals = wide.loc[ctrl_subj].values.astype(float)
            ctrl_corrs = np.array([_r(ctrl_vals[i]) for i in range(len(ctrl_subj))])
        else:
            ctrl_corrs = np.zeros(1)
        return float(case_corrs.mean() - ctrl_corrs.mean())

    else:
        raise ValueError(
            f"Unknown method '{method}'. "
            "Choose from: 'mean_difference', 'auc', 'gsea', 'template_correlation'."
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


def compute_resistance(
    df: pd.DataFrame,
    feature: str,
    perturbation_tp: int,
    baseline_window: Optional[tuple] = None,
    outcome_col: Optional[str] = "outcome",
) -> pd.DataFrame:
    """
    Compute per-subject resistance scores: signed peak deflection from baseline.

    In ecological terms, resistance is the ability of a system to remain
    unchanged when perturbed. Here it is operationalised as the signed
    peak deflection from the pre-perturbation baseline — the timepoint at
    which the feature departs furthest (in absolute terms) from baseline.
    Positive values indicate elevation above baseline; negative values
    indicate depression below baseline.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value.
    feature : str
        Feature to evaluate.
    perturbation_tp : int
        Timepoint of perturbation onset. Post-perturbation window is all
        timepoints >= perturbation_tp.
    baseline_window : tuple of (int, int), optional
        (start, end) timepoints (inclusive) defining the pre-perturbation
        baseline. Defaults to all timepoints strictly before perturbation_tp.
    outcome_col : str or None
        Column to carry through to the output for downstream group comparisons.
        Pass None to omit.

    Returns
    -------
    pd.DataFrame
        One row per subject with columns:
        subject_id, baseline_mean, peak_value, resistance [, outcome_col]

        resistance = value_at_peak_deflection − baseline_mean  (signed)
    """
    feat_df = df[df["feature"] == feature]
    timepoints = sorted(feat_df["timepoint"].unique())

    if baseline_window is None:
        pre_tps = [t for t in timepoints if t < perturbation_tp]
    else:
        pre_tps = [t for t in timepoints if baseline_window[0] <= t <= baseline_window[1]]

    post_tps = [t for t in timepoints if t >= perturbation_tp]

    if not pre_tps:
        raise ValueError(
            f"No baseline timepoints found before perturbation_tp={perturbation_tp}. "
            "Provide a baseline_window or use a perturbation_tp > 0."
        )
    if not post_tps:
        raise ValueError(
            f"No post-perturbation timepoints at or after perturbation_tp={perturbation_tp}."
        )

    baseline = (
        feat_df[feat_df["timepoint"].isin(pre_tps)]
        .groupby("subject_id")["value"]
        .mean()
        .rename("baseline_mean")
    )

    # Find the signed deflection with largest absolute magnitude.
    post_df = feat_df[feat_df["timepoint"].isin(post_tps)].copy()
    post_df = post_df.join(baseline, on="subject_id")
    post_df["deflection"] = post_df["value"] - post_df["baseline_mean"]

    def _peak_signed(grp):
        idx = grp["deflection"].abs().idxmax()
        return pd.Series({
            "peak_value": grp.loc[idx, "value"],
            "resistance": grp.loc[idx, "deflection"],
        })

    result = post_df.groupby("subject_id").apply(_peak_signed).reset_index()
    result = result.join(baseline, on="subject_id")

    if outcome_col is not None and outcome_col in feat_df.columns:
        outcome_map = feat_df.groupby("subject_id")[outcome_col].first()
        result[outcome_col] = result["subject_id"].map(outcome_map)

    return result[["subject_id", "baseline_mean", "peak_value", "resistance"]
                  + ([outcome_col] if outcome_col is not None and outcome_col in result.columns else [])]


def compute_resilience(
    df: pd.DataFrame,
    feature: str,
    perturbation_tp: int,
    baseline_window: Optional[tuple] = None,
    recovery_threshold: float = 0.2,
    outcome_col: Optional[str] = "outcome",
) -> pd.DataFrame:
    """
    Compute per-subject resilience: how quickly each subject returns to baseline.

    Resilience is operationalised as the speed of recovery after a perturbation.
    For each subject, the function identifies the timepoint of peak deflection
    from baseline, then finds the first subsequent timepoint at which the
    feature has returned to within recovery_threshold × |peak_deflection| of
    the baseline mean. Subjects that do not recover within the observation
    window receive time_to_recovery = inf and resilience_index = 0.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value.
    feature : str
        Feature to evaluate.
    perturbation_tp : int
        Timepoint of perturbation onset.
    baseline_window : tuple of (int, int), optional
        (start, end) timepoints defining the pre-perturbation baseline.
        Defaults to all timepoints strictly before perturbation_tp.
    recovery_threshold : float
        Fraction of peak deflection that counts as "recovered". A subject is
        considered recovered at the first post-peak timepoint where
        |value − baseline_mean| ≤ recovery_threshold × |peak_deflection|.
        Default 0.2 (i.e., within 20 % of the peak excursion).
    outcome_col : str or None
        Column to carry through to the output.

    Returns
    -------
    pd.DataFrame
        One row per subject with columns:
        subject_id, peak_tp, time_to_recovery, resilience_index [, outcome_col]

        time_to_recovery — number of timepoint steps from peak to recovery
                           (inf if not recovered within the window)
        resilience_index — 1 / time_to_recovery; 0 for non-recoverers;
                           higher = faster recovery = greater resilience
    """
    feat_df = df[df["feature"] == feature]
    timepoints = sorted(feat_df["timepoint"].unique())

    if baseline_window is None:
        pre_tps = [t for t in timepoints if t < perturbation_tp]
    else:
        pre_tps = [t for t in timepoints if baseline_window[0] <= t <= baseline_window[1]]

    post_tps = [t for t in timepoints if t >= perturbation_tp]

    if not pre_tps:
        raise ValueError(
            f"No baseline timepoints before perturbation_tp={perturbation_tp}."
        )
    if not post_tps:
        raise ValueError(
            f"No post-perturbation timepoints at or after perturbation_tp={perturbation_tp}."
        )

    baseline = (
        feat_df[feat_df["timepoint"].isin(pre_tps)]
        .groupby("subject_id")["value"]
        .mean()
    )

    post_df = feat_df[feat_df["timepoint"].isin(post_tps)].copy().sort_values("timepoint")
    post_df = post_df.join(baseline.rename("baseline_mean"), on="subject_id")
    post_df["deflection"] = post_df["value"] - post_df["baseline_mean"]

    records = []
    for subj, grp in post_df.groupby("subject_id"):
        grp = grp.sort_values("timepoint")
        abs_deflections = grp["deflection"].abs()
        peak_idx = abs_deflections.idxmax()
        peak_tp = grp.loc[peak_idx, "timepoint"]
        peak_abs = abs_deflections[peak_idx]
        thr = recovery_threshold * max(peak_abs, 1e-8)

        # Timepoints after the peak
        after_peak = grp[grp["timepoint"] > peak_tp]
        recovered = after_peak[after_peak["deflection"].abs() <= thr]

        if len(recovered) > 0:
            recovery_tp = recovered["timepoint"].iloc[0]
            # time_to_recovery in timepoint units
            ttp = recovery_tp - peak_tp
            ri = 1.0 / max(ttp, 1)
        else:
            ttp = float("inf")
            ri = 0.0

        records.append({
            "subject_id": subj,
            "peak_tp": peak_tp,
            "time_to_recovery": ttp,
            "resilience_index": ri,
        })

    result = pd.DataFrame(records)

    if outcome_col is not None and outcome_col in feat_df.columns:
        outcome_map = feat_df.groupby("subject_id")[outcome_col].first()
        result[outcome_col] = result["subject_id"].map(outcome_map)

    return result


def compare_recovery(
    df: pd.DataFrame,
    feature: str,
    perturbation_tp: int,
    baseline_window: Optional[tuple] = None,
    recovery_threshold: float = 0.2,
    outcome_col: str = "outcome",
) -> dict:
    """
    Compare resistance and resilience between cases and controls.

    Computes per-subject resistance and resilience scores, then summarises
    and compares them between case (outcome=1) and control (outcome=0)
    groups using the Mann-Whitney U test (non-parametric, robust to small
    samples and non-normal distributions).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with outcome column.
    feature : str
        Feature to evaluate.
    perturbation_tp : int
        Timepoint of perturbation onset.
    baseline_window : tuple of (int, int), optional
        Pre-perturbation baseline window. Defaults to all timepoints
        before perturbation_tp.
    recovery_threshold : float
        Recovery threshold fraction passed to compute_resilience().
    outcome_col : str
        Column containing binary outcome labels (1 = case, 0 = control).

    Returns
    -------
    dict
        resistance:
            case_mean, ctrl_mean, case_sd, ctrl_sd, statistic, p_value
        resilience:
            case_mean, ctrl_mean, case_sd, ctrl_sd, statistic, p_value
            (finite time_to_recovery values only; inf excluded)
        n_cases, n_controls, feature, perturbation_tp
    """
    res_df = compute_resistance(df, feature, perturbation_tp, baseline_window, outcome_col)
    sil_df = compute_resilience(df, feature, perturbation_tp, baseline_window,
                                recovery_threshold, outcome_col)

    cases_r = res_df.loc[res_df[outcome_col] == 1, "resistance"].values
    ctrl_r = res_df.loc[res_df[outcome_col] == 0, "resistance"].values

    # Use finite time_to_recovery only for resilience comparison
    finite_sil = sil_df[sil_df["time_to_recovery"] != float("inf")]
    cases_ttp = finite_sil.loc[finite_sil[outcome_col] == 1, "time_to_recovery"].values
    ctrl_ttp = finite_sil.loc[finite_sil[outcome_col] == 0, "time_to_recovery"].values

    def _mw(a, b):
        if len(a) < 2 or len(b) < 2:
            return float("nan"), float("nan")
        stat, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(stat), float(p)

    r_stat, r_p = _mw(cases_r, ctrl_r)
    ttp_stat, ttp_p = _mw(cases_ttp, ctrl_ttp)

    return {
        "feature": feature,
        "perturbation_tp": perturbation_tp,
        "n_cases": int((res_df[outcome_col] == 1).sum()),
        "n_controls": int((res_df[outcome_col] == 0).sum()),
        "resistance": {
            "case_mean": round(float(np.mean(cases_r)), 6) if len(cases_r) else float("nan"),
            "ctrl_mean": round(float(np.mean(ctrl_r)), 6) if len(ctrl_r) else float("nan"),
            "case_sd": round(float(np.std(cases_r)), 6) if len(cases_r) else float("nan"),
            "ctrl_sd": round(float(np.std(ctrl_r)), 6) if len(ctrl_r) else float("nan"),
            "statistic": round(r_stat, 4) if not np.isnan(r_stat) else float("nan"),
            "p_value": round(r_p, 4) if not np.isnan(r_p) else float("nan"),
        },
        "resilience": {
            "case_mean": round(float(np.mean(cases_ttp)), 6) if len(cases_ttp) else float("nan"),
            "ctrl_mean": round(float(np.mean(ctrl_ttp)), 6) if len(ctrl_ttp) else float("nan"),
            "case_sd": round(float(np.std(cases_ttp)), 6) if len(cases_ttp) else float("nan"),
            "ctrl_sd": round(float(np.std(ctrl_ttp)), 6) if len(ctrl_ttp) else float("nan"),
            "statistic": round(ttp_stat, 4) if not np.isnan(ttp_stat) else float("nan"),
            "p_value": round(ttp_p, 4) if not np.isnan(ttp_p) else float("nan"),
        },
    }


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

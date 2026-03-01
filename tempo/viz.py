"""
tempo/viz.py

Publication-ready visualisation functions for TEMPO analysis results.

    plot_motifs      — per-subject trajectory overlays with group mean ± SD bands
                       and motif window shading; one subplot per feature.

    plot_enrichment  — two-panel summary of harbinger() results: enrichment scores
                       (with significance stars) and -log10(p) bars.

    plot_survival    — Kaplan-Meier survival curves for two groups stratified by
                       motif score (motif-positive vs motif-negative). No external
                       survival library required.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Optional

# Consistent colour palette used across both functions
_CASE_COLOR = "#e05c5c"
_CTRL_COLOR = "#5c8ae0"
_WINDOW_COLOR = "gold"
_SIG_COLOR = "#c0392b"
_NS_COLOR = "#aaaaaa"


def plot_motifs(
    df: pd.DataFrame,
    features: Optional[list] = None,
    motif_window: Optional[tuple] = None,
    highlight_cases: bool = True,
    baseline_normalize: bool = False,
    show_individuals: bool = True,
    ribbon_type: str = "sd",
    ax=None,
    figsize: tuple = (10, 4),
) -> plt.Figure:
    """
    Plot trajectory overlays for selected features with motif window highlighting.

    Each feature gets one subplot. Per-subject trajectories are optionally drawn
    as thin semi-transparent lines; group mean ± SD or SEM bands are overlaid as
    thicker lines with shaded ribbons. If a motif window is provided it is shaded
    in gold with dashed boundary lines.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature,
        value. An outcome column (0/1) is required for case/control colouring.
    features : list of str, optional
        Features to plot. Defaults to the first 4 features (alphabetical).
    motif_window : tuple of (int, int), optional
        (start, end) timepoint range to highlight. No shading if None.
    highlight_cases : bool
        If True, draw cases in red and controls in blue. If False, all
        trajectories are drawn in a neutral grey (useful when there is no
        binary outcome column).
    baseline_normalize : bool
        If True, subtract each subject's value at the earliest timepoint before
        plotting. This converts the y-axis to change-from-baseline (Δvalue),
        which reduces between-subject baseline variance and makes group-level
        temporal trends more visible. Default False.
    show_individuals : bool
        If True (default), draw thin per-subject lines behind the group ribbons.
        Set to False to show only group mean ± ribbon — useful when individual
        variability is high relative to the group effect.
    ribbon_type : str
        Width of the shaded ribbon around the group mean: ``'sd'`` (default)
        for ±1 standard deviation, or ``'sem'`` for ±1 standard error of the
        mean. SEM ribbons are narrower (SD / √n) and better reflect the
        precision of the group mean rather than individual spread.
    ax : matplotlib.axes.Axes, optional
        Pre-existing axes to draw into. Only used when exactly one feature is
        requested; ignored for multi-feature layouts.
    figsize : tuple
        (width, height) in inches per *row* of subplots. Total figure height
        scales with the number of rows required.

    Returns
    -------
    matplotlib.figure.Figure

    Raises
    ------
    ValueError
        If ribbon_type is not 'sd' or 'sem'.
    """
    if ribbon_type not in ("sd", "sem"):
        raise ValueError("ribbon_type must be 'sd' or 'sem'.")

    if features is None:
        features = sorted(df["feature"].unique())[:4]

    n_feats = len(features)
    has_outcome = "outcome" in df.columns and highlight_cases

    # ── Figure layout ────────────────────────────────────────────────────────
    if n_feats == 1 and ax is not None:
        fig = ax.get_figure()
        axes_flat = [ax]
    else:
        ncols = min(n_feats, 4)
        nrows = (n_feats + ncols - 1) // ncols
        fig, axes_arr = plt.subplots(
            nrows, ncols,
            figsize=(figsize[0] * ncols / min(n_feats, 4), figsize[1] * nrows),
            squeeze=False,
        )
        axes_flat = axes_arr.flatten().tolist()
        for unused in axes_flat[n_feats:]:
            unused.set_visible(False)
        axes_flat = axes_flat[:n_feats]

    timepoints = sorted(df["timepoint"].unique())
    first_tp = timepoints[0]

    # ── One subplot per feature ──────────────────────────────────────────────
    for ax_, feat in zip(axes_flat, features):
        feat_df = df[df["feature"] == feat].copy()

        # Baseline-normalize: subtract each subject's value at the first timepoint
        if baseline_normalize:
            baselines = (
                feat_df[feat_df["timepoint"] == first_tp]
                .set_index("subject_id")["value"]
            )
            feat_df["value"] = feat_df["value"] - feat_df["subject_id"].map(baselines)

        # Individual subject lines (optional)
        if show_individuals:
            for _, grp in feat_df.groupby("subject_id"):
                if has_outcome:
                    outcome = grp["outcome"].iloc[0]
                    color = _CASE_COLOR if outcome == 1 else _CTRL_COLOR
                    alpha = 0.45 if outcome == 1 else 0.28
                else:
                    color, alpha = "#888888", 0.3
                ax_.plot(grp["timepoint"], grp["value"],
                         color=color, alpha=alpha, lw=1.0, zorder=2)

        # Group mean ± ribbon bands
        if has_outcome:
            for outcome, color in [(1, _CASE_COLOR), (0, _CTRL_COLOR)]:
                grp_agg = feat_df[feat_df["outcome"] == outcome].groupby("timepoint")["value"]
                means = grp_agg.mean()
                stds = grp_agg.std()
                if ribbon_type == "sem":
                    counts = grp_agg.count()
                    spread = stds / np.sqrt(counts)
                else:
                    spread = stds
                ax_.plot(means.index, means.values, color=color, lw=2.5, zorder=4)
                ax_.fill_between(means.index, means - spread, means + spread,
                                 color=color, alpha=0.25, zorder=3)

        # Motif window shading
        if motif_window is not None:
            ax_.axvspan(motif_window[0], motif_window[1],
                        alpha=0.13, color=_WINDOW_COLOR, zorder=1)
            for boundary in motif_window:
                ax_.axvline(boundary, color="goldenrod", lw=1.0, ls="--", zorder=1)

        ax_.set_title(feat, fontsize=10)
        ax_.set_xlabel("Timepoint")
        ax_.set_ylabel("Δ Value (from baseline)" if baseline_normalize else "Value")
        ax_.set_xticks(timepoints)

    # ── Shared legend ────────────────────────────────────────────────────────
    handles = []
    if has_outcome:
        handles += [
            mpatches.Patch(color=_CASE_COLOR, alpha=0.75, label="Cases"),
            mpatches.Patch(color=_CTRL_COLOR, alpha=0.75, label="Controls"),
        ]
    if motif_window is not None:
        handles.append(mpatches.Patch(color=_WINDOW_COLOR, alpha=0.5, label="Motif window"))

    if handles:
        fig.legend(handles=handles, loc="upper right",
                   bbox_to_anchor=(1.01, 1.0), framealpha=0.9)

    fig.tight_layout()
    return fig


def plot_enrichment(
    results: pd.DataFrame,
    top_k: int = 20,
    colormap: str = "RdBu_r",
    ax=None,
    figsize: tuple = (11, 6),
) -> plt.Figure:
    """
    Two-panel summary of Harbinger analysis results.

    Left panel — enrichment scores as horizontal bars, coloured by
    significance (red = p < 0.05, grey = not significant), with
    significance stars appended to feature labels.

    Right panel — -log10(p-value) bars with a dashed line at the
    p = 0.05 threshold, making it easy to see which features clear
    the significance hurdle.

    Parameters
    ----------
    results : pd.DataFrame
        Output of harbinger(), with columns: feature, motif_window,
        enrichment_score, p_value. Any extra columns are ignored.
    top_k : int
        Maximum number of features to display, taken from the top of
        results (which is already sorted by enrichment_score descending).
    colormap : str
        Matplotlib colormap name. Currently used to tint the enrichment
        score bars by score magnitude. Default "RdBu_r".
    ax : matplotlib.axes.Axes, optional
        If provided, only the enrichment-score panel is drawn into this
        axes (single-panel mode). The -log10 panel is omitted.
    figsize : tuple
        (width, height) in inches. Applies only when ax is None.

    Returns
    -------
    matplotlib.figure.Figure
    """
    plot_df = results.head(top_k).copy().reset_index(drop=True)

    # Reverse so highest enrichment appears at top of horizontal bar chart
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)

    plot_df["neg_log_p"] = -np.log10(plot_df["p_value"].clip(lower=1e-10))
    plot_df["stars"] = plot_df["p_value"].apply(_sig_stars)
    plot_df["label"] = plot_df["feature"] + "  " + plot_df["stars"]
    plot_df["bar_color"] = plot_df["p_value"].apply(
        lambda p: _SIG_COLOR if p < 0.05 else _NS_COLOR
    )

    sig_threshold = -np.log10(0.05)

    # ── Figure / axes setup ──────────────────────────────────────────────────
    if ax is not None:
        fig = ax.get_figure()
        ax0 = ax
        two_panel = False
    else:
        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=figsize)
        two_panel = True

    y_pos = np.arange(len(plot_df))

    # ── Left panel: enrichment scores ────────────────────────────────────────
    ax0.barh(y_pos, plot_df["enrichment_score"],
             color=plot_df["bar_color"], alpha=0.85, height=0.65)
    ax0.axvline(0, color="black", lw=0.8)

    ax0.set_yticks(y_pos)
    ax0.set_yticklabels(plot_df["label"], fontsize=9)
    ax0.set_xlabel("Enrichment score\n(mean case − ctrl in motif window)")
    ax0.set_title("Enrichment scores")

    sig_patch = mpatches.Patch(color=_SIG_COLOR, alpha=0.85, label="p < 0.05")
    ns_patch = mpatches.Patch(color=_NS_COLOR, alpha=0.85, label="p ≥ 0.05")
    ax0.legend(handles=[sig_patch, ns_patch], loc="lower right", fontsize=8)

    # ── Right panel: −log10(p) ────────────────────────────────────────────────
    if two_panel:
        ax1.barh(y_pos, plot_df["neg_log_p"],
                 color=plot_df["bar_color"], alpha=0.85, height=0.65)
        ax1.axvline(sig_threshold, color="red", lw=1.2, ls="--",
                    label=f"p = 0.05  ($-\\log_{{10}}$ = {sig_threshold:.2f})")
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels([])
        ax1.set_xlabel("$-\\log_{10}$(p-value)")
        ax1.set_title("Statistical significance")
        ax1.legend(fontsize=8)

    fig.tight_layout()
    return fig


def plot_survival(
    df: pd.DataFrame,
    feature: str,
    motif_window: tuple,
    time_col: str = "time_to_event",
    event_col: str = "outcome",
    figsize: tuple = (7, 5),
) -> plt.Figure:
    """
    Kaplan-Meier survival curves stratified by trajectory motif strength.

    Subjects are split into "Motif+" (window score ≥ median) and "Motif−"
    (window score < median) groups. KM curves are estimated using a
    pure-numpy implementation — no external survival library required.

    Censored observations are marked with a "+" symbol at their censoring
    time and the current survival probability.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature,
        value, plus time_col and event_col. Typically from
        simulate_longitudinal(outcome_type="survival") or
        simulate_continuous(outcome_type="survival").
    feature : str
        Feature to use for motif-score stratification.
    motif_window : tuple of (int, int)
        (start, end) timepoints defining the motif window.
    time_col : str
        Column containing time-to-event values.
    event_col : str
        Column containing the event indicator (1 = event, 0 = censored).
    figsize : tuple
        (width, height) in inches.

    Returns
    -------
    matplotlib.figure.Figure
    """
    # Per-subject mean feature value in the motif window
    feat_df = df[df["feature"] == feature]
    window_df = feat_df[feat_df["timepoint"].between(motif_window[0], motif_window[1])]
    subj_scores = window_df.groupby("subject_id")["value"].mean().reset_index()
    subj_scores.columns = ["subject_id", "score"]

    # Join survival metadata (one row per subject)
    survival = (
        df[["subject_id", time_col, event_col]]
        .drop_duplicates("subject_id")
        .set_index("subject_id")
    )
    data = subj_scores.set_index("subject_id").join(survival).reset_index()

    threshold = float(data["score"].median())
    data["group"] = (data["score"] >= threshold).map({True: "Motif+", False: "Motif\u2212"})

    fig, ax = plt.subplots(figsize=figsize)
    palette = {"Motif+": _CASE_COLOR, "Motif\u2212": _CTRL_COLOR}

    for group_name, color in palette.items():
        grp = data[data["group"] == group_name]
        if len(grp) == 0:
            continue
        t_km, s_km = _km_estimator(grp[time_col].values, grp[event_col].values)
        ax.step(t_km, s_km, where="post", color=color, lw=2.0,
                label=f"{group_name}  (n={len(grp)})")

        # Censor tick marks
        for tc in grp.loc[grp[event_col] == 0, time_col].values:
            idx = max(0, int(np.searchsorted(t_km, tc, side="right")) - 1)
            ax.plot(tc, s_km[idx], "+", color=color, ms=9, mew=1.8, zorder=5)

    ax.set_xlabel(f"Time ({time_col})")
    ax.set_ylabel("Survival probability")
    ax.set_title(
        f"Kaplan-Meier by motif score — {feature}\n"
        f"Window {motif_window}  |  split at score median = {threshold:.3f}"
    )
    ax.set_ylim(0, 1.08)
    ax.legend()
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sig_stars(p: float) -> str:
    """Return significance stars for a p-value."""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _km_estimator(times, events):
    """
    Kaplan-Meier survival estimator (pure numpy, no external dependencies).

    Parameters
    ----------
    times : array-like of float
        Time-to-event or censoring times.
    events : array-like of int
        Event indicator: 1 = event occurred, 0 = censored.

    Returns
    -------
    t_out : np.ndarray
        Time points starting at 0.
    s_out : np.ndarray
        Survival probability at each time point; starts at 1.0.
    """
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)

    order = np.argsort(times, kind="stable")
    t_sorted = times[order]
    e_sorted = events[order]

    n_at_risk = len(t_sorted)
    s = 1.0
    t_out = [0.0]
    s_out = [1.0]

    i = 0
    while i < len(t_sorted):
        ti = t_sorted[i]
        d = 0      # events at this time
        n_tied = 0
        j = i
        while j < len(t_sorted) and t_sorted[j] == ti:
            if e_sorted[j] == 1:
                d += 1
            n_tied += 1
            j += 1

        if d > 0:
            s = s * (n_at_risk - d) / n_at_risk
            t_out.append(float(ti))
            s_out.append(float(s))

        n_at_risk -= n_tied
        i = j

    return np.array(t_out, dtype=float), np.array(s_out, dtype=float)

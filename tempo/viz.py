"""
tempo/viz.py

Visualization functions for TEMPO trajectory motifs and enrichment results.

Produces publication-ready plots for:
    - Individual and group trajectory overlays with motif window highlighting
    - Enrichment heatmaps across features and time windows
    - Matrix profile plots showing motif locations
    - Survival curves stratified by motif presence
"""

import pandas as pd
import numpy as np
from typing import Optional, Union


def plot_motifs(
    df: pd.DataFrame,
    features: Optional[list] = None,
    motif_window: Optional[tuple] = None,
    highlight_cases: bool = True,
    ax=None,
    figsize: tuple = (10, 4),
) -> object:
    """
    Plot trajectory overlays for selected features with optional motif highlighting.

    Draws one subplot per feature showing per-subject trajectories over time.
    Case and control subjects are colored differently. If a motif window is
    provided, it is highlighted with a shaded region.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format dataframe with columns: subject_id, timepoint, feature, value, outcome.
    features : list of str, optional
        Features to plot. If None, plots up to the first 4 features.
    motif_window : tuple of (int, int), optional
        (start, end) timepoint range to highlight as the motif region.
        If None, no highlighting is applied.
    highlight_cases : bool
        If True, draw case subjects in a distinct color from controls.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. If None, a new figure is created.
    figsize : tuple
        Figure size (width, height) in inches. Used only if ax is None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the trajectory plots.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Examples
    --------
    >>> from tempo import simulate, plot_motifs
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> fig = plot_motifs(df, features=["feature_000", "feature_001"], motif_window=(3, 6))
    >>> fig.savefig("motifs.png")
    """
    raise NotImplementedError(
        "plot_motifs() is not yet implemented. "
        "Planned: matplotlib/seaborn trajectory overlay with motif window shading."
    )


def plot_enrichment(
    results: pd.DataFrame,
    top_k: int = 20,
    colormap: str = "RdBu_r",
    ax=None,
    figsize: tuple = (8, 6),
) -> object:
    """
    Plot an enrichment heatmap of Harbinger analysis results.

    Visualizes enrichment scores across features and time windows,
    with p-value annotations and significance thresholding.

    Parameters
    ----------
    results : pd.DataFrame
        Output of harbinger(), with columns: feature, motif_window,
        enrichment_score, p_value.
    top_k : int
        Number of top features to display (by enrichment score).
    colormap : str
        Matplotlib colormap name for the heatmap. Default "RdBu_r" shows
        positive enrichment in red, negative in blue.
    ax : matplotlib.axes.Axes, optional
        Axes object to draw on. If None, a new figure is created.
    figsize : tuple
        Figure size (width, height) in inches. Used only if ax is None.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object containing the enrichment heatmap.

    Raises
    ------
    NotImplementedError
        This function is not yet implemented.

    Examples
    --------
    >>> from tempo import simulate, harbinger, plot_enrichment
    >>> df = simulate.simulate_longitudinal(seed=0)
    >>> results = harbinger(df, window_size=3)
    >>> fig = plot_enrichment(results, top_k=10)
    >>> fig.savefig("enrichment.png")
    """
    raise NotImplementedError(
        "plot_enrichment() is not yet implemented. "
        "Planned: seaborn heatmap with significance annotations."
    )

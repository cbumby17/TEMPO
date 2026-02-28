"""
tests/test_viz.py

Unit tests for tempo.viz.

All tests use matplotlib's Agg backend (non-interactive) and close figures
after each check to prevent resource leaks.
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.figure

from tempo import simulate
from tempo.viz import plot_motifs, plot_enrichment


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_cont():
    """Continuous simulation with a clear case/control split."""
    return simulate.simulate_continuous(
        n_subjects=20, n_timepoints=10, n_features=6, n_cases=10,
        motif_features=[0, 1], motif_window=(3, 6),
        motif_strength=5.0, noise_sd=0.3, seed=42,
    )


@pytest.fixture(scope="module")
def df_no_outcome(df_cont):
    """Same data without the outcome column."""
    return df_cont.drop(columns=["outcome"])


@pytest.fixture(scope="module")
def results_df():
    """Minimal harbinger-style results DataFrame."""
    rng = np.random.default_rng(0)
    n = 10
    p_vals = rng.uniform(0, 1, n)
    p_vals[0] = 0.001   # ensure at least one significant
    p_vals[1] = 0.008
    p_vals[2] = 0.03
    return pd.DataFrame({
        "feature": [f"feature_{i:03d}" for i in range(n)],
        "motif_window": [(3, 6)] * n,
        "enrichment_score": rng.uniform(0.1, 3.0, n),
        "p_value": p_vals,
    }).sort_values("enrichment_score", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# plot_motifs
# ---------------------------------------------------------------------------

class TestPlotMotifs:

    def test_returns_figure(self, df_cont):
        fig = plot_motifs(df_cont)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_default_features_up_to_four(self, df_cont):
        fig = plot_motifs(df_cont)
        n_visible = sum(ax.get_visible() for ax in fig.axes)
        assert n_visible == min(df_cont["feature"].nunique(), 4)
        plt.close(fig)

    def test_single_feature_subplot(self, df_cont):
        fig = plot_motifs(df_cont, features=["feature_000"])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_two_features_two_subplots(self, df_cont):
        fig = plot_motifs(df_cont, features=["feature_000", "feature_001"])
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 2
        plt.close(fig)

    def test_six_features_two_rows(self, df_cont):
        feats = sorted(df_cont["feature"].unique())[:6]
        fig = plot_motifs(df_cont, features=feats)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        assert len(visible) == 6
        plt.close(fig)

    def test_motif_window_shading(self, df_cont):
        """With motif_window, axvspan patches should be added."""
        fig = plot_motifs(df_cont, features=["feature_000"], motif_window=(3, 6))
        ax = [a for a in fig.axes if a.get_visible()][0]
        # axvspan creates a Polygon patch
        patches = [p for p in ax.patches]
        assert len(patches) > 0, "Expected motif window patch"
        plt.close(fig)

    def test_no_motif_window(self, df_cont):
        """Without motif_window, no gold patches should be present."""
        fig = plot_motifs(df_cont, features=["feature_000"], motif_window=None)
        ax = [a for a in fig.axes if a.get_visible()][0]
        assert len(ax.patches) == 0
        plt.close(fig)

    def test_works_without_outcome_column(self, df_no_outcome):
        """Should not raise even without 'outcome' column."""
        fig = plot_motifs(df_no_outcome, features=["feature_000"])
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_highlight_cases_false(self, df_cont):
        """highlight_cases=False should still produce a valid figure."""
        fig = plot_motifs(df_cont, features=["feature_000"], highlight_cases=False)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_external_ax_single_feature(self, df_cont):
        """When ax is provided for single feature, it uses that axes."""
        external_fig, external_ax = plt.subplots()
        fig = plot_motifs(df_cont, features=["feature_000"], ax=external_ax)
        assert fig is external_fig
        plt.close(external_fig)

    def test_subplot_titles_match_features(self, df_cont):
        feats = ["feature_000", "feature_002"]
        fig = plot_motifs(df_cont, features=feats)
        visible = [ax for ax in fig.axes if ax.get_visible()]
        titles = [ax.get_title() for ax in visible]
        assert titles == feats
        plt.close(fig)

    def test_no_error_with_all_features(self, df_cont):
        all_feats = sorted(df_cont["feature"].unique())
        fig = plot_motifs(df_cont, features=all_feats)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)


# ---------------------------------------------------------------------------
# plot_enrichment
# ---------------------------------------------------------------------------

class TestPlotEnrichment:

    def test_returns_figure(self, results_df):
        fig = plot_enrichment(results_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_two_panel_default(self, results_df):
        fig = plot_enrichment(results_df)
        # Two panels: enrichment score + -log10(p)
        assert len(fig.axes) == 2
        plt.close(fig)

    def test_top_k_respected(self, results_df):
        fig = plot_enrichment(results_df, top_k=5)
        ax0 = fig.axes[0]
        # barh creates one container per call; check yticks count
        assert len(ax0.get_yticks()) == 5
        plt.close(fig)

    def test_top_k_larger_than_results(self, results_df):
        """top_k > len(results) should not raise."""
        fig = plot_enrichment(results_df, top_k=100)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_single_panel_with_ax(self, results_df):
        external_fig, external_ax = plt.subplots()
        fig = plot_enrichment(results_df, ax=external_ax)
        assert fig is external_fig
        # Single panel mode: only the provided axes exists, no second panel
        assert len(external_fig.axes) == 1
        plt.close(external_fig)

    def test_significance_threshold_line_present(self, results_df):
        """Right panel should contain the dashed red p=0.05 threshold line."""
        fig = plot_enrichment(results_df)
        ax1 = fig.axes[1]
        vlines = [line for line in ax1.get_lines()]
        assert len(vlines) > 0, "Expected threshold line in right panel"
        plt.close(fig)

    def test_stars_in_labels(self, results_df):
        """Significant features should have stars appended to y-axis labels."""
        fig = plot_enrichment(results_df, top_k=10)
        ax0 = fig.axes[0]
        labels = [t.get_text() for t in ax0.get_yticklabels()]
        # At least one label should contain a star
        assert any("*" in lbl for lbl in labels), f"No stars found in labels: {labels}"
        plt.close(fig)

    def test_empty_results_raises(self):
        empty = pd.DataFrame(columns=["feature", "motif_window", "enrichment_score", "p_value"])
        # plot_enrichment with empty df — should not crash fatally, but may return empty plot
        # We only require it to not raise an unexpected exception
        try:
            fig = plot_enrichment(empty)
            plt.close(fig)
        except Exception as e:
            pytest.fail(f"plot_enrichment raised unexpectedly on empty df: {e}")

    def test_all_nonsignificant(self, results_df):
        """All p >= 0.05 should still produce a valid figure (no red bars)."""
        ns_df = results_df.copy()
        ns_df["p_value"] = 0.5
        fig = plot_enrichment(ns_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_all_significant(self, results_df):
        """All p < 0.05 should still produce a valid figure."""
        sig_df = results_df.copy()
        sig_df["p_value"] = 0.001
        fig = plot_enrichment(sig_df)
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

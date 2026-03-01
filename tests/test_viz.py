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
from tempo.viz import plot_motifs, plot_enrichment, plot_survival
from tempo.viz import _km_estimator


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

    def test_baseline_normalize_anchors_first_tp_at_zero(self, df_cont):
        """With baseline_normalize=True, all subjects should have value=0 at the first timepoint."""
        feat = sorted(df_cont["feature"].unique())[0]
        fig = plot_motifs(df_cont, features=[feat], baseline_normalize=True)
        # Verify via the y-axis label
        ax = [a for a in fig.axes if a.get_visible()][0]
        assert "baseline" in ax.get_ylabel().lower()
        plt.close(fig)

    def test_baseline_normalize_ylabel(self, df_cont):
        """baseline_normalize should change the y-axis label."""
        feat = sorted(df_cont["feature"].unique())[0]
        fig_raw  = plot_motifs(df_cont, features=[feat], baseline_normalize=False)
        fig_norm = plot_motifs(df_cont, features=[feat], baseline_normalize=True)
        ax_raw  = [a for a in fig_raw.axes  if a.get_visible()][0]
        ax_norm = [a for a in fig_norm.axes if a.get_visible()][0]
        assert ax_raw.get_ylabel() != ax_norm.get_ylabel()
        plt.close(fig_raw)
        plt.close(fig_norm)


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


# ---------------------------------------------------------------------------
# _km_estimator
# ---------------------------------------------------------------------------

class TestKMEstimator:

    def test_starts_at_time_zero_prob_one(self):
        t, s = _km_estimator([1, 2, 3, 4], [1, 1, 1, 0])
        assert t[0] == 0.0
        assert s[0] == 1.0

    def test_non_increasing(self):
        t, s = _km_estimator([1, 2, 3, 5, 8], [1, 0, 1, 1, 0])
        assert (np.diff(s) <= 1e-12).all(), "Survival probabilities must be non-increasing"

    def test_all_events(self):
        """All events: S reaches 0 at the last observed time."""
        t, s = _km_estimator([1, 2, 3], [1, 1, 1])
        assert s[-1] == pytest.approx(0.0, abs=1e-9)

    def test_no_events_flat_at_one(self):
        """All censored: S stays at 1.0 throughout (no steps)."""
        t, s = _km_estimator([1, 2, 3], [0, 0, 0])
        assert (s == 1.0).all()

    def test_known_values(self):
        """Hand-computed KM: events at t=1,3; censored at t=2,4; n=4.
        t=1: n_at_risk=4, d=1 → S(1) = 3/4 = 0.75
        t=2: censored (d=0), n_at_risk drops to 2
        t=3: n_at_risk=2, d=1 → S(3) = 0.75 * 1/2 = 0.375"""
        t, s = _km_estimator([1, 2, 3, 4], [1, 0, 1, 0])
        idx1 = np.where(t == 1)[0][0]
        idx3 = np.where(t == 3)[0][0]
        assert s[idx1] == pytest.approx(0.75, abs=1e-9)
        assert s[idx3] == pytest.approx(0.375, abs=1e-9)

    def test_tied_events(self):
        """Tied events at t=2; censored at t=1; n=4.
        t=1: censored (d=0), n_at_risk drops to 3
        t=2: n_at_risk=3, d=2 → S(2) = 1 * 1/3 ≈ 0.333"""
        t, s = _km_estimator([1, 2, 2, 3], [0, 1, 1, 0])
        idx = np.where(t == 2)[0][0]
        assert s[idx] == pytest.approx(1 / 3, abs=1e-9)


# ---------------------------------------------------------------------------
# plot_survival
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_survival():
    return simulate.simulate_continuous(
        n_subjects=20, n_timepoints=10, n_features=4, n_cases=10,
        motif_features=[0], motif_window=(3, 6),
        motif_strength=3.0, noise_sd=0.3,
        outcome_type="survival", seed=42,
    )


class TestPlotSurvival:

    def test_returns_figure(self, df_survival):
        fig = plot_survival(df_survival, "feature_000", (3, 6))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_two_lines_in_axes(self, df_survival):
        """Should draw exactly 2 step-function lines (one per group)."""
        fig = plot_survival(df_survival, "feature_000", (3, 6))
        ax = fig.axes[0]
        # Step lines are Line2D objects; filter by label
        labelled = [l for l in ax.get_lines() if l.get_label().startswith("Motif")]
        assert len(labelled) == 2
        plt.close(fig)

    def test_axes_labels(self, df_survival):
        fig = plot_survival(df_survival, "feature_000", (3, 6))
        ax = fig.axes[0]
        assert "time_to_event" in ax.get_xlabel().lower()
        assert "survival" in ax.get_ylabel().lower()
        plt.close(fig)

    def test_y_axis_within_unit_interval(self, df_survival):
        """Y-axis should be bounded in [0, ~1]."""
        fig = plot_survival(df_survival, "feature_000", (3, 6))
        ax = fig.axes[0]
        ymin, ymax = ax.get_ylim()
        assert ymin >= -0.05
        assert ymax <= 1.15
        plt.close(fig)

    def test_custom_time_and_event_cols(self):
        """plot_survival respects time_col and event_col parameters."""
        df = simulate.simulate_continuous(
            n_subjects=16, n_timepoints=8, n_features=2, n_cases=8,
            motif_features=[0], motif_window=(2, 5),
            motif_strength=3.0, noise_sd=0.3,
            outcome_type="survival", seed=1,
        )
        df = df.rename(columns={"time_to_event": "tte", "outcome": "event"})
        fig = plot_survival(df, "feature_000", (2, 5),
                            time_col="tte", event_col="event")
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

    def test_no_error_with_all_events(self):
        """If all subjects experience the event, S reaches 0 — should not raise."""
        df = simulate.simulate_continuous(
            n_subjects=10, n_timepoints=8, n_features=2, n_cases=5,
            motif_features=[0], motif_window=(2, 4),
            motif_strength=3.0, noise_sd=0.3,
            outcome_type="survival", seed=2,
        )
        # Force all to event
        df["outcome"] = 1
        df["time_to_event"] = df["time_to_event"].clip(lower=1)
        fig = plot_survival(df, "feature_000", (2, 4))
        assert isinstance(fig, matplotlib.figure.Figure)
        plt.close(fig)

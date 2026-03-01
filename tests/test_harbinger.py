"""
tests/test_harbinger.py

Unit tests for tempo.harbinger.

Uses continuous simulation (no compositional constraint) so the enrichment
signal is clean and deterministic. A small n_permutations is used throughout
to keep tests fast.
"""

import pytest
import numpy as np
import pandas as pd
from tempo import simulate
from tempo.harbinger import harbinger, compute_matrix_profile


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def df_strong():
    """Strong continuous motif — motif features should rank clearly at top."""
    return simulate.simulate_continuous(
        n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
        motif_features=[0, 1], motif_window=(3, 6),
        motif_type="step", motif_strength=8.0, noise_sd=0.3,
        seed=42,
    )


@pytest.fixture
def df_small():
    """Small dataset for fast schema / structural tests."""
    return simulate.simulate_continuous(
        n_subjects=10, n_timepoints=8, n_features=4, n_cases=4,
        motif_features=[0], motif_window=(2, 5),
        motif_type="step", motif_strength=5.0, noise_sd=0.3,
        seed=0,
    )


# ---------------------------------------------------------------------------
# harbinger — output schema
# ---------------------------------------------------------------------------

class TestHarbingerSchema:

    def test_returns_dataframe(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert set(result.columns) == {
            "feature", "window_size", "motif_window", "enrichment_score",
            "p_value", "matrix_profile_min"
        }

    def test_top_k_respected(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=2, n_permutations=50, seed=0)
        assert len(result) <= 2

    def test_sorted_by_enrichment_descending(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        scores = result["enrichment_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_p_values_in_unit_interval(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_motif_window_is_tuple(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        for w in result["motif_window"]:
            assert isinstance(w, tuple)
            assert len(w) == 2
            assert w[0] <= w[1]

    def test_window_size_equals_window_span(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        for _, row in result.iterrows():
            start, end = row["motif_window"]
            assert end - start + 1 == row["window_size"]

    def test_matrix_profile_min_positive(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert (result["matrix_profile_min"] >= 0).all()

    def test_invalid_window_size_raises(self, df_small):
        n_tp = df_small["timepoint"].nunique()
        with pytest.raises(ValueError, match="window_size"):
            harbinger(df_small, window_size=n_tp, n_permutations=10)


# ---------------------------------------------------------------------------
# harbinger — signal recovery
# ---------------------------------------------------------------------------

class TestHarbingerSignal:

    def test_motif_features_rank_top(self, df_strong):
        """With a strong motif, known motif features should appear in the top 2."""
        result = harbinger(df_strong, window_size=4, top_k=6, n_permutations=200, seed=0)
        top2 = set(result.head(2)["feature"].tolist())
        assert "feature_000" in top2
        assert "feature_001" in top2

    def test_motif_features_have_significant_p_values(self, df_strong):
        result = harbinger(df_strong, window_size=4, top_k=6, n_permutations=200, seed=0)
        motif_rows = result[result["feature"].isin(["feature_000", "feature_001"])]
        assert (motif_rows["p_value"] < 0.05).all()

    def test_noise_features_have_high_p_values(self, df_strong):
        result = harbinger(df_strong, window_size=4, top_k=6, n_permutations=200, seed=0)
        noise_rows = result[~result["feature"].isin(["feature_000", "feature_001"])]
        # Noise features should not ALL be significant
        assert (noise_rows["p_value"] > 0.05).any()

    def test_motif_features_have_positive_enrichment(self, df_strong):
        result = harbinger(df_strong, window_size=4, top_k=6, n_permutations=200, seed=0)
        motif_rows = result[result["feature"].isin(["feature_000", "feature_001"])]
        assert (motif_rows["enrichment_score"] > 0).all()

    def test_reproducible_with_same_seed(self, df_strong):
        r1 = harbinger(df_strong, window_size=4, top_k=6, n_permutations=100, seed=7)
        r2 = harbinger(df_strong, window_size=4, top_k=6, n_permutations=100, seed=7)
        pd.testing.assert_frame_equal(r1, r2)

    def test_different_seeds_may_differ_in_p_values(self, df_strong):
        r1 = harbinger(df_strong, window_size=4, top_k=6, n_permutations=50, seed=1)
        r2 = harbinger(df_strong, window_size=4, top_k=6, n_permutations=50, seed=2)
        # p-values come from random permutations — very unlikely to be identical
        assert not r1["p_value"].equals(r2["p_value"])

    def test_evaluation_report_recovers_motif(self, df_strong):
        """Top features + their window should give recall > 0 against ground truth."""
        from tempo import simulate
        result = harbinger(df_strong, window_size=4, top_k=6, n_permutations=200, seed=0)
        top2_features = result.head(2)["feature"].tolist()
        top_window = result.iloc[0]["motif_window"]
        report = simulate.evaluation_report(top2_features, top_window, df_strong)
        assert report["feature_recall"] == 1.0
        assert report["window_jaccard"] > 0.0


# ---------------------------------------------------------------------------
# compute_matrix_profile
# ---------------------------------------------------------------------------

class TestComputeMatrixProfile:

    def test_1d_self_join_shape(self):
        ts = np.random.default_rng(0).standard_normal(20)
        mp = compute_matrix_profile(ts, window_size=4)
        assert mp.shape == (20 - 4 + 1,)

    def test_1d_values_finite(self):
        ts = np.random.default_rng(0).standard_normal(20)
        mp = compute_matrix_profile(ts, window_size=4)
        assert np.all(np.isfinite(mp))

    def test_1d_values_non_negative(self):
        ts = np.random.default_rng(0).standard_normal(20)
        mp = compute_matrix_profile(ts, window_size=4)
        assert np.all(mp >= 0)

    def test_1d_ab_join_shape(self):
        rng = np.random.default_rng(0)
        ts_a = rng.standard_normal(20)
        ts_b = rng.standard_normal(20)
        mp = compute_matrix_profile(ts_a, window_size=4, cross_series=ts_b)
        assert mp.shape == (20 - 4 + 1,)

    def test_2d_pan_profile_shape(self):
        T = np.random.default_rng(0).standard_normal((5, 20))  # 5 subjects, 20 timepoints
        mp = compute_matrix_profile(T, window_size=4)
        assert mp.shape == (20 - 4 + 1,)

    def test_2d_values_finite(self):
        T = np.random.default_rng(0).standard_normal((4, 15))
        mp = compute_matrix_profile(T, window_size=3)
        assert np.all(np.isfinite(mp))

    def test_invalid_ndim_raises(self):
        T = np.zeros((2, 3, 4))
        with pytest.raises(ValueError, match="1D or 2D"):
            compute_matrix_profile(T, window_size=2)

    def test_accepts_list_input(self):
        ts = list(np.random.default_rng(0).standard_normal(15))
        mp = compute_matrix_profile(ts, window_size=3)
        assert mp.shape == (15 - 3 + 1,)


# ---------------------------------------------------------------------------
# Multi-window scanning
# ---------------------------------------------------------------------------

class TestMultiWindowScanning:

    @pytest.fixture
    def df_multi(self):
        """Small dataset used for multi-window tests."""
        return simulate.simulate_continuous(
            n_subjects=10, n_timepoints=8, n_features=4, n_cases=4,
            motif_features=[0], motif_window=(2, 5),
            motif_type="step", motif_strength=5.0, noise_sd=0.3,
            seed=0,
        )

    @pytest.fixture
    def df_strong(self):
        """Strong signal dataset for ranking tests."""
        return simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
            motif_features=[0, 1], motif_window=(3, 6),
            motif_type="step", motif_strength=8.0, noise_sd=0.3,
            seed=42,
        )

    def test_window_sizes_list_accepted(self, df_multi):
        result = harbinger(df_multi, window_sizes=[3, 5], top_k=4, n_permutations=50, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_window_sizes_column_present(self, df_multi):
        result = harbinger(df_multi, window_sizes=[3, 5], top_k=4, n_permutations=50, seed=0)
        assert "window_size" in result.columns
        assert set(result["window_size"].unique()).issubset({3, 5})

    def test_window_size_range_accepted(self, df_multi):
        result = harbinger(df_multi, window_size_range=(3, 5), top_k=4, n_permutations=50, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_window_size_range_column_values(self, df_multi):
        result = harbinger(df_multi, window_size_range=(3, 5), top_k=4, n_permutations=50, seed=0)
        assert set(result["window_size"].unique()).issubset({3, 4, 5})

    def test_backward_compat_single_size(self, df_multi):
        result = harbinger(df_multi, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert (result["window_size"] == 3).all()

    def test_default_no_param(self, df_multi):
        result = harbinger(df_multi, top_k=4, n_permutations=50, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_multiple_params_raises(self, df_multi):
        with pytest.raises(ValueError, match="at most one"):
            harbinger(df_multi, window_size=3, window_sizes=[3, 5], n_permutations=10)

    def test_motif_feature_ranks_top_with_sizes(self, df_strong):
        result = harbinger(df_strong, window_sizes=[3, 4, 5], top_k=6, n_permutations=200, seed=0)
        top2 = set(result.head(2)["feature"].tolist())
        assert "feature_000" in top2
        assert "feature_001" in top2

    def test_window_tps_match_reported_size(self, df_multi):
        result = harbinger(df_multi, window_sizes=[3, 5], top_k=4, n_permutations=50, seed=0)
        for _, row in result.iterrows():
            start, end = row["motif_window"]
            assert end - start + 1 == row["window_size"]

    def test_window_size_range_finds_better_than_single(self, df_strong):
        """Scanning a range that includes the true window should recover a window
        closer to the true window than a fixed size that misses it.
        True motif is (3, 6) → size 4.  Single size 3 is too short;
        range (3, 5) includes the optimal size 4.
        Window selection uses sum-based criterion (mean * ws) to prefer
        sustained enrichment over short-window peaks — so we test window
        recovery (Jaccard) rather than raw mean enrichment score."""
        from tempo import simulate

        result_single = harbinger(
            df_strong, window_size=3, top_k=6, n_permutations=50, seed=0
        )
        result_range = harbinger(
            df_strong, window_size_range=(3, 5), top_k=6, n_permutations=50, seed=0
        )
        motif_single = result_single[result_single["feature"] == "feature_000"]
        motif_range = result_range[result_range["feature"] == "feature_000"]
        if len(motif_single) > 0 and len(motif_range) > 0:
            jaccard_single = simulate.evaluation_report(
                ["feature_000"], motif_single.iloc[0]["motif_window"], df_strong
            )["window_jaccard"]
            jaccard_range = simulate.evaluation_report(
                ["feature_000"], motif_range.iloc[0]["motif_window"], df_strong
            )["window_jaccard"]
            # Range scan (includes true size 4) should recover the window at
            # least as well as the too-short single size 3.
            assert jaccard_range >= jaccard_single

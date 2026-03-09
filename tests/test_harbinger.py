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

class TestHarbingerMissingTimepoints:
    """harbinger() must handle subjects with sporadic missing timepoints."""

    def test_missing_timepoint_does_not_crash(self):
        """Drop one row from one subject → harbinger should still produce results."""
        df = simulate.simulate_continuous(
            n_subjects=10, n_timepoints=8, n_features=3, n_cases=4,
            motif_features=[0], motif_window=(2, 5),
            motif_strength=5.0, noise_sd=0.3, seed=1,
        )
        # Remove timepoint 3 for a single subject
        mask = ~((df["subject_id"] == df["subject_id"].unique()[0]) & (df["timepoint"] == 3))
        df_missing = df[mask].copy()
        result = harbinger(df_missing, window_size=3, top_k=3, n_permutations=20, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_missing_timepoint_result_similar_to_complete(self):
        """Top feature should still be recovered even with one missing observation."""
        df = simulate.simulate_continuous(
            n_subjects=20, n_timepoints=10, n_features=4, n_cases=8,
            motif_features=[0], motif_window=(3, 7),
            motif_strength=6.0, noise_sd=0.3, seed=2,
        )
        # Remove one timepoint from one control subject
        mask = ~((df["subject_id"] == df["subject_id"].unique()[-1]) & (df["timepoint"] == 5))
        df_missing = df[mask].copy()
        result = harbinger(df_missing, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert "feature_000" in result["feature"].values


class TestHarbingerSchema:

    def test_returns_dataframe(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert isinstance(result, pd.DataFrame)

    def test_expected_columns(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert set(result.columns) == {
            "feature", "window_size", "motif_window", "enrichment_score",
            "p_value", "q_value", "matrix_profile_min"
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

    def test_q_values_in_unit_interval(self, df_small):
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert (result["q_value"] >= 0).all()
        assert (result["q_value"] <= 1).all()

    def test_q_values_geq_p_values(self, df_small):
        """BH q-values are always >= the corresponding raw p-values."""
        result = harbinger(df_small, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert (result["q_value"] >= result["p_value"] - 1e-9).all()

    def test_bh_correct_known_values(self):
        """Verify _bh_correct against a hand-computed BH example.

        p = [0.01, 0.04, 0.20, 0.06]  (n=4)
        Sorted: [0.01, 0.04, 0.06, 0.20] at ranks 1,2,3,4
        BH raw: [0.04, 0.08, 0.08, 0.20]  (p * n / rank)
        After monotonicity enforcement: [0.04, 0.08, 0.08, 0.20]
        Mapped back: q[0]=0.04, q[1]=0.08, q[2]=0.20, q[3]=0.08
        """
        from tempo.harbinger import _bh_correct
        p = np.array([0.01, 0.04, 0.20, 0.06])
        q = _bh_correct(p)
        assert q[0] == pytest.approx(0.04, abs=1e-9)
        assert q[1] == pytest.approx(0.08, abs=1e-9)
        assert q[2] == pytest.approx(0.20, abs=1e-9)
        assert q[3] == pytest.approx(0.08, abs=1e-9)
        assert (q <= 1).all()

    def test_bh_correct_monotone(self):
        """q-values sorted by p-value must be non-decreasing."""
        from tempo.harbinger import _bh_correct
        rng = np.random.default_rng(0)
        p = rng.uniform(0, 1, 20)
        q = _bh_correct(p)
        order = np.argsort(p)
        q_sorted = q[order]
        assert (np.diff(q_sorted) >= -1e-12).all()


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
        """Scanning a range that includes the true window should give enrichment
        >= a single fixed size that misses it. True motif is (3, 6) → size 4.
        Single size 3 is one shorter; range (3, 5) includes the optimal size 4."""
        result_single = harbinger(
            df_strong, window_size=3, top_k=6, n_permutations=50, seed=0
        )
        result_range = harbinger(
            df_strong, window_size_range=(3, 5), top_k=6, n_permutations=50, seed=0
        )
        motif_single = result_single[result_single["feature"] == "feature_000"]
        motif_range = result_range[result_range["feature"] == "feature_000"]
        if len(motif_single) > 0 and len(motif_range) > 0:
            assert (
                motif_range.iloc[0]["enrichment_score"]
                >= motif_single.iloc[0]["enrichment_score"]
            )

    def test_multi_window_p_values_not_below_single(self, df_strong):
        """Max-over-candidates correction: scanning multiple windows should not
        produce systematically *lower* p-values than a fixed single window for
        noise features.  The correction makes multi-window p-values at least as
        conservative as single-window ones (on average across noise features)."""
        result_single = harbinger(
            df_strong, window_size=4, top_k=6, n_permutations=200, seed=0
        )
        result_multi = harbinger(
            df_strong, window_sizes=[3, 4, 5], top_k=6, n_permutations=200, seed=0
        )
        noise_feats = [f for f in result_single["feature"]
                       if f not in ("feature_000", "feature_001")]
        p_single = result_single[result_single["feature"].isin(noise_feats)]["p_value"]
        p_multi = result_multi[result_multi["feature"].isin(noise_feats)]["p_value"]
        if len(p_single) > 0 and len(p_multi) > 0:
            # Multi-window mean p-value should be >= single-window mean p-value,
            # i.e. the correction should not deflate p-values further.
            assert p_multi.mean() >= p_single.mean() - 0.15  # generous tolerance

    def test_single_window_and_multi_same_result_when_one_size(self, df_multi):
        """A window_sizes list with one element should behave identically to
        window_size (single candidate → standard permutation, no change)."""
        r1 = harbinger(df_multi, window_size=3, top_k=4, n_permutations=100, seed=5)
        r2 = harbinger(df_multi, window_sizes=[3], top_k=4, n_permutations=100, seed=5)
        pd.testing.assert_frame_equal(r1, r2)


# ---------------------------------------------------------------------------
# Template-correlation enrichment
# ---------------------------------------------------------------------------

class TestTemplateCorrelationEnrichment:

    @pytest.fixture
    def df_step(self):
        """Strong step motif — both methods should recover it."""
        return simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
            motif_features=[0, 1], motif_window=(3, 6),
            motif_type="step", motif_strength=8.0, noise_sd=0.3,
            seed=42,
        )

    @pytest.fixture
    def df_oscillating(self):
        """Oscillating motif — mean_difference may fail, template_correlation should succeed."""
        return simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
            motif_features=[0, 1], motif_window=(3, 6),
            motif_type="oscillating", motif_strength=8.0, noise_sd=0.3,
            seed=42,
        )

    @pytest.fixture
    def df_small(self):
        return simulate.simulate_continuous(
            n_subjects=10, n_timepoints=8, n_features=4, n_cases=4,
            motif_features=[0], motif_window=(2, 5),
            motif_type="step", motif_strength=5.0, noise_sd=0.3,
            seed=0,
        )

    def test_same_output_schema_as_mean_difference(self, df_small):
        """template_correlation returns a DataFrame with the same columns."""
        result = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=50, seed=0,
            enrichment_method="template_correlation",
        )
        assert isinstance(result, pd.DataFrame)
        assert set(result.columns) == {
            "feature", "window_size", "motif_window", "enrichment_score",
            "p_value", "q_value", "matrix_profile_min",
        }

    def test_motif_features_rank_top_on_step_data(self, df_step):
        """On strong step data both methods should put motif features in top 2."""
        result = harbinger(
            df_step, window_size=4, top_k=6, n_permutations=200, seed=0,
            enrichment_method="template_correlation",
        )
        top2 = set(result.head(2)["feature"].tolist())
        assert "feature_000" in top2
        assert "feature_001" in top2

    def test_p_values_significant_for_step_motif(self, df_step):
        """Motif features should have p < 0.05 on strong step data."""
        result = harbinger(
            df_step, window_size=4, top_k=6, n_permutations=200, seed=0,
            enrichment_method="template_correlation",
        )
        motif_rows = result[result["feature"].isin(["feature_000", "feature_001"])]
        assert (motif_rows["p_value"] < 0.05).all()

    def test_enrichment_score_range(self, df_small):
        """Template-correlation enrichment score must lie in [−2, 2]."""
        result = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=50, seed=0,
            enrichment_method="template_correlation",
        )
        assert (result["enrichment_score"] >= -2.0).all()
        assert (result["enrichment_score"] <= 2.0).all()

    def test_default_method_matches_mean_difference(self, df_small):
        """Calling harbinger() without enrichment_method should match explicit 'mean_difference'."""
        r_default = harbinger(df_small, window_size=3, top_k=4, n_permutations=100, seed=7)
        r_explicit = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=100, seed=7,
            enrichment_method="mean_difference",
        )
        pd.testing.assert_frame_equal(r_default, r_explicit)

    def test_invalid_method_raises(self, df_small):
        """An unknown enrichment_method should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown enrichment_method"):
            harbinger(
                df_small, window_size=3, n_permutations=10,
                enrichment_method="bad_method",
            )

    def test_reproducibility_with_same_seed(self, df_small):
        """Same seed → identical results."""
        r1 = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=50, seed=13,
            enrichment_method="template_correlation",
        )
        r2 = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=50, seed=13,
            enrichment_method="template_correlation",
        )
        pd.testing.assert_frame_equal(r1, r2)

    def test_works_with_window_sizes_list(self, df_small):
        """template_correlation should work via the multi-candidate code path."""
        result = harbinger(
            df_small, window_sizes=[3, 5], top_k=4, n_permutations=50, seed=0,
            enrichment_method="template_correlation",
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_p_values_in_unit_interval(self, df_small):
        result = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=50, seed=0,
            enrichment_method="template_correlation",
        )
        assert (result["p_value"] >= 0).all()
        assert (result["p_value"] <= 1).all()

    def test_sorted_by_enrichment_descending(self, df_small):
        result = harbinger(
            df_small, window_size=3, top_k=4, n_permutations=50, seed=0,
            enrichment_method="template_correlation",
        )
        scores = result["enrichment_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_near_zero_template_variance_returns_zero(self):
        """If all case values in window are identical, score should be 0.0 without crash."""
        from tempo.harbinger import _window_enrichment_template_corr
        import pandas as pd
        # Build a wide DataFrame where all cases have the same constant trajectory
        wide = pd.DataFrame({
            0: [1.0, 1.0, 1.0, 0.5, 0.6],  # cases: same value
            1: [1.0, 1.0, 1.0, 0.4, 0.7],
            2: [1.0, 1.0, 1.0, 0.3, 0.8],
        }, index=["case_0", "case_1", "case_2", "ctrl_0", "ctrl_1"])
        score = _window_enrichment_template_corr(
            wide, ["case_0", "case_1", "case_2"], ["ctrl_0", "ctrl_1"], [0, 1, 2]
        )
        assert score == 0.0

    def test_loo_template_excludes_self(self):
        """Each case is scored against the mean of all OTHER cases (LOO), not itself.

        With 2 cases whose trajectories are perfectly negatively correlated,
        the LOO template for each case equals the other case's trajectory.
        Each case will therefore have correlation −1 with its LOO template,
        giving a negative mean case correlation.  Without LOO, each case
        would correlate with the flat mean (all zeros), producing 0.0 instead.
        """
        from tempo.harbinger import _window_enrichment_template_corr
        import pandas as pd
        # case_0: [1, -1, 1], case_1: [-1, 1, -1] — perfectly anti-correlated
        wide = pd.DataFrame({
            0: [ 1.0, -1.0, 0.0],
            1: [-1.0,  1.0, 0.0],
            2: [ 1.0, -1.0, 0.0],
        }, index=["case_0", "case_1", "ctrl_0"])
        score = _window_enrichment_template_corr(
            wide, ["case_0", "case_1"], ["ctrl_0"], [0, 1, 2]
        )
        # LOO: case_0 template = case_1 = [-1,1,-1] → r(case_0, template) = -1
        #      case_1 template = case_0 = [1,-1,1]  → r(case_1, template) = -1
        # mean case_corrs = -1; ctrl scores against full_template = [0,0,0] → 0
        # score = -1 - 0 = -1
        assert score == pytest.approx(-1.0, abs=1e-6)

    def test_null_calibration_template_corr(self):
        """Under H0 (no motif), template_correlation p-values should be approximately
        uniform.  With LOO scoring the null distribution is properly calibrated.

        Uses a small number of reps to keep the test fast; checks that the
        fraction of p-values below 0.1 does not exceed 0.30 (3× the nominal rate).
        A properly calibrated test should give ~10% below 0.1.
        """
        from tempo import simulate
        from tempo.harbinger import harbinger

        rng = np.random.default_rng(0)
        p_values = []
        for rep in range(40):
            df = simulate.simulate_continuous(
                n_subjects=20, n_timepoints=10, n_features=2, n_cases=8,
                motif_features=[0], motif_window=(3, 6),
                motif_type="step", motif_strength=0.0,  # H0: no signal
                noise_sd=0.5, seed=rep,
            )
            result = harbinger(
                df, window_size=4, top_k=2, n_permutations=200,
                seed=rep, enrichment_method="template_correlation",
            )
            feat_row = result[result["feature"] == "feature_000"]
            p = feat_row.iloc[0]["p_value"] if len(feat_row) > 0 else 1.0
            p_values.append(p)

        p_arr = np.array(p_values)
        frac_below_01 = float((p_arr < 0.1).mean())
        # Calibrated null: ~10% below 0.1. Allow up to 30% to avoid flakiness.
        assert frac_below_01 <= 0.30, (
            f"template_correlation appears miscalibrated: "
            f"{frac_below_01:.0%} of H0 p-values < 0.1 (expected ~10%)"
        )


# ---------------------------------------------------------------------------
# direction parameter
# ---------------------------------------------------------------------------

class TestDirectionParameter:

    @pytest.fixture
    def df_up(self):
        """Cases elevated above controls — strong upward motif."""
        return simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
            motif_features=[0], motif_window=(3, 6),
            motif_type="step", motif_strength=8.0, noise_sd=0.3,
            seed=42,
        )

    @pytest.fixture
    def df_down(self):
        """Cases depressed below controls — achieved by negating motif_strength."""
        df = simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
            motif_features=[0], motif_window=(3, 6),
            motif_type="step", motif_strength=8.0, noise_sd=0.3,
            seed=42,
        )
        # Flip values for the motif feature in the motif window so cases go DOWN
        attrs = df.attrs.copy()
        mw = attrs["motif_window"]
        mask = (
            (df["feature"] == "feature_000")
            & df["timepoint"].between(mw[0], mw[1])
            & (df["outcome"] == 1)
        )
        df.loc[mask, "value"] = -df.loc[mask, "value"]
        df.attrs = attrs
        return df

    def test_invalid_direction_raises(self, df_up):
        with pytest.raises(ValueError, match="direction"):
            harbinger(df_up, window_size=4, direction="sideways", n_permutations=10)

    def test_up_is_default(self, df_up):
        r_default = harbinger(df_up, window_size=4, top_k=6, n_permutations=100, seed=0)
        r_up = harbinger(df_up, window_size=4, top_k=6, n_permutations=100, seed=0,
                         direction="up")
        pd.testing.assert_frame_equal(r_default, r_up)

    def test_up_scores_positive_for_upward_motif(self, df_up):
        result = harbinger(df_up, window_size=4, top_k=6, n_permutations=200, seed=0,
                           direction="up")
        motif_row = result[result["feature"] == "feature_000"]
        assert len(motif_row) > 0
        assert motif_row.iloc[0]["enrichment_score"] > 0

    def test_down_finds_depressed_motif(self, df_down):
        """direction='down' should surface the feature where cases are below controls."""
        result = harbinger(df_down, window_size=4, top_k=6, n_permutations=200, seed=0,
                           direction="down")
        motif_row = result[result["feature"] == "feature_000"]
        assert len(motif_row) > 0
        assert motif_row.iloc[0]["enrichment_score"] < 0

    def test_down_motif_significant(self, df_down):
        result = harbinger(df_down, window_size=4, top_k=6, n_permutations=200, seed=0,
                           direction="down")
        motif_row = result[result["feature"] == "feature_000"]
        assert motif_row.iloc[0]["p_value"] < 0.05

    def test_down_sorted_ascending(self, df_down):
        """direction='down' results should be sorted by enrichment_score ascending."""
        result = harbinger(df_down, window_size=4, top_k=6, n_permutations=100, seed=0,
                           direction="down")
        scores = result["enrichment_score"].tolist()
        assert scores == sorted(scores)

    def test_both_has_direction_column(self, df_up):
        result = harbinger(df_up, window_size=4, top_k=6, n_permutations=100, seed=0,
                           direction="both")
        assert "direction" in result.columns
        assert set(result["direction"].unique()).issubset({"up", "down"})

    def test_both_sorted_by_abs_score(self, df_up):
        result = harbinger(df_up, window_size=4, top_k=6, n_permutations=100, seed=0,
                           direction="both")
        abs_scores = result["enrichment_score"].abs().tolist()
        assert abs_scores == sorted(abs_scores, reverse=True)

    def test_both_no_direction_column_for_up(self, df_up):
        """direction='up' should NOT add a direction column."""
        result = harbinger(df_up, window_size=4, top_k=6, n_permutations=100, seed=0,
                           direction="up")
        assert "direction" not in result.columns

    def test_up_misses_downward_motif(self, df_down):
        """direction='up' should rank the downward motif feature near the bottom."""
        result = harbinger(df_down, window_size=4, top_k=6, n_permutations=100, seed=0,
                           direction="up")
        if len(result) > 0:
            top_feature = result.iloc[0]["feature"]
            assert top_feature != "feature_000"

    def test_empty_dataframe_has_direction_col_for_both(self):
        """Empty result with direction='both' should include the direction column."""
        df = simulate.simulate_continuous(
            n_subjects=4, n_timepoints=4, n_features=2, n_cases=1,
            motif_features=[0], motif_window=(1, 2),
            motif_strength=1.0, noise_sd=0.1, seed=0,
        )
        result = harbinger(df, window_size=3, n_permutations=10, direction="both")
        assert "direction" in result.columns


# ---------------------------------------------------------------------------
# Window recovery with wide scan ranges (issue #3)
# ---------------------------------------------------------------------------

class TestWindowRecovery:
    """Regression tests for window recovery degrading with wide scan ranges.

    The pan-matrix profile uses z-normalised Euclidean distance, which can
    place the argmin one position off from the true motif onset.  Evaluating
    argmin ± 1 lets the enrichment score resolve the ambiguity without
    requiring the user to know the exact window size in advance.
    """

    @pytest.fixture
    def df_step(self):
        return simulate.simulate_continuous(
            n_subjects=40, n_timepoints=12, n_features=4, n_cases=20,
            motif_features=[0], motif_window=(3, 7),
            motif_type='step', motif_strength=3.0, noise_sd=0.4,
            seed=42,
        )

    def _jaccard(self, detected, true_window):
        true_set = set(range(true_window[0], true_window[1] + 1))
        det_set  = set(range(detected[0],    detected[1]    + 1))
        return len(true_set & det_set) / len(true_set | det_set)

    def test_wide_scan_recovers_true_window(self, df_step):
        """Scan range (3,8) — wider than the true window — should still find (3,7)."""
        true_window = df_step.attrs['motif_window']
        result = harbinger(df_step, window_size_range=(3, 8), top_k=1,
                           n_permutations=199, seed=0)
        assert len(result) > 0
        detected = result.iloc[0]['motif_window']
        jaccard = self._jaccard(detected, true_window)
        assert jaccard >= 0.8, (
            f"Wide scan gave poor window recovery: detected={detected}, "
            f"true={true_window}, Jaccard={jaccard:.3f}"
        )

    def test_very_wide_scan_recovers_true_window(self, df_step):
        """Scan range (3,10) — much wider than the true window — should still find (3,7)."""
        true_window = df_step.attrs['motif_window']
        result = harbinger(df_step, window_size_range=(3, 10), top_k=1,
                           n_permutations=199, seed=0)
        assert len(result) > 0
        detected = result.iloc[0]['motif_window']
        jaccard = self._jaccard(detected, true_window)
        assert jaccard >= 0.8, (
            f"Very wide scan gave poor window recovery: detected={detected}, "
            f"true={true_window}, Jaccard={jaccard:.3f}"
        )

    def test_exact_size_scan_recovers_true_window(self, df_step):
        """Scan with the exact true window size should give Jaccard=1.0."""
        true_window = df_step.attrs['motif_window']
        true_ws = true_window[1] - true_window[0] + 1  # 5
        result = harbinger(df_step, window_size=true_ws, top_k=1,
                           n_permutations=199, seed=0)
        assert len(result) > 0
        detected = result.iloc[0]['motif_window']
        jaccard = self._jaccard(detected, true_window)
        assert jaccard == pytest.approx(1.0), (
            f"Exact-size scan missed true window: detected={detected}, "
            f"true={true_window}"
        )


# ---------------------------------------------------------------------------
# Stratified permutation testing (issue #13)
# ---------------------------------------------------------------------------

class TestStratifiedPermutation:
    """Tests for covariate_cols stratified permutation in harbinger()."""

    @pytest.fixture
    def df_with_sex(self):
        """Dataset with a 'sex' covariate balanced across cases and controls."""
        df = simulate.simulate_continuous(
            n_subjects=40, n_timepoints=10, n_features=4, n_cases=20,
            motif_features=[0], motif_window=(3, 6),
            motif_type='step', motif_strength=4.0, noise_sd=0.3,
            seed=1,
        )
        # Assign sex: alternate M/F across subjects so it's balanced
        subjects = sorted(df['subject_id'].unique())
        sex_map = {s: ('M' if i % 2 == 0 else 'F') for i, s in enumerate(subjects)}
        df['sex'] = df['subject_id'].map(sex_map)
        return df

    def test_covariate_cols_accepted(self, df_with_sex):
        """harbinger() should run without error when covariate_cols is given."""
        result = harbinger(
            df_with_sex, window_size=4, top_k=4, n_permutations=50,
            seed=0, covariate_cols=['sex'],
        )
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_stratified_same_schema_as_unstratified(self, df_with_sex):
        """Output schema should be identical with or without covariate_cols."""
        r_plain = harbinger(df_with_sex, window_size=4, top_k=4,
                            n_permutations=50, seed=0)
        r_strat = harbinger(df_with_sex, window_size=4, top_k=4,
                            n_permutations=50, seed=0, covariate_cols=['sex'])
        assert set(r_plain.columns) == set(r_strat.columns)

    def test_stratified_recovers_motif(self, df_with_sex):
        """Stratified permutation should still recover the planted motif feature."""
        result = harbinger(
            df_with_sex, window_size=4, top_k=4, n_permutations=200,
            seed=0, covariate_cols=['sex'],
        )
        assert 'feature_000' in result['feature'].values

    def test_stratified_significant_p_value(self, df_with_sex):
        """Motif feature should be significant under stratified permutation."""
        result = harbinger(
            df_with_sex, window_size=4, top_k=4, n_permutations=200,
            seed=0, covariate_cols=['sex'],
        )
        row = result[result['feature'] == 'feature_000']
        assert row.iloc[0]['p_value'] < 0.05

    def test_invalid_covariate_col_raises(self, df_with_sex):
        """Non-existent covariate column should raise ValueError."""
        with pytest.raises(ValueError, match="covariate_cols not found"):
            harbinger(df_with_sex, window_size=4, n_permutations=10,
                      covariate_cols=['nonexistent_col'])

    def test_continuous_covariate_warns(self, df_with_sex):
        """Covariate with >10 unique values should trigger a UserWarning."""
        # Add a continuous-looking covariate
        rng = np.random.default_rng(99)
        subjects = df_with_sex['subject_id'].unique()
        age_map = {s: float(rng.integers(20, 80)) for s in subjects}
        df_with_sex = df_with_sex.copy()
        df_with_sex['age'] = df_with_sex['subject_id'].map(age_map)
        with pytest.warns(UserWarning, match="continuous"):
            harbinger(df_with_sex, window_size=4, n_permutations=10,
                      seed=0, covariate_cols=['age'])

    def test_permute_labels_global(self):
        """Without strata, _permute_labels should behave like a global shuffle."""
        from tempo.harbinger import _permute_labels
        rng = np.random.default_rng(0)
        all_subj = np.array(['a', 'b', 'c', 'd', 'e'])
        case_subj = ['a', 'b']
        perm_case, perm_ctrl = _permute_labels(all_subj, case_subj, None, rng)
        assert len(perm_case) == 2
        assert len(perm_ctrl) == 3
        assert set(perm_case + perm_ctrl) == set(all_subj)

    def test_permute_labels_stratified_preserves_stratum_counts(self):
        """Stratified permutation must preserve n_cases per stratum."""
        from tempo.harbinger import _permute_labels
        rng = np.random.default_rng(42)
        # 4 subjects: 2 male (1 case, 1 ctrl), 2 female (1 case, 1 ctrl)
        all_subj = np.array(['m_case', 'm_ctrl', 'f_case', 'f_ctrl'])
        case_subj = ['m_case', 'f_case']
        strata = np.array(['M', 'M', 'F', 'F'])
        # Run many permutations and check stratum case counts stay at 1
        for _ in range(50):
            perm_case, _ = _permute_labels(all_subj, case_subj, strata, rng)
            # Must have exactly 1 male case and 1 female case
            m_cases = sum(1 for s in perm_case if s.startswith('m'))
            f_cases = sum(1 for s in perm_case if s.startswith('f'))
            assert m_cases == 1
            assert f_cases == 1

    def test_permute_labels_homogeneous_stratum_unchanged(self):
        """A stratum with all cases or all controls should not be permuted."""
        from tempo.harbinger import _permute_labels
        rng = np.random.default_rng(0)
        # Stratum A: all 3 subjects are cases (homogeneous) — never changes
        # Stratum B: 1 case, 1 ctrl
        all_subj = np.array(['a1', 'a2', 'a3', 'b_case', 'b_ctrl'])
        case_subj = ['a1', 'a2', 'a3', 'b_case']
        strata = np.array(['A', 'A', 'A', 'B', 'B'])
        for _ in range(20):
            perm_case, perm_ctrl = _permute_labels(all_subj, case_subj, strata, rng)
            # a1, a2, a3 must always be cases
            assert 'a1' in perm_case
            assert 'a2' in perm_case
            assert 'a3' in perm_case


# ---------------------------------------------------------------------------
# NaN value interpolation in T_case
# ---------------------------------------------------------------------------

class TestNaNInterpolation:
    """harbinger() must handle NaN values within existing timepoints.

    NaN entries in the wide pivot (value missing for a subject at a timepoint
    that otherwise exists) caused stumpy.mstump to return all-inf, silently
    dropping every feature.  The fix interpolates NaNs along the timepoint
    axis before passing to STUMPY; the enrichment score still uses the
    original (uninterpolated) wide DataFrame.
    """

    def _make_df_with_nan(self, seed=42):
        df = simulate.simulate_continuous(
            n_subjects=20, n_timepoints=8, n_features=4, n_cases=8,
            motif_features=[0], motif_window=(2, 5),
            motif_type="step", motif_strength=6.0, noise_sd=0.3,
            seed=seed,
        )
        # Inject a NaN value into a case subject's mid-series timepoint
        case_subj = df[df["outcome"] == 1]["subject_id"].unique()[0]
        mask = (df["subject_id"] == case_subj) & (df["timepoint"] == 4) & (df["feature"] == "feature_000")
        df.loc[mask, "value"] = np.nan
        return df

    def test_nan_value_does_not_crash(self):
        """NaN in a case subject value should not prevent harbinger from returning results."""
        df = self._make_df_with_nan()
        result = harbinger(df, window_size=3, top_k=4, n_permutations=20, seed=0)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_nan_value_motif_feature_still_recovered(self):
        """True motif feature should still appear in results despite a NaN."""
        df = self._make_df_with_nan()
        result = harbinger(df, window_size=3, top_k=4, n_permutations=50, seed=0)
        assert "feature_000" in result["feature"].values


# ---------------------------------------------------------------------------
# motif_candidates parameter
# ---------------------------------------------------------------------------

class TestMotifCandidates:
    """motif_candidates controls how many pan-MP positions are evaluated."""

    def test_default_none_returns_results(self):
        """Default (None) mode should work as before."""
        df = simulate.simulate_continuous(
            n_subjects=20, n_timepoints=12, n_features=4, n_cases=8,
            motif_features=[0], motif_window=(3, 7),
            motif_type="step", motif_strength=6.0, noise_sd=0.3, seed=10,
        )
        result = harbinger(df, window_size=4, n_permutations=20, seed=0,
                           motif_candidates=None)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_top_k_mode_returns_results(self):
        """Top-K mode (motif_candidates=5) should return results."""
        df = simulate.simulate_continuous(
            n_subjects=20, n_timepoints=12, n_features=4, n_cases=8,
            motif_features=[0], motif_window=(3, 7),
            motif_type="step", motif_strength=6.0, noise_sd=0.3, seed=10,
        )
        result = harbinger(df, window_size=4, n_permutations=20, seed=0,
                           motif_candidates=5)
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_top_k_recovers_motif_feature(self):
        """Top-K mode should still identify the planted motif feature."""
        df = simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=6, n_cases=12,
            motif_features=[0], motif_window=(3, 7),
            motif_type="step", motif_strength=8.0, noise_sd=0.3, seed=7,
        )
        result = harbinger(df, window_size=4, n_permutations=50, seed=0,
                           motif_candidates=5)
        assert "feature_000" in result["feature"].values

    def test_large_k_clamped_to_profile_length(self):
        """motif_candidates larger than the profile length should not crash."""
        df = simulate.simulate_continuous(
            n_subjects=20, n_timepoints=8, n_features=3, n_cases=8,
            motif_features=[0], motif_window=(2, 5),
            motif_type="step", motif_strength=6.0, noise_sd=0.3, seed=3,
        )
        # k=9999 is much larger than profile_len = n_tp - ws + 1
        result = harbinger(df, window_size=3, n_permutations=20, seed=0,
                           motif_candidates=9999)
        assert isinstance(result, pd.DataFrame)

"""
tests/test_simulate.py

Unit tests for the TEMPO simulation framework.
These tests verify that simulated data has the correct structure,
compositional properties, and embedded motif characteristics.
"""

import pytest
import numpy as np
import pandas as pd
from tempo import simulate


# ---------------------------------------------------------------------------
# Test simulate_longitudinal
# ---------------------------------------------------------------------------

class TestSimulateLongitudinal:

    def test_output_shape(self):
        df = simulate.simulate_longitudinal(
            n_subjects=10, n_timepoints=8, n_features=5, n_cases=4, seed=0
        )
        expected_rows = 10 * 8 * 5
        assert len(df) == expected_rows

    def test_columns_present(self):
        df = simulate.simulate_longitudinal(seed=0)
        assert set(["subject_id", "timepoint", "feature", "value", "outcome"]).issubset(df.columns)

    def test_compositional_constraint(self):
        """Values should sum to ~1 per subject per timepoint (after zero inflation)."""
        df = simulate.simulate_longitudinal(zero_inflation=0.0, seed=0)
        sums = df.groupby(["subject_id", "timepoint"])["value"].sum()
        np.testing.assert_allclose(sums.values, 1.0, atol=1e-6)

    def test_correct_n_cases_controls(self):
        df = simulate.simulate_longitudinal(n_subjects=20, n_cases=7, seed=0)
        subjects = df[["subject_id", "outcome"]].drop_duplicates()
        assert subjects["outcome"].sum() == 7
        assert (subjects["outcome"] == 0).sum() == 13

    def test_zero_inflation(self):
        df = simulate.simulate_longitudinal(zero_inflation=0.5, seed=0)
        zero_rate = (df["value"] == 0).mean()
        assert 0.2 < zero_rate < 0.7  # not exact due to re-normalization

    def test_no_zeros_when_zero_inflation_zero(self):
        df = simulate.simulate_longitudinal(zero_inflation=0.0, seed=0)
        assert (df["value"] == 0).sum() == 0

    def test_motif_features_have_higher_values_in_cases(self):
        """Cases should have higher values for motif features during the motif window."""
        df = simulate.simulate_longitudinal(
            motif_features=[0],
            motif_window=(3, 6),
            motif_strength=5.0,
            noise_sd=0.05,
            zero_inflation=0.0,
            seed=0
        )
        motif_tp = df[
            (df["feature"] == "feature_000") &
            (df["timepoint"].between(3, 6))
        ]
        case_mean = motif_tp[motif_tp["outcome"] == 1]["value"].mean()
        ctrl_mean = motif_tp[motif_tp["outcome"] == 0]["value"].mean()
        assert case_mean > ctrl_mean

    def test_ground_truth_metadata(self):
        df = simulate.simulate_longitudinal(motif_features=[0, 1], motif_window=(2, 5), seed=0)
        truth = simulate.get_ground_truth(df)
        assert truth["motif_features"] == ["feature_000", "feature_001"]
        assert truth["motif_window"] == (2, 5)

    def test_survival_outcome_adds_column(self):
        df = simulate.simulate_longitudinal(outcome_type="survival", seed=0)
        assert "time_to_event" in df.columns

    def test_continuous_outcome(self):
        df = simulate.simulate_longitudinal(outcome_type="continuous", seed=0)
        assert df["outcome"].dtype == float

    def test_reproducibility(self):
        df1 = simulate.simulate_longitudinal(seed=42)
        df2 = simulate.simulate_longitudinal(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = simulate.simulate_longitudinal(seed=1)
        df2 = simulate.simulate_longitudinal(seed=2)
        assert not df1["value"].equals(df2["value"])


# ---------------------------------------------------------------------------
# Test simulate_continuous
# ---------------------------------------------------------------------------

class TestSimulateContinuous:

    def test_output_shape(self):
        df = simulate.simulate_continuous(
            n_subjects=10, n_timepoints=8, n_features=5, n_cases=4, seed=0
        )
        assert len(df) == 10 * 8 * 5

    def test_motif_types(self):
        for motif_type in ["step", "ramp", "pulse"]:
            df = simulate.simulate_continuous(motif_type=motif_type, seed=0)
            assert len(df) > 0

    def test_invalid_motif_type(self):
        with pytest.raises(ValueError, match="Unknown motif_type"):
            simulate.simulate_continuous(motif_type="triangle", seed=0)

    def test_step_motif_elevates_cases(self):
        df = simulate.simulate_continuous(
            motif_features=[0],
            motif_window=(3, 6),
            motif_strength=10.0,
            noise_sd=0.1,
            motif_type="step",
            seed=0
        )
        motif_tp = df[
            (df["feature"] == "feature_000") &
            (df["timepoint"].between(3, 6))
        ]
        case_mean = motif_tp[motif_tp["outcome"] == 1]["value"].mean()
        ctrl_mean = motif_tp[motif_tp["outcome"] == 0]["value"].mean()
        assert case_mean > ctrl_mean

    def test_columns_present(self):
        df = simulate.simulate_continuous(seed=0)
        assert set(["subject_id", "timepoint", "feature", "value", "outcome"]).issubset(df.columns)


# ---------------------------------------------------------------------------
# Test evaluation_report
# ---------------------------------------------------------------------------

class TestEvaluationReport:

    def test_perfect_detection(self):
        df = simulate.simulate_longitudinal(motif_features=[0, 1], motif_window=(3, 5), seed=0)
        report = simulate.evaluation_report(
            detected_features=["feature_000", "feature_001"],
            detected_window=(3, 5),
            df=df
        )
        assert report["feature_recall"] == 1.0
        assert report["feature_precision"] == 1.0
        assert report["window_jaccard"] == 1.0

    def test_partial_detection(self):
        df = simulate.simulate_longitudinal(motif_features=[0, 1, 2], motif_window=(3, 6), seed=0)
        report = simulate.evaluation_report(
            detected_features=["feature_000"],
            detected_window=(4, 6),
            df=df
        )
        assert report["feature_recall"] < 1.0
        assert report["window_jaccard"] < 1.0

    def test_no_metadata_raises(self):
        df = pd.DataFrame({"a": [1, 2]})
        with pytest.raises(ValueError, match="simulation metadata"):
            simulate.get_ground_truth(df)

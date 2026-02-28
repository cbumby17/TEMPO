"""
tests/test_preprocess.py

Unit tests for tempo.preprocess.
"""

import pytest
import numpy as np
import pandas as pd
from tempo import simulate
from tempo.preprocess import preprocess, clr_transform, bray_curtis_trajectory


@pytest.fixture
def df_comp():
    """Compositional dataframe with no zero inflation for clean math."""
    return simulate.simulate_longitudinal(
        n_subjects=10, n_timepoints=6, n_features=8, n_cases=4,
        zero_inflation=0.0, seed=0,
    )


@pytest.fixture
def df_sparse():
    """Compositional dataframe with zero inflation."""
    return simulate.simulate_longitudinal(
        n_subjects=10, n_timepoints=6, n_features=8, n_cases=4,
        zero_inflation=0.3, seed=1,
    )


# ---------------------------------------------------------------------------
# clr_transform
# ---------------------------------------------------------------------------

class TestCLRTransform:

    def test_output_shape(self, df_comp):
        result = clr_transform(df_comp)
        assert result.shape == df_comp.shape

    def test_columns_preserved(self, df_comp):
        result = clr_transform(df_comp)
        assert list(result.columns) == list(df_comp.columns)

    def test_clr_sums_to_zero_per_sample(self, df_comp):
        """CLR values must sum to zero within each (subject, timepoint) sample."""
        result = clr_transform(df_comp)
        sums = result.groupby(["subject_id", "timepoint"])["value"].sum()
        np.testing.assert_allclose(sums.values, 0.0, atol=1e-10)

    def test_clr_values_are_real(self, df_comp):
        result = clr_transform(df_comp)
        assert result["value"].notna().all()
        assert np.isfinite(result["value"]).all()

    def test_zero_inflation_handled(self, df_sparse):
        """Should not raise even when zeros are present (pseudo_count handles it)."""
        result = clr_transform(df_sparse)
        assert np.isfinite(result["value"]).all()

    def test_attrs_preserved(self, df_comp):
        result = clr_transform(df_comp)
        assert result.attrs == df_comp.attrs

    def test_non_value_columns_unchanged(self, df_comp):
        result = clr_transform(df_comp)
        for col in ["subject_id", "timepoint", "feature", "outcome"]:
            pd.testing.assert_series_equal(
                result[col].reset_index(drop=True),
                df_comp[col].reset_index(drop=True),
            )

    def test_larger_pseudo_count_shifts_values(self, df_comp):
        r1 = clr_transform(df_comp, pseudo_count=1e-6)
        r2 = clr_transform(df_comp, pseudo_count=0.5)
        assert not r1["value"].equals(r2["value"])

    def test_does_not_mutate_input(self, df_comp):
        original_values = df_comp["value"].copy()
        clr_transform(df_comp)
        pd.testing.assert_series_equal(df_comp["value"], original_values)


# ---------------------------------------------------------------------------
# bray_curtis_trajectory
# ---------------------------------------------------------------------------

class TestBrayCurtisTrajectory:

    def test_output_columns(self, df_comp):
        result = bray_curtis_trajectory(df_comp)
        assert set(result.columns) == {"subject_id", "timepoint", "distance", "outcome"}

    def test_n_rows(self, df_comp):
        """Should have (n_timepoints - 1) rows per subject."""
        result = bray_curtis_trajectory(df_comp)
        n_subjects = df_comp["subject_id"].nunique()
        n_timepoints = df_comp["timepoint"].nunique()
        assert len(result) == n_subjects * (n_timepoints - 1)

    def test_distance_range(self, df_comp):
        """Bray-Curtis must be in [0, 1]."""
        result = bray_curtis_trajectory(df_comp)
        assert (result["distance"] >= 0).all()
        assert (result["distance"] <= 1).all()

    def test_identical_timepoints_give_zero(self):
        """If consecutive timepoints are identical, BC distance should be 0."""
        # Build a df where t=0 and t=1 have the same composition
        df = pd.DataFrame([
            {"subject_id": "s1", "timepoint": 0, "feature": "f1", "value": 0.5, "outcome": 0},
            {"subject_id": "s1", "timepoint": 0, "feature": "f2", "value": 0.5, "outcome": 0},
            {"subject_id": "s1", "timepoint": 1, "feature": "f1", "value": 0.5, "outcome": 0},
            {"subject_id": "s1", "timepoint": 1, "feature": "f2", "value": 0.5, "outcome": 0},
        ])
        result = bray_curtis_trajectory(df)
        assert result["distance"].iloc[0] == pytest.approx(0.0)

    def test_completely_different_gives_one(self):
        """Non-overlapping compositions should give BC = 1."""
        df = pd.DataFrame([
            {"subject_id": "s1", "timepoint": 0, "feature": "f1", "value": 1.0, "outcome": 0},
            {"subject_id": "s1", "timepoint": 0, "feature": "f2", "value": 0.0, "outcome": 0},
            {"subject_id": "s1", "timepoint": 1, "feature": "f1", "value": 0.0, "outcome": 0},
            {"subject_id": "s1", "timepoint": 1, "feature": "f2", "value": 1.0, "outcome": 0},
        ])
        result = bray_curtis_trajectory(df)
        assert result["distance"].iloc[0] == pytest.approx(1.0)

    def test_subjects_filter(self, df_comp):
        all_subjects = df_comp["subject_id"].unique().tolist()
        subset = all_subjects[:3]
        result = bray_curtis_trajectory(df_comp, subjects=subset)
        assert set(result["subject_id"].unique()) == set(subset)

    def test_outcome_preserved(self, df_comp):
        result = bray_curtis_trajectory(df_comp)
        # Each subject should have one unique outcome in the result
        per_subject = result.groupby("subject_id")["outcome"].nunique()
        assert (per_subject == 1).all()

    def test_attrs_preserved(self, df_comp):
        result = bray_curtis_trajectory(df_comp)
        assert result.attrs == df_comp.attrs

    def test_no_outcome_column(self):
        """Should work gracefully when outcome column is absent."""
        df = pd.DataFrame([
            {"subject_id": "s1", "timepoint": 0, "feature": "f1", "value": 0.6},
            {"subject_id": "s1", "timepoint": 0, "feature": "f2", "value": 0.4},
            {"subject_id": "s1", "timepoint": 1, "feature": "f1", "value": 0.3},
            {"subject_id": "s1", "timepoint": 1, "feature": "f2", "value": 0.7},
        ])
        result = bray_curtis_trajectory(df)
        assert "outcome" not in result.columns
        assert len(result) == 1


# ---------------------------------------------------------------------------
# preprocess (orchestrator)
# ---------------------------------------------------------------------------

class TestPreprocess:

    def test_method_none_returns_copy(self, df_comp):
        result = preprocess(df_comp, method="none")
        pd.testing.assert_frame_equal(result, df_comp)
        assert result is not df_comp  # should be a copy

    def test_method_clr(self, df_comp):
        result = preprocess(df_comp, method="clr")
        sums = result.groupby(["subject_id", "timepoint"])["value"].sum()
        np.testing.assert_allclose(sums.values, 0.0, atol=1e-10)

    def test_method_bray_curtis(self, df_comp):
        result = preprocess(df_comp, method="bray_curtis")
        assert "distance" in result.columns
        assert (result["distance"] >= 0).all()
        assert (result["distance"] <= 1).all()

    def test_method_bray_curtis_with_clr(self, df_comp):
        result = preprocess(df_comp, method="bray_curtis", clr=True)
        assert "distance" in result.columns

    def test_invalid_method_raises(self, df_comp):
        with pytest.raises(ValueError, match="Unknown method"):
            preprocess(df_comp, method="manhattan")

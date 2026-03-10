"""
tests/test_stats.py

Unit tests for tempo.stats.

Uses continuous simulation for clean signal and compositional simulation
for survival tests (which have time_to_event by design).
"""

import pytest
import numpy as np
import pandas as pd
from tempo import simulate
from tempo.stats import (
    permutation_test, enrichment_score, survival_test,
    compute_resistance, compute_resilience, compare_recovery,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def df_cont():
    """Strong continuous motif in feature_000 only."""
    return simulate.simulate_continuous(
        n_subjects=30, n_timepoints=12, n_features=5, n_cases=12,
        motif_features=[0], motif_window=(3, 7),
        motif_strength=6.0, noise_sd=0.3, seed=0,
    )


@pytest.fixture(scope="module")
def df_surv():
    """Compositional simulation with survival outcome."""
    return simulate.simulate_longitudinal(
        n_subjects=40, n_cases=15,
        motif_features=[0], motif_window=(3, 7),
        motif_strength=2.5, outcome_type="survival", seed=1,
    )


# ---------------------------------------------------------------------------
# enrichment_score
# ---------------------------------------------------------------------------

class TestEnrichmentScore:

    def test_mean_difference_motif_positive(self, df_cont):
        s = enrichment_score(df_cont, "feature_000", (3, 7), method="mean_difference")
        assert s > 0, "Motif feature should have positive mean_difference"

    def test_mean_difference_noise_near_zero(self, df_cont):
        s = enrichment_score(df_cont, "feature_004", (3, 7), method="mean_difference")
        assert abs(s) < 1.5, "Noise feature mean_difference should be small"

    def test_mean_difference_motif_greater_than_noise(self, df_cont):
        motif = enrichment_score(df_cont, "feature_000", (3, 7), method="mean_difference")
        noise = enrichment_score(df_cont, "feature_004", (3, 7), method="mean_difference")
        assert motif > noise

    def test_auc_motif_near_one(self, df_cont):
        s = enrichment_score(df_cont, "feature_000", (3, 7), method="auc")
        assert s > 0.9, f"AUC for strong motif should be near 1, got {s:.3f}"

    def test_auc_noise_near_half(self, df_cont):
        s = enrichment_score(df_cont, "feature_004", (3, 7), method="auc")
        assert 0.2 < s < 0.8, f"AUC for noise feature should be near 0.5, got {s:.3f}"

    def test_auc_in_unit_interval(self, df_cont):
        for feat in ["feature_000", "feature_001", "feature_004"]:
            s = enrichment_score(df_cont, feat, (3, 7), method="auc")
            assert 0.0 <= s <= 1.0

    def test_gsea_motif_high(self, df_cont):
        s = enrichment_score(df_cont, "feature_000", (3, 7), method="gsea")
        assert s > 0.7, f"GSEA for strong motif should be high, got {s:.3f}"

    def test_gsea_in_unit_interval(self, df_cont):
        for feat in ["feature_000", "feature_004"]:
            s = enrichment_score(df_cont, feat, (3, 7), method="gsea")
            assert 0.0 <= s <= 1.0, f"GSEA out of [0,1] for {feat}: {s}"

    def test_gsea_motif_greater_than_noise(self, df_cont):
        motif = enrichment_score(df_cont, "feature_000", (3, 7), method="gsea")
        noise = enrichment_score(df_cont, "feature_004", (3, 7), method="gsea")
        assert motif > noise

    def test_invalid_method_raises(self, df_cont):
        with pytest.raises(ValueError, match="Unknown method"):
            enrichment_score(df_cont, "feature_000", (3, 7), method="bad")


# ---------------------------------------------------------------------------
# enrichment_score — template_correlation method
# ---------------------------------------------------------------------------

class TestEnrichmentScoreTemplateCorr:

    def test_motif_feature_scores_positive(self, df_cont):
        """Motif feature should have a positive template_correlation score."""
        s = enrichment_score(df_cont, "feature_000", (3, 7), method="template_correlation")
        assert s > 0, f"Expected positive template_correlation for motif feature, got {s}"

    def test_returns_float(self, df_cont):
        s = enrichment_score(df_cont, "feature_000", (3, 7), method="template_correlation")
        assert isinstance(s, float)

    def test_score_in_valid_range(self, df_cont):
        """template_correlation score must lie in [−2, 2]."""
        for feat in ["feature_000", "feature_004"]:
            s = enrichment_score(df_cont, feat, (3, 7), method="template_correlation")
            assert -2.0 <= s <= 2.0, f"Score out of range for {feat}: {s}"

    def test_motif_greater_than_noise(self, df_cont):
        """Motif feature should score higher than a noise feature."""
        motif = enrichment_score(df_cont, "feature_000", (3, 7), method="template_correlation")
        noise = enrichment_score(df_cont, "feature_004", (3, 7), method="template_correlation")
        assert motif > noise

    def test_invalid_method_raises(self, df_cont):
        with pytest.raises(ValueError, match="Unknown method"):
            enrichment_score(df_cont, "feature_000", (3, 7), method="template_corr")

    def test_returns_float(self, df_cont):
        for method in ["mean_difference", "auc", "gsea", "template_correlation"]:
            result = enrichment_score(df_cont, "feature_000", (3, 7), method=method)
            assert isinstance(result, float)


# ---------------------------------------------------------------------------
# permutation_test
# ---------------------------------------------------------------------------

class TestPermutationTest:

    def test_returns_dict_with_expected_keys(self, df_cont):
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=0)
        assert set(result.keys()) == {
            "observed_score", "p_value", "null_mean", "null_sd", "n_permutations"
        }

    def test_n_permutations_recorded(self, df_cont):
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=0)
        assert result["n_permutations"] == 99

    def test_p_value_in_unit_interval(self, df_cont):
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=0)
        assert 0.0 <= result["p_value"] <= 1.0

    def test_motif_feature_significant(self, df_cont):
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=499, seed=0)
        assert result["p_value"] < 0.05, f"Motif feature p={result['p_value']}"

    def test_noise_feature_not_significant(self, df_cont):
        result = permutation_test(df_cont, "feature_004", (3, 7), n_permutations=499, seed=0)
        assert result["p_value"] > 0.05, f"Noise feature p={result['p_value']}"

    def test_observed_score_positive_for_motif(self, df_cont):
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=0)
        assert result["observed_score"] > 0

    def test_null_mean_near_zero(self, df_cont):
        """Permutation null should be centred around zero."""
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=499, seed=0)
        assert abs(result["null_mean"]) < 1.0

    def test_null_sd_positive(self, df_cont):
        result = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=0)
        assert result["null_sd"] > 0

    def test_reproducible_with_same_seed(self, df_cont):
        r1 = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=7)
        r2 = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=99, seed=7)
        assert r1 == r2

    def test_different_seeds_give_different_null_distributions(self, df_cont):
        r1 = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=50, seed=1)
        r2 = permutation_test(df_cont, "feature_000", (3, 7), n_permutations=50, seed=2)
        # Different seeds → different null samples → different null_sd
        assert r1["null_sd"] != r2["null_sd"]


# ---------------------------------------------------------------------------
# survival_test
# ---------------------------------------------------------------------------

class TestSurvivalTest:

    def test_returns_dict_with_expected_keys(self, df_surv):
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        required = {"statistic", "p_value", "method", "feature", "motif_window",
                    "n_motif_positive", "n_motif_negative", "score_threshold"}
        assert required.issubset(result.keys())

    def test_method_recorded(self, df_surv):
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        assert result["method"] == "logrank"

    def test_feature_recorded(self, df_surv):
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        assert result["feature"] == "feature_000"

    def test_motif_window_recorded(self, df_surv):
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        assert result["motif_window"] == (3, 7)

    def test_p_value_in_unit_interval(self, df_surv):
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        assert 0.0 <= result["p_value"] <= 1.0

    def test_groups_sum_to_total_subjects(self, df_surv):
        n_subjects = df_surv["subject_id"].nunique()
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        assert result["n_motif_positive"] + result["n_motif_negative"] == n_subjects

    def test_motif_feature_significant(self, df_surv):
        """
        Motif feature should separate survival groups — motif-positive subjects
        have earlier events (cases) so the log-rank test should be significant.
        This test uses a strong motif so may be sensitive to the random seed.
        """
        result = survival_test(df_surv, "feature_000", (3, 7), method="logrank")
        assert result["p_value"] < 0.05, (
            f"Expected significant survival split for motif feature, got p={result['p_value']}"
        )

    def test_noise_feature_less_significant_than_motif(self, df_surv):
        motif_p = survival_test(df_surv, "feature_000", (3, 7), method="logrank")["p_value"]
        noise_p = survival_test(df_surv, "feature_010", (3, 7), method="logrank")["p_value"]
        assert motif_p < noise_p

    def test_invalid_method_raises(self, df_surv):
        with pytest.raises(ValueError, match="Unknown method"):
            survival_test(df_surv, "feature_000", (3, 7), method="kaplan")

    def test_cox_raises_without_lifelines(self, df_surv):
        """Cox should raise ImportError when lifelines is not installed."""
        try:
            import lifelines  # noqa: F401
            pytest.skip("lifelines is installed; skipping ImportError test")
        except ImportError:
            with pytest.raises(ImportError, match="lifelines"):
                survival_test(df_surv, "feature_000", (3, 7), method="cox")


# ---------------------------------------------------------------------------
# Stratified permutation_test (issue #13)
# ---------------------------------------------------------------------------

class TestStratifiedPermutationTest:

    @pytest.fixture
    def df_with_batch(self):
        df = simulate.simulate_continuous(
            n_subjects=30, n_timepoints=10, n_features=3, n_cases=15,
            motif_features=[0], motif_window=(3, 6),
            motif_type='step', motif_strength=5.0, noise_sd=0.3,
            seed=7,
        )
        subjects = sorted(df['subject_id'].unique())
        batch_map = {s: f'batch_{i % 3}' for i, s in enumerate(subjects)}
        df['batch'] = df['subject_id'].map(batch_map)
        return df

    def test_stratified_returns_dict(self, df_with_batch):
        result = permutation_test(
            df_with_batch, 'feature_000', (3, 6),
            n_permutations=100, seed=0, covariate_cols=['batch'],
        )
        assert isinstance(result, dict)
        assert set(result.keys()) == {
            'observed_score', 'p_value', 'null_mean', 'null_sd', 'n_permutations'
        }

    def test_stratified_p_value_significant(self, df_with_batch):
        result = permutation_test(
            df_with_batch, 'feature_000', (3, 6),
            n_permutations=500, seed=0, covariate_cols=['batch'],
        )
        assert result['p_value'] < 0.05

    def test_stratified_same_keys_as_unstratified(self, df_with_batch):
        r_plain = permutation_test(df_with_batch, 'feature_000', (3, 6),
                                   n_permutations=100, seed=0)
        r_strat = permutation_test(df_with_batch, 'feature_000', (3, 6),
                                   n_permutations=100, seed=0,
                                   covariate_cols=['batch'])
        assert set(r_plain.keys()) == set(r_strat.keys())

    def test_stratified_p_value_in_unit_interval(self, df_with_batch):
        result = permutation_test(
            df_with_batch, 'feature_000', (3, 6),
            n_permutations=100, seed=0, covariate_cols=['batch'],
        )
        assert 0.0 <= result['p_value'] <= 1.0


# ---------------------------------------------------------------------------
# compute_resistance / compute_resilience / compare_recovery (issue #16)
# ---------------------------------------------------------------------------

class TestResistanceResilience:
    """Tests for the resistance/resilience metric functions."""

    @pytest.fixture(scope="class")
    def df_pulse_decay(self):
        """Strong pulse_decay motif — cases spike and slowly return."""
        return simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=3, n_cases=15,
            motif_features=[0], motif_window=(3, 10),
            motif_type="pulse_decay", motif_strength=5.0, noise_sd=0.3,
            seed=42,
        )

    @pytest.fixture(scope="class")
    def df_pulse_plateau(self):
        """pulse_plateau motif — cases spike and stay elevated through observation end."""
        return simulate.simulate_continuous(
            n_subjects=30, n_timepoints=12, n_features=3, n_cases=15,
            motif_features=[0], motif_window=(3, 11),  # extends to final timepoint
            motif_type="pulse_plateau", motif_strength=5.0, noise_sd=0.3,
            seed=42,
        )

    # ── compute_resistance ──────────────────────────────────────────────────

    def test_resistance_returns_dataframe(self, df_pulse_decay):
        result = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert isinstance(result, pd.DataFrame)

    def test_resistance_columns(self, df_pulse_decay):
        result = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert {"subject_id", "baseline_mean", "peak_value", "resistance", "outcome"}.issubset(result.columns)

    def test_resistance_one_row_per_subject(self, df_pulse_decay):
        n_subj = df_pulse_decay["subject_id"].nunique()
        result = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert len(result) == n_subj

    def test_cases_have_lower_resistance_than_controls(self, df_pulse_decay):
        """Cases have embedded pulse_decay motif → larger displacement → lower resistance."""
        result = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3)
        case_mean = result.loc[result["outcome"] == 1, "resistance"].mean()
        ctrl_mean = result.loc[result["outcome"] == 0, "resistance"].mean()
        assert case_mean < ctrl_mean

    def test_resistance_no_outcome_col(self, df_pulse_decay):
        result = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3,
                                    outcome_col=None)
        assert "outcome" not in result.columns
        assert "resistance" in result.columns

    def test_resistance_baseline_window_param(self, df_pulse_decay):
        r1 = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3)
        r2 = compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=3,
                                baseline_window=(0, 2))
        # Both should succeed and return same n rows
        assert len(r1) == len(r2)

    def test_resistance_invalid_perturbation_tp_raises(self, df_pulse_decay):
        with pytest.raises(ValueError, match="No baseline timepoints"):
            compute_resistance(df_pulse_decay, "feature_000", perturbation_tp=0)

    # ── compute_resilience ──────────────────────────────────────────────────

    def test_resilience_returns_dataframe(self, df_pulse_decay):
        result = compute_resilience(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert isinstance(result, pd.DataFrame)

    def test_resilience_columns(self, df_pulse_decay):
        result = compute_resilience(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert {"subject_id", "peak_tp", "time_to_recovery", "resilience_index", "outcome"}.issubset(result.columns)

    def test_resilience_one_row_per_subject(self, df_pulse_decay):
        n_subj = df_pulse_decay["subject_id"].nunique()
        result = compute_resilience(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert len(result) == n_subj

    def test_resilience_index_nonnegative(self, df_pulse_decay):
        result = compute_resilience(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert (result["resilience_index"] >= 0).all()

    def test_plateau_subjects_often_not_recovered(self, df_pulse_plateau):
        """pulse_plateau keeps feature elevated → many cases should not recover."""
        result = compute_resilience(df_pulse_plateau, "feature_000", perturbation_tp=3,
                                    recovery_threshold=0.2)
        case_result = result[result["outcome"] == 1]
        n_not_recovered = (case_result["time_to_recovery"] == float("inf")).sum()
        # At least some cases should fail to recover with a plateau motif
        assert n_not_recovered > 0

    def test_resilience_index_zero_when_not_recovered(self, df_pulse_plateau):
        result = compute_resilience(df_pulse_plateau, "feature_000", perturbation_tp=3)
        not_recovered = result[result["time_to_recovery"] == float("inf")]
        assert (not_recovered["resilience_index"] == 0.0).all()

    # ── compare_recovery ───────────────────────────────────────────────────

    def test_compare_recovery_returns_dict(self, df_pulse_decay):
        result = compare_recovery(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert isinstance(result, dict)

    def test_compare_recovery_keys(self, df_pulse_decay):
        result = compare_recovery(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert {"feature", "perturbation_tp", "n_cases", "n_controls",
                "resistance", "resilience"}.issubset(result.keys())

    def test_compare_recovery_resistance_keys(self, df_pulse_decay):
        result = compare_recovery(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert {"case_mean", "ctrl_mean", "case_sd", "ctrl_sd",
                "statistic", "p_value"}.issubset(result["resistance"].keys())

    def test_compare_recovery_n_counts_correct(self, df_pulse_decay):
        result = compare_recovery(df_pulse_decay, "feature_000", perturbation_tp=3)
        assert result["n_cases"] == 15
        assert result["n_controls"] == 15

    def test_compare_recovery_resistance_p_significant(self, df_pulse_decay):
        """Strong pulse_decay motif → cases more displaced → significantly lower resistance."""
        result = compare_recovery(df_pulse_decay, "feature_000", perturbation_tp=3)
        p = result["resistance"]["p_value"]
        assert p < 0.05, f"Expected significant resistance difference, got p={p}"

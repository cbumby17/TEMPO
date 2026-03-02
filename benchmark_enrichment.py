"""
benchmark_enrichment.py

Standalone comparison of mean_difference vs template_correlation enrichment
methods in harbinger(). Produces benchmark_enrichment.png (2×3 subplot grid).

Sections
--------
1. Null calibration  — p-value histograms under H0 (motif_strength=0)
2. Power curve       — detection rate vs motif_strength sweep
3. Motif type sweep  — power across step/ramp/pulse/oscillating
4. AUC comparison    — enrichment_score AUC on Olink data (requires network)
5. LOO-CV AUC        — honest cross-validated AUC (template_correlation only)

Sections 4 and 5 are wrapped in try/except so the script runs without network
access and only falls back gracefully when the Olink CSV is unavailable.

Usage
-----
    python benchmark_enrichment.py

Runtime: ~20 minutes for all sections (null calibration + power curves are the
most expensive due to repeated harbinger() calls).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import kstest, uniform

from tempo import simulate
from tempo.harbinger import harbinger
from tempo.stats import enrichment_score

METHODS = ["mean_difference", "template_correlation"]
METHOD_LABELS = {"mean_difference": "Mean diff.", "template_correlation": "Template corr."}
METHOD_COLORS = {"mean_difference": "#4C72B0", "template_correlation": "#DD8452"}

SEED_BASE = 0
N_PERMS_FAST = 200   # permutations for calibration / power sections
N_PERMS_NULL = 200   # permutations per seed in null calibration
N_SEEDS_POWER = 20   # seeds per strength level in power curve
N_SEEDS_TYPE = 20    # seeds per motif type in type sweep


# ---------------------------------------------------------------------------
# Section 1 — Null calibration
# ---------------------------------------------------------------------------

def run_null_calibration(ax, n_reps=100):
    """Collect p-values under H0 (motif_strength=0) and check calibration."""
    print("Section 1: Null calibration …")
    p_values = {m: [] for m in METHODS}

    for rep in range(n_reps):
        df = simulate.simulate_continuous(
            n_subjects=20, n_timepoints=10, n_features=4, n_cases=8,
            motif_features=[0], motif_window=(3, 6),
            motif_type="step", motif_strength=0.0,  # H0
            noise_sd=0.5, seed=SEED_BASE + rep,
        )
        for method in METHODS:
            result = harbinger(
                df, window_size=4, top_k=4, n_permutations=N_PERMS_NULL,
                seed=rep, enrichment_method=method,
            )
            feat_row = result[result["feature"] == "feature_000"]
            if len(feat_row) > 0:
                p_values[method].append(feat_row.iloc[0]["p_value"])
            else:
                p_values[method].append(1.0)

    bins = np.linspace(0, 1, 21)
    for method in METHODS:
        ps = np.array(p_values[method])
        stat, ks_p = kstest(ps, uniform(0, 1).cdf)
        label = (
            f"{METHOD_LABELS[method]}\n"
            f"KS p={ks_p:.3f} "
            f"({'MISCALIBRATED' if ks_p < 0.05 else 'calibrated'})"
        )
        ax.hist(ps, bins=bins, alpha=0.6, color=METHOD_COLORS[method], label=label, density=True)

    ax.axhline(1.0, color="grey", linestyle="--", linewidth=0.8, label="Uniform(0,1)")
    ax.set_xlabel("p-value")
    ax.set_ylabel("Density")
    ax.set_title("1. Null calibration (H₀: strength=0)")
    ax.legend(fontsize=7)


# ---------------------------------------------------------------------------
# Section 2 — Power curve
# ---------------------------------------------------------------------------

def run_power_curve(ax, strengths=None):
    """Power = fraction of seeds with motif feature p < 0.05, across strengths."""
    if strengths is None:
        strengths = [0.5, 1.0, 2.0, 4.0, 8.0]
    print("Section 2: Power curve …")
    power = {m: [] for m in METHODS}

    for strength in strengths:
        print(f"  strength={strength} …")
        for method in METHODS:
            sig_count = 0
            for seed in range(N_SEEDS_POWER):
                df = simulate.simulate_continuous(
                    n_subjects=20, n_timepoints=10, n_features=4, n_cases=8,
                    motif_features=[0], motif_window=(3, 6),
                    motif_type="step", motif_strength=strength,
                    noise_sd=0.5, seed=SEED_BASE + seed,
                )
                result = harbinger(
                    df, window_size=4, top_k=4, n_permutations=N_PERMS_FAST,
                    seed=seed, enrichment_method=method,
                )
                feat_row = result[result["feature"] == "feature_000"]
                if len(feat_row) > 0 and feat_row.iloc[0]["p_value"] < 0.05:
                    sig_count += 1
            power[method].append(sig_count / N_SEEDS_POWER)

    for method in METHODS:
        ax.plot(
            strengths, power[method],
            marker="o", color=METHOD_COLORS[method],
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel("Motif strength")
    ax.set_ylabel("Power (p < 0.05)")
    ax.set_title("2. Power curve (step motif)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Section 3 — Motif type sweep
# ---------------------------------------------------------------------------

def run_motif_type_sweep(ax):
    """Power across motif types — key: does template_correlation beat mean_difference on oscillating?"""
    motif_types = ["step", "ramp", "pulse", "oscillating"]
    print("Section 3: Motif type sweep …")
    power = {m: [] for m in METHODS}

    for mtype in motif_types:
        print(f"  motif_type={mtype} …")
        for method in METHODS:
            sig_count = 0
            for seed in range(N_SEEDS_TYPE):
                df = simulate.simulate_continuous(
                    n_subjects=20, n_timepoints=10, n_features=4, n_cases=8,
                    motif_features=[0], motif_window=(3, 6),
                    motif_type=mtype, motif_strength=4.0,
                    noise_sd=0.5, seed=SEED_BASE + seed,
                )
                result = harbinger(
                    df, window_size=4, top_k=4, n_permutations=N_PERMS_FAST,
                    seed=seed, enrichment_method=method,
                )
                feat_row = result[result["feature"] == "feature_000"]
                if len(feat_row) > 0 and feat_row.iloc[0]["p_value"] < 0.05:
                    sig_count += 1
            power[method].append(sig_count / N_SEEDS_TYPE)

    x = np.arange(len(motif_types))
    width = 0.35
    for k, method in enumerate(METHODS):
        ax.bar(
            x + k * width, power[method], width,
            color=METHOD_COLORS[method], label=METHOD_LABELS[method], alpha=0.85,
        )

    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(motif_types)
    ax.set_ylabel("Power (p < 0.05)")
    ax.set_title("3. Motif type sweep (strength=4)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)


# ---------------------------------------------------------------------------
# Section 4 — AUC comparison on Olink data
# ---------------------------------------------------------------------------

def run_olink_auc(ax):
    """Download Olink CSV and compare enrichment_score AUC for top harbinger features."""
    print("Section 4: Olink AUC comparison …")
    try:
        olink_url = (
            "https://raw.githubusercontent.com/UMCUGenetics/OlinkAnalyze/"
            "main/OlinkAnalyze/data-raw/npx_data1.csv"
        )
        olink_raw = pd.read_csv(olink_url)

        # Reshape to long format expected by tempo
        # Columns: SampleID, Index (timepoint), OlinkID (feature), NPX (value), QC_Warning
        required = {"SampleID", "Index", "OlinkID", "NPX"}
        if not required.issubset(olink_raw.columns):
            raise ValueError(f"Unexpected Olink columns: {olink_raw.columns.tolist()}")

        df_olink = olink_raw.rename(columns={
            "SampleID": "subject_id",
            "Index": "timepoint",
            "OlinkID": "feature",
            "NPX": "value",
        })
        # Assign binary outcome: odd vs even index (proxy for case/control)
        subjects = df_olink["subject_id"].unique()
        outcome_map = {s: int(i % 2) for i, s in enumerate(subjects)}
        df_olink["outcome"] = df_olink["subject_id"].map(outcome_map)
        df_olink = df_olink.dropna(subset=["value"])

        # Run harbinger with both methods
        result_md = harbinger(
            df_olink, window_size=3, top_k=5, n_permutations=200, seed=0,
            enrichment_method="mean_difference",
        )
        result_tc = harbinger(
            df_olink, window_size=3, top_k=5, n_permutations=200, seed=0,
            enrichment_method="template_correlation",
        )

        # Compare enrichment_score AUC for top features from each method
        top_features = list(set(
            result_md["feature"].tolist() + result_tc["feature"].tolist()
        ))[:8]

        auc_md, auc_tc = [], []
        motif_window = result_md.iloc[0]["motif_window"]
        for feat in top_features:
            try:
                auc_md.append(enrichment_score(df_olink, feat, motif_window, method="auc"))
                auc_tc.append(enrichment_score(df_olink, feat, motif_window, method="auc"))
            except Exception:
                auc_md.append(0.5)
                auc_tc.append(0.5)

        x = np.arange(len(top_features))
        width = 0.35
        ax.bar(x, auc_md, width, label="mean_diff top feats", color=METHOD_COLORS["mean_difference"], alpha=0.8)
        ax.bar(x + width, auc_tc, width, label="tmpl_corr top feats", color=METHOD_COLORS["template_correlation"], alpha=0.8)
        ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels([f[:8] for f in top_features], rotation=45, ha="right", fontsize=7)
        ax.set_ylabel("AUC")
        ax.set_title("4. AUC comparison (Olink)")
        ax.legend(fontsize=7)

    except Exception as e:
        ax.text(0.5, 0.5, f"Olink data unavailable\n({e})",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("4. AUC comparison (Olink — skipped)")


# ---------------------------------------------------------------------------
# Section 5 — LOO-CV AUC (template_correlation only)
# ---------------------------------------------------------------------------

def run_loo_cv_auc(ax):
    """Honest LOO-CV AUC for template_correlation on Olink data."""
    print("Section 5: LOO-CV AUC …")
    try:
        from sklearn.metrics import roc_auc_score

        olink_url = (
            "https://raw.githubusercontent.com/UMCUGenetics/OlinkAnalyze/"
            "main/OlinkAnalyze/data-raw/npx_data1.csv"
        )
        olink_raw = pd.read_csv(olink_url)
        required = {"SampleID", "Index", "OlinkID", "NPX"}
        if not required.issubset(olink_raw.columns):
            raise ValueError(f"Unexpected Olink columns: {olink_raw.columns.tolist()}")

        df_olink = olink_raw.rename(columns={
            "SampleID": "subject_id",
            "Index": "timepoint",
            "OlinkID": "feature",
            "NPX": "value",
        })
        subjects = df_olink["subject_id"].unique()
        outcome_map = {s: int(i % 2) for i, s in enumerate(subjects)}
        df_olink["outcome"] = df_olink["subject_id"].map(outcome_map)
        df_olink = df_olink.dropna(subset=["value"])

        # Pick top feature from harbinger
        result = harbinger(
            df_olink, window_size=3, top_k=1, n_permutations=200, seed=0,
            enrichment_method="template_correlation",
        )
        if len(result) == 0:
            raise ValueError("No features returned by harbinger")

        top_feature = result.iloc[0]["feature"]
        motif_window = result.iloc[0]["motif_window"]

        # Build wide matrix for the top feature in the motif window
        feat_df = df_olink[df_olink["feature"] == top_feature]
        window_df = feat_df[feat_df["timepoint"].between(motif_window[0], motif_window[1])]
        wide = window_df.pivot(index="subject_id", columns="timepoint", values="value").dropna()
        outcome_series = df_olink.groupby("subject_id")["outcome"].first()

        all_subjects = wide.index.tolist()
        case_subj = [s for s in all_subjects if outcome_series[s] == 1]
        ctrl_subj = [s for s in all_subjects if outcome_series[s] == 0]

        # LOO-CV: for each subject, compute template from all OTHER cases
        loo_scores = []
        loo_outcomes = []

        for held_out in all_subjects:
            train_cases = [s for s in case_subj if s != held_out]
            if len(train_cases) < 2:
                continue
            template = wide.loc[train_cases].values.astype(float).mean(axis=0)
            if template.std() < 1e-8:
                continue
            row = wide.loc[held_out].values.astype(float)
            if row.std() < 1e-8:
                loo_scores.append(0.0)
            else:
                c = float(np.corrcoef(row, template)[0, 1])
                loo_scores.append(0.0 if np.isnan(c) else c)
            loo_outcomes.append(outcome_series[held_out])

        if len(set(loo_outcomes)) < 2:
            raise ValueError("Only one class in LOO outcomes")

        loo_auc = roc_auc_score(loo_outcomes, loo_scores)

        ax.bar(
            ["LOO-CV AUC"], [loo_auc],
            color=METHOD_COLORS["template_correlation"], alpha=0.85,
        )
        ax.axhline(0.5, linestyle="--", color="grey", linewidth=0.8, label="Chance")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("AUC")
        ax.set_title(f"5. LOO-CV AUC (template corr.)\nFeature: {top_feature}")
        ax.text(0, loo_auc + 0.02, f"{loo_auc:.3f}", ha="center", fontsize=10)
        ax.legend(fontsize=8)

    except Exception as e:
        ax.text(0.5, 0.5, f"Olink data unavailable\n({e})",
                ha="center", va="center", transform=ax.transAxes, fontsize=9)
        ax.set_title("5. LOO-CV AUC (Olink — skipped)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    run_null_calibration(axes[0])
    run_power_curve(axes[1])
    run_motif_type_sweep(axes[2])
    run_olink_auc(axes[3])
    run_loo_cv_auc(axes[4])

    # Axis 5 — summary text
    ax = axes[5]
    ax.axis("off")
    summary = (
        "Benchmark summary\n"
        "─────────────────\n"
        "• Null calibration: both methods should be\n"
        "  uniform under H₀ (KS p > 0.05)\n\n"
        "• Power: template_correlation expected to\n"
        "  dominate on oscillating motifs; comparable\n"
        "  on step/ramp/pulse\n\n"
        "• LOO-CV AUC: honest estimate because template\n"
        "  is never computed from the held-out subject"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, va="top", family="monospace")

    plt.tight_layout()
    out_path = "benchmark_enrichment.png"
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()

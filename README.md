# TEMPO
### **T**rajectory **E**nrichment via **M**atrix **P**rofile **O**utcomes

TEMPO is a Python package for detecting shared trajectory patterns in longitudinal biological data and linking them to outcomes. It implements **Harbinger analysis** — a non-parametric, motif-based framework for identifying discriminative temporal signatures in high-dimensional biological time series.

---

## Motivation

Longitudinal biological data — microbiome compositions, flow cytometry panels, gene expression profiles — often contains rich temporal structure that predicts outcomes. Standard approaches compare timepoints cross-sectionally and miss the *shape* of a response. TEMPO detects **trajectory motifs**: recurring patterns of change shared within outcome groups and absent in others.

---

## Key Features

- **Flexible input**: accepts longitudinal data in long format for any biological feature (taxa, cell populations, gene transcripts)
- **Compositional-aware preprocessing**: Bray-Curtis trajectory distances, CLR transformation, zero-robust normalization
- **Harbinger analysis**: matrix profile-based motif discovery via [STUMPY](https://stumpy.readthedocs.io/)
- **Three statistical testing frameworks**:
  - Permutation-based motif enrichment (general purpose)
  - Harbinger enrichment scoring (GSEA-style, for ranked outcomes)
  - Survival-integrated testing (time-to-event outcomes)
- **Dimensionality reduction hooks**: accepts raw features or pre-reduced representations (PCA, PCoA)
- **Visualization**: trajectory plots with motif highlighting, enrichment heatmaps

---

## Installation

```bash
pip install tempo-bio
```

Or from source:
```bash
git clone https://github.com/yourusername/TEMPO.git
cd TEMPO
pip install -e .
```

---

## Quickstart

```python
import tempo

# Load longitudinal data in long format
# Required columns: subject_id, timepoint, feature, value, outcome
df = tempo.load_example_data()

# Preprocess: compute trajectories relative to baseline
trajectories = tempo.preprocess(
    df,
    feature="Bifidobacterium",
    data_type="compositional",       # handles zero-robust normalization
    baseline_timepoint=0,
    transform="bray_curtis"
)

# Run Harbinger analysis
result = tempo.harbinger(
    trajectories,
    window_sizes=[3, 5, 7],          # scan across window sizes
    outcome_col="t1d_status",
    stat_test="permutation",         # or "enrichment" or "survival"
    n_permutations=1000
)

# Visualize
tempo.plot_motifs(result)
tempo.plot_enrichment(result)
```

---

## Data Types Supported

| Data Type | Preprocessing | Zero Handling |
|---|---|---|
| 16S / microbiome | CLR or Bray-Curtis trajectory | Multiplicative replacement or presence/absence track |
| Flow cytometry | Z-score normalization | N/A |
| Bulk RNA-seq | Log2 normalization | Pseudocount |
| Single-cell (pseudobulk) | Log2 normalization | Pseudocount |

---

## Statistical Testing: Choosing the Right Framework

| Framework | Use When |
|---|---|
| **Permutation enrichment** | Binary or categorical outcomes, general purpose |
| **Harbinger enrichment score** | Continuous or ranked outcomes (analogous to GSEA) |
| **Survival-integrated** | Time-to-event outcomes (diagnosis date, death, relapse) |

---

## Validation

TEMPO is validated on:
1. **Simulated data**: synthetic longitudinal compositional time series with known embedded motifs, used to characterize sensitivity and specificity across noise levels and window sizes
2. **DIABIMMUNE cohort**: longitudinal infant gut microbiome data (16S) linked to type 1 diabetes development — tests whether Harbinger analysis recovers known early-life microbiome signatures of T1D risk

---

## Citation

> *Coming soon*

---

## License

MIT

---

## Acknowledgments

TEMPO builds on [STUMPY](https://stumpy.readthedocs.io/) for matrix profile computation. The Harbinger analysis framework was developed in the context of transplant immunology and microbiome research at Tulane University.

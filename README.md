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

# Load the bundled example dataset (40 subjects, 12 timepoints, 15 features)
# Long format: subject_id, timepoint, feature, value, outcome
df = tempo.load_example_data()

# Preprocess: CLR-transform compositional values so Euclidean distances are valid
df_clr = tempo.clr_transform(df)

# Run Harbinger analysis — scan window sizes 3–6, rank all features
results = tempo.harbinger(df_clr, window_size_range=(3, 6), top_k=15, n_permutations=999)
print(results[['feature', 'motif_window', 'enrichment_score', 'p_value']].head())

# Visualize the top motif features
fig = tempo.plot_motifs(df_clr, features=results['feature'].head(4).tolist(),
                        motif_window=results['motif_window'].iloc[0])

# Enrichment summary (two-panel: scores + –log10 p-values)
fig = tempo.plot_enrichment(results)
```

For a full walkthrough with biological context and interpretation guidance, see the **[vignette notebook](vignette.ipynb)**.

---

## Vignette

`vignette.ipynb` is a self-contained tutorial notebook covering the complete Harbinger workflow on the bundled microbiome dataset:

| Section | What it covers |
|---------|---------------|
| §1 Setup | Load data, inspect ground truth metadata |
| §2 Explore | Raw trajectory plots, group mean ± SD |
| §3 CLR preprocess | Why CLR is necessary; sanity checks |
| §4 Harbinger | Matrix profile analysis, multi-window scanning |
| §5 Plot motifs | `plot_motifs` — case vs control trajectory overlays |
| §6 Enrichment summary | `plot_enrichment` — scores and p-values |
| §7 Permutation test | Fixed-window confirmatory testing |
| §8 Evaluate | Feature recall, precision, window Jaccard vs ground truth |

Each section includes prose explaining the biological and statistical reasoning, following the [Seurat vignette](https://satijalab.org/seurat/articles/pbmc3k_tutorial) convention. Run it with:

```bash
jupyter notebook vignette.ipynb
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

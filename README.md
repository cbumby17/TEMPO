# TEMPO
### **T**rajectory **E**nrichment via **M**atrix **P**rofile **O**utcomes

TEMPO is a Python package for detecting outcome-associated trajectory motifs in longitudinal biological data. It implements **Harbinger analysis** — a non-parametric, matrix profile-based framework for identifying temporal patterns of change that are shared among cases and absent in controls.

---

## Motivation

Many longitudinal studies apply the same perturbation to all subjects — a diet, a drug, a vaccine, a transplant, an infection — and measure how biological features evolve over time. At baseline, cases and controls are often indistinguishable. After the perturbation, their trajectories diverge: cases follow a distinctive pattern of change that controls do not, and that pattern is associated with eventually developing the outcome.

Standard approaches compare groups at individual timepoints and miss the *shape* of this divergence — the timing, the rate of change, the window during which cases and controls differ. TEMPO asks: **which features followed a distinctive trajectory in cases, and when did that trajectory diverge from controls?**

It works on any feature set that can be measured repeatedly over time: immune cell populations, cytokine profiles, metabolite abundances, gene expression, clinical measurements, or any high-dimensional longitudinal readout.

---

## Key Features

- **Flexible input**: accepts longitudinal data in long format for any biological feature (cell populations, proteins, metabolites, gene transcripts)
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
# outcome=1 = cases (developed the outcome), outcome=0 = controls
df = tempo.load_example_data()

# Preprocess: CLR-transform compositional values so Euclidean distances are valid
# Skip this step if your data is not compositional (e.g. raw concentrations)
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

`vignette.ipynb` is a self-contained tutorial notebook covering the complete Harbinger workflow on the bundled example perturbation-response dataset:

| Section | What it covers |
|---------|---------------|
| 1 Setup | Load data, inspect ground truth metadata |
| 2 Explore | Raw trajectory plots, group mean ± SD |
| 3 CLR preprocess | Why CLR is necessary for compositional data; sanity checks |
| 4 Harbinger | Matrix profile analysis, multi-window scanning |
| 5 Plot motifs | `plot_motifs` — responder vs non-responder trajectory overlays |
| 6 Enrichment summary | `plot_enrichment` — scores and p-values |
| 7 Permutation test | Fixed-window confirmatory testing |
| 8 Evaluate | Feature recall, precision, window Jaccard vs ground truth |

Each section includes prose explaining the biological and statistical reasoning.
The notebook is pre-executed — visualizations are visible directly on GitHub
or in any notebook viewer without running any code.

---

## Data Types Supported

| Data Type | Preprocessing | Notes |
|---|---|---|
| Compositional (cell fractions, relative abundances) | CLR or Bray-Curtis trajectory | Values sum to 1 at each timepoint |
| Flow cytometry (raw counts / MFI) | Z-score normalization | Non-compositional; no CLR needed |
| Bulk RNA-seq / proteomics | Log2 normalization | Pseudocount for zeros |
| Clinical measurements | None or Z-score | Depends on scale and units |

---

## Statistical Testing: Choosing the Right Framework

| Framework | Use When |
|---|---|
| **Permutation enrichment** | Binary outcomes (case/control), general purpose |
| **Harbinger enrichment score** | Continuous or ranked outcomes (analogous to GSEA) |
| **Survival-integrated** | Time-to-event outcomes (diagnosis date, relapse, graft failure) |

---

## Validation

TEMPO is validated on:
1. **Simulated data**: synthetic longitudinal time series with known embedded motifs, used to characterize sensitivity and specificity across noise levels, window sizes, and study designs

---

## Citation

> *Coming soon*

---

## License

MIT

---

## Acknowledgments

TEMPO builds on [STUMPY](https://stumpy.readthedocs.io/) for matrix profile computation. The Harbinger analysis framework was developed at Tulane University.

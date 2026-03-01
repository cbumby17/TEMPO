"""
test_real_data.py — TEMPO validation on Wastyk et al. 2021 (Cell)
Olink plasma proteomics: fiber vs fermented-food dietary intervention.

Study design:
  - 35 subjects randomised to high-fiber (n=17) or high-fermented-food (n=18) diet
  - Blood collected at 5 timepoints: weeks -3, 0, 6, 10, 14 (labelled 01-06)
  - 67 plasma proteins measured by Olink proximity extension assay (NPX units)
  - Non-compositional: no CLR needed, pass raw NPX values to harbinger()

TEMPO use case:
  - Both arms start from the same baseline (randomised, baseline equivalence expected)
  - Question: which proteins show distinctive temporal trajectories in the
    fermented-food arm vs fiber arm, and during which weeks?
  - outcome = 1 (Fermented), outcome = 0 (Fiber)
"""

import urllib.request
import numpy as np
import pandas as pd
import sys, warnings
import os

os.environ['KMP_WARNINGS'] = 'FALSE'
sys.path.insert(0, '/Users/caitlinbumby/Documents/Projects/TEMPO')

import tempo
from tempo import simulate

print("=" * 65)
print("TEMPO real-data validation — Wastyk et al. 2021 (Olink)")
print("=" * 65)

# ── 1. Download ───────────────────────────────────────────────────────────────
print("\n[1] Loading Olink data from Sonnenburg lab GitHub...")
url = ("https://raw.githubusercontent.com/SonnenburgLab/"
       "fiber-fermented-study/master/data/Olink/cleaned/olink_data_cleaned.csv")
with urllib.request.urlopen(url) as r:
    raw = pd.read_csv(r)

proteins = [c for c in raw.columns if c not in ('Participant', 'Timepoint', 'Group')]
print(f"    Raw shape: {raw.shape}  ({len(proteins)} proteins)")

# ── 2. Long format ────────────────────────────────────────────────────────────
long = raw.melt(
    id_vars=['Participant', 'Timepoint', 'Group'],
    value_vars=proteins,
    var_name='feature',
    value_name='value',
)
long = long.rename(columns={'Participant': 'subject_id',
                             'Timepoint':   'timepoint'})
long['timepoint'] = long['timepoint'].astype(int)
long['outcome']   = (long['Group'] == 'Fermented').astype(int)
long = long[['subject_id', 'timepoint', 'feature', 'value', 'outcome']].copy()
long['subject_id'] = long['subject_id'].astype(str)

print(f"    Long format: {long.shape}")
print(f"    Timepoints : {sorted(long['timepoint'].unique())}")
print(f"    Fermented  : {long[long['outcome']==1]['subject_id'].nunique()} subjects")
print(f"    Fiber      : {long[long['outcome']==0]['subject_id'].nunique()} subjects")

# ── 3. Baseline equivalence ───────────────────────────────────────────────────
print("\n[2] Baseline equivalence check (timepoint 1 = week -3)...")
bl = tempo.check_baseline(long, timepoint=1)
sig_bl = bl[bl['significant']]
print(f"    {len(sig_bl)}/{len(bl)} proteins differ at baseline (α=0.05)")
if len(sig_bl):
    print("    Significant at baseline:")
    print(sig_bl[['feature', 'case_mean', 'ctrl_mean', 'p_value']].to_string(index=False))
else:
    print("    PASS — no proteins differ at baseline.")

# ── 4. Harbinger analysis ─────────────────────────────────────────────────────
print("\n[3] Running harbinger (window_sizes=[2,3], n_permutations=999)...")
print("    (Non-compositional data — no CLR; raw NPX values passed directly)")
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    results = tempo.harbinger(
        long,
        window_sizes=[2, 3],
        top_k=20,
        n_permutations=999,
        seed=42,
    )

print(f"\n    Top 20 features by enrichment score:")
print(results[['feature', 'window_size', 'motif_window',
               'enrichment_score', 'p_value', 'q_value']].to_string(index=False))

# ── 5. Significant hits ───────────────────────────────────────────────────────
sig = results[results['q_value'] < 0.05]
print(f"\n    Significant (q < 0.05): {len(sig)} proteins")
if len(sig):
    print(sig[['feature', 'motif_window', 'enrichment_score', 'p_value', 'q_value']].to_string(index=False))

# ── 6. Nominal hits (p < 0.05 before correction) ─────────────────────────────
nom = results[results['p_value'] < 0.05]
print(f"    Nominal     (p < 0.05): {len(nom)} proteins")
if len(nom):
    print(nom[['feature', 'motif_window', 'enrichment_score', 'p_value', 'q_value']].to_string(index=False))

# ── 7. Cross-check with paper ─────────────────────────────────────────────────
# Wastyk et al. Fig 4: fermented food diet decreased 19 inflammatory markers.
# Key hits mentioned: CXCL10, IL-6, IL-10, IL-12, CXCL11, MCP-1, MCP-3, CX3CL1
paper_hits = ['CXCL10', 'IL6', 'IL10', 'IL-10', 'CXCL11',
              'MCP-1', 'MCP-3', 'CX3CL1', 'IL-12B', 'OSM', 'EN-RAGE']
recovered = [h for h in paper_hits if h in results['feature'].values]
print(f"\n[4] Cross-check with Wastyk et al. Fig 4 key inflammatory hits:")
print(f"    Paper hits in TEMPO top-20: {recovered if recovered else 'checking with exact names...'}")
# Also check with the exact column names from the data
all_feats = set(results['feature'].values)
for h in paper_hits:
    close = [f for f in all_feats if h.lower().replace('-','') in f.lower().replace('-','')]
    if close:
        print(f"      '{h}' → found as: {close}")

print("\n[5] Value range sanity check:")
print(f"    NPX min: {long['value'].min():.3f}")
print(f"    NPX max: {long['value'].max():.3f}")
print(f"    NPX mean: {long['value'].mean():.3f}")
print("\nDone.")

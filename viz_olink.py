"""Visualize TEMPO top hits on Wastyk 2021 Olink data."""
import urllib.request, warnings, sys, os
os.environ['KMP_WARNINGS'] = 'FALSE'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

sys.path.insert(0, '/Users/caitlinbumby/Documents/Projects/TEMPO')
import tempo

# ── Load & wrangle ────────────────────────────────────────────────────────────
url = ("https://raw.githubusercontent.com/SonnenburgLab/"
       "fiber-fermented-study/master/data/Olink/cleaned/olink_data_cleaned.csv")
with urllib.request.urlopen(url) as r:
    raw = pd.read_csv(r)

proteins = [c for c in raw.columns if c not in ('Participant','Timepoint','Group')]
long = raw.melt(id_vars=['Participant','Timepoint','Group'],
                value_vars=proteins, var_name='feature', value_name='value')
long = long.rename(columns={'Participant':'subject_id','Timepoint':'timepoint'})
long['timepoint'] = long['timepoint'].astype(int)
long['outcome']   = (long['Group'] == 'Fermented').astype(int)
long['subject_id'] = long['subject_id'].astype(str)
long = long[['subject_id','timepoint','feature','value','outcome']]

# Timepoint labels (study weeks)
tp_labels = {1: 'Week −3\n(baseline)', 2: 'Week 0\n(diet start)',
             4: 'Week 6', 5: 'Week 10', 6: 'Week 14\n(washout)'}

# ── Re-run harbinger ──────────────────────────────────────────────────────────
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    results = tempo.harbinger(long, window_sizes=[2, 3], top_k=20,
                              n_permutations=999, seed=42)

top_feats  = results['feature'].head(6).tolist()
top_window = results['motif_window'].iloc[0]   # (1, 4)

print("Top features:", top_feats)
print("Motif window:", top_window)

# ── Fig 1: plot_motifs (baseline-normalised) ──────────────────────────────────
fig1 = tempo.plot_motifs(long, features=top_feats, motif_window=top_window,
                         baseline_normalize=True, show_individuals=False,
                         ribbon_type='sem', figsize=(11, 4))
# Relabel x-ticks with week names on all visible axes
for ax in [a for a in fig1.axes if a.get_visible()]:
    ax.set_xticks(sorted(long['timepoint'].unique()))
    ax.set_xticklabels([tp_labels[t] for t in sorted(long['timepoint'].unique())],
                       fontsize=7)
    ax.set_ylabel('ΔNPX from baseline')
    ax.set_xlabel('')
    ax.axhline(0, color='gray', lw=0.6, ls=':', zorder=0, alpha=0.6)

# Override generic legend with study-specific labels
case_color = '#e07b30'
ctrl_color = '#5c8ae0'
fig1.legends[0].remove()
fig1.legend(handles=[
    mpatches.Patch(color=case_color, alpha=0.75, label='Fermented food'),
    mpatches.Patch(color=ctrl_color, alpha=0.75, label='High fiber'),
    mpatches.Patch(color='gold',     alpha=0.5,  label='Motif window'),
], loc='upper right', bbox_to_anchor=(1.01, 1.0), framealpha=0.9)

fig1.suptitle(
    'TEMPO top hits — Wastyk et al. 2021\n'
    'Fermented food (orange) vs High fiber (blue) | Olink plasma proteomics',
    fontsize=10, y=1.02,
)
fig1.savefig('olink_motifs.png', dpi=150, bbox_inches='tight')
print("Saved olink_motifs.png")

# ── Fig 2: plot_enrichment ────────────────────────────────────────────────────
fig2 = tempo.plot_enrichment(results, top_k=20)
fig2.suptitle(
    'Enrichment summary — Olink proteomics (Fermented vs Fiber)\n'
    'Enrichment score = mean NPX difference in motif window (Fermented − Fiber)',
    fontsize=9, y=1.02,
)
fig2.savefig('olink_enrichment.png', dpi=150, bbox_inches='tight')
print("Saved olink_enrichment.png")

# ── Fig 3: mean ± SD ribbon for top 6 hits (baseline-normalised) ─────────────
case_color = '#e07b30'
ctrl_color = '#5c8ae0'
tps = sorted(long['timepoint'].unique())
first_tp = tps[0]

# Baseline-normalize: subtract each subject×feature value at the first timepoint
baselines = (
    long[long['timepoint'] == first_tp]
    .set_index(['subject_id', 'feature'])['value']
)
long_norm = long.copy()
long_norm['value'] = (
    long_norm['value']
    - long_norm.set_index(['subject_id', 'feature']).index.map(baselines)
)

fig3, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=False)
axes_flat = axes.flatten()

for ax, feat in zip(axes_flat, top_feats):
    feat_df = long_norm[long_norm['feature'] == feat]
    row = results[results['feature'] == feat].iloc[0]
    win = row['motif_window']
    p   = row['p_value']
    q   = row['q_value']

    for outcome, label, color in [(1, 'Fermented', case_color),
                                   (0, 'Fiber',     ctrl_color)]:
        grp = feat_df[feat_df['outcome'] == outcome]
        # Mean ± SEM ribbon (no individual lines — effect size is small)
        agg = grp.groupby('timepoint')['value']
        means = agg.mean()
        sems  = agg.std() / np.sqrt(agg.count())
        ax.plot(means.index, means.values, color=color, lw=2.5,
                label=label, zorder=4)
        ax.fill_between(means.index, means - sems, means + sems,
                        color=color, alpha=0.30, zorder=3)

    # Zero reference line
    ax.axhline(0, color='gray', lw=0.6, ls=':', zorder=0, alpha=0.6)
    # Window shading
    ax.axvspan(win[0], win[1], alpha=0.12, color='gold', zorder=1)
    ax.axvline(win[0], color='goldenrod', lw=1.0, ls='--', zorder=1)
    ax.axvline(win[1], color='goldenrod', lw=1.0, ls='--', zorder=1)
    # Diet-start marker
    ax.axvline(2, color='gray', lw=0.8, ls=':', zorder=1, alpha=0.6)

    ax.set_title(f'{feat}\np={p:.3f}  q={q:.3f}', fontsize=9)
    ax.set_xticks(tps)
    ax.set_xticklabels([tp_labels[t] for t in tps], fontsize=7)
    ax.set_ylabel('ΔNPX from baseline', fontsize=8)

# Shared legend
case_p = mpatches.Patch(color=case_color, alpha=0.8, label='Fermented food')
ctrl_p = mpatches.Patch(color=ctrl_color, alpha=0.8, label='High fiber')
win_p  = mpatches.Patch(color='gold',     alpha=0.5, label='Motif window')
diet_l = plt.Line2D([0], [0], color='gray', lw=0.8, ls=':', label='Diet start (wk 0)')
fig3.legend(handles=[case_p, ctrl_p, win_p, diet_l],
            loc='upper right', bbox_to_anchor=(1.01, 0.98), fontsize=9)

fig3.suptitle(
    'Top TEMPO hits: plasma protein trajectories\n'
    'Wastyk et al. 2021 — Fermented food vs High fiber diet',
    fontsize=11, y=1.01,
)
plt.tight_layout()
fig3.savefig('olink_ribbons.png', dpi=150, bbox_inches='tight')
print("Saved olink_ribbons.png")

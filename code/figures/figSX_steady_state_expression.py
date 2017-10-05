import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mwc_induction_utils as mwc
import pandas as pd
from tqdm import tqdm
mwc.set_plotting_style()


# Load the data file.
data = pd.read_csv('data/20171003_O2_timeseries_MACSQuant.csv',
                   comment='#')
plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
strain_colors = {'auto': 'b', 'delta': 'g', 'RBS1027': 'r'}
grouped = data.groupby(['strain', 'delta_t'])

for g, d in grouped:
    ax[0].plot(d['delta_t'], d['FITC-A'], 'o', color=strain_colors[g[0]],
               alpha=0.5, markersize=5)
    if g[0] == 'RBS1027':
        ax[1].plot(d['delta_t'], d['fold_change_A'], 'o', alpha=0.5,
                   markerfacecolor='w', markeredgecolor='r', markeredgewidth=1,
                   markersize=5)

ax[1].set_xlim([5.5, 12.5])
ax[0].set_yscale('log')
ax[0].set_ylabel('mean YFP fluorescence (a.u.)', fontsize=11)
ax[1].set_ylim([0, 1])
ax[1].set_ylabel('fold-change', fontsize=11)
ax[0].set_yticks([1E3, 3E3, 1E4, 3E4, 1E5])

# Set (A) and (B) labels.
fig.text(0, 0.95, '(A)', fontsize=15)
fig.text(0.5, 0.95, '(B)', fontsize=15)
for a in ax:
    a.set_xlabel('growth time (hr)', fontsize=11)
    a.xaxis.set_tick_params(labelsize=10)
    a.yaxis.set_tick_params(labelsize=10)
    a.grid(linewidth=1)
plt.tight_layout()
plt.savefig('figures/SI_figs/figSX_steady_state.pdf')

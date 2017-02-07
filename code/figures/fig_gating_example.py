# For numerical computing
import numpy as np
import scipy.stats

# For plotting
import matplotlib.pyplot as plt
import seaborn as sns

# Custom written utilities
import mwc_induction_utils as mwc
mwc.set_plotting_style()

# For OS interaction and reading of data files.
import glob
import pandas as pd

# Load example flow cytometry data
flow_data = pd.read_csv('../../data/flow/csv/20160813/20160813_r2_wt_O2_auto_0.1uMIPTG.csv', comment='#')

# Set the colors
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Set the range  of alpha.
alpha_range = [0.8, 0.6, 0.4, 0.25, 0.05]

# Generate an understandable legend.
plt.close('all')
plt.plot([], [], 'ko', label=100)
for i, a in enumerate(alpha_range):
    plt.plot([], [], 'o', color=colors[i], label=a * 100)
leg = plt.legend(title=r'''percentile
  ($\alpha \times 100$)''', ncol=1, loc='upper left',
                 fontsize=12)
leg.get_title().set_fontsize('15')
means = []

plt.plot(flow_data['FSC-A'], flow_data['SSC-A'], 'ko', markersize=2, rasterized=True)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('forward scatter (a.u.)')
plt.ylabel('side scatter (a.u.)')

# Plot the selected cells for a range of alpha
for i, a in enumerate(alpha_range):
    gated_cells = mwc.auto_gauss_gate(flow_data, a)
    plt.plot(gated_cells['FSC-A'], gated_cells['SSC-A'], 'o',
    markersize=2, color=colors[i], rasterized=True)
plt.tight_layout()
# Save the figure.j
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/fig_example_gating.pdf', bbox_inches='tight')

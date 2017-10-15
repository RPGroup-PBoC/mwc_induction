import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mwc_induction_utils as mwc
import pandas as pd
import glob
from tqdm import tqdm
from collections import OrderedDict
mwc.set_plotting_style()


# Load the data file.
data = pd.read_csv('../../data/20171013_O2_timeseries_MACSQuant.csv',
                   comment='#')
plt.close('all')
fig, ax = plt.subplots(1, 3, figsize=(6, 2.5), gridspec_kw={
                       'width_ratios': [2, 1.2, 2]})
# note that the second subplot (ax[1]) will remain blank for legend
strain_colors = {'auto': 'b', 'delta': 'g', 'RBS1027': 'r'}
grouped = data.groupby(['strain', 'delta_t'])

print(data[data.strain == 'auto']['FITC-A'])
for g, d in grouped:
    ax[0].plot(d['delta_t'], d['FITC-A'], 'o', color=strain_colors[g[0]],
               alpha=0.5, markersize=5, label=g[0])
    if g[0] == 'RBS1027':
        ax[2].plot(d['delta_t'], d['fold_change_A'], 'o', alpha=0.5,
                   markerfacecolor='w', markeredgecolor='r', markeredgewidth=1,
                   markersize=5)

ax[2].set_xlim([5.5, 12.5])
ax[0].set_yscale('log')
ax[0].set_ylabel('mean YFP fluorescence (a.u.)', fontsize=11)
ax[2].set_ylim([0, 1])
ax[2].set_ylabel('fold-change', fontsize=11)
ax[0].set_yticks([1E3, 3E3, 1E4, 3E4, 1E5])
ax[0].set_xticks([6, 8, 10, 12])
ax[2].set_xticks([6, 8, 10, 12])

# Set (A) and (B) labels.
fig.text(0.05, 0.96, '(A)', fontsize=15)
fig.text(0.6, 0.96, '(B)', fontsize=15)
for a in ax:
    if a == 1:
        continue
    a.set_xlabel('growth time (hr)', fontsize=11)
    a.xaxis.set_tick_params(labelsize=10)
    a.yaxis.set_tick_params(labelsize=10)
    a.grid(linewidth=1)
ax[1].axis('off')

handles = []
labels = []
for i, _ in enumerate(ax):
    if i != 0:
        # remove if I want to collect labels for fold-change
        continue
    handles_temp, labels_temp = ax[i].get_legend_handles_labels()
    handles = np.append(handles, handles_temp)
    labels = np.append(labels, labels_temp)

# convert labels into more descriptive label
strain_labels = {'auto': 'autofluorescence', 'delta': '$\Delta lacI$ strain',
                 'RBS1027': 'rep. / cell = 260', 'fold_change_A': 'fold change'}
labels = np.array([strain_labels[key] for key in labels])

# create legend with unique strain labels
by_label = OrderedDict(zip(labels, handles))
lgd = ax[0].legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.02, 1), loc=2,
                   title=r"$\Delta\varepsilon_{RA} = -13.9$ $k_BT$" + "\n" + r"$c$ = $50$ $\mu$M", fontsize=8)
plt.setp(lgd.get_title(), fontsize=10)

fig.subplots_adjust(wspace=.55)
fig.savefig('../../figures/SI_figs/figSX_steady_state_round_two.pdf',
            bbox_inches='tight')

# # %% Make a figure showing the ECDFs of all of the experiemental runs.
#
# fig, ax = plt.subplots(4, 6, figsize=(5, 9))
#
# # Set up the axes dictionary.
# axes = {'auto': ax[0], 'delta': ax[1], 'RBS1027': ax[2]}
#
# # Go through all of the files.
# files = glob.glob('data/flow/csv/20171003*.csv')
# alpha = 0.4
# for i, f in enumerate(files):
#     # Split the file name to get the strain information.
#     strain = f.split('/')[-1].split('_')[2]
#
#     # Load the data  and gate.
#     flow_data = pd.read_csv(f)
#     gate = mwc.auto_gauss_gate(flow_data, alpha)
#
#     # Compute the ECDF and plot as thin black lines.
#     x, y = np.sort(gate['FITC-A']), np.arange(0, len(gate), 1) / len(gate)
#
#     axes[strain].plot(x, y, 'k-', lw=0.5, alpha=0.5, rasterized=True)
#
# for a in ax:
#     a.set_xscale('log')
#     a.set_yscale('log')
# plt.savefig('/Users/gchure/Desktop/test.pdf')

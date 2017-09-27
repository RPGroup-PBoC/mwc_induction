"""
Creation Date:
    20161026
Last Modified:
    20170215
Author(s):
    Griffin Chure
Purpose:
    This script generates the plot from the supplementary figure S3 showing
    that there is no effect of running the flow-cytometry samples in 'reverse
    mode'. Please see SI Section B2 for more information.

"""
# Import standard dependencies.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import re

# Custom written modules.
import mwc_induction_utils as mwc

# Set the plotting environments.
mwc.set_plotting_style()

# Load in all data files.
data_sets = glob.glob('../../data/2016*O2_*titration*')
with open('../../data/datasets_ignore.csv') as f:
    ignored_sets = f.readlines()
    ignored_sets = ['../../data/' + z.rstrip() for z in ignored_sets]
with open('../../data/reversed_plates_O2.csv') as f:
    control_sets = f.readlines()
    control_sets = ['../../data/' + z.rstrip() for z in control_sets]
forward_sets, reversed_sets = [], []
for entry in data_sets:
    if entry not in ignored_sets:
        samp = pd.read_csv(entry, comment='#')
        if 'r2' in entry:
            samp.insert(1, 'exp_run', 2)
        else:
            samp.insert(1, 'exp_run', 1)

        if entry in control_sets:
            reversed_sets.append(samp)
        else:
            forward_sets.append(samp)

# Add columns.
fwd = pd.concat(forward_sets, axis=0)
rev = pd.concat(reversed_sets, axis=0)
fwd.insert(np.shape(fwd)[1], 'reversed', 0)
rev.insert(np.shape(rev)[1], 'reversed', 1)

# Make the final data frame.
df = pd.concat([fwd, rev], axis=0)
df = df[df['rbs'] == 'RBS1027']

# Group the data frame by the important bits.
grouped = pd.groupby(df, ['reversed', 'date', 'exp_run', 'rbs'])

# Plot the fold change as a function of IPTG coloring the reverse runs.
plt.figure(figsize=(6, 6))
for group, data in grouped:
    if group[0] == 0:
        fwd, = plt.plot(data['IPTG_uM']/1E6, data['fold_change_A'], 'ko',
        markersize=8, alpha=0.75, label='forward')

    else:
        rev, = plt.plot(data['IPTG_uM']/1E6, data['fold_change_A'], 'ro',
        markersize=8, alpha=0.75, label='reverse')


# Do some formatting.
legend = plt.legend(handles=[fwd, rev], loc='upper left', fontsize=14,
                    title=r"""rep. / cell = 260
$\Delta\varepsilon_{RA} = -13.9$ $k_BT$""")
plt.setp(legend.get_title(), fontsize=15)
plt.xlabel('[IPTG] (M)', fontsize=20)
plt.ylabel('fold-change', fontsize=20)
plt.setp(legend.get_title(), fontsize=15)
plt.tick_params(labelsize=16)
plt.xscale('log')
plt.ylim([0, 1.1])
plt.tight_layout()
plt.xlim([1E-8, 1E-2])
ax = plt.gca()
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS8.pdf', bbox_inches='tight')

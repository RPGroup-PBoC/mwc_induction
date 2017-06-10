import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob
import sys
sys.path.insert(0, '../../analysis/')
import mwc_induction_utils as mwc
import seaborn as sns
mwc.set_plotting_style()

# Load the ignore data sets file.
with open('../../data/datasets_ignore.csv', 'r') as file:
    ignored_files = file.read().splitlines()
    for i, s in enumerate(ignored_files):
        ignored_files[i] = '../../data/' + s

# Set up the plotting environment.
operators = ['O3', 'O2', 'O1']
energies = {'O3': -9.7, 'O2':-13.9, 'O1': -17.0}
ops = len(operators)
colors = sns.color_palette('viridis', n_colors=len(operators))
color_key = {operators[i]: colors[i] for i, _ in enumerate(operators)}
plt.close('all')
fig, ax = plt.subplots(1, 1)
ax.plot([-100], [-100], 'k-', label='individual experiment')
ax.errorbar([-100], [-100], [-100], fmt='o', label='mean + standard error', color='k')

for o in operators:
    ax.plot([-100], [-100], '-o', color=color_key[o], label='{0}, {1} $k_BT$'.format(o, energies[o]))
ax.plot([-100], [-100], 'ro-', label='autofluorescence')
ax.legend(bbox_to_anchor=(0.4, 0.32), ncol=1, fontsize=12)
dfs = []
for i, op in enumerate(operators):
    # Grab all of the data files.
    data_files = glob.glob('../../data/*_{0}_*MACSQuant.csv'.format(op))

    # Loop through each file and parse .
    samples = []
    for j, g in enumerate(data_files):
        if g not in ignored_files:
            # Load in the file and only keep the
            # delta
            data = pd.read_csv(g, comment='#')
            data = data[(data['rbs'] == 'delta') | (data['rbs']=='auto')]
            dfs.append(data)
            # Get the IPTG values.
            data = data.sort_values('IPTG_uM')
            iptg = data['IPTG_uM'].unique()/1E6

            # Plot the fold change vs iptg.
            ax.plot(iptg, np.log(data[(data['rbs']=='delta')]['mean_YFP_A']), '-',
                    color=color_key[op], alpha=0.3, lw=1)

            ax.plot(iptg, np.log(data[(data['rbs']=='auto')]['mean_YFP_A']), '-',
                    color='r', alpha=0.2, lw=1)

# Concatenate the dataframes, group by operator and IPTG and compute
# the means.
op_data = pd.concat(dfs, axis=0)
grouped = op_data.groupby(['operator', 'rbs', 'IPTG_uM'])

# Plot the means.
for g, d in grouped:
    mean_val = np.mean(np.log(d['mean_YFP_A']))
    sem_val = np.std(np.log(d['mean_YFP_A'])) / np.sqrt(len(d))
    if g[1] == 'delta':
        ax.errorbar(g[2]/1E6, mean_val, yerr=sem_val, linestyle='none', color=color_key[g[0]], fmt='o', markersize=5)

auto_data = op_data[op_data['rbs']=='auto'].groupby('IPTG_uM')
for g, d in auto_data:
    mean_val = np.mean(np.log(d['mean_YFP_A']))
    sem_val = np.std(np.log(d['mean_YFP_A'])) / np.sqrt(len(d))
    ax.errorbar(g/1E6, mean_val, yerr=sem_val, linestyle='none', fmt='o', color='r', markersize=5)
ax.set_ylim([7.75, 11])
ax.set_xscale('log')
ax.set_xlabel('IPTG (Μ)', fontsize=14)
ax.set_ylabel('log YFP intensity (a.u.)', fontsize=14)
plt.tight_layout()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/figR1_iptg_influence.pdf', bbox_inches='tight')

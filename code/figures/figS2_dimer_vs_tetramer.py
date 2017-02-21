"""
Title:
    figS2_dimer_vs_tetramer.py
Author:
    Griffin Chure
Creation Date:
    20170216
Last Modified:
    20170216
Purpose:
    This script generates the plots shown in supplementary figure S2. It
    demonstrates the predicted behavior of an allosterically dependent
    repression in a dimeric vs tetrameric state.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Load in the data.
df = pd.read_csv('../../data/flow_master.csv', comment='#')
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

# Define the parameters.
operators = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7}
R_range = [1740, 1220, 260, 124, 60, 22]
IPTG_range = np.logspace(-8, -2, 500)
ka_n4 = 57.2E-6
ki_n4 = 3.5E-6
ep_ai = 4.5
ops = ['O1', 'O2', 'O3']
# Set the colors for the plot.
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]


# Plot the predictions in the tetrameric state.
fig, ax = plt.subplots(2, 3, figsize=(9, 6.2), sharey=True)
for i, R in enumerate(R_range):
    ax[0, -1].plot([], [], 'o', color=colors[i], label=R)

leg = ax[0, -1].legend(bbox_to_anchor=(1.9, 0.3), title="""repressors / cell
 #tetramer data""", fontsize=12)
leg.get_title().set_fontsize(12)
for i, op in enumerate(ops):
    for j, R in enumerate(R_range):
        R = np.array(R) / 2
        fc_dimer = mwc.fold_change_log(IPTG_range, -np.log(ka_n4),
                                       -np.log(ki_n4), ep_ai, R, operators[op],
                                       n=4)

        fc_tetramer = mwc.fold_change_log(IPTG_range, -np.log(ka_n4),
                                          -np.log(ki_n4), ep_ai, R,
                                          operators[op], n=2)

        ax[0, i].plot(IPTG_range, fc_dimer, '-', color=colors[j])
        ax[1, i].plot(IPTG_range, fc_tetramer, '-', color=colors[j])
    ax[0, i].set_xscale('log')
    ax[1, i].set_xscale('log')
    ax[1, i].set_xlabel('[IPTG] (M)')
    ax[0, i].set_xlabel('[IPTG] (M)')
    ax[0, 0].set_ylabel('fold-change')
    ax[1, 0].set_ylabel('fold-change')
    ax[0, 1].set_ylim([0, 1.1])
    ax[1, i].set_ylim([0, 1.1])
    ax[1, i].xaxis.set_ticks([1E-8, 1E-6, 1E-4, 1E-2])
    ax[0, i].xaxis.set_ticks([1E-8, 1E-6, 1E-4, 1E-2])
    for k in range(2):
        ax[k, i].set_title(r'%s $\Delta\varepsilon_{RA} = %s\, k_BT$' %(op, operators[op]), fontsize=16, position=(0.5,1.05), backgroundcolor='#ffedce')


# Now plot the data.
for i, op in enumerate(ops):
    for j, R in enumerate(R_range):
        data = df[(df['repressors'] == R/2) & (df['operator'] == op)]

        grouped_data = pd.groupby(data, ['IPTG_uM']).fold_change_A
        for group, d in grouped_data:
            mean_fluo = np.mean(d)
            std_fluo = np.std(d) / np.sqrt(len(d))
            for k in range(2):
                if k!=1:
                    alpha = 1
                    ax[k, i].plot(group/1E6, mean_fluo, 'o', color=colors[j], markersize=5, alpha=alpha)
                    ax[k, i].errorbar(group/1E6, mean_fluo, std_fluo, linestyle='none', color=colors[j], alpha=alpha)
plt.show()
plt.tight_layout()
plt.subplots_adjust(hspace=1.0)
plt.figtext(0, 1.02, 'A', fontsize=20)
plt.figtext(0, 0.5, 'B', fontsize=20)
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/dimer_vs_tetramer.pdf', bbox_inches='tight')

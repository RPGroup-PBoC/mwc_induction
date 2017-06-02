import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import pickle
import glob
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc
import pandas as pd
mwc.set_plotting_style()

# Load through all of the MCMC flatchains and extract the mode and HPD.
chains = np.sort(glob.glob('../../data/mcmc/SI_I_*.pkl'))
dfs = []
for i, c in enumerate(chains):
    with open(c, 'rb') as file:
        unpickler = pickle.Unpickler(file)
        flatchain = unpickler.load()
        flatlnprob = unpickler.load()
        max_idx = np.argmax(flatlnprob, axis=0)
        ea, ei, sigma = flatchain[max_idx]
        ka_mode = np.exp(-ea)
        ki_mode = np.exp(-ei)
        ka = np.exp(-flatchain[:, 0])
        ki = np.exp(-flatchain[:, 1])
        ka_hpd = mwc.hpd(ka, mass_frac=0.95)
        ki_hpd = mwc.hpd(ki, mass_frac=0.95)
    # Parse the file name for operator and repressor copy number.
    split = c.split('_')
    op = split[2]
    R = int(split[3].rstrip('.pkl')[1:])

    # Make a DataFrame
    df = pd.DataFrame([op, R, ka_mode, ki_mode, ka_hpd[0], ka_hpd[1],
                       ki_hpd[0], ki_hpd[1]]).T
    df.columns = ['operator', 'repressors', 'Ka_uM', 'Ki_uM', 'Ka_low',
                  'Ka_high', 'Ki_low', 'Ki_high']
    dfs.append(df)
df = pd.concat(dfs, ignore_index=True)
df = df.sort_values('repressors')
# Set up the colors and figure axis.
ops = df['operator'].unique()
colors = sns.color_palette('viridis', n_colors=len(ops))
for i, op in enumerate(ops):
    if i == 0:
        color_dict = {op: colors[i]}
    else:
        color_dict[op] = colors[i]
color_dict = {'O1': colors[0], 'O2': colors[1], 'O3': colors[2]}
# Group the dataframe.
grouped = df.groupby(['repressors', 'operator'])

# Set up the figure axis.
plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(8, 2), sharex=True)
for o in color_dict.keys():
    ax[0].plot([], [], 'o', markersize=5.5, markerfacecolor='w', markeredgecolor=color_dict[o], markeredgewidth=2, label=o)
ax[0].legend(loc='lower center', ncol=len(ops))

i = 1
j = 0
for g, d in grouped:
    ax[0].plot(i, d['Ka_uM'], 'o', markerfacecolor='w', markeredgecolor=color_dict[g[1]], markeredgewidth=2, ms=5.5, alpha=0.75)
    ax[0].vlines(i, d['Ka_low'], d['Ka_high'], color=color_dict[g[1]],
                 lw=1.5)

    ax[1].plot(i, d['Ki_uM'], 'o', markerfacecolor='w', markeredgecolor=color_dict[g[1]], markeredgewidth=2, ms=5.5, alpha=0.75)
    ax[1].vlines(i, d['Ki_low'], d['Ki_high'], color=color_dict[g[1]],
                 lw=1.5)
    j += 1
    if j % len(ops) == 0:
        i += 1
repressors = df['repressors'].unique()
sel_dat = df[(df['repressors'] == 260) & (df['operator']=='O2')]
ka_paper = sel_dat['Ka_uM']
ka_hpd = np.linspace(sel_dat['Ka_low'], sel_dat['Ka_high'], 1000)
ki_paper = sel_dat['Ki_uM']
ki_hpd = np.linspace(sel_dat['Ki_low'], sel_dat['Ki_high'], 1000)
x_vals = np.linspace(0, i, 1000)
ax[0].hlines(ka_paper, 0.8, len(repressors) + .2, linestyle='--', color='c',zorder=1)
ax[1].hlines(ki_paper, 0.8, len(repressors) + .2, linestyle='--', color='c',zorder=1)
ax[0].fill_between(np.linspace(0, i, 1000), sel_dat['Ka_low'].values[0], sel_dat['Ka_high'].values[0], color='c', alpha=0.4)
ax[1].fill_between(np.linspace(0, i, 1000), sel_dat['Ki_low'].values[0],
                   sel_dat['Ki_high'].values[0], color='c', alpha=0.4)

for j, a in enumerate(ax):
    a.set_xlim([1-.2, (i -1) + .2])
    a.set_xticks(np.arange(1, i, 1))
    a.set_xticklabels(np.sort(df['repressors'].unique()), fontsize=13)
    a.set_yscale('log')

ax[0].set_ylim([1E-3, 1E4])
ax[1].set_ylim([0.5E-3, 1.6E0])


# Set the labels.
ax[0].set_ylabel('$K_A\,\,(\mu\mathrm{M})$', fontsize=14)
ax[1].set_ylabel('$K_I\,\,(\mu\mathrm{M})$', fontsize=14)
ax[0].set_xlabel('repressors per cell', fontsize=14)
ax[1].set_xlabel('repressors per cell', fontsize=14)
plt.show()
plt.tight_layout()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/ka_ki_vals.svg', bbox_inches='tight')

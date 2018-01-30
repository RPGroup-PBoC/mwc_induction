import numpy as np
import mwc_induction_utils as mwc
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
mwc.set_plotting_style()
# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Set the parameters.
ep_ai = 4.5
ops = {'O1': -15, 'O2': -13.9, 'O3': -9.3}
reps = [22, 60, 124, 260, 1220, 1740]
c = colors[0:6].reverse()
reps.reverse()

Ka = 139
Ki = 0.53
c_range = np.logspace(-2, 4, 500)

# Load in the experimental data.
data = pd.read_csv('../../data/flow_master.csv')

# Take the fold-change for O2
fit_strain = data[(data['operator'] == 'O2') & (data['rbs'] == 'RBS1027')]
grouped = fit_strain.groupby('IPTG_uM')['fold_change_A'].mean()
#%% Set up the figure canvas.

fig, ax = plt.subplots(3, 1, figsize=(2, 5), sharex=True,
                       sharey=True)
for a in ax:
    a.set_ylim([-0.1, 1.1])
    a.set_yticks([0, 0.5, 1.0])
    a.set_xscale('log')
    a.set_xlim([1E-2, 8E3])
    a.set_xticks([1E-2, 1, 1E2, 1E4])

# Add the necessary labels.
ax[1].set_ylabel('fold-change')
ax[2].set_xlabel('[IPTG] (µM)')


# Plot the theory curves.
for i, o in enumerate(ops.keys()):
    for j, r in enumerate(reps):
        theo = mwc.fold_change_log(c_range, -np.log(Ka), -np.log(Ki), ep_ai, np.array([r]),
                                   ops[o])
        ax[i].plot(c_range, theo, color=colors[j], lw=1, label=r)


# Plot the O2 data.
_leg = ax[0].legend(title='rep. per cell', fontsize=8,
                    loc='upper left')
plt.setp(_leg.get_title(), fontsize=8)

_ = ax[1].plot(grouped, 'o', markerfacecolor='w', markeredgecolor='r',
               markeredgewidth=1, markersize=4)


plt.savefig('graphical_abstract.svg', bbox_inches='tight')


# %%

# compute the data collapse
F_range = np.linspace(-8, 9, 500)
theo = (1 + np.exp(-F_range))**-1
computed_bohr = mwc.bohr_fn(data, -np.log(Ka), -np.log(Ki))

fig, ax = plt.subplots(1, 1, figsize=(4, 2.5))
ax.set_ylabel('fold-change')
ax.set_yticks([0, 0.5, 1.0])
ax.set_xlabel('Bohr parameter ($k_BT$ units)')
ax.plot([], [], 'k-', label='theory')
ax.plot([], [], 'b.', label='experiment')
ax.legend(loc='upper left')
# Group the data and compute the bohr parameter.
titration_data = data[data['repressors'] != 0]

grouped = titration_data.groupby(['operator', 'repressors', 'IPTG_uM'])
for g, d in grouped:
    computed_bohr = mwc.bohr_fn(d, -np.log(Ka), -np.log(Ki))
    _ = ax.plot(np.mean(computed_bohr), d['fold_change_A'].mean(), '.')

_ = ax.plot(F_range, theo, 'k-')

plt.savefig('graphical_abstract_collapse.svg', bbox_inches='tight')

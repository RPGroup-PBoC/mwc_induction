import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mwc_induction_utils as mwc
import seaborn as sns
mwc.set_plotting_style()
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
sns.set_palette(colors)

# %matplotlib inline

# Load the growth curve data.
growth_data = pd.read_csv('../../data/20171011_steady_state_OD600.csv')
growth_data.loc[:, 'strain'] = growth_data['strain'].str.lower()

# Plot the growth curves for each strain.
grouped = growth_data.groupby(['strain'])

fig, ax = plt.subplots(1, 1)
colors = {'auto': 'g', 'delta': 'b', 'rbs1027': 'r'}
for g, d in grouped:
    _ = ax.plot(d['delta_t_min'], d['OD_600'] * d['dilution_factor'], 'o',
                color=colors[g], alpha=0.5)
growth_data.head()


# %%
# Load the flow data.
flow_data = pd.read_csv('../../data/20171011_O2_timeseries_MACSQuant.csv',
                        comment='#')

# Group by strain and plot the intensity values.
flow_grouped = flow_data.groupby(['strain'])
fig, ax = plt.subplots(3, 1, sharex=True)

_ = ax[0].set_title('growth curve', backgroundcolor='#FFEDCE')
_ = ax[1].set_title('raw fluorescence values', backgroundcolor='#FFEDCE')
_ = ax[2].set_title('fold-change', backgroundcolor='#FFEDCE')
_ = ax[0].set_ylabel(r'OD$_{600\,\mathrm{nm}}$')
_ = ax[2].set_xlabel('time from inoculation (hr)')
_ = ax[1].set_ylabel('fluorescence (a.u.)')
_ = ax[2].set_ylabel('fold-change')

for g, d in grouped:
    _ = ax[0].plot(d['delta_t_min'] / 60, d['OD_600'] * d['dilution_factor'], 'o',
                   color=colors[g], alpha=0.5)


for g, d in flow_grouped:
    ax[1].plot(d['delta_t'], d['FITC-A'], 'o', color=colors[g.lower()],
               alpha=0.5)
    if g.lower() == 'rbs1027':
        ax[2].plot(d['delta_t'], d['fold_change_A'],  'o', alpha=0.5,
                   markerfacecolor='w', markeredgecolor='r',
                   markeredgewidth=1.5)

ka = 139E-6
ki = 0.53E-6
numer = (1 + 50E-6 / ka)**2
denom = numer + np.exp(-4.5) * (1 + 50E-6 / ki)**2
pact = numer / denom
fc = (1 + pact * (260 / 4.6E6) * np.exp(13.9))**-1

ax[2].hlines(fc, 0, 13, 'r')
ax[2].set_ylim([0, 1])
ax[1].set_ylim([0, 40000])
ax[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0, 1.25])
ax[0].set_ylim([-0.01, 1.25])

for a in ax:
    a.vlines(8, a.get_ylim()[0], a.get_ylim()[
             1], color='slategray', zorder=1, alpha=0.5, linewidth=20)
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figSX_steady_state.pdf',
            bbox_inches='tight')
flow_data.head()

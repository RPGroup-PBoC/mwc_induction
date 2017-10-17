import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mwc_induction_utils as mwc
import seaborn as sns
import pymc3 as pm
import theano.tensor as tt
mwc.set_plotting_style()
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
sns.set_palette(colors)

# %matplotlib inline

# Load the growth curve data.
growth_data = pd.read_csv('../../data/20171013_steady_state_OD600.csv')
growth_data.loc[:, 'strain'] = growth_data['strain'].str.lower()

# Plot the growth curves for each strain.
grouped = growth_data.groupby(['strain'])
plt.close('all')
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
colors = {'auto': 'g', 'delta': 'b', 'rbs1027': 'r'}
labels = {'auto':  'autofluorescence', 'delta': '$\Delta$ lacI',
          'rbs1027': '$R=260$\n' + r'$\Delta\varepsilon_{RA}=-13.9\, k_BT$'}
for g, d in grouped:
    _ = ax[0].plot(d['delta_t_min'] / 60, d['OD_600'] * d['dilution_factor'], 'o',
                   color=colors[g], alpha=0.5, label=labels[g])
ax[0].set_yscale('log')
ax[0].legend(loc='upper left')

# Compute and plot the fold-change.
# Load the flow data.
flow_data = pd.read_csv('../../data/20171013_O2_timeseries_MACSQuant.csv',
                        comment='#')
flow_data = flow_data[flow_data['fold_change_A'] > 0]
# Group by strain and plot the intensity values.
flow_grouped = flow_data.groupby(['strain'])

for g, d in flow_grouped:
    if g.lower() == 'rbs1027':
        ax[1].plot(d['delta_t'], d['fold_change_A'],  'o', alpha=0.5,
                   markerfacecolor='w', markeredgecolor='r',
                   markeredgewidth=1.5)

for a in ax:
    a.set_xlabel('time (hr)')
    a.vlines(8, 0, 8, color='slategray',
             alpha=0.5, linewidth=20)
ax[0].set_title('spectrophotometry', backgroundcolor='#FFEDC0')
ax[1].set_title('flow cytometry', backgroundcolor='#FFEDC0')
ax[1].set_xlim([0, 13])
ax[1].set_ylim([0, 1])
ax[0].set_ylim([0, 4])
ax[0].set_ylabel('OD$_{600\mathrm{nm}}$')

ax[1].set_ylabel('fold-change')
# mwc.scale_plot(fig, 'one_row')
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figSX_example_growth_curve.pdf')

#%%


# %%
# Load the flow data.
    flow_data = pd.read_csv('../../data/20171013_O2_timeseries_MACSQuant.csv',
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
    _ = ax[0].plot(d['delta_t_min'] / 60, d['OD_600'] * d['dilution_factor'], 'd-',
                   color=colors[g], alpha=0.5)

# %%
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
ax[0].set_ylim([-0.01, 4.25])

for a in ax:
    a.vlines(8, a.get_ylim()[0], a.get_ylim()[
             1], color='slategray', zorder=1, alpha=0.5, linewidth=20)

mwc.scale_plot(fig, 'three_row')
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figSX_steady_state_growth.pdf',
            bbox_inches='tight')


# #####################################################################
# %%
# Nathan's idea of plotting Fluorescence v OD.
growth_data = pd.read_csv('../../data/20171013_steady_state_OD600.csv')
growth_data.loc[:, 'strain'] = growth_data['strain'].str.lower()
# Load the flow data.
flow_data = pd.read_csv('../../data/20171013_O2_timeseries_MACSQuant.csv',
                        comment='#')

flow_data.sort_values(by=['delta_t', 'replicate_no'], inplace=True)
growth_data.sort_values(by=['delta_t_min', 'replicate_number'], inplace=True)

# Set up a list of the strains for hacky-slicing.
strains = ['auto', 'delta', 'RBS1027']

plt.close('all')
fig, ax = plt.subplots(1, 1)
_ = ax.set_xlabel('OD$_{600\mathrm{nm}}$')
_ = ax.set_ylabel('fluorescence (a.u.)')

ax.plot([], [], 'bo', label='ΔlacI')
ax.plot([], [], 'go', label='autofluorescence')
ax.plot([], [], 'ro', label=r'$R=260$, $\Delta\varepsilon_{RA}=-13.9\,k_bT$')
ax.legend(loc='upper left')
for i, st in enumerate(strains):
    flow_slc = flow_data[flow_data['strain'] == st]
    growth_slc = growth_data[growth_data['strain'] == st.lower()]
    mean_od_vec = []
    mean_fluo_vec = []
    for j in range(0, len(flow_slc), 3):
        # growth_data.iloc[j:j + 3]
        ods = growth_slc.iloc[j:j + 3]['OD_600'] * \
            growth_slc.iloc[j:j + 3]['dilution_factor']
        mean_OD = np.mean(ods)
        sem_OD = np.std(ods) / np.sqrt(len(ods))

        fluos = flow_slc.iloc[j:j + 3]['FITC-A']
        mean_fluo = np.mean(fluos)
        fluo_sem = np.std(fluos) / np.sqrt(len(fluos))
        ax.errorbar(mean_OD, mean_fluo, yerr=fluo_sem, xerr=sem_OD,
                    fmt='o', color=colors[st.lower()], alpha=0.75)

# %% Now load the second data set and do the same thing.
growth_data = pd.read_csv('../../data/20171011_steady_state_OD600.csv')
growth_data.loc[:, 'strain'] = growth_data['strain'].str.lower()
# Load the flow data.
flow_data = pd.read_csv('../../data/20171011_O2_timeseries_MACSQuant.csv',
                        comment='#')
flow_data.sort_values(by=['delta_t', 'replicate_no'], inplace=True)
growth_data.sort_values(by=['delta_t_min', 'replicate_number'], inplace=True)

# Set up a list of the strains for hacky-slicing.
strains = ['auto', 'delta', 'RBS1027']
for i, st in enumerate(strains):
    flow_slc = flow_data[flow_data['strain'] == st]
    growth_slc = growth_data[growth_data['strain'] == st.lower()]
    for j in range(0, len(flow_slc), 3):
        ods = growth_slc.iloc[j:j + 3]['OD_600'] * \
            growth_slc.iloc[j:j + 3]['dilution_factor']
        mean_OD = np.mean(ods)
        sem_OD = np.std(ods) / np.sqrt(len(ods))
        fluos = flow_slc.iloc[j:j + 3]['FITC-A']
        mean_fluo = np.mean(fluos)
        fluo_sem = np.std(fluos) / np.sqrt(len(fluos))
        ax.errorbar(mean_OD, mean_fluo, yerr=fluo_sem, xerr=sem_OD,
                    fmt='D', markerfacecolor='w', markeredgecolor=colors[st.lower()], color=colors[st.lower()],
                    markeredgewidth=1, alpha=0.75,
                    linestyle='none')
ax.vlines(0.3, 2000, 35000, color='slategray', alpha=0.5,
          linewidth=20, zorder=1)

ax.set_ylim(2000, 35000)
ax.set_xlim([5E-4, 3.3])
ax.set_xscale('log')
# mwc.scale_plot(fig, 'single_plot_wide')
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figSX_fluo_v_od.pdf')

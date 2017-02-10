import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mwc_induction_utils as mwc
import pandas as pd
sns.set_context('paper')
mwc.set_plotting_style()
# Load in the datasets
data_a = pd.read_csv('../../data/figs9_part_A.csv')
data_b = pd.read_csv('../../data/figs9_part_b.csv')
data_O2 = pd.read_csv('../../data/flow_master.csv')
data_O2 = data_O2[(data_O2.operator == 'O2') & (data_O2.rbs=='RBS1027')]
plt.close('all')

fig, ax  = plt.subplots(1, 2)
# Make the plots for part A
ka = data_a[data_a.parameter=='logKA']
ki = data_a[data_a.parameter=='logKI']
ax[0].plot(ka.ep, ka.bestfit, '-', label=r'$\mathrm{log}\, \frac{K_A}{1\mathrm{M}}$')
ax[0].plot(ki.ep, ki.bestfit, '-', label=r'$\mathrm{log}\, \frac{K_A}{1\mathrm{M}}$')
ax[0].set_xlabel(r'allosteric parameter $\Delta\varepsilon_{AI}~(k_BT)$')
ax[0].set_ylabel(r'best-fit parameter value')
ax[0].legend(loc='lower left', fontsize=15)


IPTG_range = np.logspace(-8, -2, 500)
R = 260
ep_r = -13.9
# Define the function for the second plot.
def foldchange(c, R, ep_ai, ep_r, ep_a, ep_i):
    mwc_term = (1 + c / ep_a)**2 * (1 + np.exp(-ep_ai)) / ((1 + c / ep_a)**2 +
                np.exp(-ep_ai) * (1 + c / ep_i)**2)
    fc= (1 + mwc_term * (R / 4.6E6) * np.exp(-ep_r))**-1
    return fc


# Loop through the ep_ai in part b.
colors = sns.color_palette('viridis', n_colors=len(data_b))
for i in range(len(data_b)):
    ep_ai = data_b.iloc[i]['ep_ai']
    log_ka = np.exp(data_b.iloc[i]['log_ka'])
    log_ki = np.exp(data_b.iloc[i]['log_ki'])
    fc = foldchange(IPTG_range, R, ep_ai, ep_r, log_ka, log_ki)
    ax[1].plot(IPTG_range, fc, label=ep_ai, color=colors[i])

# plot the data.
grouped = pd.groupby(data_O2, ['IPTG_uM']).fold_change_A
for group, data in grouped:
    mean_fc = np.mean(data)
    mean_sem = np.std(data) / np.sqrt(len(data))
    ax[1].plot(group/1E6, mean_fc, 'o', color=colors[0])
    ax[1].errorbar(group/1E6, mean_fc, mean_sem, linestyle='none', color=colors[0])

ax[1].set_xscale('log')
ax[1].set_xlabel('[IPTG] (M)')
ax[1].set_ylabel('fold-change')
leg = ax[1].legend(loc='upper left', title=r"""allosteric parameter
    $\Delta\varepsilon_{AI}\, (k_BT)$""", fontsize=15)
leg.get_title().set_fontsize(18)

plt.show()

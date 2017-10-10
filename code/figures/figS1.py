import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Import the project utils
import sys
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc
mwc.set_plotting_style()


# Define the necessary functions:
def foldchange(c, R, ep_ai, ep_r, ep_a, ep_i):
    mwc_term = (1 + c / ep_a)**2 * (1 + np.exp(-ep_ai)) / ((1 + c / ep_a)**2 +
                                                           np.exp(-ep_ai) * (1 + c / ep_i)**2)
    fc = (1 + mwc_term * (R / 4.6E6) * np.exp(-ep_r))**-1
    return fc


# Load in the datasets
data_a = pd.read_csv('../../data/figs1_part_A.csv')
data_b = pd.read_csv('../../data/figs1_part_b.csv')
data_O2 = pd.read_csv('../../data/flow_master.csv')
data_O2 = data_O2[(data_O2.operator == 'O2') & (data_O2.rbs == 'RBS1027')]
plt.close('all')


# Define necessary parameters.
ka = data_a[data_a.parameter == 'logKA']
ki = data_a[data_a.parameter == 'logKI']
IPTG_range = np.logspace(-8, -2, 500)
R = 260
ep_r = -13.9
colors = sns.color_palette('viridis', n_colors=len(data_b))

# Define the figure axis and labels.
fig, ax = plt.subplots(1, 2, figsize=(4.5, 2.25))
_ = ax[0].set_xlabel(r'allosteric parameter $\Delta\varepsilon_{AI}\,(k_BT)$')
_ = ax[0].set_ylabel(r'best-fit parameter value')
_ = ax[1].set_xscale('log')
_ = ax[1].set_xlabel('[IPTG] (M)')
_ = ax[1].set_ylabel('fold-change')

# Plots for panel (A)
_ = ax[0].plot(ka.ep, ka.bestfit, '-',
               label=r'$\mathrm{log}\, \frac{K_A}{1\mathrm{M}}$')
_ = ax[0].plot(ki.ep, ki.bestfit, '-',
               label=r'$\mathrm{log}\, \frac{K_A}{1\mathrm{M}}$')

# Plots for panel (B)
# Plot the curves
for i in range(len(data_b)):
    ep_ai = data_b.iloc[i]['ep_ai']
    log_ka = np.exp(data_b.iloc[i]['log_ka'])
    log_ki = np.exp(data_b.iloc[i]['log_ki'])
    fc = foldchange(IPTG_range, R, ep_ai, ep_r, log_ka, log_ki)
    _ = ax[1].plot(IPTG_range, fc, label=ep_ai, color=colors[i])

# plot the data.
grouped = data_O2.groupby(['IPTG_uM']).fold_change_A
for group, data in grouped:
    mean_fc = np.mean(data)
    mean_sem = np.std(data) / np.sqrt(len(data))
    _ = ax[1].errorbar(group / 1E6, mean_fc, mean_sem,
                       linestyle='none', color=colors[0], fmt='o')


# Add the legends and labels.
_ = ax[0].legend(loc='lower left')
leg = ax[1].legend(loc='upper left', title=r"""allosteric parameter
       $\Delta\varepsilon_{AI}$ $(k_BT)$""", bbox_to_anchor=(1, 1))
leg.get_title().set_fontsize(8)
fig.text(-0.025, 0.92, '(A)', fontsize=15)
fig.text(0.5, 0.92, '(B)', fontsize=15)

# Format and save the figure.
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS1.pdf', bbox_inches='tight')

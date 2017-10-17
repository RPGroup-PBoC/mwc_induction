# Import numerical and plotting modules/functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.optimize import fsolve

# Import the project utils
import sys
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc
mwc.set_plotting_style()
colors = sns.color_palette('colorblind').as_hex()
colors[4] = sns.xkcd_palette(['dusty purple']).as_hex()[0]
sns.set_palette(colors)


# Define functions to be used in figure
def fugacity_leakiness(R, Ns, e_s, e_AI=4.5, Nc=0, e_c=0):
    '''
    Solves for the leakiness of a simple repression construct with
    multiple promoter copies (Ns, with energy e_s) or competitor sites
    (Nc, with energy e_c).
    Parameters
    ----------
    R : float
        Number of repressors per cell
    e_AI : float
        Energetic difference between the active and inactive state
    Ns : float
        Number of specific operators available for repressor binding
    Nc : float
        Number of competitor operators available for repressor binding
    e_s : float
        Binding energy between specific operator and repressor as inferred in
        Garcia 2011
    e_c : float
        Binding energy between competitor operator and repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    leakiness
    '''
    NNS = 4.6E6
    p_A = 1 / (1 + np.exp(-e_AI))
    # Reff = R * p_A
    leakiness = []
    for r in R:
        Reff = r * p_A

        def func(x): return -Reff + Ns * (x * np.exp(-e_s)) / (1 + x *
                                                               np.exp(-e_s)) +\
            NNS * (x) / (1 + x) + \
            Nc * (x * np.exp(-e_c)) / (1 + x * np.exp(-e_c))
        lam = fsolve(func, 0)
        leakiness.append(1 / (1 + lam * np.exp(-(e_s))))
    return np.array(leakiness)


# Set parameter values
reps = np.logspace(0, 3, 100)
op_H = -15.3
N = 10
N_vals = [64, 52, 10]
ops = [-15.3, -15.3, -17.0]
e_AI_vals = [-4, -2, 0, 2, 4]

# Load data from Brewster et al. 2014
data_file = '../../data/tidy_lacI_multiple_operator_data.csv'
df = pd.read_csv(data_file)

# Plot figures
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 2.5))

for i, val in enumerate(e_AI_vals):
    op_true = op_H - np.log(1 + np.exp(-val))
    ax1.plot(reps, fugacity_leakiness(reps, N, op_true, e_AI=val),
             label=val, color=colors[i])
    x_coord = N * (1 + np.exp(-val))
    y_coord = fugacity_leakiness([x_coord], N, op_true, e_AI=val)
    _ = ax1.plot(x_coord, y_coord, marker='^', fillstyle='full',
                 markerfacecolor='white', markeredgecolor=colors[i],
                 markeredgewidth=1.5, zorder=(i + 5))
    _ = ax1.plot([x_coord, x_coord], [0, y_coord], '--', color=colors[i])


for j, val in enumerate(N_vals):
    energy = df.energy[df.N == val].unique()[0]
    R = np.array(df.repressor[df.N == val])
    fc = np.array(df.fold_change[df.N == val])
    _ = ax2.plot(R, fc, 'o', color=colors[j], label=None)
    _ = ax2.plot(reps, fugacity_leakiness(reps, val, ops[j], e_AI=4.5),
                 label=('%s, %s' % (str(ops[j]), val)), color=colors[j])
# Make labels
title_dict = {ax1: r'$\Delta \varepsilon_{AI}\ (k_BT)$',
              ax2: r'$\Delta \varepsilon_{RA}\ (k_BT),\ N$'}
for ax in (ax1, ax2):
    _ = ax.set_xscale('log')
    _ = ax.set_yscale('log')
    _ = ax.set_xlabel('repressors/cell', fontsize=8)
    _ = ax.set_ylabel('fold-change', fontsize=8)
    _ = ax.set_xlim(0, 1000)
    _ = ax.set_ylim(1E-4, 1)
    ax.tick_params(labelsize=6)
    leg = ax.legend(loc='lower left', title=title_dict[ax], fontsize=7)
    leg.get_title().set_fontsize(7)

# Add panel labels.
plt.figtext(0.005, 0.95, '(A)', fontsize=12)
plt.figtext(0.5, 0.95, '(B)', fontsize=12)

# Format and save
plt.tight_layout()
mwc.scale_plot(fig, 'one_row')
plt.savefig('../../figures/SI_figs/figS2.pdf', bbox_inches='tight')

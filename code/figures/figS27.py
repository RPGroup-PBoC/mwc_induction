import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines

import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Functions for calculating fold change, EC50, and effective Hill coefficient


def pact(IPTG, K_A, K_I, e_AI):
    '''
    Computes the probability that a repressor is active
    Parameters
    ----------
    IPTG : array-like
        Array of IPTG concentrations in uM
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    probability that repressor is active
    '''
    pact = (1 + IPTG * 1 / K_A)**2 / \
        (((1 + IPTG * 1 / K_A))**2 + np.exp(-e_AI) * (1 + IPTG * 1 / K_I)**2)
    return pact


def fold_change(IPTG, K_A, K_I, e_AI, R, Op):
    '''
    Computes fold-change for simple repression
    Parameters
    ----------
    IPTG : array-like
        Array of IPTG concentrations in uM
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    R : float
        Number of repressors per cell
    Op : float
        Operator binding energy
    Returns
    -------
    probability that repressor is active
    '''
    return 1 / (1 + R / 5E6 * pact(IPTG, K_A, K_I, e_AI) * np.exp(-Op))


def EC50(K_A, K_I, e_AI, R, Op):
    '''
    Computes the concentration at which half of the repressors are in the active state
    Parameters
    ----------
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    Concentration at which half of repressors are active (EC50)
    '''
    t = 1 + (R / 5E6) * np.exp(-Op) + (K_A / K_I)**2 * \
        (2 * np.exp(-e_AI) + 1 + (R / 5E6) * np.exp(-Op))
    b = 2 * (1 + (R / 5E6) * np.exp(-Op)) + \
        np.exp(-e_AI) + (K_A / K_I)**2 * np.exp(-e_AI)
    return K_A * ((K_A / K_I - 1) / (K_A / K_I - (t / b)**(1 / 2)) - 1)


def effective_Hill(K_A, K_I, e_AI, R, Op):
    '''
    Computes the effective Hill coefficient
    Parameters
    ----------
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    effective Hill coefficient
    '''
    c = EC50(K_A, K_I, e_AI, R, Op)
    return 2 / (fold_change(c, K_A, K_I, e_AI, R, Op) - fold_change(0, K_A, K_I, e_AI, R, Op)) *\
        (-(fold_change(c, K_A, K_I, e_AI, R, Op))**2 * R / 5E6 * np.exp(-Op) *
         2 * c * np.exp(-e_AI) * (1 / K_A * (1 + c / K_A) * (1 + c / K_I)**2 - 1 / K_I *
                                  (1 + c / K_A)**2 * (1 + c / K_I)) / ((1 + c / K_A)**2 + np.exp(-e_AI) * (1 + c / K_I)**2)**2)


# Define parameters
K_A = 139
K_I = 0.53
e_AI = 4.5
ops_range = np.linspace(-18, -8, 50)
Reps = np.array([1740, 1220, 260, 124, 60, 22])
O1 = -15.3
O2 = -13.9
O3 = -9.7
ops = [O1, O2, O3]
markers = ['o', 'D', 's']
names = ['O1', 'O2', 'O3']
op_dict = dict(zip(ops, markers))

# Set color palette
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 4))

for i in range(len(Reps)):
    ax[0].semilogy(ops_range, EC50(K_A, K_I, e_AI, Reps[i],
                                   ops_range) * 1E-6, color=colors[i], label=Reps[i])
    ax[1].plot(ops_range, effective_Hill(
        K_A, K_I, e_AI, Reps[i], ops_range), color=colors[i])
    for op in ops:
        ax[0].semilogy(op, EC50(K_A, K_I, e_AI, Reps[i], op) * 1E-6, op_dict[op],
                       markerfacecolor='white', markeredgecolor=colors[i], markeredgewidth=1, markersize=5)
        ax[1].plot(op, effective_Hill(K_A, K_I, e_AI, Reps[i], op), op_dict[op],
                   markerfacecolor='white', markeredgecolor=colors[i], markeredgewidth=1, markersize=5)

# Plot legend
ims = []
for i in range(3):
    ims.append(mlines.Line2D([], [], color='gray',
                             marker=markers[i], label=names[i], linestyle='None'))
ax[0].add_artist(ax[0].legend(handles=ims, ncol=3, loc='upper left',
                              handletextpad=0, columnspacing=0.5, fontsize=8))
leg = ax[0].legend(title='rep./cell')
leg.get_title().set_fontsize(15)

# Set axes properties
ax[0].set_ylim(3.5E-6, 1E-3)
ax[0].set_xlabel(r'binding energy $\Delta \varepsilon_{AI}\ (k_BT)$')
ax[1].set_xlabel(r'binding energy $\Delta \varepsilon_{AI}\ (k_BT)$')
ax[0].set_ylabel(r'[$EC_{50}$]')
ax[1].set_ylabel('effective Hill coefficient')

plt.figtext(0.0, 0.95, '(A)', fontsize=12)
plt.figtext(0.5, 0.95, '(B)', fontsize=12)
mwc.scale_plot(fig, 'one_row')
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS27.pdf', bbox_inches='tight')

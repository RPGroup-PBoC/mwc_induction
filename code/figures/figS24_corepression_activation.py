"""
Title: figS24_corepression_activation.py
Author: Griffin Chure
Last Modified: 2016-02-14
Purpose: This script generates the predicted titration curves shown in figure
         S24 for corepression and activation.
"""

import numpy as np
import matplotlib.pyplot as plt
import mwc_induction_utils as mwc
import seaborn as sns


# Define the fold-change functions for corepression and activation
def fc_repression(c, R, ep_ra, ep_ai, k_a, k_i, n_ns=4.6E6):
    """
    Computes the fold-change equation. See SI section "Applications"
    for more information. This function assumes two allosterically independent
    corepressor binding sites.

    Parameters
    ----------
    c : 1d-array
        Range of corepressor concentrations over which to evaluate the
        fold-change. This should be in the same units as k_a and k_i.
    R : int
        Number of repressor molecules per cell.
    ep_ra : float
        Binding energy of the repressor to the DNA. Should be in units of
        k_BT.
    ep_ai : float
        Allosteric energy between the active and inactive states of the
        repressor.
    k_a : float
        Dissociation constant of the corepressor to the active repressor
        molecule. Should be in the same units as c.
    k_i : float
        Dissociation constant of the corepressor to the inactive repressor
        molecule. Should be in teh same units as c.
    n_ns: int, optional
        Number of nonspecific binding sites available to the repressor. The
        default value is 4.6x10^6, the length of the E. coli genome in
        base pairs.

    Returns
    -------
    fold_change : 1d-array
        Theoretical fold-change values at each value provided in c.
    """
    # Compute the allosteric term.
    p_act = ((1 + c / k_a)**2 / ((1 + c / k_a)**2 +
                                 np.exp(ep_ai)*(1 + c / k_i)**2))
    return (1 + p_act * (R / n_ns) * np.exp(-ep_ra))**-1


def fc_activation(c, A, ep_aa, ep_ai, ep_pa, k_a, k_i, n_ns=4.6E6):
    """
    Computes the fold-change equation. See SI section "Applications"
    for more information. This function assumes two allosterically independent
    corepressor binding sites.

    Parameters
    ----------
    c : 1d-array
        Range of coactivator concentrations over which to evaluate the
        fold-change. This should be in the same units as k_a and k_i.
    A : int
        Number of activator molecules per cell.
    ep_aa : float
        Binding energy of the activator to the DNA. Should be in units of
        k_BT.
    ep_ai : float
        Allosteric energy between the active and inactive states of the
        activator.
    ep_pa : float
        Interaction energy between the activator and the polymerase.
    k_a : float
        Dissociation constant of the coactivator to the active activator
        molecule. Should be in the same units as c.
    k_i : float
        Dissociation constant of the corepressor to the inactive activator
        molecule. Should be in teh same units as c.
    n_ns: int, optional
        Number of nonspecific binding sites available to the activator. The
        default value is 4.6x10^6, the length of the E. coli genome in
        base pairs.

    Returns
    -------
    fold_change : 1d-array
        Theoretical fold-change values at each value provided in c.
    """
    # Compute the allosteric term
    p_act = (1 + c / k_a)**2 / ((1 + c / k_a)**2 + np.exp(ep_ai) *
                                (1 + c / k_i)**2)

    # Compute the fold-change.
    numerator = 1 + p_act * (A / n_ns) * np.exp(-ep_aa) * np.exp(-ep_pa)
    denominator = 1 + p_act * (A / n_ns) * np.exp(-ep_aa)
    return numerator / denominator


# Define the parameters used for the corepression case.
c_range = np.logspace(-10, 0, 500)  # In units of M.
R_range = [1, 10, 100, 200, 500, 1000]
ep_ra_range = [-8, -10, -12, -15, -18, -20]
ep_ai = 5
k_a = 50E-9
k_i = 200E-6

# Set the colors
viridis = sns.color_palette('viridis', n_colors=6)
magma = sns.color_palette('magma', n_colors=6)


# Instantiate the figure
fig, ax = plt.subplots(1, 2, figsize=(9,5), sharey=True)

# Plot the predictions from varying R.
for i, R in enumerate(R_range):
    fold_change = fc_repression(c_range, R, ep_ra_range[2], ep_ai, k_a, k_i)
    ax[0].plot(c_range / k_a, fold_change, '-', color=viridis[i], label=R)

a0_leg = ax[0].legend(bbox_to_anchor=(0.90, -.20), title='repressors / cell', ncol=3)
a0_leg.get_title().set_fontsize(15)

# Compute the corepression with varying ep_ra
for i, ep in enumerate(ep_ra_range):
    fold_change = fc_repression(c_range, R_range[2], ep, ep_ai, k_a, k_i)
    ax[1].plot(c_range / k_a, fold_change, '-', color=magma[i], label=ep)
a1_leg = ax[1].legend(bbox_to_anchor=(0.95, -.20),
                      title=r'$\Delta\varepsilon_{R}^*\,(k_BT)$', ncol=3)
a1_leg.get_title().set_fontsize(15)


# Format the axes.
min_c = np.min(c_range / k_a)
max_c = np.max(c_range / k_a)
ax[0].set_xlabel(r'$\frac{c}{K_A}$', fontsize=22)
ax[1].set_xlabel(r'$\frac{c}{K_A}$', fontsize=22)
ax[0].set_ylabel('fold-change')
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_ylim([0, 1.1])
ax[1].set_ylim([0, 1.1])
ax[0].set_xlim([min_c, max_c])
ax[1].set_xlim([min_c, max_c])
ax[0].set_xticks([1E-2, 1, 1E2, 1E4, 1E6])
ax[1].set_xticks([1E-2, 1, 1E2, 1E4, 1E6])
ax[0].tick_params(labelsize=17)
ax[1].tick_params(labelsize=17)
plt.tight_layout()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/corepression_predictions.pdf', bbox_inches='tight')


# Define the parameters for the activation case.
A_range = [1, 10, 100, 200, 500, 1000]
ep_aa_range = [-8, -10, -12, -15, -18, -20]
ep_pa = -3

# Instantiate the figure
fig, ax = plt.subplots(1, 2, figsize=(9, 5), sharey=True)

# Compute the corepression with varying A
for i, A in enumerate(A_range):
    fold_change = fc_activation(c_range, A, ep_aa_range[2], ep_ai, ep_pa,
                                  k_a, k_i)
    ax[0].plot(c_range / k_a, fold_change, '-', color=viridis[i], label=A)
leg0 = ax[0].legend(bbox_to_anchor=(0.5, -.215), loc='upper center',
                    title='activators / cell', ncol=3)
leg0.get_title().set_fontsize(15)

# Compute the corepression with varying ep_aa
for i, ep in enumerate(ep_aa_range):
    fold_change = fc_activation(c_range, A_range[2], ep, ep_ai, ep_pa, k_a,
                                  k_i)
    ax[1].plot(c_range / k_a, fold_change, '-', color=magma[i], label=ep)
leg1 = ax[1].legend(bbox_to_anchor=(0.5, -.21), loc='upper center',
                    title=r'$\Delta\varepsilon_{A}^*\,(k_BT)$', ncol=3)
leg1.get_title().set_fontsize(15)


# Format the axes.
ax[0].set_xlabel(r'$\frac{c}{K_A}$', fontsize=22)
ax[1].set_xlabel(r'$\frac{c}{K_A}$', fontsize=22)
ax[0].set_ylabel('fold-change')
ax[0].set_xscale('log')
ax[1].set_xscale('log')
ax[0].set_xlim([min_c, max_c])
ax[1].set_xlim([min_c, max_c])
ax[0].set_ylim([0, 22])
ax[1].set_ylim([0, 22])
ax[0].set_xticks([1E-2, 1, 1E2, 1E4, 1E6])
ax[1].set_xticks([1E-2, 1, 1E2, 1E4, 1E6])
ax[0].tick_params(labelsize=17)
ax[1].tick_params(labelsize=17)

plt.tight_layout()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/activation_predictions.pdf', bbox_inches='tight')

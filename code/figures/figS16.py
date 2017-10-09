
# (c) 2017 the authors. This work is licensed under a [Creative Commons
# Attribution License CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
# All code contained herein is licensed under an [MIT
# license](https://opensource.org/licenses/MIT).

# For operating system interaction
import re

# For loading .pkl files.
import pickle

# For scientific computing
import numpy as np
import pandas as pd
import scipy.special

# Import custom utilities
import mwc_induction_utils as mwc

# Useful plotting libraries
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
mwc.set_plotting_style()


# Variability  in fold-change as parameter change.

def fold_change_oo(Ka, Ki, R, era, eai=4.5, Nns=4.6E6):
    '''
    computes the gene expression fold change for a simple repression architecture
    in the limit where the inducer concentration goes to infinity
    Parameters
    ----------
    Ka, Ki : float.
        Dissociation constants of the ligand to the active and inactive state
        of the repressor respectively.
    R : float.
        Mean repressor copy number per cell
    era : float.
        Repressor-DNA binding energy
    eai : float.
        Energy difference between active and inactive state of repressor
    Nns : float.
        Number of non-specific binding sites.
    Returns
    -------
    fold-change
    '''
    return (1 + 1 / (1 + np.exp(-eai) * (Ka / Ki)**2) * R / Nns *
            np.exp(-era))**-1


# Let us now define the numerical values for all the needed parameters
era_num = np.array([-15.3, -13.9, -9.7])  # kBT
Ka_num = 139.96  # µM
Ki_num = 0.54  # µM


# Let's now plot the change in fold-change as $K_A$ and $K_I$ vary for
# different energies and repressor copy numbers.

# Factor by which the Ka and Ki are varied
factor = 2

Ka_array = np.logspace(np.log10(Ka_num / factor),
                       np.log10(Ka_num * factor), 100)
Ki_array = np.logspace(np.log10(Ki_num / factor),
                       np.log10(Ki_num * factor), 100)


# Initialize plot
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax = ax.ravel()

# Fixed R, variable ∆e_RA

# Loopt through binding energies
colors = sns.color_palette('Oranges', n_colors=4)[::-1]
rep = 260  # repressors per cell
for i, eRA in enumerate(era_num):
    # compute the ∆fold-change_Ka
    delta_fc = fold_change_oo(Ka_num, Ki_num, rep, eRA) - \
               fold_change_oo(Ka_array, Ki_num, rep, eRA)

    ax[0].plot(np.log10(Ka_array / Ka_num), delta_fc,
               label=r'{:.1f}'.format(eRA), color=colors[i])

    # compute the ∆fold-change_KI
    delta_fc = fold_change_oo(Ka_num, Ki_num, rep, eRA) - \
        fold_change_oo(Ka_num, Ki_array, rep, eRA)

    ax[1].plot(np.log10(Ki_array / Ki_num), delta_fc,
               label=r'{:.1f}'.format(eRA), color=colors[i])

# Format Ka plot
ax[0].set_xlabel(r'$\log_{10} \frac{K_A}{K_A^{fit}}$')
ax[0].set_ylabel(r'$\Delta$fold-change$_{K_A}$')
ax[0].margins(0.01)

# Format Ki plot
ax[1].set_xlabel(r'$\log_{10} \frac{K_I}{K_I^{fit}}$')
ax[1].set_ylabel(r'$\Delta$fold-change$_{K_I}$')
ax[1].margins(0.01)
ax[1].legend(loc='center left', title=r'$\Delta\varepsilon_{RA}$ ($k_BT$)',
             ncol=1, fontsize=11, bbox_to_anchor=(1, 0.5))

# Fixed R, variable ∆e_RA

# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

repressors = [22, 60, 124, 260, 1220, 1740][::-1]
eRA = -15.3
for i, rep in  enumerate(repressors):
# compute the ∆fold-change_Ka
    delta_fc = fold_change_oo(Ka_num, Ki_num, rep, eRA) - \
        fold_change_oo(Ka_array, Ki_num, rep, eRA)

    ax[2].plot(np.log10(Ka_array / Ka_num), delta_fc,
               label=str(rep), color=colors[i])

    # compute the ∆fold-change_KI
    delta_fc = fold_change_oo(Ka_num, Ki_num, rep, eRA) - \
        fold_change_oo(Ka_num, Ki_array, rep, eRA)

    ax[3].plot(np.log10(Ki_array / Ki_num), delta_fc,
               label=str(rep), color=colors[i])

# Format Ka plot
ax[2].set_xlabel(r'$\log_{10} \frac{K_A}{K_A^{fit}}$')
ax[2].set_ylabel(r'$\Delta$fold-change$_{K_A}$')
ax[2].margins(0.01)

# # Format Ki plot
ax[3].set_xlabel(r'$\log_{10} \frac{K_I}{K_I^{fit}}$')
ax[3].set_ylabel(r'$\Delta$fold-change$_{K_I}$')
ax[3].margins(0.01)
ax[3].legend(loc='center left', title=r'repressors / cell', ncol=1,
             fontsize=11, bbox_to_anchor=(1, 0.5))


# Label plot
plt.figtext(0.0, .95, '(A)', fontsize=20)
plt.figtext(0.50, .95, '(B)', fontsize=20)
plt.figtext(0.0, .46, '(C)', fontsize=20)
plt.figtext(0.50, .46, '(D)', fontsize=20)

plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS16.pdf', bbox_inches='tight')

# %%
# Importing libraries

import os
import glob
import pickle
import re

# Our numerical workhorses
import numpy as np
import pandas as pd

# Import the project utils
import sys
sys.path.insert(0, '../../analysis/')
import mwc_induction_utils as mwc

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

mwc.set_plotting_style()


# %% Define function to compute the fold-change accounting for multiple
# promoters
def fold_change_double_R_adjust(R, iptg, epsilon_r, f,
                                ea=-np.log(139), ei=-np.log(0.53), epsilon=4.5,
                                Ns=2, Nns=4.6E6):
    '''
    Computes the fold-change for a promoter that spends f fraction of the time
    with two copies of the promoter and (1-f) with one promoter.
    Parameters
    ----------
    R : array-like.
        Number of repressors per cell.
    iptg : array-like.
        Inducer concentration.
    epsilon_r : float.
        Repressor-DNA binding energy.
    f : float.
        Fraction of the cell cycle that the cell spends with multiple copies of
        the promoter.
    ea, ei : float.
        -log(Ka) and -log(Ki) respectively.
    epsilon : float.
        Energy difference between active and inactive state of repressor.
    Ns : int.
        Number of promoters that the cell has f fraction of the time.
    Nns : int.
        Number of non-specific binding sites

    Returns
    -------
    fold_change : array-like.
        fold-change in gene expression.
    '''
    # Compute the number of active repressors
    Ract = mwc.pact_log(iptg=iptg, ea=ea, ei=ei, epsilon=epsilon) * R

    Reff = Ract / (1 + f)

    # Define the coefficients of the polynomial
    a = -2 * Nns * np.exp(-epsilon_r)
    b = np.exp(-epsilon_r) * (2 * Reff - Ns) - 2 * Nns
    c = 2 * Reff

    # Initialize array to save
    lam = np.empty_like(Ract)
    # Find the value of lambda
    for i, r in enumerate(Ract):
        lam[i] = np.max(np.roots([a, b[i], c[i]]))

    # Compute and return fold-change
    single_fc = 1 / (1 + Reff / Nns * np.exp(-epsilon_r))
    double_fc = 1 / (1 + lam * np.exp(-epsilon_r))
    fold_change = 1 / (1 + f) * ((1 - f) * single_fc + 2 * f * double_fc)

    return fold_change

# %%
# Read the data
# Define working directory
datadir = '../../../data/'
# List files to be read
files = ['flow_master.csv', 'merged_Oid_data_foldchange.csv']
# Read flow cytometry data
df_Oid = pd.read_csv(datadir + files[1], comment='#')
# make an extra column to have consistent labeling
df_Oid['fold_change_A'] = df_Oid.fold_change
# Remove manually the outlier with an unphysical fold-change
df_Oid = df_Oid[df_Oid.fold_change_A <= 1]
# Read the flow cytometry data
df = pd.read_csv(datadir + files[0], comment='#')
# Attach both data frames into a single one
df = pd.concat([df, df_Oid])
# Drop rows containing NA values
df.dropna(axis=1, inplace=True)

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

# %%
# Load MCMC flatchain

# Load the flat-chain
with open('../../../data/mcmc/main_text_KaKi.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]

ka_fc = np.exp(-gauss_flatchain[:, 0])
ki_fc = np.exp(-gauss_flatchain[:, 1])

# %%
# Plot the theory vs data for all 4 operators with the credible region

# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-7, -2, 100)
IPTG_lin = np.array([0, 1E-7])

# Define parameters for the multi-promoter model
f = 1 / 3
Ns = 2
# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3', 'Oid']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17.0}

# Initialize the plot to set the size
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax = ax.ravel()

# Loop through operators
for i, op in enumerate(operators):
    print(op)
    data = df[df.operator == op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # Check if the RBS was measured for this operator
        if rbs in data.rbs.unique():
            # plot the theory using the parameters from the fit.
            # SINGLE PROMOTER
            # Log scale
            ax[i].plot(IPTG, mwc.fold_change_log(IPTG * 1E6,
                                                 ea=ea,
                                                 ei=ei,
                                                 epsilon=4.5,
                                R=df[(df.rbs == rbs)].repressors.unique(),
                                                 epsilon_r=energies[op]),
                       color=colors[j])
            # Linear scale
            ax[i].plot(IPTG_lin, mwc.fold_change_log(IPTG_lin * 1E6,
                       ea=ea, ei=ei, epsilon=4.5,
                                R=df[(df.rbs == rbs)].repressors.unique(),
                epsilon_r=energies[op]),
                color=colors[j], linestyle='--')
            # MULTIPLE PROMOTERS
            # Log scale
            ax[i].plot(IPTG, fold_change_double_R_adjust(iptg=IPTG * 1E6, f=f,
                                                         Ns=Ns, ea=ea, ei=ei,
                                                         epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique() * 2,
            epsilon_r=energies[op]),
                       color=colors[j], linestyle=':')
            # Linear scale
            ax[i].plot(IPTG_lin, fold_change_double_R_adjust(iptg=IPTG_lin *
                                                             1E6, f=f, Ns=Ns,
                                                             ea=ea, ei=ei,
                                                             epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique() * 2,
            epsilon_r=energies[op]),
                color=colors[j], linestyle='--')
        # MULTIPLE PROMOTERES

        # Plot mean and standard error of the mean for the flow data
        if op != 'Oid':
            # compute the mean value for each concentration
            fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
            # compute the standard error of the mean
            fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
            np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())

            # plot the experimental data
            ax[i].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
                yerr=fc_err, fmt='o',
                label=df[df.rbs==rbs].repressors.unique()[0] * 2,
                color=colors[j])
        # Plot the raw data for Oid
        else:
            ax[i].plot(data[data.rbs==rbs].IPTG_uM / 1E6,
                    data[data.rbs==rbs].fold_change_A, marker='o', lw=0,
                    color=colors[j])

    # Add operator and binding energy labels.
    ax[i].text(0.8, 0.09, r'{0}'.format(op), transform=ax[i].transAxes,
            fontsize=13)
    ax[i].text(0.67, 0.02,
            r'$\Delta\varepsilon_{RA} = %s\,k_BT$' %energies[op],
            transform=ax[i].transAxes, fontsize=13)
    ax[i].set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    ax[i].set_xlabel('IPTG (M)', fontsize=15)
    ax[i].set_ylabel('fold-change', fontsize=16)
    ax[i].set_ylim([-0.01, 1.1])
    ax[i].set_xlim(left=-5E-9)
    ax[i].tick_params(labelsize=14)

ax[0].legend(loc='upper left', title='repressors / cell')
# add plot letter labels
plt.figtext(0.0, .95, '(A)', fontsize=20)
plt.figtext(0.50, .95, '(B)', fontsize=20)
plt.figtext(0.0, .46, '(C)', fontsize=20)
plt.figtext(0.50, .46, '(D)', fontsize=20)
plt.tight_layout()
plt.savefig('../../../figures/extras/figSxx_double_promoter_double_R.pdf',
            bbox_inches='tight')

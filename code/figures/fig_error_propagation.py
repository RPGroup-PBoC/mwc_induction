import os
import glob
import pickle
import re

# Our numerical workhorses
import numpy as np
import pandas as pd

# Import the project utils
import sys
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

mwc.set_plotting_style()

#=============================================================================== 
# Set output directory based on the graphicspath.tex file to print in dropbox
#=============================================================================== 
dropbox = open('../../doc/induction_paper/graphicspath.tex')
output = dropbox.read()
output = re.sub('\\graphicspath{{', '', output)
output = output[1::]
output = re.sub('}}\n', '', output)

#=============================================================================== 
# Read the data
#=============================================================================== 

datadir = '../../data/'
df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta') & (df.operator != 'Oid')]

#=============================================================================== 
# Load MCMC flatchain
#=============================================================================== 
# Load the flat-chain
with open('../../data/mcmc/error_prop_pool_data_large_sigma.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()   

# Generate a Pandas Data Frame with the mcmc chain
columns = np.concatenate([['ea', 'ei', 'sigma'],\
          [df[df.repressors==r].rbs.unique()[0] for r in \
              np.sort(df.repressors.unique())],
          [df[df.binding_energy==o].operator.unique()[0] for o in \
              np.sort(df.binding_energy.unique())]])

mcmc_df = pd.DataFrame(gauss_flatchain, columns=columns)
# Generate data frame with mode values for each parameter
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
param_fit = pd.DataFrame(gauss_flatchain[max_idx, :], index=columns,
                         columns=['mode'])
# map value of the parameters
map_param = param_fit['mode'].to_dict()

#=============================================================================== 
# Plot the theory vs data for all 4 operators with the credible region
#=============================================================================== 
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-7, -2, 100)
IPTG_lin = np.array([0, 1E-7])

# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize the plot to set the size
fig = plt.figure(figsize=(11, 8))

# Define the GridSpec to center the lower plot
ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=2)
ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
ax = [ax1, ax2, ax3]

# Loop through operators
for i, op in enumerate(operators):
    print(op)
    data = df[df.operator==op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # plot the theory using the parameters from the fit.
        # Log scale
        ax[i].plot(IPTG, mwc.fold_change_log(IPTG * 1E6, 
            ea=map_param['ea'], ei=map_param['ei'], epsilon=4.5,
            R=map_param[rbs],
            epsilon_r=map_param[op]),
            color=colors[j])
        # Linear scale
        ax[i].plot(IPTG_lin, mwc.fold_change_log(IPTG_lin * 1E6, 
            ea=map_param['ea'], ei=map_param['ei'], epsilon=4.5,
            R=map_param[rbs],
            epsilon_r=map_param[op]),
            color=colors[j], linestyle='--')
        # plot 95% HPD region using the variability in the parameters
        # Log scale
        flatchain = np.array(mcmc_df[['ea', 'ei', rbs, op]])
        cred_region = mwc.mcmc_cred_reg_error_prop(IPTG * 1E6, 
            flatchain, epsilon=4.5)
        ax[i].fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                        alpha=0.3, color=colors[j])
        # Linear scale
        flatchain = np.array(mcmc_df[['ea', 'ei', rbs, op]])
        cred_region = mwc.mcmc_cred_reg_error_prop(IPTG_lin * 1E6, 
            flatchain, epsilon=4.5)
        ax[i].fill_between(IPTG_lin, cred_region[0,:], cred_region[1,:],
                        alpha=0.3, color=colors[j])

        # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())
        
        # plot the experimental data
        ax[i].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
            yerr=fc_err, fmt='o', label=df[df.rbs==rbs].repressors.unique()[0],
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
plt.figtext(0.25, .95, 'A', fontsize=20)
plt.figtext(0.0, .46, 'B', fontsize=20)
plt.figtext(0.50, .46, 'C', fontsize=20)
plt.tight_layout()
plt.savefig(output + '/fig_error_propagation.pdf', bbox_inches='tight')

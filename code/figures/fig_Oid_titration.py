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
files = ['flow_master.csv', 'merged_Oid_data_foldchange.csv']
df_Oid = pd.read_csv(datadir + files[1], comment='#')
df_Oid['fold_change_A'] = df_Oid.fold_change
df = pd.read_csv(datadir + files[0], comment='#')
df = pd.concat([df, df_Oid])
df.dropna(axis=1, inplace=True)

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

#=============================================================================== 
# Load MCMC flatchain
#=============================================================================== 
# O2 - RBS1027 fit
# Load the flat-chain
with open('../../data/mcmc/O2_RBS1027.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain_O2 = unpickler.load()
    gauss_flatlnprobability_O2 = unpickler.load()
    
# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability_O2, axis=0)
ea, ei, sigma = gauss_flatchain_O2[max_idx]

# Global fit
# Load the flat-chain
with open('../../data/mcmc/error_prop_global_Oid.pkl', 'rb') as file:
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
operators = ['Oid']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize the plot to set the size
fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))

# Loop through operators
for i, op in enumerate(operators):
    print(op)
    data = df[df.operator==op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # Check if the RBS was measured for this operator
        if rbs in data.rbs.unique():
        # plot the theory using the parameters from the fit.
        ## O2 - 1027 fit ##
        # Log-scale
            ax.plot(IPTG, mwc.fold_change_log(IPTG * 1E6,
                ea=ea, ei=ei, epsilon=4.5,
                R=np.array(df[(df.rbs == rbs)].repressors.unique()),
                epsilon_r=energies[op]),
                color=colors[j], label=None, zorder=1)
            # Linear scale
            ax.plot(IPTG_lin, mwc.fold_change_log(IPTG_lin * 1E6,
                ea=ea, ei=ei, epsilon=4.5,
                R=np.array(df[(df.rbs == rbs)].repressors.unique()),
                epsilon_r=energies[op]),
                color=colors[j], label=None, zorder=1, linestyle='--')

        ## Global fit ##
        # Log scale
            ax.plot(IPTG, mwc.fold_change_log(IPTG * 1E6, 
                ea=map_param['ea'], ei=map_param['ei'], epsilon=4.5,
                R=map_param[rbs],
                epsilon_r=map_param[op]),
                color=colors[j], linestyle=(0, (5, 1)), lw=2.5)
            # Linear scale
            ax.plot(IPTG_lin, mwc.fold_change_log(IPTG_lin * 1E6, 
                ea=map_param['ea'], ei=map_param['ei'], epsilon=4.5,
                R=map_param[rbs],
                epsilon_r=map_param[op]),
                color=colors[j], linestyle='--')
        
            # plot 95% HPD region using the variability in the MWC parameters
            ## O2 - 1027 fit ##
            # Log scale
            cred_region = mwc.mcmc_cred_region(IPTG * 1e6,
                gauss_flatchain_O2, epsilon=4.5,
                R=df[(df.rbs == rbs)].repressors.unique(),
                epsilon_r=energies[op])
            ax.fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                            alpha=0.3, color=colors[j])
            ## Global fit ##
            # Log scale
            flatchain = np.array(mcmc_df[['ea', 'ei', rbs, op]])
            cred_region = mwc.mcmc_cred_reg_error_prop(IPTG * 1E6, 
                flatchain, epsilon=4.5)
            ax.fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                            alpha=0.3, color=colors[j])
            # Linear scale
            flatchain = np.array(mcmc_df[['ea', 'ei', rbs, op]])
            cred_region = mwc.mcmc_cred_reg_error_prop(IPTG_lin * 1E6, 
                flatchain, epsilon=4.5)
            ax.fill_between(IPTG_lin, cred_region[0,:], cred_region[1,:],
                            alpha=0.3, color=colors[j])

        # Plot the raw data for Oid
        if rbs in data.rbs.unique():
            label=df[df.rbs==rbs].repressors.unique()[0] * 2
        else:
            label=''
        ax.plot(data[data.rbs==rbs].IPTG_uM / 1E6,
                data[data.rbs==rbs].fold_change_A, marker='o', lw=0,
                color=colors[j], label=label)

    # Add operator and binding energy labels.
    ax.text(0.8, 0.02, r'{0}'.format(op), transform=ax.transAxes, 
            fontsize=13)
    ax.set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    ax.set_xlabel('IPTG (M)', fontsize=15)
    ax.set_ylabel('fold-change', fontsize=16)
    ax.set_ylim([-0.01, 1.1])
    ax.set_xlim(left=-5E-9)
    ax.tick_params(labelsize=14)

main_legend = ax.legend(loc='upper left', title='repressors / cell')

l1 = ax.plot([], [], color='k')
l2 = ax.plot([], [], color='k', linestyle=(0, (5, 1)))
extra_legend = [l1, l2]
extra_labels = [r'HG et al. 2011: -17', 
                r'fit: {:.1f}'.format(map_param['Oid'])]
ax.legend([l[0] for l in extra_legend], extra_labels,
          loc='center left', 
          title='binding energy $\Delta \epsilon _{RA}$ ($k_BT$)')
# add main legend
plt.gca().add_artist(main_legend)

plt.tight_layout()
plt.savefig(output + '/fig_Oid_titration.pdf', bbox_inches='tight')

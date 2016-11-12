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

sns.set_palette("deep", color_codes=True)
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
# read the list of data-sets to ignore
data_ignore = pd.read_csv(datadir + 'datasets_ignore.csv', header=None).values
# read the all data sets except for the ones in the ignore list
all_files = glob.glob(datadir + '*' + '_IPTG_titration' + '*csv')
ignore_files = [f for f in all_files for i in data_ignore if i[0] in f]
read_files = [f for f in all_files if f not in ignore_files]
print('Number of unique data-sets: {:d}'.format(len(read_files)))
df = pd.concat(pd.read_csv(f, comment='#') for f in read_files)

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta') & (df.operator != 'Oid')]

#=============================================================================== 
# Load MCMC flatchain
#=============================================================================== 
# Load the flat-chain
with open('../../data/mcmc/' + '20161021' + \
                  '_error_prop_pool_data_larger_sigma.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    
# Generate a Pandas Data Frame with the mcmc chain
columns = np.concatenate([['ea', 'ei', 'sigma'],\
          [df[df.repressors==r].rbs.unique()[0] for r in \
              np.sort(df.repressors.unique())],
          [df[df.binding_energy==o].operator.unique()[0] for o in \
              np.sort(df.binding_energy.unique())]])

mcmc_df = pd.DataFrame(gauss_flatchain, columns=columns)
# map value of the parameters
map_param = dict(mcmc_df.mean())

#=============================================================================== 
# Plot the theory vs data for all 4 operators with the credible region
#=============================================================================== 
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-8, -2, 100)

# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']# 'Oid']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize subplots
fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax = ax.ravel()

# Loop through operators
for i, op in enumerate(operators):
    data = df[df.operator==op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # plot the theory using the parameters from the fit.
        ax[i].plot(IPTG, mwc.fold_change_log(IPTG * 1E6, 
            ea=map_param['ea'], ei=map_param['ei'], epsilon=4.5,
            R=map_param[rbs],
            epsilon_r=map_param[op]),
            color=colors[j])

        # plot 95% HPD region using the variability in the parameters
        flatchain = np.array(mcmc_df[['ea', 'ei', rbs, op]])
        cred_region = mwc.mcmc_cred_reg_error_prop(IPTG * 1E6, 
            flatchain, epsilon=4.5)
        ax[i].fill_between(IPTG, cred_region[0,:], cred_region[1,:],
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
    ax[i].set_xscale('log')
    ax[i].set_xlabel('IPTG (M)')
    ax[i].set_ylabel('fold-change')
    ax[i].set_ylim([-0.01, 1.2])
    ax[i].set_title(op)
ax[0].legend(loc='upper left', title='repressors / cell')
plt.tight_layout()
plt.savefig(output + '/fig_error_propagation.pdf')

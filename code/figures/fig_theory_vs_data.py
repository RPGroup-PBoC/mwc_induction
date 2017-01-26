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
# read the list of data-sets to ignore
data_ignore = pd.read_csv(datadir + 'datasets_ignore.csv', header=None).values
# read the all data sets except for the ones in the ignore list
all_files = glob.glob(datadir + '*' + '_IPTG_titration_MACSQuant' + '*csv')
ignore_files = [f for f in all_files for i in data_ignore if i[0] in f]
read_files = [f for f in all_files if f not in ignore_files and 'dimer' not in f]
print('Number of unique data-sets: {:d}'.format(len(read_files)))
df = pd.concat(pd.read_csv(f, comment='#') for f in read_files)

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

#===============================================================================
# O2 RBS1027
#===============================================================================
with open('../../data/mcmc/' + '20161208' + \
                  '_gauss_homoscedastic_RBS1027.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()
    
# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]

#===============================================================================
# Plot the theory vs data for all 4 operators with the credible region
#===============================================================================
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-11, -2, 100)

# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7}

# Initialize subplots
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

ax = ax.ravel()

# Loop through operators
for i, op in enumerate(operators):
    print(op)
    data = df[df.operator==op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # plot the theory using the parameters from the fit.
        ax[i+1].plot(IPTG, mwc.fold_change_log(IPTG * 1E6,
            ea=ea, ei=ei, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op]),
            color=colors[j], label=None)
        # plot 95% HPD region using the variability in the MWC parameters
        cred_region = mwc.mcmc_cred_region(IPTG * 1e6,
            gauss_flatchain, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op])
        ax[i+1].fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                        alpha=0.3, color=colors[j])
        # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())

        # plot the experimental data
        # Distinguish between the fit data and the predictions
        if (op == 'O2') & (rbs == 'RBS1027'):
            ax[i+1].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                    fc_mean, yerr=fc_err, linestyle='none', color=colors[j])
            ax[i+1].plot(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                       fc_mean, marker='o', linestyle='none',
                       markeredgewidth=2, markeredgecolor=colors[j],
                       markerfacecolor='w', 
                       label=df[df.rbs=='RBS1027'].repressors.unique()[0] * 2)
        else:
            ax[i+1].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                    fc_mean, yerr=fc_err,
                    fmt='o', label=df[df.rbs==rbs].repressors.unique()[0] * 2,
                color=colors[j])

        # Add operator and binding energy labels.
        ax[i+1].text(0.8, 0.08, r'{0}'.format(op), transform=ax[i+1].transAxes, 
                fontsize=14)
        ax[i+1].text(0.65, 0.02,
                r'$\Delta\varepsilon_{RA} = %s\,k_BT$' %energies[op],
                transform=ax[i+1].transAxes, fontsize=14)
    ax[i+1].set_xscale('symlog', linthreshx=1E-7)
    ax[i+1].set_xlabel('IPTG (M)', fontsize=15)
    ax[i+1].set_ylabel('fold-change', fontsize=16)
    ax[i+1].set_ylim([-0.01, 1.1])
    ax[i+1].set_xlim(left=-5E-9)
    ax[i+1].tick_params(labelsize=14)
ax[1].legend(loc='upper left', title='repressors / cell')
ax[0].set_axis_off()
# add plot letter labels
plt.figtext(0.0, .95, 'A', fontsize=20)
plt.figtext(0.5, .95, 'B', fontsize=20)
plt.figtext(0.0, .46, 'C', fontsize=20)
plt.figtext(0.5, .46, 'D', fontsize=20)
plt.tight_layout()
plt.savefig(output + '/fig_predictions_O2_RBS1027_fit_with_data.pdf', 
        bbox_inches='tight')


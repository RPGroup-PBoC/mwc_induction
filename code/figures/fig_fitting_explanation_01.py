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

# favorite Seaborn settings for notebooks
rc={'lines.linewidth': 2, 
    'axes.labelsize' : 16, 
    'axes.titlesize' : 18,
    'axes.facecolor' : 'F4F3F6',
    'axes.edgecolor' : '000000',
    'axes.linewidth' : 1.2,
    'xtick.labelsize' : 13,
    'ytick.labelsize' : 13,
    'grid.linestyle' : ':',
    'grid.color' : 'a6a6a6'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)
sns.set_palette("deep", color_codes=True)

#=============================================================================== 
# Set output directory based on the graphicspath.tex file to print in dropbox
#=============================================================================== 
dropbox = open('../../doc/induction_paper/graphicspath.tex')
output = dropbox.read()
output = re.sub('\\graphicspath{{', '', output)
output = output[1::]
output = re.sub('}}\n', '', output + '/extra_figures')

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
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

#=============================================================================== 
# O2 RBS1027
#=============================================================================== 
# Load the flat-chain
with open('../../data/mcmc/' + '20160815' + \
                  '_gauss_homoscedastic_RBS1027.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    
# map value of the parameters
ea, ei = np.mean(gauss_flatchain[:, [0, 1]], axis=0)

#=============================================================================== 
# Plot the theory vs data for all 4 operators with the credible region
#=============================================================================== 
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-8, -2, 100)

# Set the colors for the strains
colors = sns.color_palette(n_colors=7)

# Define the operators and their respective energies
operators = ['O2']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize subplots
fig = plt.figure()
ax = plt.subplot(111)

# Loop through operators
for i, op in enumerate(operators):
    data = df[df.operator==op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # Plot only for RBS1027
        if rbs=='RBS1027':
            # compute the mean value for each concentration
            fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
            # compute the standard error of the mean
            fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
            np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())
            
            # plot the experimental data
            ax.errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
                yerr=fc_err, fmt='o', label=df[df.rbs==rbs].repressors.unique()[0],
                color=colors[j])

ax.set_xscale('log')
ax.set_xlabel('IPTG (M)')
ax.set_ylabel('fold-change')
ax.set_ylim([-0.01, 1.2])
ax.set_xlim([1E-8, 1E-2])
ax.set_title(op)
ax.legend(loc='upper left', title='repressors / cell')
plt.tight_layout()
plt.savefig(output + '/fig_fit_explanation_01.pdf')

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

# Compute the Bohr parameter
df['bohr_1027'] = mwc.bohr_fn(df, ea, ei)

#=============================================================================== 
# Plot the theory vs data for all 4 operators with the credible region
#=============================================================================== 

# Given this result let's plot all the curves using this parameters.
colors = sns.color_palette('colorblind', n_colors=4)
F = np.linspace(-8, 10, 200)
plt.figure(figsize=(8, 6))
plt.plot(F, 1 / (1 + np.exp(-F)), '-', color='black')
for i, operator in enumerate(df.operator.unique()):
    data = df[df.operator==operator]
    plt.errorbar([], [], label=operator, color=colors[i], fmt='o')
    for j, rbs in enumerate(df.rbs.unique()):
        # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())
        bohr = data[data.rbs==rbs].groupby('IPTG_uM').bohr_1027.mean()
        plt.errorbar(bohr, fc_mean, yerr=fc_err, fmt='o', color=colors[i],
                label=None)
plt.xlabel(r'Bohr parameter ($k_BT$ units)')
plt.ylabel('fold-change')
plt.ylim([-0.01, 1.1])
plt.legend(loc='upper left', ncol=1, title='operator', fontsize=13)
plt.tight_layout()
plt.savefig(output + '/fig_data_collapse_O2_RBS1027_fit.pdf')

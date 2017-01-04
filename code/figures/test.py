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
import matplotlib.patches as pch

# Seaborn, useful for graphics
import seaborn as sns

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
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
# Define the operators to use in the plot
operators = ['O1', 'O2', 'O3']
markers = ['o', 'D', 's']
F = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 6))
plt.plot(F, 1 / (1 + np.exp(-F)), '-', color='black')

# Instantiate the legend.
label_col = ['R', 1740, 1220, 260, 124, 60, 22]
label_O1 = ['O1']
label_O2 = ['O2']
label_O3 = ['O3']
label_empty = ['']
handles = []
with sns.axes_style('white'):
    for i, operator in enumerate(operators):
        data = df[df.operator==operator]
        # plt.errorbar([], [], label=operator, color=colors[i], fmt='o')
        for j, rbs in enumerate(df.rbs.unique()):
            # compute the mean value for each concentration
            fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
            # compute the standard error of the mean
            fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
            np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())
            bohr = data[data.rbs==rbs].groupby('IPTG_uM').bohr_1027.mean()
            plt.errorbar(bohr, fc_mean, yerr=fc_err, fmt=markers[i],
                        color='k', label=None, markersize=8, alpha=0.75,
                        capsize=0)
            _p, = plt.plot(bohr, fc_mean, linestyle='none', marker=markers[i],
                          markeredgewidth=2, markeredgecolor='k',
                          markerfacecolor='w', alpha=0.75)
    
            handles.append(_p)
    plt.xlabel(r'Bohr parameter ($k_BT$ units)', fontsize=18)
    plt.ylabel('fold-change', fontsize=18)
    plt.ylim([-0.01, 1.1])
    
    # Generate the legend handles. Extra is empty space.
    extra = pch.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                          linewidth=0)
    leg_handles = [extra, extra, extra, extra, extra, extra, extra]
    slc = [0, 6, 12, 18]
    for i in range(len(slc) - 1):
        leg_handles.append(extra)
        sel_handles = handles[slc[i]:slc[i + 1]]
        for j in range(len(sel_handles)):
            leg_handles.append(sel_handles[j])
    labels = np.concatenate([label_col, label_O1, label_empty * 6, label_O2, label_empty * 6, label_O3, label_empty * 6])
    #plt.legend(leg_handles, labels, loc='upper left', ncol=4, fontsize=13,
    #           handletextpad=-1.5)
    plt.tight_layout()
    plt.tick_params(labelsize=16)
    # output = '/Users/gchure/Dropbox/mwc_induction'
    plt.savefig('/Users/gchure/Desktop/collapse.pdf')

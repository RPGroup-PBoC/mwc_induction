import os
import glob
# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy
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
# define variables to use over the script
date = 20160725
username = 'mrazomej'

# read the CSV file with the mean fold change
df = pd.read_csv('output/' + str(date) + '_lacI_titration_MACSQuant.csv')
rbs = df.rbs.unique()
replica = df.replica.unique()
#=============================================================================== 

# compute the theoretical repression level
repressor_array = np.logspace(1, 3, 100)
epsilon_array = np.array([-15.3, -13.9, -9.7, -17])
operators = np.array(['O1', 'O2', 'O3', 'Oid'])

#=============================================================================== 
colors = sns.hls_palette(len(operators), l=.3, s=.8)
# plot theoretical curve
# First for the A channel
plt.figure(figsize=(7, 7))
for i, o in enumerate(operators):
    fold_change_theor = 1 / (1 + 2 * repressor_array / 5E6 \
            * np.exp(-epsilon_array[i]))
    plt.plot(repressor_array, fold_change_theor, label=o,
            color=colors[i])
    plt.scatter(df[(df.operator == o) & (df.rbs != 'auto') & \
            (df.rbs != 'delta')].repressors, 
            df[(df.operator == o) & (df.rbs != 'auto') & \
            (df.rbs != 'delta')].fold_change_A,
            marker='o', linewidth=0, color=colors[i], 
            label=o + ' flow cytometer',
            alpha=0.7)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('repressor copy number')
plt.ylabel('fold-change')
plt.title('FITC-A')
plt.xlim([10, 1E3])
plt.legend(loc='lower left')
plt.tight_layout()
plt.savefig('output/lacI_titration_FITC_A.png')

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
date = 20161015
username = 'nbellive'
run = 'r1'
operators = {'O1' : -15.3, 'O2' : -13.9}

# read the CSV file with the mean fold change
df = pd.read_csv('output/' + str(date) + '_' + run + '_' + 'O1O2dimer' + \
                 '_IPTG_titration_MACSQuant.csv')
rbs = df.rbs.unique()

#===============================================================================

# plot all raw data
plt.figure()
for op in operators.keys():
    for strain in rbs[np.array([r != 'auto' and r != 'delta' for r in rbs])]:
        plt.plot(df[(df.operator == op) & \
                (df.rbs == strain)].sort_values(by='IPTG_uM').IPTG_uM * 1E-6,
                df[(df.operator == op) & \
                (df.rbs == strain)].sort_values(by='IPTG_uM').fold_change_A,
                marker='o', linewidth=1, linestyle='--',
                label=op + ' ' + strain)
plt.xscale('log')
plt.xlabel('IPTG (M)')
plt.ylabel('fold-change')
plt.ylim([-0.01, 1.2])
plt.xlim([1E-8, 1E-2])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('output/' + 'O1O2dimer' + '_IPTG_titration_data.png')

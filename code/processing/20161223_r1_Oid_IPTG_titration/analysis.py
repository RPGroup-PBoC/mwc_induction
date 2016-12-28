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
date = 20161223
username = 'sbarnes'
run = 'r1'
operator = 'Oid'

# read the CSV file with the mean fold change
df = pd.read_csv('output/' + str(date) + '_' + run + '_' + operator + \
                 '_IPTG_titration_MACSQuant.csv')
rbs = df.rbs.unique()

#=============================================================================== 

# plot all raw data
plt.figure()
for strain in rbs[np.array([r != 'auto' and r != 'delta' for r in rbs])]:
    plt.plot(df[df.rbs == strain].sort_values(by='IPTG_uM').IPTG_uM * 1E-6,
            df[df.rbs == strain].sort_values(by='IPTG_uM').fold_change_A,
            marker='o', linewidth=1, linestyle='--', label=strain)
plt.xscale('log')
plt.xlabel('IPTG (M)')
plt.ylabel('fold-change')
plt.ylim([-0.01, 1.2])
plt.xlim([1E-8, 1E-2])
plt.legend(loc='upper left')
plt.tight_layout()
plt.savefig('output/' + operator + '_IPTG_titration_data.png')

#=============================================================================== 

# plot the curve for the 0 IPTG cultures
repressor_array = np.logspace(0, 3, 200)
binding_energy = df.binding_energy.unique()
fc_theory = 1 / (1 + 2 * repressor_array / 5E6 * np.exp(- binding_energy))

plt.figure(figsize=(7, 7))
plt.plot(repressor_array, fc_theory)
no_iptg = df.groupby('IPTG_uM').get_group(0)
plt.plot(no_iptg.repressors, no_iptg.fold_change_A, marker='o', linewidth=0)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('repressor copy number')
plt.ylabel('fold-change')
plt.tight_layout()
plt.savefig('output/' + operator + '_lacI_titration_ctrl.png')

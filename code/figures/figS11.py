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
# Read the data
#===============================================================================

datadir = '../../data/'
df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta') & (df.IPTG_uM==0)]

# Let's import the data from HG 2011 and RB 2014
df_old = pd.read_csv(datadir + 'tidy_lacI_titration_data.csv', comment='#')
df_old = df_old[df_old.operator!='Oid']
#===============================================================================
# O2 RBS1027
#===============================================================================
# Load the flat-chain
with open('../../data/mcmc/main_text_KaKi.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]

#===============================================================================
# Plot the theory vs data for all 3 operators
#===============================================================================

## Flow Data ##
# Define the number of repressors for the theoretical predictions
r_array = np.logspace(0, 3.5, 100)
# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=4)

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize the plot to set the size
#fig = plt.figure(figsize=(4.5, 4.5))
sns.set_context('paper')
fig = plt.figure()
ax = plt.subplot(111, aspect='equal')
plt.axis('scaled')

## HG and RB data ##
df_group = df_old.groupby('operator')
i = 0
for group, data in df_group:
    # Extract HG data
    garcia = data[data.author=='garcia']
    ax.plot(garcia.repressor, garcia.fold_change, color=colors[i],
                lw=0, marker='o', label='', alpha=0.75)
    # Extract RB data
    brewster = data[data.author=='brewster']
    ax.plot(brewster.repressor, brewster.fold_change, color=colors[i],
                lw=0, marker='D', label='', alpha=0.75)
    i+=1

# Group data by operator
df_group = df.groupby('operator')

# initialize counter (because of df_group)
i = 0

for group, data in df_group:
    # Compute the theoretical fold change for this operator
    fc = mwc.fold_change_log(np.array([0]), ea, ei, 4.5,
                                r_array / 2, data.binding_energy.unique())
    ax.plot(r_array, fc, color=colors[i],
            label=group + r' $\Delta\varepsilon_{RA} =$' + \
            str(data.binding_energy.unique()[0]) + ' $k_BT$')
    # compute the mean value for each concentration
    fc_mean = data.groupby('repressors').fold_change_A.mean()
    # compute the standard error of the mean
    fc_err = data.groupby('repressors').fold_change_A.std() / \
    np.sqrt(data.groupby('repressors').size())
    log_fc_err = fc_mean - 10**(np.log10(fc_mean) - \
                           fc_err / fc_mean / np.log(10))

    log_fc_err= np.vstack([log_fc_err, 10**(np.log10(fc_mean) + \
                                       fc_err / fc_mean / np.log(10)) -\
                                       fc_mean])
    # plot the experimental data
    ax.errorbar(fc_mean.index * 2, fc_mean,
                yerr=log_fc_err,
                fmt='o', markeredgecolor=colors[i], label='',
                markerfacecolor='white', markeredgewidth=2)
    i+=1

ax.plot([], [], marker='o',
        markeredgecolor='k', markerfacecolor='w', markeredgewidth=2,
        label='flow cytometry', lw=0)
ax.plot([], [], marker='o', color='k', alpha=0.75,
        label='HG & RP 2011,\nMiller assay', lw=0)
ax.plot([], [], marker='D', color='k', alpha=0.75,
        label='RB et al. 2014,\ntime lapse microscopy', lw=0)
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('repressors / cell')
ax.set_ylabel('fold-change')
ax.set_xlim(right=10**3.5)
ax.set_ylim(top=2)
leg = ax.legend(loc='lower left', fontsize=8)
leg.set_zorder(1)

plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS11.pdf', bbox_inches='tight')

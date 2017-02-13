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
# Define working directory
datadir = '../../data/'
# List files to be read
files = ['flow_master.csv', 'merged_Oid_data_foldchange.csv']
# Read flow cytometry data
df_Oid = pd.read_csv(datadir + files[1], comment='#')
# make an extra column to have consistent labeling
df_Oid['fold_change_A'] = df_Oid.fold_change
# Remove manually the outlier with an unphysical fold-change
df_Oid = df_Oid[df_Oid.fold_change_A <= 1]
# Read the flow cytometry data
df = pd.read_csv(datadir + files[0], comment='#')
# Attach both data frames into a single one
df = pd.concat([df, df_Oid])
# Drop rows containing NA values
df.dropna(axis=1, inplace=True)
# Now we remove the autofluorescence, delta, and higher IPTG values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta') & (df.IPTG_uM==0)]

# Let's import the data from HG 2011 and RB 2014
df_old = pd.read_csv(datadir + 'tidy_lacI_titration_data.csv', comment='#')
#===============================================================================
# Global fit
#===============================================================================
# Load the flat-chain
with open('../../data/mcmc/error_prop_global_large_sigma.pkl', 'rb') as file:
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
# Plot the theory vs data for all 3 operators
#===============================================================================
## Flow Data ##
# Define the number of repressors for the theoretical predictions
r_array = np.logspace(0, 3.5, 100)
# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=4)

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3', 'Oid']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize the plot to set the size
#fig = plt.figure(figsize=(4.5, 4.5))
sns.set_context('paper')
fig = plt.figure()
ax = plt.subplot(111, aspect=2/3)
#plt.axis('scaled')

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
    #### Hernan energies ####
    fc = mwc.fold_change_log(np.array([0]), map_param['ea'],
                                            map_param['ei'], 4.5,
                                r_array / 2, data.binding_energy.unique())
    ax.plot(r_array, fc, color=colors[i], 
            label=group)

    #### Global fit energies ####
    fc = mwc.fold_change_log(np.array([0]), map_param['ea'],
                                            map_param['ei'], 4.5,
                                r_array / 2, map_param[group])
    ax.plot(r_array, fc, color=colors[i], linestyle='--',
            label='')# str(np.round(map_param[group], 1))

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

main_legend = ax.legend(loc='center left', title='operator')

l1 = ax.plot([], [], marker='o',
        markeredgecolor='k', markerfacecolor='w', markeredgewidth=2,
        label='flow cytometry', lw=0)
l2 = ax.plot([], [], marker='o', color='k', alpha=0.75,
        label='HG & RP 2011,\nMiller assay', lw=0)
l3 = ax.plot([], [], marker='D', color='k', alpha=0.75,
        label='RB et al. 2014,\ntime lapse microscopy', lw=0)
l4 = ax.plot([], [], color='k', alpha=0.75,
        label='HG & RP 2011 fit')
l5 = ax.plot([], [], color='k', alpha=0.75, linestyle='--',
        label='global fit')
extra_legend = [l1, l2, l3, l4, l5]

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('repressors / cell')
ax.set_ylabel('fold-change')
ax.set_xlim(right=10**3.5)
ax.set_ylim(top=2)

# Add the extra legend
labels = ['this study',
          'HG & RP 2011',
          'RB et al. 2014',
          'HG & RP 2011 fit',
          'global fit']
leg = ax.legend([l[0] for l in extra_legend], labels,
                loc='lower left', fontsize=8)
plt.gca().add_artist(main_legend)
                
leg.set_zorder(1)

plt.tight_layout()
plt.savefig(output + '/fig_error_propagation_lacI_titration.pdf', bbox_inches='tight')


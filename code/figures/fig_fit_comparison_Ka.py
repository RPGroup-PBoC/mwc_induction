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
df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values and Oid data sets
df = df[(df.rbs != 'auto') & (df.rbs != 'delta') & (df.operator != 'Oid')]


#===============================================================================
# We first need to summarized the Ka, Ki, and associated hpd values
# Generate DataFrame to save parameter estimates
#===============================================================================

param_summary = pd.DataFrame()

# Loop through each strain in groups and calculate HPD
groups = df.groupby(['operator', 'rbs'])
for g, subdata in groups:
    if g[0] == 'Oid':
        continue
    with open('../../data/mcmc/' + '20170209' + \
              '_gauss_' + g[0] + '_' + g[1] + '.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        gauss_flatchain = unpickler.load()
        gauss_flatlnprobability = unpickler.load()

    # map value of the parameters
    max_idx = np.argmax(gauss_flatlnprobability, axis=0)
    ea, ei = gauss_flatchain[max_idx, [0, 1]]

    # ea range
    ea_hpd = mwc.hpd(gauss_flatchain[:, 0],0.95)
    ei_hpd = mwc.hpd(gauss_flatchain[:, 1],0.95)

    # add values to dataframe
    param_summary_temp = pd.DataFrame({'rbs':g[1],
                                'operator':g[0],
                                'repressors':df[df['rbs']==g[1]].repressors.unique()[0],
                                'ka_mode':[ea],
                                'ka_hpd_max':[ea_hpd[0]],
                                'ka_hpd_min':[ea_hpd[1]],
                                'ki_mode':[ei],
                                'ki_hpd_max':[ei_hpd[0]],
                                'ki_hpd_min':[ei_hpd[1]]})

    param_summary = param_summary.append(param_summary_temp, ignore_index=True)


#==============
# Load in flatchain from global fit
#==============

with open('../../data/mcmc/' + 'error_prop_global_Oid' + '.pkl', 'rb') as file:
    unpickler_global = pickle.Unpickler(file)
    gauss_flatchain_global = unpickler_global.load()
    gauss_flatlnprobability_global = unpickler_global.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability_global, axis=0)
Ka_global, Ki_global = np.exp(-gauss_flatchain_global[max_idx, [0, 1]])

# ea range
Ka_global_range = np.exp(-mwc.hpd(gauss_flatchain_global[:, 0],0.95))
Ki_global_range = np.exp(-mwc.hpd(gauss_flatchain_global[:, 1],0.95))


#===============================================================================
# Plot the fitted Ka and Ki along with global fit for each operator strain set.
#===============================================================================

# Define array of IPTG concentrations
IPTG = np.logspace(-9, -1, 200)

# current default color palette
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
gs_dict = {'hspace': 0.1, 'wspace':0.1}


for op in df.operator.unique():
    # Load operator specific data from df
    param_summary_op= param_summary[param_summary.operator==op]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,2), gridspec_kw=gs_dict)

    # plot kA
    for i, rbs_fit in enumerate(df_plot.rbs.unique()[::-1]):
        Ka = np.exp(-param_summary_op[param_summary_op['rbs']==rbs_fit].ka_mode)
        Ka_hpd_max = np.exp(-param_summary_op[param_summary_op['rbs']==rbs_fit].ka_hpd_max)
        Ka_hpd_min = np.exp(-param_summary_op[param_summary_op['rbs']==rbs_fit].ka_hpd_min)

        plt.errorbar(i,Ka,
                     yerr=[np.abs(Ka-Ka_hpd_min.values[0]),np.abs(Ka-Ka_hpd_max.values[0])],
                     fmt='o', color=colors[-(i+3)])

    # Lets also plot the global best value of Ki using all
    # data across all operators
    plt.plot(np.arange(-2,7),np.ones(9)*Ka_global, linestyle='--',alpha=0.6)
    plt.fill_between(np.arange(-2,7) , Ka_global_range[1], Ka_global_range[0],
                             alpha=0.5, color=colors[i])

    ax.set_yscale('log')
    plt.xticks(np.arange(0,6), 2*df_plot.repressors.unique()[::-1])#, rotation='vertical')
    plt.xlim(-0.2,5.2)
    plt.ylim(1E-1,1E4)
    plt.ylabel('Best fit for $K_A$ ($\mu M$)')
    plt.xlabel('LacI copy number')
    plt.savefig(output + '/' + today + \
               '_Ka_summary_' + op + '_wglobal.pdf',bbox_inches='tight')

    # plot Ki
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,2), gridspec_kw=gs_dict)

    for i, rbs_fit in enumerate(df_plot.rbs.unique()[::-1]):

        Ki = np.exp(-param_summary_op[param_summary_op['rbs']==rbs_fit].ki_mode)
        Ki_hpd_max = np.exp(-param_summary_op[param_summary_op['rbs']==rbs_fit].ki_hpd_max)
        Ki_hpd_min = np.exp(-param_summary_op[param_summary_op['rbs']==rbs_fit].ki_hpd_min)

        plt.errorbar(i,Ki,
                     yerr=[np.abs(Ki-Ki_hpd_min.values[0]),np.abs(Ki-Ki_hpd_max.values[0])],
                     fmt='o', color=colors[-(i+3)])

    # Lets also plot the global best value of Ki using all
    # data across all operators
    plt.plot(np.arange(-2,7),np.ones(9)*Ki_global, linestyle='--',alpha=0.6)
    plt.fill_between(np.arange(-2,7) , Ki_global_range[1], Ki_global_range[0],
                             alpha=0.5, color=colors[i])

    ax.set_yscale('log')
    plt.xticks(np.arange(0,6), 2*df_plot.repressors.unique()[::-1])#, rotation='vertical')
    plt.xlim(-0.2,5.2)
    plt.ylim(5E-3,2)
    plt.ylabel('Best fit for $K_I$ ($\mu M$)')
    plt.xlabel('LacI copy number')
    plt.savefig(output + '/' + today + \
               '_Ki_summary_' + op + '_wglobal.pdf',bbox_inches='tight')

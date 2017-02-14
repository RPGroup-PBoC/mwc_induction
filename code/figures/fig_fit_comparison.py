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
# Plot the theory vs data across all strains for each operator
#===============================================================================

# Define array of IPTG concentrations
IPTG = np.logspace(-9, -1, 200)

# current default color palette
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
gs_dict = {'hspace': 0.1, 'wspace':0.1}


for op in df.operator.unique():

    # Load operator specific data from df
    df_plot = df[df.operator==op]
    # Initialize plot
    fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(12,12), gridspec_kw=gs_dict)

    for j, rbs_fit in enumerate(df_plot.rbs.unique()):

        with open('../../data/mcmc/' + '20170209' + \
              '_gauss_' + op + '_' + rbs_fit + '.pkl', 'rb') as file:
            unpickler = pickle.Unpickler(file)
            gauss_pool_flatchain = unpickler.load()


        # map value of the parameters
        ea, ei = np.mean(gauss_pool_flatchain[:, [0, 1]], axis=0)

        for i, rbs in enumerate(df_plot.rbs.unique()):
            # plot the theory using the parameters from the fit.

            ax[j,i].plot(IPTG , mwc.fold_change_log(IPTG * 1E6,
                ea=ea, ei=ei, epsilon=4.5,
                R=df_plot[(df_plot.rbs == rbs)].repressors.unique(),
                epsilon_r=df_plot.binding_energy.unique()),
                color=colors[j])

            # plot 95% HPD region using the variability in the MWC parameters

            cred_region = mwc.mcmc_cred_region(IPTG * 1E6,
                gauss_pool_flatchain, epsilon=4.5,
                R=df_plot[(df_plot.rbs == rbs)].repressors.unique(),
                epsilon_r=df_plot.binding_energy.unique())
            ax[j,i].fill_between(IPTG , cred_region[0,:], cred_region[1,:],
                            alpha=0.5, color=colors[j])

            # compute the mean value for each concentration

            fc_mean = df_plot[df_plot.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()

            # compute the standard error of the mean

            fc_err = df_plot[df_plot.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
            np.sqrt(df_plot[df_plot.rbs==rbs].groupby('IPTG_uM').size())

            # plot the experimental data
            if i == j:
                ax[j,i].errorbar(np.sort(df_plot[df_plot.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
                    yerr=fc_err, linestyle='none', label=rbs, color=colors[i])
                ax[j,i].plot(np.sort(df_plot[df_plot.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
                            'o', markeredgewidth=1, markeredgecolor=colors[i],
                             markerfacecolor='w', markersize=7)
                ax[j,i].set_axis_bgcolor("#DFDFE5")
            else:
                ax[j,i].errorbar(np.sort(df_plot[df_plot.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
                    yerr=fc_err, fmt='o', markersize=7, label=rbs, color=colors[i])

            if i == 0:
                ax[j,i].set_ylabel('R = %s' %(df_plot[df_plot.rbs==rbs_fit].repressors.unique()[0] * 2), fontsize=14)
            if j == 0:
                ax[j,i].set_title('R = %s' %(df_plot[df_plot.rbs==rbs].repressors.unique()[0] * 2), fontsize=14)

            # adjust axes, etc to clean up plots
            ax[j,i].set_xlim(5E-8,9E-3)
            ax[j,i].set_ylim(-0.05,1.1)
            ax[j,i].set_xscale('log')
            ax[j,i].set_xticks([1E-7, 1E-5, 1E-3])
            ax[j,i].set_yticks([0, 0.5, 1])

            if i > 0 and i < 6 and j < 5:
                ax[j, i].set_xticklabels([])
                ax[j, i].set_yticklabels([])
            if i==0 and j < 5:
                ax[j, i].set_xticklabels([])
            if j==5 and i > 0 and i < 6:
                ax[j, i].set_yticklabels([])

    fig.text(0.01, 0.5, '$K_A, K_I$ fit strain', va='center', rotation='vertical',fontsize=18)
    fig.text(0.05, 0.5, 'fold-change', va='center', rotation='vertical',fontsize=15)
    fig.text(0.4, 0.05, 'IPTG concentration (M)', va='bottom',fontsize=15)
    fig.suptitle('comparison strain', y=0.95, fontsize=18)

    plt.savefig(output + '/fig_fitcompare_summary_' + op + '_large.pdf', bbox_inches='tight')

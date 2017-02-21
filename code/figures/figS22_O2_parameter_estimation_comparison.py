# (c) 2017 the authors. This work is licensed under a [Creative Commons
# Attribution License CC-BY 4.0](https://creativecommons.org/licenses/by/4.0/).
# All code contained herein is licensed under an [MIT
# license](https://opensource.org/licenses/MIT).

import os
import glob
import pickle
import datetime
# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy.special
import numba
# Library to perform MCMC runs
import emcee
import matplotlib.gridspec as gridspec
import sys
sys.path.append(os.path.abspath('../'))
import mwc_induction_utils as mwc

# Useful plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
import corner

mwc.set_plotting_style()

# Magic function to make matplotlib inline; other style specs must come AFTER
%matplotlib inline

# This enables SVG graphics inline (only use with static plots (non-Bokeh))
%config InlineBackend.figure_format = 'svg'

# Generate a variable with the day that the script is run
today = str(datetime.datetime.today().strftime('%Y%m%d'))

#===============================================================================
# Defining the problem

# In the SI_parameter_estimation.py we performed parameter estimations of
# K_A and K_I for each strain in our data. Now we will load in the MCMC
# flatchains to extract parameter estimates. We will summarize these in plots
# and also plot the strain specific predictions within each operator strain set.

# For details of the Bayesian parameter estimation and the Bayesian approach
# that is applied here, see the 'bayesian_parameter_estimation' notebook.
#===============================================================================


#===============================================================================
# Load in the data from all strains
#===============================================================================
datadir = '../../data/'

df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

# Restart index
df = df.reset_index()

# Lets take a look at the first few rows to make sure
# the data looks okay.
df.head()

#===============================================================================
# Extract the parameter estimates from each strain as well as their highest
# probability density regions (i.e. 95th percentile credible regions).

## Note on determining credible region using MCMC traces:

# To report the output of the MCMC routine we will use the HPD. As explained in
# (http://bebi103.caltech.edu/2015/tutorials/l06_credible_regions.html)
# This method uses the highest posterior density interval, or HPD. If we're
# considering a 95% confidence interval, the HPD interval is the shortest
# interval that contains 95% of the probability of the posterior. So, we report
# the mode and then the bounds on the HPD interval.

# We will use the same funciton used in the tutorial to compute the HPD from the
# MCMC chain. Now that we not only know the MAP value of the MWC parameters, but
# also the credible intervals for them we can properly reflect that uncertainty
# on our plots.

# We use the function 'mcmc_cred_region' in our mwc_utils to calculate this
# credible region.
#===============================================================================

# We named each .pkl chain as SI_I_x_Rj.pkl where x=(O1,O2,O3)
# and j refers to the repressor copy number, R=(22, 60, 124,
# 260, 1220, 1740). Create dictionary to to convert between
# strain name and R.
dict_R = {'HG104': '22', 'RBS1147': '60', 'RBS446' : '124', \
          'RBS1027': '260', 'RBS1': '1220', 'RBS1L': '1740'}

# generate DataFrame to save parameter estimates
param_summary = pd.DataFrame()

# Loop through each strain in groups and calculate HPD
groups = df.groupby(['operator', 'rbs'])
for g, subdata in groups:
    with open('../../data/mcmc/' + \
              'SI_I_' + g[0] + '_R' + dict_R[g[1]] + '.pkl', 'rb') as file:
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

#===============================================================================
# Here we consider the strains containing an O1 oeprator. For each estimate of
# K_A and K_I (for a specific R strain), we plot the fold-change prediction
# across each R in our dataset along with the assocaited data for that strain. #===============================================================================

# Pick operator to look at.
op = 'O2'

df_plot = df[df.operator==op]
param_summary_op= param_summary[param_summary.operator==op]


# Define array of IPTG concentrations
IPTG = np.logspace(-9, -1, 200)

# current default color palette
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
gs_dict = {'hspace': 0.1, 'wspace':0.1}

fig, ax = plt.subplots(nrows=6, ncols=6, figsize=(12,12), gridspec_kw=gs_dict)

for j, rbs_fit in enumerate(df_plot.rbs.unique()[::-1]):

    with open('../../data/mcmc/' + \
              'SI_I_' + g[0] + '_R' + dict_R[g[1]] + '.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        gauss_pool_flatchain = unpickler.load()


    # map value of the parameters
    ea, ei = np.mean(gauss_pool_flatchain[:, [0, 1]], axis=0)

    for i, rbs in enumerate(df_plot.rbs.unique()[::-1]):
        # plot the theory using the parameters from the fit.

        ax[j,i].plot(IPTG , mwc.fold_change_log(IPTG * 1E6,
            ea=ea, ei=ei, epsilon=4.5,
            R=df_plot[(df_plot.rbs == rbs)].repressors.unique(),
            epsilon_r=df_plot.binding_energy.unique()),
            color=colors[-(j+3)])

        # plot 95% HPD region using the variability in the MWC parameters

        cred_region = mwc.mcmc_cred_region(IPTG * 1E6,
            gauss_pool_flatchain, epsilon=4.5,
            R=df_plot[(df_plot.rbs == rbs)].repressors.unique(),
            epsilon_r=df_plot.binding_energy.unique())
        ax[j,i].fill_between(IPTG , cred_region[0,:], cred_region[1,:],
                        alpha=0.5, color=colors[-(j+3)])

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
                        'o', markeredgewidth=1, markeredgecolor=colors[-(i+3)],
                         markerfacecolor='w', markersize=7)
            ax[j,i].set_axis_bgcolor("#DFDFE5")
        else:
            ax[j,i].errorbar(np.sort(df_plot[df_plot.rbs==rbs].IPTG_uM.unique()) / 1E6, fc_mean,
                yerr=fc_err, fmt='o', markersize=7, label=rbs, color=colors[-(i+3)])

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

plt.savefig('../../data/mcmc/figS22_' + today + \
                '.pdf',bbox_inches='tight')

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
# Load in flatchain from global fit
#===============================================================================

with open('../../data/mcmc/' + 'alldata_Ka_Ki.pkl', 'rb') as file:
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
# We choose to plot the strains associated with each operator, though of course,
# we could have taken a different approach.
# We plot the K_A and K_I associated with each R strain for operators O1,O2, and
# O3.
#===============================================================================

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
    plt.savefig(output + '/figS20_' + today + \
               '_Ka_summary_' + op + '.pdf',bbox_inches='tight')

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
    plt.savefig(output + '/figS20_' + today + \
               '_Ki_summary_' + op + '.pdf',bbox_inches='tight')

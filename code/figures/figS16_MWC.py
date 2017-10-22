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

# Set the plotting style.
mwc.set_plotting_style()
# %matplotlib inline
colors = sns.color_palette('colorblind', n_colors=6)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
colors.reverse()
# Magic function to make matplotlib inline; other style specs must come AFTER
# %matplotlib inline

# This enables SVG graphics inline (only use with static plots (non-Bokeh))
# %config InlineBackend.figure_format = 'svg'

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
# Load in the MCMC traces from all strains
#===============================================================================

# We named each .pkl chain as SI_I_x_Rj.pkl where x=(O1,O2,O3)
# and j refers to the repressor copy number, R=(22, 60, 124,
# 260, 1220, 1740). Create dictionary to to convert between
# strain name and R.
dict_R = {'HG104': '22', 'RBS1147': '60', 'RBS446' : '124', \
          'RBS1027': '260', 'RBS1': '1220', 'RBS1L': '1740'}

# # generate DataFrame to save parameter estimates
# param_summary = pd.DataFrame()
#
# # Loop through each strain in groups and calculate HPD
# groups = df.groupby(['operator', 'rbs'])
# for g, subdata in groups:
#     with open('../../data/mcmc/' + \
#               'SI_I_' + g[0] + '_R' + dict_R[g[1]] + '.pkl', 'rb') as file:
#         unpickler = pickle.Unpickler(file)
#         gauss_flatchain = unpickler.load()
#         gauss_flatlnprobability = unpickler.load()
#
#     # map value of the parameters
#     max_idx = np.argmax(gauss_flatlnprobability, axis=0)
#     ea, ei = gauss_flatchain[max_idx, [0, 1]]
#
#     # ea range
#     ea_hpd = mwc.hpd(gauss_flatchain[:, 0],0.95)
#     ei_hpd = mwc.hpd(gauss_flatchain[:, 1],0.95)
#
#     # add values to dataframe
#     param_summary_temp = pd.DataFrame({'rbs':g[1],
#                                 'operator':g[0],
#                                 'repressors':df[df['rbs']==g[1]].repressors.unique()[0],
#                                 'ka_mode':[ea],
#                                 'ka_hpd_max':[ea_hpd[0]],
#                                 'ka_hpd_min':[ea_hpd[1]],
#                                 'ki_mode':[ei],
#                                 'ki_hpd_max':[ei_hpd[0]],
#                                 'ki_hpd_min':[ei_hpd[1]]})
#
#     param_summary = param_summary.append(param_summary_temp, ignore_index=True)


# Load the parameters and data
# params = pd.read_csv('../../data/hill_params.csv')
data = pd.read_csv('../../data/flow_master.csv')

# Ignore any data with repressors no repressors.
params = params[params['repressors'] > 0]
data = data[data['repressors'] > 0]

# Define the concentration range over which to plot.
c_range = np.logspace(-9, -2, 1000)  # in M.
lin_ind = np.where(c_range > 1E-7)[0][0]
# Group parameters by rbs and operator.
grouped_params = params.groupby(['repressors', 'operator'])
reps = params.repressors.unique()

# Instantiate the figure axis and set labels.
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
ops = ['operator O1', 'operator O2', 'operator O3']
panel_labels = ['(A)', '(B)', '(C)']
for i, a in enumerate(ax[:-1]):
    a.set_xlabel('[IPTG] (M)')
    a.set_ylabel('fold-change')
    a.set_title(ops[i], backgroundcolor='#FFEDC0')
    a.set_xlim([1E-9, 1E-2])
    a.set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    a.text(-0.24, 1.03, panel_labels[i], fontsize=12, transform=a.transAxes)
    ax[-1].set_axis_off()

# Define the axes identifiers and the correct colors
axes = {'O1': ax[0], 'O2': ax[1], 'O3': ax[2]}
color_key = {i: j for i, j in zip(reps, colors)}

# Define array of IPTG concentrations
IPTG = np.logspace(-9, -1, 200)

for g, d in grouped_params:
    print('plotting: ',g[0],g[1])
    # if g[1] != 'O3':
    #     continue
    with open('../../data/mcmc/' + \
              'SI_I_' + g[1] + '_R' + str(g[0]) + '.pkl', 'rb') as file:
        unpickler = pickle.Unpickler(file)
        gauss_pool_flatchain = unpickler.load()

    # map value of the parameters
    ea, ei = np.mean(gauss_pool_flatchain[:, [0, 1]], axis=0)

    # Compute the fit value.
    fit = mwc.fold_change_log(c_range * 1E6,
        ea=ea, ei=ei, epsilon=4.5,
        R=df[(df.repressors == g[0]/2)&(df.operator == g[1])].repressors.unique(),
        epsilon_r=df[(df.repressors == g[0]/2)&(df.operator == g[1])].binding_energy.unique())

    # Compute the credible region
    cred_region = mwc.mcmc_cred_region(c_range * 1E6,
        gauss_pool_flatchain, epsilon=4.5,
        R=df[(df.repressors == g[0]/2)&(df.operator == g[1])].repressors.unique(),
        epsilon_r=df[(df.repressors == g[0]/2)&(df.operator == g[1])].binding_energy.unique())

    # Plot it
    _ = axes[g[1]].plot(c_range[lin_ind:], fit[lin_ind:], '-',
                        color=color_key[g[0]],
                        lw=1, label='__nolegend__')
    _ = axes[g[1]].plot(c_range[:lin_ind], fit[:lin_ind], '--',
                        color=color_key[g[0]],
                        lw=1, label='__nolegend__')
    _ = axes[g[1]].fill_between(c_range, cred_region[0, :], cred_region[1, :],
                                color=color_key[g[0]], alpha=0.4)
# Plot the data.
grouped_data = data.groupby(['repressors', 'operator'])
for g, d in grouped_data:

    # Arguments for calculation of mean and SEM
    args = [np.mean, np.std, len]
    _d = d.groupby('IPTG_uM')['IPTG_uM', 'fold_change_A'].agg(args)
    sem = _d['fold_change_A']['std'] / np.sqrt(_d['fold_change_A']['len'])

    # Plot the data.
    _ = axes[g[1]].errorbar(_d['IPTG_uM']['mean'] / 1E6,
                            _d['fold_change_A']['mean'], sem, fmt='o',
                            color=color_key[2 * g[0]], markersize=3, lw=0.75,
                            label=2 * g[0])

# set the legend.
_ = ax[0].legend(title='rep. per cell', loc='upper left')

# Scale, add panel labels, and save.
mwc.scale_plot(fig, 'two_row')
fig.set_size_inches(6, 4.5)
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS16_MWC.svg')


########################################################################
# if all fails, this code is useful for plotting
########################################################################

    # print(df[(df.repressors == g[0]/2)&(df.operator == g[1])].binding_energy.unique())
#
# # for i, rbs in enumerate(df_plot.rbs.unique()[::-1]):
#     # plot the theory using the parameters from the fit.
#
# axes[g[1]].plot(IPTG , mwc.fold_change_log(IPTG * 1E6,
#     ea=ea, ei=ei, epsilon=4.5,
#     R=df[(df.repressors == g[0]/2)&(df.operator == g[1])].repressors.unique(),
#     epsilon_r=df[(df.repressors == g[0]/2)&(df.operator == g[1])].binding_energy.unique()),
#     lw=1, label='__nolegend__',
#     color=color_key[g[0]])
#
# # plot 95% HPD region using the variability in the MWC parameters
#
# cred_region = mwc.mcmc_cred_region(IPTG * 1E6,
#     gauss_pool_flatchain, epsilon=4.5,
#     R=df[(df.repressors == g[0]/2)&(df.operator == g[1])].repressors.unique(),
#     epsilon_r=df[(df.repressors == g[0]/2)&(df.operator == g[1])].binding_energy.unique())
#
# axes[g[1]].fill_between(IPTG , cred_region[0,:], cred_region[1,:],
#                 alpha=1, color=color_key[g[0]],
#                 lw=1, label='__nolegend__', alpha=0.4)
#
# # compute the mean value for each concentration
#
# fc_mean = df[(df.repressors == g[0])&(df.operator == g[1])].groupby('IPTG_uM').fold_change_A.mean()
#
# # compute the standard error of the mean
#
# fc_err = df[(df.repressors == g[0])&(df.operator == g[1])].groupby('IPTG_uM').fold_change_A.std() / \
# np.sqrt(df[(df.repressors == g[0])&(df.operator == g[1])].groupby('IPTG_uM').size())

#

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
sys.path.append(os.path.abspath('../analysis/'))
import mwc_induction_utils as mwc

# Useful plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import seaborn as sns
import corner

import pymc3 as pm
import theano.tensor as tt

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

#---------------------------------------------------------------------
# for Hill
#---------------------------------------------------------------------

# Define the necessary functions
def jeffreys_prior(val):
    return -tt.log(val)


def gen_hill(params, c):
    ep, a, b, n = params
    numer = (c * np.exp(-ep))**n
    denom = 1 + numer
    return a + b * numer / denom


# Load the data and ignore those with no repressors..
data = pd.read_csv('../../data/flow_master.csv')
data = data[data['repressors'] > 0]

#-------------------------------------------------------------------------------
# script to run MCMC for hill fits.
#-------------------------------------------------------------------------------

# # %% Group the data by operator and repressor.
# samples_df = {}
# grouped_data = data.groupby(['repressors', 'operator'])
# for g, d in grouped_data:
#     with pm.Model() as model:
#
#         # Define the priors.
#         a = pm.Uniform('a', lower=0, upper=1, testval=0.1)
#         b = pm.Uniform('b', lower=-1, upper=1, testval=0)
#         ep = pm.Normal('ep', mu=0, sd=10, testval=3)
#         sigma = pm.DensityDist('sigma', jeffreys_prior, testval=1)
#         n = pm.Normal('n', mu=0, sd=100,  testval=2)
#
#         # Define the likelihood
#         mu = gen_hill([ep, a, b, n], d['IPTG_uM'].values)
#         like = pm.Normal('like', mu=mu, sd=sigma,
#                          observed=d['fold_change_A'].values)
#
#         # # Do the sampling.
#         trace = pm.sample(draws=500, tune=1000)
#
#         # Convert it to a DataFrame.
#
#         df = pm.trace_to_dataframe(trace)
#         df['logp'] = pm.stats._log_post_trace(
#             model=model, trace=trace).sum(axis=1)
#
#         # Compute the statistics.
#         samples_df[g] = df

#-------------------------------------------------------------------------------
# load in MCMC traces from Hill fits
#-------------------------------------------------------------------------------

# samples_df = pd.read_csv('../../data/.csv')

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


# Load the parameters and data
params = pd.read_csv('../../data/hill_params.csv')
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
fig, ax = plt.subplots(3, 2)
ax = ax.ravel()
ops = ['operator O1', 'operator O1',
       'operator O2', 'operator O2',
       'operator O3', 'operator O3']
panel_labels = ['(A)', '(B)', '(C)']
for i, a in enumerate(ax[:-1]):
    a.set_xlabel('[IPTG] (M)')
    a.set_ylabel('fold-change')
    a.set_title(ops[i], backgroundcolor='#FFEDC0')
    a.set_xlim([1E-9, 1E-2])
    a.set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    a.text(-0.24, 1.03, panel_labels[i], fontsize=12, transform=a.transAxes)
    ax[-1].set_axis_off()


# ---------------- ---------------- ---------------- ----------------
# for Hill function
# ---------------- ---------------- ---------------- ----------------


# Define the axes identifiers and the correct colors
axes = {'O1': ax[0], 'O2': ax[2], 'O3': ax[4]}
color_key = {i: j for i, j in zip(reps, colors)}
stat_df = pd.DataFrame([], columns=['operator', 'repressors',
                                    'param', 'mode', 'hpd_min',
                                    'hpd_max'])
for g, d in grouped_params:
    # Properly package the parameters for the function.
    slc = d.loc[:, ['param', 'mode']]
    fit_df = samples_df[(g[0] / 2, g[1])]

    fit_df['k'] = np.exp(df['ep'])
    stats = mwc.compute_statistics(fit_df)
    stats['k']

    # Make the stats a DataFrame.
    keys = ['k', 'ep', 'a', 'b', 'n']
    for _, key in enumerate(keys):
        param_dict = dict(operator=g[1], repressors=g[0],
                          mode=stats[key][0], hpd_min=stats[key][1],
                          hpd_max=stats[key][2], param=key)
        _stats = pd.Series(param_dict)
        stat_df = stat_df.append(_stats, ignore_index=True)

    _df = pd.DataFrame(fit_stats)
    _df.insert(0, 'repressors', 2 * g[0])
    _df.insert(0, 'operator', g[1])
    stat_df.append(_df)
    modes = [stats['ep'][0] - np.log(1E6),
             stats['a'][0], stats['b'][0],
             stats['n'][0]]

    # Compute the fit value.
    fit = gen_hill(modes, c_range)

    # Compute the credible region
    cred_region = np.zeros((2, len(c_range)))
    fit_df['k'] = fit_df['ep'] - np.log(1E6)
    param_vals = fit_df.loc[:, ['k', 'a', 'b', 'n']].values
    param_vals = param_vals.T
    for i, c in enumerate(c_range):
        fc = gen_hill(param_vals, c)
        cred_region[:, i] = mwc.hpd(fc, mass_frac=0.95)

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

# ---------------- ---------------- ---------------- ----------------
# for MWC function
# ---------------- ---------------- ---------------- ----------------

# Define the axes identifiers and the correct colors
axes = {'O1': ax[1], 'O2': ax[3], 'O3': ax[5]}
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
mwc.scale_plot(fig, 'three_row_two_column')
fig.set_size_inches(6, 4.5)
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS16_MWC_combined_good.svg')

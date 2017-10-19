import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mwc_induction_utils as mwc
import pymc3 as pm
import seaborn as sns
import theano.tensor as tt
import scipy.optimize

# Set the plotting style.
mwc.set_plotting_style()
%matplotlib inline
colors = sns.color_palette('colorblind', n_colors=6)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
colors.reverse()

# %%
# ######################


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

# %% Group the data by operator and repressor.
stat_df = []
samples_df = {}
grouped_data = data.groupby(['repressors', 'operator'])
for g, d in grouped_data:
    with pm.Model() as model:
        g
        # Define the priors.
        a = pm.Normal('a', mu=1, sd=10, testval=0.1)
        b = pm.Normal('b', mu=0, sd=10, testval=0)
        ep = pm.Normal('ep', mu=0, sd=10, testval=3)
        sigma = pm.DensityDist('sigma', jeffreys_prior, testval=1)
        n = pm.Normal('n', mu=0, sd=100,  testval=2)

        # Define the likelihood
        mu = gen_hill([ep, a, b, n], d['IPTG_uM'].values)
        like = pm.Normal('like', mu=mu, sd=sigma,
                         observed=d['fold_change_A'].values)

        # # Do the sampling.
        trace = pm.sample(draws=5000, tune=10000)

        # Convert it to a DataFrame.

        df = pm.trace_to_dataframe(trace)
        df['logp'] = pm.stats._log_post_trace(
            model=model, trace=trace).sum(axis=1)

        # Compute the statistics.
        stats = mwc.compute_statistics(df)
        stat_df.append(pd.DataFrame(stats))
        samples_df[g] = df

# %%
# Load the parameters and data
params = pd.read_csv('../../data/hill_params.csv')
data = pd.read_csv('../../data/flow_master.csv')

# Ignore any data with repressors no repressors.
params = params[params['repressors'] > 0]
data = data[data['repressors'] > 0]

# Define the concentration range over which to plot.
c_range = np.logspace(-9, -2, 500)  # in M.
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
    # a.set_xscale('log')
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

# compute and plot the fits.
for g, d in grouped_params:
    # Properly package the parameters for the function.
    slc = d.loc[:, ['param', 'mode']]
    fit_df = samples_df[(g[0] / 2, g[1])]
    fit_stats = mwc.compute_statistics(fit_df)
    modes = [fit_stats['ep'][0] - np.log(1E6),
             fit_stats['a'][0], fit_stats['b'][0],
             fit_stats['n'][0]]

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

# set the legend.
_ = ax[0].legend(title='rep. per cell', loc='upper left')

# Scale, add panel labels, and save.
mwc.scale_plot(fig, 'two_row')
fig.set_size_inches(6, 4.5)
plt.tight_layout()
# plt.savefig('../../figures/SI_figs/figS16.svg')

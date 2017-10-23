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


# Define the necessary functions
def jeffreys_prior(val):
    return -tt.log(val)


def gen_hill(params, c):
    ep, a, b, n = params
    numer = (c * np.exp(ep))**n
    denom = 1 + numer
    return a + b * numer / denom


def sample_mcmc(df, iptg='IPTG_uM', fc='fold_change_A',
                draws=1500, tune=10000):
    with pm.Model() as model:

        # Define the priors.
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        a = BoundedNormal('a', mu=0.5, sd=5, testval=0.1)
        b = pm.Uniform('b', lower=-1, upper=1, testval=0.5)
        sigma = pm.HalfNormal('sigma', sd=1)
        ep = pm.Normal('ep', mu=0, sd=100, testval=7)
        n = pm.Uniform('n', lower=0, upper=100, testval=2)

        # Set the data.
        c_vals = df[iptg].values
        obs = df[fc].values
        params = [ep, a, b, n]
        theo = gen_hill(params, c_vals)

        # Define the likelihood and sample.
        like = pm.Normal('like', mu=theo, sd=sigma, observed=obs)
        trace = pm.sample(draws=draws, tune=tune)

        # Convert to DataFrame and return.
        df = pm.trace_to_dataframe(trace)
        df['logp'] = pm.stats._log_post_trace(
            trace=trace, model=model).sum(axis=1)
    return df


# Load the data and ignore those with no repressors..
data = pd.read_csv('../../data/flow_master.csv')
data = data[data['repressors'] > 0]

# %% Group the data by operator and repressor.
samples_df = {}
concat_df = []
grouped_data = data.groupby(['repressors', 'operator'])
for g, d in grouped_data:
    try:
        df = sample_mcmc(d)
        df.insert(0, 'repressors', g[0])
        df.insert(0, 'operator', g[1])
        samples_df[g] = df
        concat_df.append(df)
    except:
        print('Rerun sample {0}'.format(g))
        samples_df[g] = None
# %%
# Rerun the two aberrant samples.
reruns = [(610, 'O2')]
for g in reruns:
    samples_df[g] = None

for g in reruns:
    slc_data = data[(data['repressors'] == g[0]) & (data['operator'] == g[1])]
    with pm.Model() as model:

        # Define the priors.
        BoundedNormal = pm.Bound(pm.Normal, lower=0, upper=1)
        a = BoundedNormal('a', mu=0.5, sd=5, testval=0.1)
        b = pm.Uniform('b', lower=-1, upper=1, testval=0.5)
        sigma = pm.HalfNormal('sigma', sd=1)
        ep = pm.Normal('ep', mu=0, sd=100, testval=7)
        n = pm.Normal('n', mu=0, sd=100, testval=2)

        # Set the data.
        c_vals = slc_data['IPTG_uM'].values
        obs = slc_data['fold_change_A'].values
        params = [ep, a, b, n]
        theo = gen_hill(params, c_vals)

        # Define the likelihood and sample.
        like = pm.Normal('like', mu=theo, sd=sigma, observed=obs)
        trace = pm.sample(draws=1500, tune=10000)

        # Convert to DataFrame and return.
        df = pm.trace_to_dataframe(trace)
        df['logp'] = pm.stats._log_post_trace(
            trace=trace, model=model).sum(axis=1)
        df.insert(0, 'repressors', g[0])
        df.insert(0, 'operator', g[1])
        samples_df[g] = df
        concat_df.append(df)

all_traces = pd.concat(concat_df, axis=0)
all_traces.to_csv('../../data/generic_hill_traces.csv', index=False)
# %%
# Load the parameters and data
data = pd.read_csv('../../data/flow_master.csv')
traces = pd.read_csv('../../data/generic_hill_traces.csv')


# Ignore any data with repressors no repressors.
data = data[data['repressors'] > 0]
grouped_data = data.groupby(by=['repressors', 'operator'])

# Define the concentration range over which to plot.
c_range = np.logspace(-3, 4, 500)  # in M.
lin_ind = np.where(c_range / 1E6 > 1E-7)[0][0]
# Group parameters by rbs and operator.
reps = data.repressors.unique()

# Instantiate the figure axis and set labels.
fig, ax = plt.subplots(2, 2)
ax = ax.ravel()
ops = ['operator O1', 'operator O2', 'operator O3']
panel_labels = ['(A)', '(B)', '(C)']
for i, a in enumerate(ax[:-1]):
    a.set_xlabel('[IPTG] (M)')
    a.set_ylabel('fold-change')
    a.set_title(ops[i], backgroundcolor='#FFEDC0')
    # a.set_xlim([1E-9, 1E-2])
    a.set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    a.text(-0.24, 1.03, panel_labels[i], fontsize=12, transform=a.transAxes)
    ax[-1].set_axis_off()

# Define the axes identifiers and the correct colors
axes = {'O1': ax[0], 'O2': ax[1], 'O3': ax[2]}
color_key = {i: j for i, j in zip(reps, colors)}
stat_df = pd.DataFrame([], columns=['operator', 'repressors',
                                    'param', 'mode', 'hpd_min',
                                    'hpd_max'])
for g, d in grouped_data:
    # Properly package the parameters for the function.
    # slc = d.loc[:, ['param', 'mode']]
    # fit_df = samples_df[(g[0], g[1])]
    fit_df = traces[(traces['repressors'] == g[0])
                    & (traces['operator'] == g[1])]
    fit_df['k'] = np.exp(df['ep'])

    # Find the max of the logp
    ind = np.argmax(fit_df['logp'])
    a_mode, b_mode, ep_mode, _, n_mode, _, _, _ = traces.iloc[ind].values
    k_mode = np.exp(ep_mode)

    stats = mwc.compute_statistics(fit_df)

    # Make the stats a DataFrame.
    keys = ['k', 'ep', 'a', 'b', 'n']
    for _, key in enumerate(keys):
        param_dict = dict(operator=g[1], repressors=g[0],
                          mode=stats[key][0], hpd_min=stats[key][1],
                          hpd_max=stats[key][2], param=key)
        _stats = pd.Series(param_dict)
        stat_df = stat_df.append(_stats, ignore_index=True)

    # _df = pd.DataFrame(fit_stats)
    # _df.insert(0, 'repressors', 2 * g[0])
    # _df.insert(0, 'operator', g[1])
    # stat_df.append(_df)
    modes = [stats['ep'][0],
             stats['a'][0], stats['b'][0],
             stats['n'][0]]

    # Compute the fit value.
    fit = gen_hill(modes, c_range)

    # Compute the credible region
    cred_region = np.zeros((2, len(c_range)))
    # fit_df['k'] = fit_df['ep'] - np.log(1E6)
    param_vals = fit_df.loc[:, ['ep', 'a', 'b', 'n']].values
    param_vals = param_vals.T
    for i, c in enumerate(c_range):
        fc = gen_hill(param_vals, c)
        cred_region[:, i] = mwc.hpd(fc, mass_frac=0.95)

    # Plot it
    _ = axes[g[1]].plot(c_range[lin_ind:] / 1E6, fit[lin_ind:], '-',
                        color=color_key[g[0]],
                        lw=1, label='__nolegend__')
    _ = axes[g[1]].plot(c_range[:lin_ind] / 1E6, fit[:lin_ind], '--',
                        color=color_key[g[0]],
                        lw=1, label='__nolegend__')
    _ = axes[g[1]].fill_between(c_range / 1E6, cred_region[0, :], cred_region[1, :],
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
                            color=color_key[g[0]], markersize=3, lw=0.75,
                            label=2 * g[0])

# set the legend.
_ = ax[0].legend(title='rep. per cell', loc='upper left')

# Scale, add panel labels, and save.
mwc.scale_plot(fig, 'two_row')
fig.set_size_inches(6, 4.5)
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS16.svg')

# %%
import corner
%matplotlib inline
a = samples_df[(11, 'O3')]

_ = corner.corner(a.drop('logp', axis=1))
samples_df[(11, 'O3')]
# %% Save the general hill parameters.
stat_df
stat_df.to_csv('../../data/general_hill_params.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mwc_induction_utils as mwc
import scipy.optimize
from tqdm import tqdm
import glob

import pymc3 as pm
import theano.tensor as tt
import pandas as pd

mwc.set_plotting_style()
colors = sns.color_palette('colorblind')
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
colors.reverse()
# %% Define the relevant functions.


def generic_hill_fn(a, b, c, ep, n):
    numer = (c * np.exp(-ep))**n
    denom = 1 + numer
    return a + b * numer / denom


def jeffreys(val):
    return -tt.log(val)


def trace_to_df(trace, model):
    """
    Converts the trace from a pymc3 sampling trace to a
    Pandas DataFrame.
    """

    def compute_logp(chain):
        """
        Computes the log probability of the provided trace
        at a given chain.
        """
        names = trace.varnames
        var_dict = {}
        for n in names:
            var_dict[n] = trace.get_values(n, chains=chain)
        sample_df = pd.DataFrame(var_dict)

        logp = [model.logp(sample_df.iloc[step]
                           ) for step in range(len(sample_df))]
        return logp

    chains = trace.chains
    for c in tqdm(chains, desc='Processing chains'):
        logp = compute_logp(c)
        if c == 0:
            df = pm.trace_to_dataframe(trace, chains=c)
            df.insert(np.shape(df)[1], 'logp', logp)
        else:
            _df = pm.trace_to_dataframe(trace, chains=c)
            _df.insert(np.shape(_df)[1], 'logp', logp)
            df.append(_df, ignore_index=True)

    return df


def compute_statistics(df, ignore_vars='logp'):
    """
    Computes the mode and highest probability density (hpd)
    of the parameters in a given dataframe.  """  # Set up the multi indexing.
    var_names = np.array(df.keys())
    if ignore_vars is not None:
        var_names = var_names[var_names != ignore_vars]

    # Generate arrays for indexing and zip as tuples.
    names = [var for var in var_names] * 3
    stats = ['mode', 'hpd_min', 'hpd_max']
    stats = np.array([[s] * len(var_names) for s in stats]).flatten()
    tuples = list(zip(*[names, stats]))

    # Define the index.
    index = pd.MultiIndex.from_tuples(tuples, names=['var', 'stat'])

    # Determine the mode for each
    mode_ind = np.argmax(df['logp'])
    stat_vals = [df.iloc[mode_ind][var] for var in var_names]
    # Compute the min and max vals of the HPD.
    hpd_min, hpd_max = [], []
    for i, var in enumerate(var_names):
        _min, _max = mwc.hpd(df[var], 0.95)
        hpd_min.append(_min)
        hpd_max.append(_max)
    for _ in hpd_min:
        stat_vals.append(_)
    for _ in hpd_max:
        stat_vals.append(_)

    # Add them to the array for the multiindex
    flat_vals = np.array([stat_vals]).flatten()
    var_stats = pd.Series(flat_vals, index=index)

    return var_stats


# Load in the data files.
data = pd.read_csv('data/flow_master.csv')

# Separate only the O2 RBS 1027 files.
data = data[(data['operator'] == 'O2') & (data['rbs'] == 'RBS1027')]

# %%Set up the inference using PyMC3.
model = pm.Model()
with model:
    # Set the pirors as uniform.
    a = pm.Uniform('a', lower=-1.0, upper=1.0, testval=0.1)
    b = pm.Uniform('b', lower=-7.0, upper=7.0, testval=0)
    ep = pm.Uniform('ep', lower=-10, upper=10, testval=3)
    sigma = pm.DensityDist('sigma', jeffreys, testval=1)
    n = pm.Uniform('n', lower=0, upper=100,  testval=2)

    # Define the concentration constant
    IPTG = data['IPTG_uM'].unique()

    # Compute the expected value.
    fc = generic_hill_fn(a, b, data['IPTG_uM'].values, ep, n)

    # Define the likelihood.
    like = pm.Normal('like', mu=fc, sd=sigma,
                     observed=data['fold_change_A'].values)

    # Sample the distribution.
    step = pm.Metropolis()
    start = pm.find_MAP(model=model, fmin=scipy.optimize.fmin_powell)
    burn = pm.sample(draws=10000, njobs=10, step=step, start=start)
    step = pm.Metropolis()
    trace = pm.sample(draws=50000, tune=100000, njobs=10, step=step,
                      start=burn[-1])

    # Convert the trace to a dataframe.
    df = trace_to_df(trace, model)
    df['kd'] = np.exp(df['ep'])

    # Compute the statistics.
    stats = compute_statistics(df)

# %%
stats = compute_statistics(df)
stats['kd']
# Compute the credible region for the fold-change curve.
c_range = np.logspace(-2, 4, 500)
cred_region = np.zeros([2, len(c_range)])
for i, c in enumerate(c_range):
    params = (df['a'], df['b'], c, df['ep'], df['n'])
    fc_val = generic_hill_fn(*params)
    cred_region[:, i] = mwc.hpd(fc_val, 0.95)

# Extract the modes and plot the fit + credible regions.
modes = {}
grouped = stats.groupby('var')
for g, d in grouped:
    modes[g] = d[0]

fit = generic_hill_fn(modes['a'], modes['b'], c_range, modes['ep'], modes['n'])


fig, ax = plt.subplots(1, 1, figsize=(5, 3))
grouped = data.groupby('IPTG_uM')
for g, d in grouped:
    sem = np.std(d['fold_change_A']) / np.sqrt(len(d))
    mean = d['fold_change_A'].mean()
    if g == 5000.0:
        label = 'data'
    else:
        label = '__nolegend__'
    ax.errorbar(g / 1E6, mean, yerr=sem, linestyle='none', color='r', zorder=1,
                label='__nolegend__')
    ax.plot(g / 1E6, mean, marker='o', markerfacecolor='w', markersize=4,
            markeredgewidth=2, markeredgecolor='r', label=label, zorder=2,
            linestyle='none')

ax.fill_between(c_range / 1E6, cred_region[0, :], cred_region[1, :],
                color='r', label='__nolegend__', alpha=0.5)
ax.plot(c_range / 1E6, fit, color='r', label='Hill function fit', zorder=1)
ax.legend(loc='upper left')
ax.set_xlabel('[IPTG] (M)', fontsize=11)
ax.set_ylabel('fold-change', fontsize=11)
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
ax.set_xscale('log')
ax.set_ylim([-0.01, 1])
ax.set_xlim([1E-8, 1E-2])
ax.set_title('$R$ = 260, Operator O2', backgroundcolor='#FFEDCE', y=1.03,
             fontsize=11)

plt.savefig('figures/SI_figs/figS16_generic_hill.pdf', bbox_inches='tight')


# %% Computing the parameters for all eighteen strains.
data = pd.read_csv('data/flow_master.csv')

# Set up the inference using PyMC3.
kd_df = pd.DataFrame([], columns=['operator', 'repressors', 'param',
                                  'mode', 'hpd_min', 'hpd_max'])

grouped = data.groupby(['operator', 'repressors'])
_stats = []
for g, d in tqdm(grouped):
    model = pm.Model()
    with model:
        # Set the pirors as uniform.
        a = pm.Uniform('a', lower=0, upper=1.0, testval=0.1)
        b = pm.Uniform('b', lower=0, upper=1.0, testval=0)
        ep = pm.Uniform('ep', lower=-10, upper=10, testval=3)
        sigma = pm.DensityDist('sigma', jeffreys, testval=1)
        n = pm.Uniform('n', lower=0, upper=100,  testval=2)

        # Compute the expected value.
        fc = generic_hill_fn(a, b, d['IPTG_uM'].values, ep, n)

        # Define the likelihood.
        like = pm.Normal('like', mu=fc, sd=sigma,
                         observed=d['fold_change_A'].values)

        # Sample the distribution.
        step = pm.Metropolis()
        start = pm.find_MAP(model=model, fmin=scipy.optimize.fmin_powell)
        burn = pm.sample(draws=10000, njobs=10, step=step, start=start)
        step = pm.Metropolis()
        trace = pm.sample(draws=50000, tune=100000, njobs=1, step=step,
                          start=burn[-1])

        # Convert the trace to a dataframe.
        df = trace_to_df(trace, model)
        df['kd'] = np.exp(df['ep'])

        # Compute the statistics.
        stats = compute_statistics(df)
        param_keys = ['kd', 'a', 'b', 'n']
        for _, key in enumerate(param_keys):
            param_dict = dict(operator=g[0], repressors=2 * g[1], param=key,
                              mode=stats[key][0], hpd_min=stats[key][1],
                              hpd_max=stats[key][2])
            _df = pd.Series(param_dict)
            kd_df = kd_df.append(_df, ignore_index=True)

kd_df.to_csv('data/hill_params.csv', index=False)

# %%
# load the cv file of all of the parameters.
params = pd.read_csv('../../data/hill_params.csv')

# Keep only those with repressors greater than zero
params = params[params['repressors'] > 0]


# %%Plot the parameter values.
plt.close('all')
fig, ax = plt.subplots(2, 2, figsize=(6, 4))
# Add figure text labels
fig.text(0.0, 0.99, '(A)', fontsize=12)
fig.text(0.5, 0.99, '(B)', fontsize=12)
fig.text(0.0, 0.5, '(C)', fontsize=12)
fig.text(0.5, 0.5, '(D)', fontsize=12)

ax = ax.ravel()

# Group by operator and repressor count.
grouped = params.groupby(['operator', 'repressors'])
params
axes = {'O1': ax[0], 'O2': ax[1], 'O3': ax[2]}
reps = np.sort(params['repressors'].unique())
positions = np.arange(1, len(reps) + 1, 1)
pos = {i: j for i, j in zip(reps, positions)}
offset = {'O1': -0.2, 'O2': 0, 'O3': 0.2}
glyphs = {'O1': 'o', 'O2': 'D', 'O3': 's'}
ylabels = {'a': 'leakiness', 'b': 'dynamic range',
           'n': 'Hill coefficient', 'kd': r'$K$ ($\mu$M)'}
color_dict = {i: j for i, j in zip(params['repressors'].unique(), colors)}
c_range = np.logspace(-2, 4)


for d in glyphs.keys():
    ax[0].plot([], [], marker=glyphs[d], markerfacecolor='w',
               markeredgecolor='k', markersize=3, markeredgewidth=1.5, label=d,
               linestyle='none')
ax[0].legend(ncol=3, bbox_to_anchor=(1.0, 1.3), fontsize=10)

ax_dict = {'a': 0, 'b': 1, 'kd': 2, 'n': 3}
for a in ax:
    a.set_xlim([0.7, 6.3])
for g, d in grouped:

    for i, p in enumerate(d['param'].unique()):
        position = pos[g[1]] + offset[g[0]]
        param_id = d[d['param'] == p]
        mode = param_id['mode'].unique()
        min_val = param_id['hpd_min'].unique()
        max_val = param_id['hpd_max'].unique()

        if min_val < 0:
            min_val = 0
        if max_val < 0:
            max_val = 0
        ax[ax_dict[p]].vlines(position, min_val, max_val,
                              color=color_dict[g[1]],
                              linewidth=1)
        ax[ax_dict[p]].plot(position, mode, marker=glyphs[g[0]],
                            markerfacecolor='w', markeredgecolor=color_dict[g[1]],
                            markersize=5,
                            markeredgewidth=1.5)
        ax[ax_dict[p]].set_ylabel(ylabels[p], fontsize=11)

    # a.xaxis.set_ticklabels(params['repressors'].unique(), y=-0.06)

    a = d[d['param'] == 'a']
    b = d[d['param'] == 'b']
    kd = d[d['param'] == 'kd']
    n = d[d['param'] == 'n']
    # fit = generic_hill_fn(a, b, c_range, kd, n)
    position = pos[g[1]] + offset[g[0]]
    # kd = d[d['param'] == 'n']
    mode = kd['mode'].unique()
    hpd_min = kd['hpd_min'].unique()
    hpd_max = kd['hpd_max'].unique()


ax[2].set_yscale('linear')

for a in ax:
    a.xaxis.grid(False)
    a.xaxis.set_ticks([0.7, 1, 2, 3, 4, 5, 6, 6.3])
    a.xaxis.set_tick_params(labelsize=10)
    a.yaxis.set_tick_params(labelsize=10)
    a.set_xlabel('repressors per cell', fontsize=11)
    a.xaxis.set_ticklabels(
        ['', '22', '60', '124', '260', '1220', '1740', ''], y=-0.06)

for a in [ax[2],  ax[3]]:
    a.set_yscale('log')

plt.tight_layout()
plt.savefig('../../figures/SI_figs/figSX_hill_params.pdf', bbox_inches='tight')

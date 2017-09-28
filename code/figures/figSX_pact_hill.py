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
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
# colors.reverse()

# Define the relevant functions.
def generic_hill_fn(a, b, c, ep, n):
    """
    Computes a generic Hill function of the form
    h = a + b * ( (c * exp(-ep))^n / 1 + (c * exp(-ep))^n)
    """
    numer = (c * np.exp(-ep))**n
    denom = 1 + numer
    return a + b * numer / denom


def fc_function(R, ep_r, hill_args, n_ns=4.6E6):
    h_c = generic_hill_fn(*hill_args)
    return (1 + h_c * (R / n_ns) * np.exp(-ep_r))**-1

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
    of the parameters in a given dataframe.
    """

    # Set up the multi indexing.
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

# Load the data set.
data = pd.read_csv('../../data/flow_master.csv', comment='#')

# Slice out the O2 RBS1027 data.
leak_data = data[(data['IPTG_uM'] == 0) & (data['operator']=='O2') &
                 (data['repressors'] == 130)]

# Fit parameter "a" just from the leakiness.
model = pm.Model()
with model:
    a = pm.Uniform('a', lower=0, upper=10, testval=0.1)
    R = 2 * leak_data['repressors'].values
    ep_r = leak_data['binding_energy'].unique()
    sigma = pm.DensityDist('sigma', jeffreys, testval=1)

    # Compute the expected value.
    fc_exp = fc_function(R, ep_r, (a, 0, 0, 0, 0))

    # Compute the likelihood.
    obs = leak_data['fold_change_A'].values
    like = pm.Normal('likelihood', mu=fc_exp, sd=sigma,
                     observed=obs)
    # find the MAP as the starting position.
    start = pm.find_MAP(model=model, fmin=scipy.optimize.fmin_powell)
    step = pm.Metropolis()
    burn = pm.sample(draws=10000, step=step, start=start, njobs=None)
    step = pm.Metropolis()
    trace = pm.sample(draws=50000, step=step, start=burn[-1], njobs=None)

    # Convert the trace to a dataframe and compute the statistics.
    df = trace_to_df(trace, model=model)
    stats = compute_statistics(df)



O2_data = data[(data['operator'] == 'O2') & (data['rbs'] == 'RBS1027')]
# Set up the MCMC.
model = pm.Model()
with model:
    # Define the priors
    # a = pm.Uniform('a', lower=0, upper=1, testval=0.1)
    a = 1.0
    b = pm.Uniform('b', lower=-1.0, upper=1.0, testval=0)
    ep = pm.Uniform('ep', lower=-7, upper=7, testval=4)
    n = pm.Uniform('n', lower=0, upper=10, testval=2)
    sigma = pm.DensityDist('sigma', jeffreys, testval=1)

    # Define the constants.
    R = 2 * O2_data['repressors'].unique()
    ep_r = O2_data['binding_energy'].unique()
    IPTG = O2_data['IPTG_uM'].values

    # Compute the expected value.
    hill_args = (a, b, IPTG, ep, n)
    fc_exp = fc_function(R, ep_r, hill_args)

    # Compute the likelihood.
    like = pm.Normal('likelihood', mu=fc_exp, sd=sigma,
                     observed=O2_data['fold_change_A'].values)

    # Find the MAP and sample.
    start = pm.find_MAP(model=model, fmin=scipy.optimize.fmin_powell)
    step = pm.Metropolis()
    burn = pm.sample(draws=10000, start=start, step=step, njobs=None)
    step = pm.Metropolis()
    trace = pm.sample(draws=50000, njobs=None,
                      start=burn[-1], step=step)

    # Convert the trace to a dataframe and compute the statistics.
    df = trace_to_df(trace,  model)
    df['kd'] = np.exp(df['ep'])
    stats = compute_statistics(df)



# Extract the modes.
modes = {}
grouped = stats.groupby('var')
for g, d in grouped:
    modes[g] = d[0]


# Define the figure axis.
plt.close('all')
fig, ax = plt.subplots(2, 2, figsize=(6, 4))
ax = ax.ravel()

# Define the data and plotting information.
data = pd.read_csv('../../data/flow_master.csv', comment='#')
data = data[data['repressors'] > 0]
axes = {'O1': ax[0], 'O2': ax[1], 'O3': ax[2]}
binding_energy = {i: j for i, j in zip(data['operator'].unique(),
                                  data['binding_energy'].unique())}
color_key = {i:j for i, j in zip(np.sort(data['repressors'].unique()), colors)}


# Set the concentrations overwhich to plot.
c_range = np.logspace(-2, 4, 500)

# plot the fits, data, and credible regions.
grouped = data.groupby(['operator', 'repressors'])
for g, d in grouped:
    # Compute the mode fit and plot.
    fit = fc_function(2 * g[1], binding_energy[g[0]], (modes['a'], modes['b'], c_range,
                                   modes['ep'], modes['n']))

    # Compute the credible regions.
    cred_region = np.zeros([2, len(c_range)])
    for i, c in enumerate(c_range):
        val = fc_function(2 * g[1], binding_energy[g[0]],
                          (df['a'], df['b'], c, df['ep'], df['n']))
        cred_region[:, i] = mwc.hpd(val, mass_frac=0.95)


    axes[g[0]].plot(c_range / 1E6, fit, color=color_key[g[1]], zorder=1,
                    label='__nolegend__')
    axes[g[0]].fill_between(c_range / 1E6, fit, cred_region[0,:],
                            cred_region[1, :], color=color_key[g[1]],
                            alpha=0.5, zorder=0)

    # Groupby the concentration and plot the mean/sem
    _grouped = d.groupby('IPTG_uM')
    for _g, _d, in _grouped:
        # Compute the mean and sem.
        mean_fc = _d['fold_change_A'].mean()
        mean_sem = _d['fold_change_A'].std() / np.sqrt(len(_d))

        # Determine the glyph type.
        if (g[1] == 130) & (g[0] == 'O2'):
            face = 'w'
        else:
            face = color_key[g[1]]

        axes[g[0]].errorbar(_g / 1E6, mean_fc, yerr=mean_sem,
                            linestyle='none', color=color_key[g[1]])

        axes[g[0]].plot(_g / 1E6, mean_fc, 'o', markerfacecolor=face,
                                markersize=5,
                            markeredgecolor=color_key[g[1]],
                            markeredgewidth=1)

for a in ax:
    a.set_xscale('log')
    a.set_xlabel('[IPTG] (M)')
    a.set_ylabel('fold-change')

ax[-1].set_axis_off()
plt.tight_layout()
plt.show()

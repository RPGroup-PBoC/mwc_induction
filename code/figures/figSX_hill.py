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
# Define the relevant functions.
def generic_hill_fn(a, b, c, ep, n, log=True):
    if log == True:
        numer = (c * np.exp(-ep))**n
    elif log == False:
        numer = (c / ep)**n
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

def cred_region(df, mass_frac=0.95):
    cred_region = np.zeros([])


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
        hpd_min.append(stat_vals[i] - _min)
        hpd_max.append(_max - stat_vals[i])
    for _ in hpd_min:
        stat_vals.append(_)
    for _ in hpd_max:
        stat_vals.append(_)

    # Add them to the array for the multiindex
    flat_vals = np.array([stat_vals]).flatten()
    var_stats = pd.Series(flat_vals, index=index)

    return var_stats


# Load in the data files.
data = pd.read_csv('../../data/flow_master.csv')

# Separate only the O2 RBS 1027 files.
data = data[(data['operator'] == 'O2') & (data['rbs'] == 'RBS1027')]

# Set up the inference using PyMC3.
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
    burn = pm.sample(draws=10000, njobs=None, step=step, start=start)
    step = pm.Metropolis()
    trace = pm.sample(draws=50000, tune=100000, njobs=None, step=step,
                      start=burn[-1])

    # Convert the trace to a dataframe.
    df = trace_to_df(trace, model)
    df['kd'] = np.exp(df['ep'])
# Compute the statistics.
    stats = compute_statistics(df)


# Compute the credible region for the fold-change curve.
c_range = np.logspace(-3, 4, 500)
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

fig, ax = plt.subplots(1,1)
grouped = data.groupby('IPTG_uM')
for g, d in grouped:
    sem = np.std(d['fold_change_A']) / np.sqrt(len(d))
    mean = d['fold_change_A'].mean()
    if g == 5000.0:
        label='data'
    else:
        label='__nolegend__'
    ax.errorbar(g/1E6, mean, yerr=sem, linestyle='none', color='r', zorder=1,
                     label='__nolegend__')
    ax.plot(g/1E6, mean, marker='o', markerfacecolor='w',
            markeredgewidth=2, markeredgecolor='r', label=label, zorder=2,
            linestyle='none')

ax.fill_between(c_range / 1E6, cred_region[0, :], cred_region[1, :],
                color='r', label='__nolegend__', alpha=0.5)
ax.plot(c_range/1E6, fit, color='r', label='Hill function fit', zorder=1)
ax.legend(loc='upper left')
ax.set_xlabel('[IPTG] (M)')
ax.set_ylabel('fold-change')
ax.set_xscale('log')
ax.text

ax.set_ylim([-0.01,1])
ax.set_xlim([1E-9, 1E-2])
ax.set_title('$R$ = 260, Operator O2', backgroundcolor='#FFEDCE', y=1.03)

plt.savefig('../../figures/SI_figs/figSX_generic_hill.pdf', bbox_inches='tight')

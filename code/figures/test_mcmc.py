import numpy as np
import matplotlib.pyplot as plt
import mwc_induction_utils as mwc
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
mwc.set_plotting_style()


def Jeffreys(val):
    return -tt.log(val)


def generic_hill_fn(a, b, c, n, ep):
    numer = (c * np.exp(ep))**n
    denom = 1 + numer
    return a + b * (numer / denom)


# Load the data.
data = pd.read_csv('../../data/flow_master.csv')
O2_data = data[(data['operator'] == 'O3') & (data['rbs'] == 'HG104')]


def sample_mcmc(df, iptg='IPTG_uM', fc='fold_change_A'):
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
        theo = generic_hill_fn(a, b, c_vals, n, ep)

        # Define the likelihood and sample.
        like = pm.Normal('like', mu=theo, sd=sigma, observed=obs)
        trace = pm.sample(draws=1000, tune=10000)

        # Convert to DataFrame and return.
        df = pm.trace_to_dataframe(trace)
        df['logp'] = pm.stats._log_post_trace(
            trace=trace, model=model).sum(axis=1)
    return df


%matplotlib inline
#%%  Compute the statistics and plot it.
stats = mwc.compute_statistics(df)
var_names = ['a', 'b', 'n', 'ep']
modes = {}
for v in var_names:
    modes[v] = stats[v][0]
modes
# %%
fig, ax = plt.subplots(1, 1)
ax.set_xscale('log')

grouped = O2_data.groupby(['IPTG_uM'])['IPTG_uM',
                                       'fold_change_A'].agg([np.mean, np.std, len])
sem = grouped['fold_change_A']['std'] / \
    np.sqrt(grouped['fold_change_A']['len'])
c_range = grouped['IPTG_uM']['mean']

# Plot
_ = ax.errorbar(grouped['IPTG_uM']['mean'], grouped['fold_change_A']['mean'],
                sem, fmt='o')
ax.set_ylim([0, 1.2])

_ = pm.traceplot(trace)
# Compute the fit.
c_vals = np.logspace(-3, 4, 500)
fit = generic_hill_fn(modes['a'], modes['b'], c_vals, modes['n'], modes['ep'])
ax.plot(c_vals, fit, 'b-')

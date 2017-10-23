import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import emcee
import mwc_induction_utils as mwc
import corner
%matplotlib inline


# Define the negative log posterior and priors.
def gen_hill_fn(p, c):
    a, b, ep, n = p
    return a + b * (c * np.exp(ep))**n / (1 + (c * np.exp(ep))**n)


def log_prior(params):
    a, b, ep, n, sigma = params
    lnprior = 0
    if (a < 0) or (a > 1) or (b < 0) or (b > 0):
        return -np.inf
    if (ep < -10) or (ep > 10):
        return -np.inf
    return lnprior


def log_likelihood(params, data):
    a, b, ep, n, sigma = params
    c = data['IPTG_uM']
    fc = data['fold_change_A']
    theory = gen_hill_fn([a, b, ep, n], c)
    return -np.sum((theory - fc)**2) / (2 / sigma**2)


def log_post(params, data):
    a, b, ep, n, sigma = params
    lnp = log_prior(params)
    if lnp == -np.inf:
        return -np.inf
    else:
        return -(len(data) + 1) * np.log(sigma) +\
            log_likeihood(params, data) + lnp


# Load the data.
data = pd.read_csv('../../data/flow_master.csv')

O2_data = data[(data['operator'] == 'O2') & (data['repressors'] == 130)]

# Set up the mcmc.
n_dim = 5
n_walkers = 50
n_burn = 5000
n_steps = 10000

p0 = np.empty((n_walkers, n_dim))
p0[:, [0, 1]] = np.random.uniform(0, 1, size=(n_walkers, 2))
p0[:, 2] = np.random.normal(loc=0, scale=5, size=n_walkers)
p0[:, 3] = np.random.exponential(scale=5, size=n_walkers)
p0[:, 4] = np.random.uniform(1E-5, 0.2, size=n_walkers)

# Instantiate the sampler.
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post, args=(
    O2_data.loc[:, ['IPTG_uM', 'fold_change_A']],), threads=6)

# Perform the burn in.
pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)

# run the mcmc.
out = sampler.run_mcmc(pos, n_steps)


# Look at the corner plot.
_ = corner.corner(sampler.flatchain)

# %%
import pymc3 as pm
with pm.Model() as model:
    # Define the priors.
    a = pm.Uniform('a',  lower=0, upper=1)
    b = pm.Uniform('b', lower=0, upper=1)
    ep = pm.Uniform('ep', lower=-7, upper=7)
    n = pm.Uniform('n', lower=0, upper=100)
    sigma = pm.HalfNormal('sigma', sd=1)

    # Define the observed variables.
    IPTG = O2_data['IPTG_uM'].values
    observed = O2_data['fold_change_A'].values

    # Compute the expected value and likelihood.
    mu = gen_hill_fn((a, b, ep, n))
    likelihood = pm.Normal('likelihood', mu=mu, sd=sigma, observed=obs)

    # Sample it.
    trace = pm.sample(draws=5000, tune=10000, n_jobs=6)

    df = pm.trace_to_dataframe(trace)

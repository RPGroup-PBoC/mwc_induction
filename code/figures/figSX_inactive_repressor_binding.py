import numpy as np
import pandas as pd
import mwc_induction_utils as mwc
import matplotlib.pyplot as plt
import seaborn as sns

import scipy.optimize
import pymc3 as pm
import theano.tensor as tt


# %% Define the necessary functions.
def fold_change_inact_rep(c_range, R, ep_ra, ep_ri, ep_a, ep_i, ep_ai=4.5,
                          n_ns=4.6E6, n_sites=2):
    """
    Computes the fold-change in gene expression for a simple repression
    circuit with binding of active and inactive repressor.
    """

    # Compute the probability of being active.
    prob_act = mwc.pact_log(c_range, ep_a, ep_i, ep_ai=ep_ai, n=n_sites)

    # Compute the repression.
    R_act = prob_act * (R / n_ns) * np.exp(-ep_ra)
    R_inact = (1 - prob_act) * (R / n_ns) * np.exp(-ep_ri)
    fold_change = (1 + R_act + R_inact)**-1
    return fold_change


def jeffreys(val):
    return -tt.log(val)


# Load in the data.
data = pd.read_csv('data/flow_master.csv')

# Look only at O3 data.
O3_data = data[data['operator'] == 'O3']

# %% Define the model and sample the distribution.
model = pm.Model()
with model:
    # Define the priors.
    ep_ra = pm.Uniform('ep_ra', lower=-15, upper=15, testval=-10)
    ep_ri = pm.Uniform('ep_ri', lower=-15, upper=15, testval=-6)
    ep_a = pm.Uniform('ep_a', lower=-15, upper=15, testval=4.5)
    ep_i = pm.Uniform('ep_i', lower=-15, upper=15, testval=-0.6)
    sigma = pm.DensityDist('sigma', jeffreys, testval=1)

    # Compute the expected values.
    IPTG = O3_data['IPTG_uM'].values
    R = O3_data['repressors'].values
    obs = O3_data['fold_change_A'].values
    fc_theo = fold_change_inact_rep(IPTG, R, ep_ra, ep_ri, ep_a, ep_ai)

    # Compute the likelihood.
    like = pm.Normal('likelihood', mu=fc_theo, sd=sigma, observed=obs)

    # Sample the posterior.
    trace = pm.Sample(draws=5000, tune=10000)

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

mwc.set_plotting_style()

# Magic function to make matplotlib inline; other style specs must come AFTER
%matplotlib inline

# This enables SVG graphics inline (only use with static plots (non-Bokeh))
%config InlineBackend.figure_format = 'svg'

# Generate a variable with the day that the script is run
today = str(datetime.datetime.today().strftime('%Y%m%d'))

#===============================================================================
# Defining the problem

# In this notebook we will perform Bayesian parameter estimation of the
# dissociation constants of the inducer binding to the repressor in the active
# and inactive state, or $K_A$ and $K_I$, respectively. While the main text of
# the paper estimated these values from measurements of fold-change from a
# strain containing the O2 operator and R=260 per cell, here we consider this
# inference for each of the individual strains. Specifically, we make multiple
# estimates of $K_A$ and $K_I$ by applying parameter estimation using the
# fold-change measurements as a function of inducer concentration, $c$, for each
# combination of operator binding energy and LacI copy number considered.
# Following parameter estimation from each strain's dataset, we then compare the
# theoretical predictions with our experimental data in a similar manner to that
# of the main text.

# For details of the Bayesian parameter estimation and the Bayesian approach
# that is applied here, see the 'bayesian_parameter_estimation' notebook.

# In order to minimize the data frame parsing that the log-posterior has to do
# when performing the MCMC, we also run the data through a pre-processing
# function that will parse the data once such that the output can be feed to the
# log-posterior function.
#===============================================================================


#===============================================================================
# Define pre-processing function, log likelihood, prior, and posterior
#===============================================================================

def log_likelihood(param, param_idx, unique_var, data, epsilon=4.5):
    '''
    Computes the log-likelihood
    Parameters
    ----------
    param : array-like
        Array with the value of all the parameters/dismensions on which the
        MCMC walkers should walk. The array follows this order:
        ea, ei, sigma
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
    unique_var : : list.
        This is used by the MCMC function to determine how many dimensions
        the walkers should walk in.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : binding_energy
    epsilon : float.
        Energetic difference between the active and inactive state.
    Returns
    -------
    log likelihood probability
    '''
    # unpack parameters
    ea, ei, sigma = param[0:param_idx[0]] # MWC parameters
    rep = unique_var[0] # Repressor copy numbers
    eps_r = unique_var[1] # Represor energies

    # Initialize the log_likelihood
    log_like = 0
    # loop through the parameters to fit in order to compute the
    # theoretical fold change using the right parameters for each strain
    for i, r in enumerate(rep):
        for j, eps in enumerate(eps_r):
            data_block = data[(data[:, 2]==r) & (data[:, 3]==eps), :]
            # compute the theoretical fold-change
            fc_theory = mwc.fold_change_log(data_block[:, 1],
                                            ea, ei, epsilon,
                                            rep[i], eps_r[j])
            # compute the log likelihood for this block of data
            log_like -=  np.sum((fc_theory - data_block[:, 0])**2) / 2 / sigma**2

    return log_like

def log_prior(param, param_idx, unique_var, data, epsilon=4.5):
    '''
    Computes the log-prior probability
    Parameters
    ----------
    param : array-like
        Array with the value of all the parameters/dismensions on which the
        MCMC walkers should walk. The array follows this order:
        ea, ei, sigma
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
    unique_var : : list.
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : binding_energy
    epsilon : float.
        Energetic difference between the active and inactive state.
    Returns
    -------
    log prior probability
    '''

    # unpack parameters
    ea, ei, sigma = param[0:param_idx[0]] # MWC parameters
    rep = unique_var[0] # Repressor copy numbers
    eps_r = unique_var[1] # Represor energies

    # Initialize the log_prior
    log_prior = 0

    # Check the bounds on the parameters
    # Here we have set bounds on our priors
    # for ea and ei.
    if np.any(rep <= 0) or (sigma <= 0):
        return -np.inf
    if (-7 >= ea) or (ea >= 7):
        return -np.inf
    if (-7 >= ei) or (ei >= 7):
        return -np.inf

    return log_prior

def log_post(param, param_idx, unique_var, data, epsilon=4.5):
    '''
    Computes the log posterior probability.
    Parameters
    ----------
    param : array-like
        Array with the value of all the parameters/dismensions on which the
        MCMC walkers should walk. The array follows this order:
        ea, ei, sigma
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
    unique_var : : list.
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
        The second element is the list of unique binding energies also found
        in the DataFrame.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : binding_energy
    epsilon : float.
        Energetic difference between the active and inactive state.
    Returns
    -------
    The log posterior probability
    '''
    # unpack parameters
    ea, ei, sigma = param[0:param_idx[0]] # MWC parameters

    lnp = log_prior(param, param_idx, unique_var, data, epsilon)
    # Check before computing the likelihood if one of the boundaries set by
    # the prior was not satisfied. If that is the case don't waste time
    # computing the likelihood and return -inf
    if lnp == -np.inf:
        return lnp

    return -(len(data) + 1) * np.log(sigma)\
           + log_likelihood(param, param_idx, unique_var, data, epsilon)\
           + lnp


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
# Perform parameter estimation of $K_A$ and $K_I$ from each strain's fold-change
# measurements.
#===============================================================================

#===============================================================================
# We will first define a function that initializes the walkers for each MCMC.
#===============================================================================

def init_walkers(df, n_walkers, unique_var, param_idx):
    '''
    Initialize walkers according to however many dimensions will be explored
    by the MCMC
    Parameters
    ----------
    df : pandas DataFrame
        Data frame containing the data that will be used for fitting the
        parameters
    n_walkers : int
        Number of walkers for the MCMC.
    unique_var : : list
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
        The second element is the list of unique binding energies also found
        in the DataFrame.
        This is used by the MCMC function to determine how many dimensions
        the walkers should walk in.
    param_idx : array-like
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
        This variable indicates the position of each of these variables such
        that  the function is robust and it works for a DataFrame with 1 RBS
        mutant and 1 energy as well as for multiple mutants and multiple enrgies.
    n_dim : int
        Number of dimensions that the MCMC walkers will walk on.

    Returns
    -------
    [p0, ndim] : list
        The maximum a-priori value from optimization and the number of parameters
        used for the MCMC execution.
    '''

    #Define the parameters for emcee
    n_dim = 3

    # Perform a non-linear regression
    map_param =  mwc.non_lin_reg_mwc(df, p0=[1, 7], diss_const=False)
    mean = [map_param[0], map_param[2]]
    cov = np.array([[map_param[1], 0], [0, map_param[3]]])

    # Initialize walkers
    p0 = np.empty((n_walkers, n_dim))
    # Initialize walkers
    p0 = np.empty((n_walkers, n_dim))
    p0[:,[0, 1]] = np.random.multivariate_normal(mean, cov/10.0, n_walkers)# ea, ei
    p0[:,2] = np.random.uniform(1E-5, 0.2, n_walkers)# sigma

    return p0, n_dim

#===============================================================================
# Perform parameter estimation by MCMC for each strain in our dataset.
#===============================================================================

# Determine the unique combinations of operator and LacI copy
# number in our data for us to loop through.
groups = df.groupby(['operator', 'rbs'])

# Name each .pkl chain as SI_I_x_Rj.pkl where x=(O1,O2,O3)
# and j refers to the repressor copy number, R=(22, 60, 124,
# 260, 1220, 1740). Create dictionary to to convert between
# strain name and R.
dict_R = {'HG104': '22', 'RBS1147': '60', 'RBS446' : '124', \
          'RBS1027': '260', 'RBS1': '1220', 'RBS1L': '1740'}

# Loop through each combination of operator and LacI copy
# number. Perform the parameter estimation for each dataset
# Note that teach of these consists of fold-change measurements
# from a single strain and different inducer concentrations.

for g, subdata in groups:

    # Grab data for parameter estimation
    df_temp = df[(df.operator == g[0]) & (df.rbs ==  g[1])]
    unique_var, param_idx, data = mcmc_pre_process(df_temp)

    # initialize the walkers
    n_walkers = 100
    n_burn = 500
    n_steps = 5000
    p0, n_dim = init_walkers(df_temp, n_walkers, unique_var, param_idx)

    #Call the sampler.
    sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,\
                args=(param_idx, unique_var, data, 4.5),\
                threads=6)

    #Do the burn in
    pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)

    sample = False
    if sample:
        # Perform the real MCMC
        _ = sampler.run_mcmc(pos, n_steps)
        output = open('../../data/mcmc/' + \
                      'SI_I_' + g[0] + '_R' + dict_R[g[1]] + \
                      '.pkl', 'wb')
        pickle.dump(sampler.flatchain, output)
        pickle.dump(sampler.flatlnprobability, output)

    output.close() # close it to make sure it's all been written
    print('Completed strain:', g[0], dict_R[g[1]])

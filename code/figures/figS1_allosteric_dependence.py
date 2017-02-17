"""
Title:
    figS1_allosteric_dependence.py
Author:
    Griffin Chure
Creation Date:
    20170210
Last Modified:
    20170216
Purpose:
    This script generates the plots seen in figure S1. MCMC is performed on all
    experimental data from the O1, O2, and O3 data sets to fit Ka and Ki for
    the allosterically dependent case (n=4) and the allosterically independent
    case (n=2).
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.tools.numdiff as smnd
import scipy.optimize
import pandas as pd
import glob
import emcee
import pickle

# Custom written utilities
import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Some of the funcitons from the mwc_induction_utils need to be changed to
# change the allosteric state n. We include those functions below.
# #################

def log_likelihood(param, indep_var, dep_var, epsilon=4.5):
    """
    Computes the log likelihood probability.
    Parameters
    -----------
    param : data-frame.
        The parameters to be fit by the MCMC. This must be an array of length 3
        with the following entries
        param[0] = ea == -lnKa
        param[1] = ei == -lnKi
        param[2] = sigma. Homoscedastic error associated with the Gaussian
        likelihood.
    indep_var : n x 3 array.
        series of independent variables to compute the theoretical fold-change.
        1st column : IPTG concentration
        2nd column : repressor copy number
        3rd column : repressor binding energy
    dep_var : array-like
        dependent variable, i.e. experimental fold-change. Then length of this
        array should be the same as the number of rows in indep_var.
    epsilon : float.
        Energy difference between the active and inactive state of the repressor.
    Returns
    -------
    log_like : float.
        the log likelihood.
    """
    # unpack parameters
    ea, ei, sigma = param

    # unpack independent variables
    IPTG, R, epsilon_r = indep_var.iloc[:, 0],\
                         indep_var.iloc[:, 1],\
                         indep_var.iloc[:, 2]

    # compute the theoretical fold-change
    fc_theory = mwc.fold_change_log(IPTG, ea, ei, epsilon, R, epsilon_r, n=4)

    log_like =  np.sum((fc_theory - dep_var)**2) / 2 / sigma**2
    return log_like

def regression_log_post(param, indep_var, dep_var):
    '''
    Computes the log posterior for a single set of parameters.
    Parameters
    ----------
    param : array-like.
        param[0] = epsilon_a
        param[1] = epsilon_i
    indep_var : n x 3 array.
        series of independent variables to compute the theoretical fold-change.
        1st column : IPTG concentration
        2nd column : repressor copy number
        3rd column : repressor binding energy
    dep_var : array-like
        dependent variable, i.e. experimental fold-change. Then length of this
        array should be the same as the number of rows in indep_var.

    Returns
    -------
    log_post : float.
        the log posterior probability
    '''
    # unpack parameters
    ea, ei = param

    # unpack independent variables
    IPTG, R, epsilon_r = indep_var[:, 0], indep_var[:, 1], indep_var[:, 2]

    # compute the theoretical fold-change
    fc_theory = mwc.fold_change_log(IPTG, ea, ei, 4.5, R, epsilon_r, n=4)

    # return the log posterior
    return -len(dep_var) / 2 * np.log(np.sum((dep_var - fc_theory)**2))


def log_post(param, indep_var, dep_var, epsilon=4.5,
             ea_range=[6 -6], ei_range=[6, -6], sigma_range=[0, 1]):
    '''
    Computes the log posterior probability.
    Parameters
    ----------
    param : array-like.
        The parameters to be fit by the MCMC. This must be an array of length 3
        with the following entries
        param[0] = ea == -lnKa
        param[1] = ei == -lnKi
        param[2] = sigma. Homoscedastic error associated with the Gaussian
        likelihood.
    indep_var : n x 3 array.
        Series of independent variables to compute the theoretical fold-change.
        1st column : IPTG concentration
        2nd column : repressor copy number
        3rd column : repressor binding energy
    dep_var : array-like
        Dependent variable, i.e. experimental fold-change. Then length of this
        array should be the same as the number of rows in indep_var.
    ea_range : array-like.
        Range of variables to use in the prior as boundaries for the ea parameter.
    ei_range : array-like.
        Range of variables to use in the prior as boundaries for the ei parameter.
    sigma_range : array-like.
        Range of variables to use in the prior as boundaries for the sigma param.
    epsilon : float.
        Energy difference between the active and inactive state of the repressor.
    '''
    # unpack parameters
    ea, ei, sigma = param

    # Set the prior boundaries. Since the variables have a Jeffreys prior, in
    # the log probability they have a uniform prior
    if ea > np.max(ea_range) or ea < np.min(ea_range)\
    or ei > np.max(ei_range) or ea < np.min(ei_range)\
    or sigma > np.max(sigma_range) or sigma < np.min(sigma_range):
        return -np.inf

    return -(len(indep_var) + 1) * np.log(sigma) \
    - log_likelihood(param, indep_var, dep_var, epsilon)


# #################
def resid(param, indep_var, dep_var, epsilon=4.5):
    '''
    Residuals for the theoretical fold change.

    Parameters
    ----------
    param : array-like.
        param[0] = epsilon_a
        param[1] = epsilon_i
    indep_var : n x 3 array.
        series of independent variables to compute the theoretical
        fold-change.
        1st column : iptg concentration
        2nd column : repressor copy number
        3rd column : repressor binding energy
    dep_var : array-like
        dependent variable, i.e. experimental fold-change. Then length of
        this array should be the same as the number of rows in indep_var.

    Returns
    -------
    fold-change_exp - fold-change_theory
    '''
    # unpack parameters
    ea, ei = param

    # unpack independent variables
    iptg, R, epsilon_r = indep_var[:, 0], indep_var[:, 1], indep_var[:, 2]

    # compute the theoretical fold-change
    fc_theory = mwc.fold_change_log(iptg, ea, ei, epsilon, R, epsilon_r, n=4)

    # return the log posterior
    return dep_var - fc_theory

def non_lin_reg_mwc(df, p0,
                    indep_var=['IPTG_uM', 'repressors', 'binding_energy'],
                    dep_var='fold_change_A', epsilon=4.5, diss_const=False):
    '''
    Performs a non-linear regression on the lacI IPTG titration data assuming
    Gaussian errors with constant variance. Returns the parameters
    e_A == -ln(K_A)
    e_I == -ln(K_I)
    and it's corresponding error bars by approximating the posterior distribution
    as Gaussian.
    Parameters
    ----------
    df : DataFrame.
        DataFrame containing all the titration information. It should at minimum
        contain the IPTG concentration used, the repressor copy number for each
        strain and the binding energy of such strain as the independent variables
        and obviously the gene expression fold-change as the dependent variable.
    p0 : array-like (length = 2).
        Initial guess for the parameter values. The first entry is the guess for
        e_A == -ln(K_A) and the second is the initial guess for e_I == -ln(K_I).
    indep_var : array-like (length = 3).
        Array of length 3 with the name of the DataFrame columns that contain
        the following parameters:
        1) IPTG concentration
        2) repressor copy number
        3) repressor binding energy to the operator
    dep_var : str.
        Name of the DataFrame column containing the gene expression fold-change.
    epsilon : float.
        Value of the allosteric parameter, i.e. the energy difference between
        the active and the inactive state.
    diss_const : bool.
        Indicates if the dissociation constants should be returned instead of
        the e_A and e_I parameteres.
    Returns
    -------
    if diss_const  == True:
        e_A : MAP for the e_A parameter.
        de_A : error bar on the e_A parameter
        e_I : MAP for the e_I parameter.
        de_I : error bar on the e_I parameter
    else:
        K_A : MAP for the K_A parameter.
        dK_A : error bar on the K_A parameter
        K_I : MAP for the K_I parameter.
        dK_I : error bar on the K_I parameter
    '''
    df_indep = df[indep_var]
    df_dep = df[dep_var]

    # Extra arguments given as tuple
    args = (df_indep.values, df_dep.values, epsilon)

    # Compute the MAP
    popt, _ = scipy.optimize.leastsq(resid, p0, args=args)

    # Extract the values
    ea, ei = popt

    # Compute the Hessian at the map
    hes = smnd.approx_hess(popt, regression_log_post,
                           args=(df_indep.values, df_dep.values))

    # Compute the covariance matrix
    cov = -np.linalg.inv(hes)

    if diss_const:
        # Get the values for the dissociation constants and their
        # respective error bars
        Ka = np.exp(-ea)
        Ki = np.exp(-ei)
        deltaKa = np.sqrt(cov[0,0]) * Ka
        deltaKi = np.sqrt(cov[1,1]) * Ki
        return Ka, deltaKa, Ki, deltaKi
    else:
        return ea, cov[0,0], ei, cov[1,1]

# Set the colors
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Load in all of the data.
df = pd.read_csv('../../data/flow_master.csv', comment='#')
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]
indep_var = df[['IPTG_uM', 'repressors', 'binding_energy']]
dep_var = df.fold_change_A

# Begin the analysis with the n=4 state.
map_param = non_lin_reg_mwc(df, p0=[0, 1], diss_const=False)

mean = [map_param[0], map_param[2]]
cov = np.array([[map_param[1], 0], [0, map_param[3]]])


# Set up the MCMC.
n_dim = 3 # number of parameters to fit
n_walkers = 50
n_burn = 500
n_steps = 5000

# Initialize the walkers.
p0 = np.empty((n_walkers, n_dim))
p0[:,[0, 1]] = np.random.multivariate_normal(mean, cov, n_walkers)
p0[:,2] = np.random.uniform(1E-5, 0.2, n_walkers)

# Execute the MCMC.
# Set the ranges for the MCMC
ea_range = [-7, 7]
ei_range = [-7, 7]
sigma_range = [0, df.groupby('rbs').fold_change_A.std().max()]
# Call the sampler.
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                args=(indep_var, dep_var, 4.5, ea_range,
                                      ei_range, sigma_range), threads=6)
sample = False
if sample:
    # Do the burn in
    pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)
    # Perform the real MCMC
    _ = sampler.run_mcmc(pos, n_steps)
    output = open('../../data/mcmc/SI_A_all_data_KaKi_allosteric_dependence.pkl', 'wb')
    pickle.dump(sampler.flatchain, output)
    pickle.dump(sampler.flatlnprobability, output)

# read the flat-chain
with open('../../data/mcmc/SI_A_all_data_KaKi_allosteric_dependence.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()


# Generate a Pandas Data Frame with the mcmc chain
index = ['ka', 'ki', 'sigma']

# Generate a data frame out of the MCMC chains
mcmc_rbs = pd.DataFrame(gauss_flatchain, columns=index)
mcmc_rbs['Ka'] = np.exp(-mcmc_rbs['ka'])
mcmc_rbs['Ki'] = np.exp(-mcmc_rbs['ki'])

# rerbsine the index with the new entries
index = mcmc_rbs.columns

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma, Ka, Ki = mcmc_rbs.ix[max_idx, :]

# ea range
Ka_hpd = mwc.hpd(mcmc_rbs.ix[:, 3], 0.95)
Ki_hpd = mwc.hpd(mcmc_rbs.ix[:, 4], 0.95)
# Print results
print("""
The most probable parameters for the MWC model
----------------------------------------------
Ka = {0:.2f} -{1:0.2f} +{2:0.2f} µM
Ki = {3:.2f} -{4:0.3f} +{5:0.3f} µM
""".format(Ka, np.abs(Ka-Ka_hpd[0]), np.abs(Ka-Ka_hpd[1]),\
           Ki,np.abs(Ki-Ki_hpd[0]), np.abs(Ki-Ki_hpd[1])))

ka_n4 = Ka_hpd[0] / 1E6
ki_n4 = Ki_hpd[0] / 1E6
# Plot the predictions.
fig, ax = plt.subplots(2, 3, figsize=(9,6.2), sharey=True)

# Define the operators and repressor copy numbers.
operators = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7}
R_range = [1740, 1220, 260, 124, 60, 22]
IPTG_range = np.logspace(-8, -2, 500)
ka_n2 = 139E-6
ki_n2 = 0.53E-6
ep_ai = 4.5


for i, R in enumerate(R_range):
    ax[0, -1].plot([], [], 'o', color=colors[i], label=R)
#
leg = ax[0, -1].legend(bbox_to_anchor=(1.9, 0.3), title="""repressors / cell
  tetramer data""", fontsize=12)
leg.get_title().set_fontsize(12)
ops = ['O1', 'O2', 'O3']
for i, op in enumerate(ops):
    for j, R in enumerate(R_range):
        R = np.array(R) / 2
        fc_dimer = mwc.fold_change_log(IPTG_range, -np.log(ka_n2), -np.log(ki_n2), ep_ai, R, operators[op])

        ax[0, i].plot(IPTG_range, fc_dimer, '-', color=colors[j])

    ax[0, i].set_xscale('log')
    ax[1, i].set_xscale('log')
    ax[1, i].set_xlabel('[IPTG] (M)')
    ax[0, i].set_xlabel('[IPTG] (M)')
    ax[0, 0].set_ylabel('fold-change')
    ax[1, 0].set_ylabel('fold-change')
    ax[0, 1].set_ylim([0, 1.1])
    ax[1, i].set_ylim([0, 1.1])
    ax[1, i].xaxis.set_ticks([1E-8, 1E-6, 1E-4, 1E-2])
    ax[0, i].xaxis.set_ticks([1E-8, 1E-6, 1E-4, 1E-2])
    for k in range(2):

        ax[k, i].set_title(r'%s $\Delta\varepsilon_{RA} = %s\, k_BT$' %(op, operators[op]), fontsize=12, backgroundcolor='#ffedce', position=(0.5,1.05), fontweight='bold')
#
for i , op in enumerate(ops):
    for j, R in enumerate(R_range):
        R = np.array(R) / 2
        fc_tetramer = mwc.fold_change_log(IPTG_range, -np.log(ka_n4),
                                         -np.log(ki_n4), ep_ai, R,
                                         operators[op], n=4)
        ax[1, i].plot(IPTG_range, fc_tetramer, '-', color=colors[j])
plt.tight_layout()
plt.show()


# Now plot the data.
for i, op in enumerate(ops):
    for j, R in enumerate(R_range):
        data = df[(df['repressors'] == R/2) & (df['operator'] == op)]
#
        grouped_data = pd.groupby(data, ['IPTG_uM']).fold_change_A
        for group, d in grouped_data:
            mean_fluo = np.mean(d)
            std_fluo = np.std(d) / np.sqrt(len(d))
            for k in range(2):
                ax[k, i].plot(group/1E6, mean_fluo, 'o', color=colors[j], markersize=5)
                ax[k, i].errorbar(group/1E6, mean_fluo, std_fluo, linestyle='none', color=colors[j])
plt.show()
plt.subplots_adjust(hspace=1.2)
plt.figtext(0, 1.02, 'A', fontsize=20)
plt.figtext(0, 0.5, 'B', fontsize=20)
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/allosteric_dependence.pdf', bbox_inches='tight')

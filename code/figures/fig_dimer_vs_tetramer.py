import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import glob
import emcee
import pickle

# Custom written utilities
import mwc_induction_utils as mwc
mwc.set_plotting_style()
# Close all other open figures.
plt.close('all')

# Define the functions for the MCMC
def mcmc_pre_process(df):
    """
    Pre-process the tidy DataFrame to prepare it for the MCMC. This is done
    separately from the log-posterior calculation to speed up the process
    avoiding parsing the DataFrame every evaluation of the posterior.
    Parameteres
    -----------
    df : pandas DataFrame.
        A tidy pandas DataFrame as standardized in the project that contains
        at least the following columns:
        fold_change_A : the experimental fold-change from channel A in the
        flow cytometer.
    IPTG_uM : 1d-array
        Concentrations of the inducer in micromolar.
    repressors : int
        The mean repressor copy number in copies per cell.
    delta_repressors : float
        The experimental standard deviation on the mean repressor copy number
    binding_energy : float
        The mean repressor binding energy
    delta_energy : float
        The experimental standard deviation on the binding energy

    Returns
    -------
    [rep_unique, eps_unique] : list
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
        The second element is the list of unique binding energies also found
        in the DataFrame.
        This is used by the MCMC function to determine how many dimensions
        the walkers should walk in.
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
        After that we have all the repressor copy numbers for each of the RBS
        mutants. Followed by all the unique binding energies in the DataFrame.
        This variable indicates the position of each of these variables such
        that  the function is robust and it works for a DataFrame with 1 RBS
        mutant and 1 energy as well as for multiple mutants and multiple enrgies.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : delta_repressors
        data[:, 4] : binding_energy
        data[:, 5] : delta_energy
    """
    # List the unique variables
    rep_unique = np.sort(df.repressors.unique())
    eps_unique = np.sort(df.binding_energy.unique())
    IPTG_unique = np.sort(df.IPTG_uM.unique())

    # determine the number of unique variables
    n_repressor = len(rep_unique)
    n_epsilon_r = len(eps_unique)
    n_IPTG = len(IPTG_unique)

    # Depending on the number of parameters determine the indexes of the
    # parameters to fit
    param_idx = np.cumsum([3, n_repressor, n_epsilon_r])

    # Sort the data frame such that the log-posterior function can
    # automatically compute the log probability with the right parameters
    # for each data point
    df_sort = df.sort(['repressors', 'binding_energy', 'IPTG_uM'])
    data = np.array(df_sort[['fold_change_A', 'IPTG_uM',
                             'repressors', 'delta_repressors',
                             'binding_energy', 'delta_energy']])
    return [rep_unique, eps_unique], param_idx, data

def log_likelihood(param, param_idx, unique_var, data, epsilon=4.5,
                   quaternary_state=2, n=2):
    '''
    Computes the log-likelihood
    Parameters
    ----------
    param : array-like
        Array with the value of all the parameters/dismensions on which the
        MCMC walkers should walk. The array follows this order:
        ea, ei, sigma : first 3 columns.
        repressor copy number : next columns.
        binding energies : final columns.
        The exact position of each of these parameters depends on the number
        of unique repressors and energies as indicated by param_idx.
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
        After that we have all the repressor copy numbers for each of the RBS
        mutants. Followed by all the unique binding energies in the DataFrame.
        This variable indicates the position of each of these variables such
        that  the function is robust and it works for a DataFrame with 1 RBS
        mutant and 1 energy as well as for multiple mutants and multiple enrgies.
    unique_var : : list.
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
        The second element is the list of unique binding energies also found
        in the DataFrame.
        This is used by the MCMC function to determine how many dimensions
        the walkers should walk in.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : delta_repressors
        data[:, 4] : binding_energy
        data[:, 5] : delta_energy
    epsilon : float.
        Energetic difference between the active and inactive state.
    Returns
    -------
    log likelihood probability
    '''
    # unpack parameters
    ea, ei, sigma = param[0:param_idx[0]] # MWC parameters
    rep = param[param_idx[0]:param_idx[1]] # Repressor copy numbers
    eps_r = param[param_idx[1]:param_idx[2]] # Represor energies

    # Initialize the log_likelihood
    log_like = 0
    # loop through the parameters to fit in order to compute the
    # theoretical fold change using the right parameters for each strain
    for i, r in enumerate(unique_var[0]):
        for j, eps in enumerate(unique_var[1]):
            data_block = data[(data[:, 2]==r) & (data[:, 4]==eps), :]
            # compute the theoretical fold-change
            fc_theory = mwc.fold_change_log(data_block[:, 1],
                                            ea, ei, epsilon,
                                            rep[i], eps_r[j], n=n,
                                            quaternary_state=quaternary_state)
            # compute the log likelihood for this block of data
            log_like -=  np.sum((fc_theory - data_block[:, 0])**2) / 2 / sigma**2

    return log_like

def log_prior(param, param_idx, unique_var, data, epsilon=4.5,
              quaternary_state=2):
    '''
    Computes the log-prior probability
    Parameters
    ----------
    param : array-like
        Array with the value of all the parameters/dismensions on which the
        MCMC walkers should walk. The array follows this order:
        ea, ei, sigma : first 3 columns.
        repressor copy number : next columns.
        binding energies : final columns.
        The exact position of each of these parameters depends on the number
        of unique repressors and energies as indicated by param_idx.
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
        After that we have all the repressor copy numbers for each of the RBS
        mutants. Followed by all the unique binding energies in the DataFrame.
        This variable indicates the position of each of these variables such
        that  the function is robust and it works for a DataFrame with 1 RBS
        mutant and 1 energy as well as for multiple mutants and multiple enrgies.
    unique_var : : list.
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
        The second element is the list of unique binding energies also found
        in the DataFrame.
        This is used by the MCMC function to determine how many dimensions
        the walkers should walk in.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : delta_repressors
        data[:, 4] : binding_energy
        data[:, 5] : delta_energy
    epsilon : float.
        Energetic difference between the active and inactive state.
    Returns
    -------
    log prior probability
    '''
    # unpack parameters
    ea, ei, sigma = param[0:param_idx[0]] # MWC parameters
    rep = param[param_idx[0]:param_idx[1]] # Repressor copy numbers
    eps_r = param[param_idx[1]:param_idx[2]] # Represor energies

    # Initialize the log_prior
    log_prior = 0
    # loop through the parameters to to fit in order to compute the appropiate
    # log prior
    for i, r in enumerate(unique_var[0]):
        for j, eps in enumerate(unique_var[1]):
            data_block = data[(data[:, 2]==r) & (data[:, 4]==eps), :]
            log_prior -= np.sum((rep[i] - data_block[:, 2])**2 / \
                         2 / data_block[:, 3]**2)
            log_prior -= np.sum((eps_r[j] - data_block[:, 4])**2 / \
                         2 / data_block[:, 5]**2)

    # check the bounds on the parameterreps
    if np.any(rep <= 0) or (sigma <= 0):
        return -np.inf

    return log_prior

def log_post(param, param_idx, unique_var, data, epsilon=4.5, n=2,
            quaternary_state=2):
    '''
    Computes the log posterior probability.
    Parameters
    ----------
    param : array-like
        Array with the value of all the parameters/dismensions on which the
        MCMC walkers should walk. The array follows this order:
        ea, ei, sigma : first 3 columns.
        repressor copy number : next columns.
        binding energies : final columns.
        The exact position of each of these parameters depends on the number
        of unique repressors and energies as indicated by param_idx.
    param_idx : array-like.
        An array that indicates in the param array where are each parameters
        located. The logic is the following:
        In the first 3 positions of the param argument for the MCMC we find
        epsilon_A, epsilon_I and sigma the error associated with the Gaussian
        likelihood.
        After that we have all the repressor copy numbers for each of the RBS
        mutants. Followed by all the unique binding energies in the DataFrame.
        This variable indicates the position of each of these variables such
        that  the function is robust and it works for a DataFrame with 1 RBS
        mutant and 1 energy as well as for multiple mutants and multiple enrgies.
    unique_var : : list.
        A list whose first element is the list of the unique mean repressor
        copy number found in the DataFrame.
        The second element is the list of unique binding energies also found
        in the DataFrame.
        This is used by the MCMC function to determine how many dimensions
        the walkers should walk in.
    data : array-like.
        Numpy array pre-arranged in the order that the log-posterior function
        expects it with the following columns:
        data[:, 0] : fold_change_A
        data[:, 1] : IPTG_uM
        data[:, 2] : repressors
        data[:, 3] : delta_repressors
        data[:, 4] : binding_energy
        data[:, 5] : delta_energy
    epsilon : float.
        Energetic difference between the active and inactive state.
    Returns
    -------
    The log posterior probability
    '''
    # unpack parameters
    ea, ei, sigma = param[0:param_idx[0]] # MWC parameters
    rep = param[param_idx[0]:param_idx[1]] # Repressor copy numbers
    eps_r = param[param_idx[1]:param_idx[2]] # Represor energies

    lnp = log_prior(param, param_idx, unique_var, data, epsilon)
    # Check before computing the likelihood if one of the boundaries set by
    # the prior was not satisfied. If that is the case don't waste time
    # computing the likelihood and return -inf
    if lnp == -np.inf:
        return lnp

    return -(len(data) + 1) * np.log(sigma)\
            + log_likelihood(param, param_idx, unique_var, data, epsilon, n=n,
                             quaternary_state=quaternary_state)\
            + lnp

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
        After that we have all the repressor copy numbers for each of the RBS
        mutants. Followed by all the unique binding energies in the DataFrame.
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
    n_dim = 3 + np.sum([len(x) for x in unique_var])

    # Perform a non-linear regression
    map_param =  mwc.non_lin_reg_mwc(df, p0=[1, 7], diss_const=False)
    mean = [map_param[0], map_param[2]]
    cov = np.array([[map_param[1], 0], [0, map_param[3]]])

    # Initialize walkers
    p0 = np.empty((n_walkers, n_dim))
    # Initialize walkers
    p0 = np.empty((n_walkers, n_dim))
    p0[:,[0, 1]] = np.random.multivariate_normal(mean, cov, n_walkers)# ea, ei
    p0[:,2] = np.random.uniform(1E-5, 0.2, n_walkers)# sigma

    # loop through the repressors
    for i, r in enumerate(unique_var[0]):
        sigma_r = df[df.repressors==r].delta_repressors.unique()
        # Check if any walker was initialized in a forbidden area
        rep_num = np.random.normal(r, sigma_r, n_walkers)
        rep_num[rep_num < 0] = 0
        p0[:, param_idx[0]+i] = rep_num
    for j, eps in enumerate(unique_var[1]):
        sigma_eps = df[df.binding_energy==eps].delta_energy.unique()
        p0[:, param_idx[1]+j] = np.random.normal(eps, sigma_eps, n_walkers)

    return p0, n_dim


# Set the colors
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Load in all of the data.
df = pd.read_csv('../../data/flow_master.csv', comment='#')
grouped = pd.groupby(df, ['operator', 'IPTG_uM'])

# Plot the predictions.
fig, ax = plt.subplots(2, 3, figsize=(9,6.2), sharey=True)

# Define the operators and repressor copy numbers.
operators = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7}
R_range = [1740, 1220, 260, 124, 60, 22]
IPTG_range = np.logspace(-8, -2, 500)
k_a = 139E-6
k_i = 0.53E-6
ep_ai = 4.5

# Perform the global fits for n=4 case.
# List the error sources as described by Garcia & Phillips PNAS 2011.
delta_R = {'HG104':2, 'RBS1147':10, 'RBS446':15, 'RBS1027':20, 'RBS1':80,
               'RBS1L':170}
delta_epsilon_r = {'O1':0.2, 'O2':0.2, 'O3':0.1, 'Oid':0.2}
# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

# Restart index
df = df.reset_index()

# Add the error columns to the data frame
df['delta_repressors'] = pd.Series([delta_R[df.iloc[x].rbs] for x\
                                    in np.arange(df.shape[0])])
df['delta_energy'] = pd.Series([delta_epsilon_r[x] for x in df.operator])

# Preprocess the data
unique_var, param_idx, data = mcmc_pre_process(df)
n_walkers = 50
n_burn = 500
n_steps = 8000
p0, n_dim = init_walkers(df, n_walkers, unique_var, param_idx)

# Call the sampler for the .
sampler = emcee.EnsembleSampler(n_walkers, n_dim, log_post,
                                args=(param_idx, unique_var, data, 4.5, 4),
                                threads=6)
sample = False
if sample:
    #Do the burn in
    print('Performing the burn-in')
    pos, prob, state = sampler.run_mcmc(p0, n_burn, storechain=False)
    # Perform the real MCMC
    print('Performing the MCMC')
    _ = sampler.run_mcmc(pos, n_steps)
    output = open('../../data/mcmc/error_prop_pool_data_allosteric_dependence.pkl',
         'wb')
    pickle.dump(sampler.flatchain, output)
    pickle.dump(sampler.flatlnprobability, output)

# Load the flat-chain
with open('../../data/mcmc/error_prop_pool_data_allosteric_dependence.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()


# Generate a Pandas Data Frame with the mcmc chain
index = np.concatenate([['ka', 'ki', 'sigma'],\
          [df[df.repressors==r].rbs.unique()[0] for r in \
              np.sort(df.repressors.unique())],
          [df[df.binding_energy==o].operator.unique()[0] for o in \
              np.sort(df.binding_energy.unique())]])

# Generate a data frame out of the MCMC chains
mcmc_df = pd.DataFrame(gauss_flatchain, columns=index)
mcmc_df['ka'] = np.exp(-mcmc_df['ka'])
mcmc_df['ki'] = np.exp(-mcmc_df['ki'])

# Generate data frame with mode values for each parameter
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
param_fit = pd.DataFrame(gauss_flatchain[max_idx, :], index=index,
                         columns=['mode'])

# Generate parameter to save the hpd for each parameter
param_hpd = pd.DataFrame(columns=['hpd_max', 'hpd_min'])

# # Loop through each parameter computing the 95% hpd
for column in mcmc_df:
    param_hpd = param_hpd.append(pd.Series(np.abs(mwc.hpd(mcmc_df[column], 0.95) - param_fit.ix[column, 'mode']),
                               index=['hpd_min', 'hpd_max'], name=column))

# # Combine the data frames into a single data frame
param_fit_n4 = pd.concat([param_fit, param_hpd], axis=1)

with open('../../data/mcmc/error_prop_pool_data.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()


# Generate a Pandas Data Frame with the mcmc chain
index = np.concatenate([['ka', 'ki', 'sigma'],\
          [df[df.repressors==r].rbs.unique()[0] for r in \
              np.sort(df.repressors.unique())],
          [df[df.binding_energy==o].operator.unique()[0] for o in \
              np.sort(df.binding_energy.unique())]])

# Generate a data frame out of the MCMC chains
mcmc_df = pd.DataFrame(gauss_flatchain, columns=index)
mcmc_df['ka'] = np.exp(-mcmc_df['ka'])
mcmc_df['ki'] = np.exp(-mcmc_df['ki'])

# Generate data frame with mode values for each parameter
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
param_fit = pd.DataFrame(gauss_flatchain[max_idx, :], index=index,
                         columns=['mode'])

# Generate parameter to save the hpd for each parameter
param_hpd = pd.DataFrame(columns=['hpd_max', 'hpd_min'])

# # Loop through each parameter computing the 95% hpd
for column in mcmc_df:
    param_hpd = param_hpd.append(pd.Series(np.abs(mwc.hpd(mcmc_df[column], 0.95) -
                                           param_fit.ix[column, 'mode']),
                               index=['hpd_min', 'hpd_max'], name=column))

#
#  Combine the data frames into a single data frame
param_fit_n2 = pd.concat([param_fit, param_hpd], axis=1)

# Make the plot of the fits.
n2_R = list(param_fit_n2['mode'][3:9])
n2_R.reverse()
n2_ka =param_fit_n2['mode'][0].astype(float)/1E6
n2_ki = param_fit_n2['mode'][1].astype(float)/1E6
n2_op = dict(param_fit_n2['mode'][-3:])
n4_R = list(param_fit_n4['mode'][3:9])
n4_R.reverse()
n4_ka = param_fit_n4['mode'][0].astype(float)/1E6
n4_ki = param_fit_n4['mode'][1].astype(float)/1E6
n4_op = dict(param_fit_n4['mode'][-3:])
for i, R in enumerate(R_range):
    ax[0, -1].plot([], [], 'o', color=colors[i], label=R)

leg = ax[0, -1].legend(bbox_to_anchor=(1.9, 0.3), title="""repressors / cell
  tetramer data""", fontsize=12)
leg.get_title().set_fontsize(12)
ops = ['O1', 'O2', 'O3']
for i, op in enumerate(ops):
    print(i)
    for j, R in enumerate(n2_R):
        R = np.array(R)
        fc_dimer = mwc.fold_change_log(IPTG_range, -np.log(n2_ka), -np.log(n2_ki),
                                       ep_ai, R, n2_op[op])

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
        print(op)
        ax[k, i].set_title(r'%s $\Delta\varepsilon_{RA} = %s\, k_BT$' %(op, operators[op]), fontsize=12, backgroundcolor='#ffedce', position=(0.5,1.05), fontweight='bold')

for i , op in enumerate(ops):
    for j, R in enumerate(n4_R):
        R = np.array(R)
        fc_tetramer = mwc.fold_change_log(IPTG_range, -np.log(n4_ka),
                                         -np.log(n4_ki), ep_ai, R,
                                         n4_op[op], n=4)
        ax[1, i].plot(IPTG_range, fc_tetramer, '-', color=colors[j])
plt.tight_layout()


# Now plot the data.
for i, op in enumerate(ops):
    for j, R in enumerate(R_range):
        data = df[(df['repressors'] == R/2) & (df['operator'] == op)]

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
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/dimer_v_tetramer.pdf', bbox_inches='tight')
#


# Now consider the case where the N=2 case has the same ka/ki from n=4 fit.
fig, ax = plt.subplots(2, 3, figsize=(9,6.2), sharey=True)
for i, R in enumerate(R_range):
    ax[0, -1].plot([], [], 'o', color=colors[i], label=R)

leg = ax[0, -1].legend(bbox_to_anchor=(1.9, 0.3), title="""repressors / cell
  tetramer data""", fontsize=12)
leg.get_title().set_fontsize(12)
for i, op in enumerate(ops):
    for j, R in enumerate(n4_R):
        R = np.array(R)
        fc_dimer = mwc.fold_change_log(IPTG_range, -np.log(n4_ka), -np.log(n4_ki),
                                       ep_ai, R, n4_op[op], n=4)

        fc_tetramer = mwc.fold_change_log(IPTG_range, -np.log(n4_ka), -np.log(n4_ki),
                                       ep_ai, R, n4_op[op], n=2)

        ax[0, i].plot(IPTG_range, fc_dimer, '-', color=colors[j])
        ax[1, i].plot(IPTG_range, fc_tetramer, '-', color=colors[j])
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
        ax[k, i].set_title(r'%s $\Delta\varepsilon_{RA} = %s\, k_BT$' %(op, operators[op]), fontsize=12, position=(0.5,1.05), backgroundcolor='#ffedce')


# Now plot the data.
for i, op in enumerate(ops):
    for j, R in enumerate(R_range):
        data = df[(df['repressors'] == R/2) & (df['operator'] == op)]

        grouped_data = pd.groupby(data, ['IPTG_uM']).fold_change_A
        for group, d in grouped_data:
            mean_fluo = np.mean(d)
            std_fluo = np.std(d) / np.sqrt(len(d))
            for k in range(2):
                if k!=1:
                    alpha = 1
                    ax[k, i].plot(group/1E6, mean_fluo, 'o', color=colors[j], markersize=5, alpha=alpha)
                    ax[k, i].errorbar(group/1E6, mean_fluo, std_fluo, linestyle='none', color=colors[j], alpha=alpha)
plt.show()
plt.tight_layout()
plt.subplots_adjust(hspace=1.0)
plt.figtext(0, 1.02, 'A', fontsize=20)
plt.figtext(0, 0.5, 'B', fontsize=20)
#
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/dimer_v_tetramer_prediction.pdf', bbox_inches='tight')

import os
import glob
import pickle
import re

# Our numerical workhorses
import numpy as np
import pandas as pd

# Import the project utils
import sys
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec  as gridspec

# Seaborn, useful for graphics
import seaborn as sns

mwc.set_plotting_style()


# dropbox = open('../../doc/induction_paper/graphicspath.tex')
# output = dropbox.read()
# output = re.sub('\\graphicspath{{', '', output)
# output = output[1::]
# output = re.sub('}}\n', '', output)
#

# Load the master data file
datadir = '../../data/'
df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]


# Load the flat-chain  used for parameter estimation
with open('../../data/mcmc/main_text_KaKi.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]
ka, ki = np.exp(-ea), np.exp(-ei)
# Convert the flatchains to units of concentration.
ka_fc = np.exp(-gauss_flatchain[:,0])
ki_fc = np.exp(-gauss_flatchain[:,1])

# Separate the data for calculation of other properties.
grouped = pd.groupby(df, 'operator')
operators = df.operator.unique()
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7}
#  Define functions for each property calculation.
def dyn_range(num_rep, ep_r, ka_ki, ep_ai=4.5, n_sites=2, n_ns=4.6E6):
    pact_leak = 1 / (1 + np.exp(-ep_ai))
    pact_sat = 1 / (1 + np.exp(-ep_ai) * (ka_ki)**n_sites)
    leak = (1 + pact_leak * (num_rep / n_ns) * np.exp(-ep_r))**-1
    sat = (1 + pact_sat * (num_rep / n_ns) * np.exp(-ep_r))**-1
    return sat - leak


def dyn_cred_region(num_rep, ka_flatchain, ki_flatchain,
                    ep_r, mass_frac=0.95, epsilon=4.5):
    cred_region = np.zeros([2, len(num_rep)])
    # Loop through each repressor copy number and compute the fold-changes
    # for each concentration.
    ka_ki = ka_flatchain / ki_flatchain
    for i, R in enumerate(num_rep):
        drng = dyn_range(R, ep_r, ka_ki, ep_ai=epsilon)
        cred_region[:, i] = mwc.hpd(drng, mass_frac)
    return cred_region

def leakiness(num_rep, ep_r, ep_ai, n_ns=4.6E6):
    pact = 1 / (1 + np.exp(-ep_ai))
    return (1 + pact * (num_rep / n_ns) * np.exp(-ep_r))**-1

def leakiness_cred_region(num_rep, ep_r, ep_ai, n_ns=4.6E6,
                         mass_frac=0.95):
    pact = 1 / (1 + np.exp(-ep_ai))
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        fc = (1 + pact * (R / n_ns) * np.exp(-ep_r))**-1
        cred_region[:, i] = mwc.hpd(fc, mass_frac=mass_frac)
    return cred_region
def saturation(num_rep, ep_r, ep_ai, ka_ki, n_sites=2, n_ns=4.6E6):
    pact = 1 / (1 + np.exp(-ep_ai) * ka_ki**n_sites)
    return (1 + pact * (num_rep/n_ns)*np.exp(-ep_r))**-1


def saturation_cred_region(num_rep, ep_r, ep_ai, ka_flatchain, ki_flatchain,
                           n_sites=2, n_ns=4.6E6, mass_frac=0.95):
    pact = 1 / (1 + np.exp(-ep_ai) * (ka_flatchain / ki_flatchain)**n_sites)
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        fc = (1 + pact * (R / n_ns) * np.exp(-ep_r))**-1
        cred_region[:, i] = mwc.hpd(fc, mass_frac)
    return cred_region

"""
The following equations are borrowed from Stephanie Barnes.
"""
def pact(IPTG, K_A, K_I, e_AI):
    '''
    Computes the probability that a repressor is active
    Parameters
    ----------
    IPTG : array-like
        Array of IPTG concentrations in uM
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    probability that repressor is active
    '''
    pact = (1 + IPTG * 1 / K_A)**2 / \
    (((1 + IPTG * 1 / K_A))**2 + np.exp(-e_AI) * (1 + IPTG * 1 / K_I)**2)
    return pact


def fold_change(IPTG, K_A, K_I, e_AI, R, Op):
    '''
    Computes fold-change for simple repression
    Parameters
    ----------
    IPTG : array-like
        Array of IPTG concentrations in uM
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    R : float
        Number of repressors per cell
    Op : float
        Operator binding energy
    Returns
    -------
    probability that repressor is active
    '''
    return 1 / (1 + R / 4.6E6 * pact(IPTG, K_A, K_I, e_AI) * np.exp(-Op))

def EC50(K_A, K_I, e_AI, R, Op):
    '''
    Computes the concentration at which half of the repressors are in the active state
    Parameters
    ----------
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state

    Returns
    -------
    Concentration at which half of repressors are active (EC50)
    '''
    t = 1 + (R / 4.6E6) * np.exp(-Op) + (K_A / K_I)**2 * (2 * np.exp(-e_AI) + 1 + (R / 4.6E6) * np.exp(-Op))
    b = 2 * (1 + (R / 4.6E6) * np.exp(-Op)) + np.exp(-e_AI) + (K_A / K_I)**2 * np.exp(-e_AI)
    return K_A * (((K_A / K_I) - 1)/((K_A / K_I) - (t/b)**(1/2)) -1)


def ec50_cred_region(num_rep, Op, e_AI, K_A, K_I,
                     mass_frac=0.95):
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        ec50_rng = EC50(K_A, K_I, e_AI, R, Op)
        cred_region[:, i] = mwc.hpd(ec50_rng, mass_frac)
    return cred_region

def effective_Hill(K_A, K_I, e_AI, R, Op):
    '''
    Computes the effective Hill coefficient
    Parameters
    ----------
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    effective Hill coefficient
    '''
    c = EC50(K_A, K_I, e_AI, R, Op)
    return 2/(fold_change(c, K_A, K_I, e_AI, R, Op) - fold_change(0, K_A, K_I, e_AI, R, Op))*\
           (-(fold_change(c, K_A, K_I, e_AI, R, Op))**2 * R / 4.6E6 * np.exp(-Op) * \
            2 * c * np.exp(-e_AI) * (1/K_A * (1 + c/K_A) * (1 + c/K_I)**2 - 1/K_I *\
           (1 + c/K_A)**2 * (1 + c/K_I)) / ((1 + c/K_A)**2 + np.exp(-e_AI) * (1 + c/K_I)**2)**2)

def effective_hill_cred(num_rep, Op, e_AI, K_A, K_I,
                        mass_frac=0.95):
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        # Compute the EC50
        c = EC50(K_A, K_I, e_AI, R, Op)
        # Compute the hill
        e_hill = 2/(fold_change(c, K_A, K_I, e_AI, R, Op) - fold_change(0, K_A, K_I, e_AI, R, Op))*\
           (-(fold_change(c, K_A, K_I, e_AI, R, Op))**2 * R / 4.6E6 * np.exp(-Op) * \
            2 * c * np.exp(-e_AI) * (1/K_A * (1 + c/K_A) * (1 + c/K_I)**2 - 1/K_I *\
           (1 + c/K_A)**2 * (1 + c/K_I)) / ((1 + c/K_A)**2 + np.exp(-e_AI) * (1 + c/K_I)**2)**2)
        cred_region[:, i] = mwc.hpd(e_hill, mass_frac)

    return cred_region


# Compute the dynamic range
drs = []
for g, d in grouped:
    unique_IPTG = d.IPTG_uM.unique()
    min_IPTG = np.min(unique_IPTG)
    max_IPTG = np.max(unique_IPTG)
    # Group the new data by repressors.
    grouped_rep = pd.groupby(d, ['rbs', 'date', 'username'])
    rbs_ind = {'HG104' : 0, 'RBS1147': 1, 'RBS446' : 2, 'RBS1027': 3,
               'RBS1': 4, 'RBS1L': 5}
    rep_dr = [[], [], [], [], [], []]
    rep_std = []
    for g_rep, d_rep in grouped_rep:
        if g_rep[2] != 'sloosbarnes':
            dr = d_rep[d_rep.IPTG_uM==max_IPTG].fold_change_A.values - d_rep[d_rep.IPTG_uM==min_IPTG].fold_change_A.values
            rep_dr[rbs_ind[g_rep[0]]].append(dr[0])

    # Compute the means.
    for i, dr in enumerate(rep_dr):
        rep_dr[i] = np.mean(dr)
        rep_std.append(np.std(dr) / np.sqrt(len(dr)))

    reps = np.sort(df.repressors.unique())
    dr_df = pd.DataFrame([reps, rep_dr, rep_std]).T
    dr_df.columns = ['repressors', 'dynamic_range', 'err']
    dr_df.insert(0, 'operator', g)
    drs.append(dr_df)
drng = pd.concat(drs, axis=0)

"""
# Load in the flatchains for the calculation of the effective hill and EC50
repressors = ['R22', 'R60', 'R124', 'R260', 'R1220', 'R1740']
flatchains = [[], [], []]
kas = [[], [], []]
kis = [[], [], []]
for i, op in enumerate(operators):
    for j, R in enumerate(repressors):
        with open('../../data/mcmc/SI_I_' + op + '_' + R + '.pkl', 'rb') as file:
            print(j)
            unpickler = pickle.Unpickler(file)
            gauss_flatchain = unpickler.load()
            flatchains[i].append(gauss_flatchain)
            gauss_flatlnprobability = unpickler.load()
            ind = np.argmax(gauss_flatlnprobability)
            kas[i].append(np.exp(-gauss_flatchain[ind, 0]))
            kis[i].append(np.exp(-gauss_flatchain[ind, 1]))


# Plot the leakiness, saturation, dynamic range, effective hill, and ec50.
rep_range = np.logspace(0, 4, 200)
ka, ki = np.exp(-ea) , np.exp(-ei)
ka_ki = ka / ki


# Set the figure.
fig, ax = plt.subplots(2,3, figsize=(8, 5))
ax = ax.ravel()
en_colors = sns.color_palette('viridis', n_colors=len(operators))
repressor_numbers = [22, 60, 124, 260, 1220, 1740]
for i, op in enumerate(operators):
    # Compute the dynamic range.
    sat = saturation(rep_range, energies[op], 4.5, np.exp(-ea)/np.exp(-ei))
    leak = leakiness(rep_range, energies[op], 4.5)
    dyn_rng = dyn_range(rep_range, energies[op], ka_ki)
    ec50 = EC50(ka/1E6, ki/1E6, 4.5, rep_range, energies[op])
    hill = effective_Hill(ka, ki, 4.5, rep_range, energies[op])
    ax[0].plot(rep_range, leak, color=en_colors[i], label='__nolegend__',
               markersize=6)
    ax[1].plot(rep_range, sat, color=en_colors[i], label='__nolegend__',
               markersize=6)
    ax[2].plot(rep_range, dyn_rng, color=en_colors[i], label='__nolegend__',
               markersize=6)
    ax[3].plot(rep_range, ec50, color=en_colors[i], label='__nolegend__')
    ax[4].plot(rep_range, hill, color=en_colors[i], label='__nolegend__')
    # Compute the credible regions.
    # sat_cred = saturation_cred_region(rep_range, energies[op], 4.5, ka_fc,
    #                                   ki_fc)
    # dyn_cred = dyn_cred_region(rep_range,
    #      ka_fc, ki_fc, epsilon=4.5,
    #      ep_r=energies[op])
    # ec50_cred = ec50_cred_region(rep_range, energies[op], 4.5, ka_fc/1E6, ki_fc/1E6)
    # hill_cred = effective_hill_cred(rep_range, energies[op], 4.5, ka_fc, ki_fc)
    # ax[0].fill_between(rep_range, sat_cred[0,:], sat_cred[1,:],
    #                    alpha=0.3, color=en_colors[i])
    # ax[1].fill_between(rep_range, dyn_cred[0,:], dyn_cred[1,:],
    #                 alpha=0.3, color=en_colors[i])
    # ax[2].fill_between(rep_range, ec50_cred[0,:], ec50_cred[1,:],
    #                    alpha=0.3, color=en_colors[i])
    # ax[4].fill_between(rep_range, hill_cred[0,:], hill_cred[1,:],
    #                    alpha=0.3, color=en_colors[i])
    #
    # Plot the inferred parameter values for EC50 and effective hill
    for j, R in enumerate(repressor_numbers):
        ec50_inf = EC50(kas[i][j], kis[i][j], 4.5, R, energies[op])
        hill_inf = effective_Hill(kas[i][j], kis[i][j], 4.5, R,
                                  energies[op])
    #     convert the flatchains to units of concentration
        _ka_fc = np.exp(-flatchains[i][j][:,0])
        _ki_fc = np.exp(-flatchains[i][j][:,1])
        ec50_cred = EC50(_ka_fc, _ki_fc, 4.5, R, energies[op])
        ec50_cred = mwc.hpd(ec50_cred, mass_frac=0.95)
        hill_cred = effective_Hill(_ka_fc, _ki_fc, 4.5, R, energies[op])
        hill_cred = mwc.hpd(hill_cred, 0.95)


        if j == 0:
            label = energies[op]
        else:
            label = '__nolegend__'

        ax[3].vlines(R, ec50_cred[0]/1E6, ec50_cred[1]/1E6, color=en_colors[i], zorder=4-i, label='__nolegend__')
        ax[4].vlines(R, hill_cred[0], hill_cred[1], color=en_colors[i], zorder=4-i, label='__nolegend__')
        ax[3].plot(R, ec50_inf/1E6, 's', markerfacecolor='w', markeredgecolor=en_colors[i], ms=6, markeredgewidth=1.5, zorder=4-i)
        ax[4].plot(R, hill_inf, 's', ms=6,markerfacecolor='w', markeredgecolor=en_colors[i], markeredgewidth=1.5, zorder=4-i)


# #
# #
# #
# Get the dynamic range data and plot.
for i, op in enumerate(operators):
    dyn_rng = drng[drng.operator==op]
    ax[2].errorbar(2 * dyn_rng.repressors, dyn_rng.dynamic_range, yerr=dyn_rng.err, color=en_colors[i], fmt='o', linestyle='none',
                    label=energies[op], markersize=5)
# #
# Plot the leakiness and saturation data.
grouped = pd.groupby(df, ['operator', 'repressors'])
#Define the colors so I don't have to make another data frame.
op_colors = {'O1': en_colors[0], 'O2': en_colors[1], 'O3': en_colors[2]}
for g, d in grouped:
#    Extract the unique IPTG values.
    unique_IPTG = d['IPTG_uM'].unique()

#    Slice the min and max IPTG values.
    leak_vals = d[d['IPTG_uM'] == np.min(unique_IPTG)].fold_change_A
    sat_vals = d[d['IPTG_uM'] == np.max(unique_IPTG)].fold_change_A

    #Compute the mean and standard errors of reach.
    mean_leak = np.mean(leak_vals)
    sem_leak = np.std(leak_vals) / np.sqrt(len(leak_vals))
    mean_sat = np.mean(sat_vals)
    sem_sat =  np.std(sat_vals) / np.sqrt(len(sat_vals))

    #Plot the data with the appropriate legends..
    if g[1] == 11:
        legend = energies[g[0]]
    else:
        legend = '__nolegend__'
    ax[0].plot(2 * g[1], mean_leak, 'o', color=op_colors[g[0]],
               markersize=5, label='__nolegend__')
    ax[1].plot(2 * g[1], mean_sat, 'o', color=op_colors[g[0]],
               markersize=5, label='__nolegend__')
    ax[0].errorbar(2 * g[1], mean_leak, sem_leak, linestyle='none',
                   color=op_colors[g[0]], fmt='o', markersize=6, label=legend)
    ax[1].errorbar(2 * g[1], mean_sat, sem_sat, linestyle='none',
                   color=op_colors[g[0]], fmt='o', markersize=6, label=legend)

#Add labels and format axes.
ylabels = ['leakiness', 'saturation', 'dynamic range', '$[EC_{50}]\, \,(\mathrm{M})$', 'effective Hill coefficient']
for i in range(len(ylabels)):
    ax[i].set_xlabel('repressors per cell', fontsize=14)
    ax[i].set_xscale('log')
    ax[i].set_ylabel(ylabels[i], fontsize=14)
ax[0].set_yscale('log')
ax[3].set_yscale('log')
ax[5].set_axis_off()

for i, a in enumerate(ax):
    a.set_xlim([1, 1E4])
    a.set_xticks([1, 10, 100, 1000, 10000])
    a.tick_params(labelsize=14)
l = ax[0].legend(loc='lower left', title='binding\n energy ($k_BT$)')
# plt.tight_layout()
# plt.show()
# #
# add plot letter labels
# plt.figtext(0., .96, 'A', fontsize=20)
# plt.figtext(0.33, .96, 'B', fontsize=20)
# plt.figtext(0.65, .96, 'C', fontsize=20)
# plt.figtext(0.0, .63, 'D', fontsize=20)
# plt.figtext(0.33, .63, 'E', fontsize=20)
# plt.figtext(0.65, .63, 'F', fontsize=20)
# plt.figtext(0.0, .32, 'G', fontsize=20)
# plt.figtext(0.33, .32, 'H', fontsize=20)
#
# plt.savefig(output + '/fig5.pdf',
#         bbox_inches='tight')
# #
# plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/theory_v_data.svg', bbox_inches='tight')


"""
# Load the global fit chains,
with open('../../data/mcmc/SI_E_global.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    flatchain = unpickler.load()
    flat_lnprob = unpickler.load()


# Set the indexing for the MCMC dataframe.
index = ['log_ka', 'log_ki', 'sigma', 'HG104', 'RBS1147', 'RBS446',
         'RBS1027', 'RBS1', 'RBS1L', 'Oid', 'O1', 'O2', 'O3']

global_df = pd.DataFrame(flatchain, columns=index)
global_df['Ka'] = np.exp(-global_df['log_ka'])
global_df['Ki'] = np.exp(-global_df['log_ki'])
index =global_df.columns


# Compute the mode and HPD for every parameter.
max_idx = np.argmax(flat_lnprob, axis=0)
param_fit = global_df.ix[max_idx, :]
param_fit = param_fit.to_frame(name='mode')
param_hpd = pd.DataFrame(columns=['hpd_min', 'hpd_max'])

# Loop through and generate the dataframe.
for column in global_df:
    param_hpd = param_hpd.append(pd.Series(mwc.hpd(global_df[column], 0.95),
                                 index=['hpd_min', 'hpd_max'], name=column))

param_fit = pd.concat([param_fit, param_hpd], axis=1)


# Make a new figure and plot the properties.
plt.close('all')
fig, ax = plt.subplots(2, 3, figsize=(9, 5))
ax = ax.ravel()
ops = ['O1', 'O2', 'O3']
colors = sns.color_palette('viridis', n_colors=len(ops))
for i, o in enumerate(ops):
    if i == 0:
        en_colors = {o: colors[i]}
    else:
        en_colors[o] = colors[i]
num_rep = np.logspace(0, 4, 500)
reps = np.array([22, 60, 124, 260, 1220, 1740])
rbs = ['HG104', 'RBS1147', 'RBS446', 'RBS1027', 'RBS1', 'RBS1L']
c= np.logspace(-9, -2, 500)
ka_gf = param_fit.loc['Ka'].loc['mode']
ki_gf = param_fit.loc['Ki'].loc['mode']

colors = sns.color_palette('deep', n_colors=len(rbs))


for op in ops:
    ax[0].plot([], [], label=np.round(param_fit.loc[op].loc['mode'], 2),
               color=en_colors[op])
ax[0].legend(title='    binding\n energy ($k_BT$)', loc='lower left')

ax[-1].set_axis_off()
for i, op in enumerate(ops):
    print(op)
    # Plot the predictions.
    ep_r = param_fit.loc[op].loc['mode']
    print(ep_r)
    leak = leakiness(num_rep, ep_r, 4.5)
    sat = saturation(num_rep, ep_r, 4.5, ka_gf/ki_gf)
    dyn_rng = dyn_range(num_rep, ep_r, ka_gf/ki_gf, ep_ai=4.5)
    ec50 = EC50(ka_gf, ki_gf, 4.5, num_rep, ep_r)
    hill = effective_Hill(ka_gf, ki_gf, 4.5, num_rep, ep_r)
    ax[0].plot(num_rep, leak, '-', color=en_colors[op])
    ax[1].plot(num_rep, sat, '-', color=en_colors[op])
    ax[2].plot(num_rep, dyn_rng, '-', color=en_colors[op])
    ax[3].plot(num_rep, ec50/1E6, '-', color=en_colors[op])
    ax[4].plot(num_rep, hill, '-', color=en_colors[op])



    # Compute and plot the credible regions.
    leak_cred = leakiness_cred_region(num_rep, global_df[op], 4.5)
    sat_cred = saturation_cred_region(num_rep, global_df[op], 4.5, global_df['Ka'], global_df['Ki'])
    dyn_cred = dyn_cred_region(num_rep, global_df['Ka'], global_df['Ki'], global_df[op])
    ec50_cred = ec50_cred_region(num_rep, global_df[op], 4.5, global_df['Ka'],
                                 global_df['Ki'])
    hill_cred = effective_hill_cred(num_rep, global_df[op], 4.5,
                                    global_df['Ka'], global_df['Ki'])

    ax[0].fill_between(num_rep, leak_cred[0, :], leak_cred[1, :],
                        color=en_colors[op], alpha=0.4)

    ax[1].fill_between(num_rep, sat_cred[0, :], sat_cred[1, :],
                       color=en_colors[op], alpha=0.4)
    ax[2].fill_between(num_rep, dyn_cred[0, :], dyn_cred[1, :],
                       color=en_colors[op], alpha=0.4)
    ax[3].fill_between(num_rep, ec50_cred[0, :]/1E6, ec50_cred[1, :]/1E6,
                       color=en_colors[op], alpha=0.4)
    ax[4].fill_between(num_rep, hill_cred[0, :], hill_cred[1, :],
                       color=en_colors[op], alpha=0.4)



    # Compute the EC50 and effective hill and plot.
    for j, R in enumerate(reps):
        # Load the single fit data.
        chain = glob.glob('../../data/mcmc/SI_G_{0}_{1}.pkl'.format(op, rbs[j]))
        with open(chain[0], 'rb') as file:
            unpickler = pickle.Unpickler(file)
            flatchain2 = unpickler.load()
            flat_lnprob2 = unpickler.load()
        max_idx = np.argmax(flat_lnprob2)
        ea, ei, _, rep, ep = flatchain2[max_idx]
        rep *= 2
        rep_cred = 2 * mwc.hpd(flatchain2[:,3], 0.95)
        print(ep)
        ka_fc, ki_fc = np.exp(-flatchain2[:,0]), np.exp(-flatchain2[:,1])
        ep_fc = flatchain2[:,4]
        ka, ki = np.exp(-ea), np.exp(-ei)


        ec502 = EC50(ka, ki, 4.5, rep, ep)
        rep = np.array([rep,])
        ec50_cred = ec50_cred_region(rep, ep_fc, 4.5, ka_fc,
                                 ki_fc)
        hill = effective_Hill(ka, ki, 4.5, rep, ep)
        hill_cred = effective_hill_cred(rep, ep_fc, 4.5, ka_fc,
                                    ki_fc)



        ax[3].plot(rep, ec502/1E6, 's', markerfacecolor='w', markeredgecolor=en_colors[op], markersize=6, markeredgewidth=1.5)



        ax[4].plot(rep, hill, 's', markerfacecolor='w', markeredgecolor=en_colors[op], markersize=6, markeredgewidth=1.5)
        ax[3].vlines(rep, ec50_cred[0]/1E6, ec50_cred[1]/1E6, color=en_colors[op])
        ax[4].vlines(rep, hill_cred[0], hill_cred[1], color=en_colors[op])

        # Load data for leakiness and saturation plots
        dat = df[(df['repressors']==(R/2)) & (df['operator']==op)]
        iptg = dat.IPTG_uM.unique()
        grouped = dat.groupby('IPTG_uM').fold_change_A.mean()

        unique_IPTG = dat['IPTG_uM'].unique()

        #   Slice the min and max IPTG values.
        leak_vals = dat[dat['IPTG_uM'] == np.min(unique_IPTG)].fold_change_A
        sat_vals = dat[dat['IPTG_uM'] == np.max(unique_IPTG)].fold_change_A

        #Compute the mean and standard errors of reach.
        mean_leak = np.mean(leak_vals)
        sem_leak = np.std(leak_vals) / np.sqrt(len(leak_vals))
        mean_sat = np.mean(sat_vals)
        sem_sat =  np.std(sat_vals) / np.sqrt(len(sat_vals))


        # Plot the data for every point.
        rep = rep[0]
        ax[0].plot(rep, mean_leak, 'o', color=en_colors[op], ms=5)
        ax[0].errorbar(rep, mean_leak, yerr=sem_leak, color=en_colors[op])
        ax[1].plot(rep, mean_sat, 'o', color=en_colors[op], ms=5)
        ax[1].errorbar(rep, mean_sat, yerr=sem_sat, color=en_colors[op])
        d = drng[(drng.repressors == (0.5 * R)) & (drng.operator == op)]
        ax[2].plot(rep, d.dynamic_range, 'o', color=en_colors[op], ms=5)
        ax[2].errorbar(rep, d.dynamic_range, yerr=d.err, color=en_colors[op])



ylabs = ['leakiness', 'saturation', 'dynamic range', '$[EC_{50}]$ (M)',
         'effective Hill coefficient', '']
for i, a in enumerate(ax):
    a.set_xscale('log')
    a.set_xlabel('repressors per cell', fontsize=12)
    a.set_ylabel(ylabs[i], fontsize=12)

ax[0].set_yscale('log')
ax[3].set_yscale('log')

plt.savefig('/Users/gchure/Dropbox/mwc_indution/resubmission figures/figSX_global_properties.svg', bbox_inches='tight')

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
import matplotlib.gridspec as gridspec
import matplotlib.colors as plc
import corner

# Seaborn, useful for graphics
import seaborn as sns
import scipy.stats
mwc.set_plotting_style()

datadir = '../../data/'
df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

#===============================================================================
# O2 RBS1027
#===============================================================================
# Load the flat-chain
with open('../../data/mcmc/main_text_KaKi.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]

ka_fc = np.exp(-gauss_flatchain[:, 0])
ki_fc = np.exp(-gauss_flatchain[:, 1])
#===============================================================================
# Plot the theory vs data for all 4 operators with the credible region
#===============================================================================

# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]


# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}


# Initialize the figure.
fig, ax = plt.subplots(2, 3, figsize=(8.5, 6))
ax = ax.ravel()


# Plot the predictions.
def leakiness(num_rep, ep_r, ep_ai, n_ns=4.6E6):
    pact = 1 / (1 + np.exp(-ep_ai))
    return (1 + pact * (num_rep / n_ns) * np.exp(-ep_r))**-1


def saturation(num_rep, ep_r, ep_ai, ka_ki, n_sites=2, n_ns=4.6E6):
    pact = 1 / (1 + np.exp(-ep_ai) * ka_ki**n_sites)
    return (1 + pact * (num_rep / n_ns) * np.exp(-ep_r))**-1


def saturation_cred_region(num_rep, ep_r, ep_ai, ka_flatchain, ki_flatchain,
                           n_sites=2, n_ns=4.6E6, mass_frac=0.95):
    pact = 1 / (1 + np.exp(-ep_ai) * (ka_flatchain / ki_flatchain)**n_sites)
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        fc = (1 + pact * (R / n_ns) * np.exp(-ep_r))**-1
        cred_region[:, i] = mwc.hpd(fc, mass_frac)
    return cred_region


def dyn_range(num_rep, ep_r, ka_ki, ep_ai=4.5, n_sites=2, n_ns=4.6E6):
    pact_leak = 1 / (1 + np.exp(-ep_ai))
    pact_sat = 1 / (1 + np.exp(-ep_ai) * (ka_ki)**n_sites)
    leak = (1 + pact_leak * (num_rep / n_ns) * np.exp(-ep_r))**-1
    sat = (1 + pact_sat * (num_rep / n_ns) * np.exp(-ep_r))**-1
    return sat - leak


# The following equations are borrowed from Stephanie Barnes.

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
    return 1 / (1 + R / 5E6 * pact(IPTG, K_A, K_I, e_AI) * np.exp(-Op))


def dyn_cred_region(num_rep, ka_flatchain, ki_flatchain, ep_r, mass_frac=0.95, epsilon=4.5):
    cred_region = np.zeros([2, len(num_rep)])
    ka_ki = ka_flatchain / ki_flatchain
    for i, R in enumerate(num_rep):
        drng = dyn_range(R, ep_r, ka_ki, ep_ai=epsilon)
        cred_region[:, i] = mwc.hpd(drng, mass_frac)
    return cred_region


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
    t = 1 + (R / 4.6E6) * np.exp(-Op) + (K_A / K_I)**2 * \
        (2 * np.exp(-e_AI) + 1 + (R / 4.6E6) * np.exp(-Op))
    b = 2 * (1 + (R / 4.6E6) * np.exp(-Op)) + \
        np.exp(-e_AI) + (K_A / K_I)**2 * np.exp(-e_AI)
    return K_A * ((K_A / K_I - 1) / (K_A / K_I - (t / b)**(1 / 2)) - 1)


def ec50_cred_region(num_rep, Op, e_AI, K_A, K_I,
                     mass_frac=0.95):
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        t = 1 + (R / 4.6E6) * np.exp(-Op) + (K_A / K_I)**2 * \
            (2 * np.exp(-e_AI) + 1 + (R / 4.6E6) * np.exp(-Op))
        b = 2 * (1 + (R / 4.6E6) * np.exp(-Op)) + \
            np.exp(-e_AI) + (K_A / K_I)**2 * np.exp(-e_AI)
        ec50_rng = K_A * ((K_A / K_I - 1) / (K_A / K_I - (t / b)**(1 / 2)) - 1)
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
    return 2 / (fold_change(c, K_A, K_I, e_AI, R, Op) - fold_change(0, K_A, K_I, e_AI, R, Op)) *\
        (-(fold_change(c, K_A, K_I, e_AI, R, Op))**2 * R / 5E6 * np.exp(-Op) *
         2 * c * np.exp(-e_AI) * (1 / K_A * (1 + c / K_A) * (1 + c / K_I)**2 - 1 / K_I *
                                  (1 + c / K_A)**2 * (1 + c / K_I)) / ((1 + c / K_A)**2 + np.exp(-e_AI) * (1 + c / K_I)**2)**2)


def effective_hill_cred(num_rep, Op, e_AI, K_A, K_I,
                        mass_frac=0.95):
    cred_region = np.zeros([2, len(num_rep)])
    for i, R in enumerate(num_rep):
        # Compute the EC50
        c = EC50(K_A, K_I, e_AI, R, Op)
        # Compute the hill
        e_hill = 2 / (fold_change(c, K_A, K_I, e_AI, R, Op) - fold_change(0, K_A, K_I, e_AI, R, Op)) *\
            (-(fold_change(c, K_A, K_I, e_AI, R, Op))**2 * R / 5E6 * np.exp(-Op) *
             2 * c * np.exp(-e_AI) * (1 / K_A * (1 + c / K_A) * (1 + c / K_I)**2 - 1 / K_I *
                                      (1 + c / K_A)**2 * (1 + c / K_I)) / ((1 + c / K_A)**2 + np.exp(-e_AI) * (1 + c / K_I)**2)**2)
        cred_region[:, i] = mwc.hpd(e_hill, mass_frac)

    return cred_region


rep_range = np.logspace(0, 4, 200)
ka_ki = np.exp(-ea) / np.exp(-ei)
en_colors = sns.color_palette('viridis', n_colors=len(operators))
titles = ['leakiness', 'saturation', 'dynamic range',
          'EC50 ($\mu$M)', 'effective Hill coefficient']
for i, op in enumerate(operators):
    # Compute the properties
    leak = leakiness(rep_range, energies[op], ep_ai=4.5)
    sat = saturation(rep_range, energies[op], 4.5, np.exp(-ea) / np.exp(-ei))
    dyn_rng = dyn_range(rep_range, energies[op], ka_ki)
    ec50 = EC50(np.exp(-ea), np.exp(-ei), 4.5, rep_range, energies[op])
    e_hill = effective_Hill(np.exp(-ea), np.exp(-ei),
                            4.5, rep_range, energies[op])

    ax[0].plot(rep_range, leak, color=en_colors[i], label='__nolegend__')
    ax[1].plot(rep_range, sat, color=en_colors[i], label='__nolegend__')
    ax[2].plot(rep_range, dyn_rng, color=en_colors[i], label='__nolegend__')
    ax[3].plot(rep_range, ec50 / 1E6, color=en_colors[i])
    ax[4].plot(rep_range, e_hill, color=en_colors[i])
    ax[i].set_xlabel('repressors per cell', fontsize=12)
    ax[i].set_ylabel(titles[i], fontsize=12)

    # Plot the credible regions
    sat_cred = saturation_cred_region(
        rep_range, energies[op], 4.5, ka_fc, ki_fc)
    dyn_cred = dyn_cred_region(rep_range,
                               ka_fc, ki_fc, epsilon=4.5,
                               ep_r=energies[op])
    ec50_cred = ec50_cred_region(rep_range, energies[op], 4.5, ka_fc,
                                 ki_fc, mass_frac=0.95)
    hill_cred = effective_hill_cred(
        rep_range, energies[op], 4.5, ka_fc, ki_fc, mass_frac=0.95)
    ax[1].fill_between(rep_range, dyn_cred[0, :], dyn_cred[1, :],
                       alpha=0.3, color=en_colors[i])
    ax[2].fill_between(rep_range, sat_cred[0, :], sat_cred[1, :],
                       alpha=0.3, color=en_colors[i])
    ax[3].fill_between(rep_range, ec50_cred[0, :] / 1E6, ec50_cred[1, :] / 1E6,
                       alpha=0.3, color=en_colors[i])
    ax[4].fill_between(rep_range, hill_cred[0, :], hill_cred[1, :],
                       alpha=0.3, color=en_colors[i])

    ax[i].set_xlim([1, 1E4])
    #


# Plot the points.


# Plot the leakiness and saturation data.
grouped = pd.groupby(df, ['operator', 'repressors'])
# Define the colors so I don't have to make another data frame.
op_colors = {'O1': en_colors[0], 'O2': en_colors[1], 'O3': en_colors[2]}
for g, d in grouped:
    # Extract the unique IPTG values.
    unique_IPTG = d['IPTG_uM'].unique()

    # Slice the min and max IPTG values.
    leak_vals = d[d['IPTG_uM'] == np.min(unique_IPTG)].fold_change_A
    sat_vals = d[d['IPTG_uM'] == np.max(unique_IPTG)].fold_change_A

    # Compute the mean and standard errors of reach.
    mean_leak = np.mean(leak_vals)
    sem_leak = np.std(leak_vals) / np.sqrt(len(leak_vals))
    mean_sat = np.mean(sat_vals)
    sem_sat = np.std(sat_vals) / np.sqrt(len(sat_vals))

    # Plot the data with the appropriate legends..
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

# Compute the dynamic range.
grouped = pd.groupby(df, 'operator')
drs = []
for g, d in grouped:
    unique_IPTG = d.IPTG_uM.unique()
    min_IPTG = np.min(unique_IPTG)
    max_IPTG = np.max(unique_IPTG)
    # Group the new data by repressors.
    grouped_rep = pd.groupby(d, ['rbs', 'date', 'username'])
    rbs_ind = {'HG104': 0, 'RBS1147': 1, 'RBS446': 2, 'RBS1027': 3,
               'RBS1': 4, 'RBS1L': 5}
    rep_dr = [[], [], [], [], [], []]
    rep_std = []
    for g_rep, d_rep in grouped_rep:
        if g_rep[2] != 'sloosbarnes':
            dr = d_rep[d_rep.IPTG_uM == max_IPTG].fold_change_A.values - \
                d_rep[d_rep.IPTG_uM == min_IPTG].fold_change_A.values
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


# Get the dynamic range data and plot.
for i, op in enumerate(operators):
    dyn_rng = drng[drng.operator == op]
    ax[2].errorbar(2 * dyn_rng.repressors, dyn_rng.dynamic_range, yerr=dyn_rng.err, color=en_colors[i], fmt='o', linestyle='none',
                   label=energies[op], markersize=5)


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

# Plot the EC50 and Effective hill
repressor_numbers = [22, 60, 124, 260, 1220, 1740]
for i, op in enumerate(operators):
    for j, R in enumerate(repressor_numbers):
        ec50_inf = EC50(kas[i][j], kis[i][j], 4.5, R, energies[op])
        hill_inf = effective_Hill(kas[i][j], kis[i][j], 4.5, R,
                                  energies[op])
        # convert the flatchains to units of concentration
        _ka_fc = np.exp(-flatchains[i][j][:, 0])
        _ki_fc = np.exp(-flatchains[i][j][:, 1])
        ec50_cred = EC50(_ka_fc, _ki_fc, 4.5, R, energies[op])
        ec50_cred = mwc.hpd(ec50_cred, mass_frac=0.95)
        hill_cred = effective_Hill(_ka_fc, _ki_fc, 4.5, R, energies[op])
        hill_cred = mwc.hpd(hill_cred, 0.95)

        if j == 0:
            label = energies[op]
        else:
            label = '__nolegend__'

        ax[3].vlines(R, ec50_cred[0] / 1E6, ec50_cred[1] / 1E6,
                     color=en_colors[i], zorder=4 - i, label='__nolegend__')
        ax[4].vlines(R, hill_cred[0], hill_cred[1],
                     color=en_colors[i], zorder=4 - i, label='__nolegend__')
        ax[3].plot(R, ec50_inf / 1E6, 's', markerfacecolor='w',
                   markeredgecolor=en_colors[i], ms=6, markeredgewidth=1.5, zorder=4 - i)
        ax[4].plot(R, hill_inf, 's', ms=6, markerfacecolor='w',
                   markeredgecolor=en_colors[i], markeredgewidth=1.5, zorder=4 - i)

ax[3].set_xlim([1, 1E4])
ax[4].set_xlim([1, 1E4])
ax[3].set_ylabel('$[EC_{50}]\,\,$(M)', fontsize=12)
ax[4].set_ylabel('effective Hill coefficient', fontsize=12)
leg_2 = ax[0].legend(title='   binding\n energy ($k_BT$)',
                     loc='lower left', fontsize=8, handlelength=1)
leg_2.get_title().set_fontsize(8)
ax[0].set_yscale('log')
ax[3].set_yscale('log')
ax[3].set_yticks([1E-8, 1E-7, 1E-6, 1E-5, 1E-4])
ax[4].set_yticks([1.2, 1.4, 1.6, 1.8])
ax[5].set_axis_off()

for i in range(len(ax)):
    ax[i].set_xscale('log')
    ax[i].set_xticks([1, 10, 100, 1000, 1E4])
    ax[i].set_xlabel('repressors per cell', fontsize=12)

plt.figtext(0., .96, 'A', fontsize=20)
plt.figtext(0.33, .96, 'B', fontsize=20)
plt.figtext(0.65, .96, 'C', fontsize=20)
plt.figtext(0.0, .63, 'D', fontsize=20)
plt.figtext(0.33, .63, 'E', fontsize=20)

plt.tight_layout()

# Add plot letter label
# plt.figtext(0.01, 0.96, '(C)', fontsize=12)
# plt.figtext(0.33, 0.96, '(D)', fontsize=12)
# plt.figtext(0.64, 0.96, '(E)', fontsize=12)
# plt.figtext(0.01, 0.65, '(F)', fontsize=12)
# plt.figtext(0.34, 0.65, '(G)', fontsize=12)
# plt.figtext(0.64, 0.65, '(H)', fontsize=12)
# plt.figtext(0.01, 0.32, '(I)', fontsize=12)
# plt.figtext(0.34, 0.32, '(J)', fontsize=12)
#
#
# plt.tight_layout()
# plt.savefig('../../figures/main_figs/fig5_curves.svg', bbox_inches='tight')

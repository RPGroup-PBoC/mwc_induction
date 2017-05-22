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

# Seaborn, useful for graphics
import seaborn as sns

mwc.set_plotting_style()

#===============================================================================
# Set output directory based on the graphicspath.tex file to print in dropbox
#===============================================================================
# dropbox = open('../../doc/induction_paper/graphicspath.tex')
# output = dropbox.read()
# output = re.sub('\\graphicspath{{', '', output)
# output = output[1::]
# output = re.sub('}}\n', '', output)
#
#===============================================================================
# Read the data
#===============================================================================

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

# Convert the flatchains to units of concentration.
ka_fc = np.exp(-gauss_flatchain[:,0])
ki_fc = np.exp(-gauss_flatchain[:,1])

#===============================================================================
# Plot the theory vs data for all 4 operators with the credible region
#===============================================================================
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-7, -2, 100)
IPTG_lin = np.array([0, 1E-7])

# Set the colors for the strains
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize the plot to set the size
# fig = plt.figure(figsize=(11, 8))

# Define the GridSpec to center the lower plot
# ax1 = plt.subplot2grid((2, 4), (0, 1), colspan=2)
# ax2 = plt.subplot2grid((2, 4), (1, 0), colspan=2)
# ax3 = plt.subplot2grid((2, 4), (1, 2), colspan=2)
# ax = [ax1, ax2, ax3]

fig, ax = plt.subplots(2, 3, figsize=(11, 8))
ax = ax.ravel()

# Loop through operators
for i, op in enumerate(operators):
    print(op)
    data = df[df.operator == op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # plot the theory using the parameters from the fit.
        # Log-scale
        ax[i].plot(IPTG, mwc.fold_change_log(IPTG * 1E6,
            ea=ea, ei=ei, epsilon=4.5,
            R=np.array(df[(df.rbs == rbs)].repressors.unique()),
            epsilon_r=energies[op]),
            color=colors[j], label=None, zorder=1)
        # Linear scale
        ax[i].plot(IPTG_lin, mwc.fold_change_log(IPTG_lin * 1E6,
            ea=ea, ei=ei, epsilon=4.5,
            R=np.array(df[(df.rbs == rbs)].repressors.unique()),
            epsilon_r=energies[op]),
            color=colors[j], label=None, zorder=1, linestyle='--')
        # plot 95% HPD region using the variability in the MWC parameters
        # Log scale
        cred_region = mwc.mcmc_cred_region(IPTG * 1e6,
            gauss_flatchain, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op])
        ax[i].fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                        alpha=0.3, color=colors[j])
        # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())

        # plot the experimental data
        # Distinguish between the fit data and the predictions
        if (op == 'O2') & (rbs == 'RBS1027'):
            ax[i].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                    fc_mean, yerr=fc_err, linestyle='none', color=colors[j])
            ax[i].plot(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                       fc_mean, marker='o', linestyle='none',
                       markeredgewidth=1, markersize=5, markeredgecolor=colors[j],
                       markerfacecolor='w',
                       label=df[df.rbs=='RBS1027'].repressors.unique()[0] * 2,
                       zorder=100)
        else:
            ax[i].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                    fc_mean, yerr=fc_err,
                    fmt='o', label=df[df.rbs==rbs].repressors.unique()[0] * 2,
                color=colors[j], zorder=100, markersize=5)

    # Add operator and binding energy labels.
    ax[i].set_title(r'%s  $\Delta\varepsilon_{RA} = %s\, k_BT$'
                    %(op, energies[op]), backgroundcolor='#ffedce',
                    fontsize=14, y=1.03)
    # ax[i].text(0.8, 0.09, r'{0}'.format(op), transform=ax[i].transAxes,
            # fontsize=13)
    # ax[i].text(0.67, 0.02,
            # r'$\Delta\varepsilon_{RA} = %s\,k_BT$' %energies[op],
            # transform=ax[i].transAxes, fontsize=13)
    ax[i].set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    ax[i].set_xlabel('IPTG (M)', fontsize=15)
    ax[i].set_ylabel('fold-change', fontsize=16)
    ax[i].set_ylim([-0.01, 1.1])
    ax[i].set_xlim(left=-5E-9)
    ax[i].tick_params(labelsize=14)




# Separate the data for the dynamic range calculation.
grouped = pd.groupby(df, 'operator')


# Plot the properties
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


# Compute the dynamic range.

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

# Plot the leakiness, saturation, and dynamic range.
rep_range = np.logspace(0, 4, 200)
ka_ki = np.exp(-ea) / np.exp(-ei)
en_colors = sns.color_palette('viridis', n_colors=len(operators))
for i, op in enumerate(operators):
    # Compute the dynamic range.
    sat = saturation(rep_range, energies[op], 4.5, np.exp(-ea)/np.exp(-ei))
    leak = leakiness(rep_range, energies[op], 4.5)
    dyn_rng = dyn_range(rep_range, energies[op], ka_ki)
    ax[3].plot(rep_range, leak, color=en_colors[i], label='__nolegend__',
               markersize=5)
    ax[4].plot(rep_range, sat, color=en_colors[i], label='__nolegend__',
               markersize=5)
    ax[5].plot(rep_range, dyn_rng, color=en_colors[i], label='_nolegend_',
               markersize=5)
    # Compute the credible regions.
    sat_cred = saturation_cred_region(rep_range, energies[op], 4.5, ka_fc,
                                      ki_fc)
    dyn_cred = dyn_cred_region(rep_range,
         ka_fc, ki_fc, epsilon=4.5,
         ep_r=energies[op])
    ax[4].fill_between(rep_range, sat_cred[0,:], sat_cred[1,:],
                       alpha=0.4, color=en_colors[i])
    ax[5].fill_between(rep_range, dyn_cred[0,:], dyn_cred[1,:],
                    alpha=0.3, color=en_colors[i])


# Get the dynamic range data and plot.
for i, op in enumerate(operators):
    dyn_rng = drng[drng.operator==op]
    ax[5].errorbar(2 * dyn_rng.repressors, dyn_rng.dynamic_range, yerr=dyn_rng.err, color=en_colors[i], fmt='o', linestyle='none',
                    label=energies[op], markersize=5)

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
    sem_sat =  np.std(sat_vals) / np.sqrt(len(sat_vals))

    # Plot the data with the appropriate legends..
    if g[1] == 11:
        legend = energies[g[0]]
    else:
        legend = '__nolegend__'
    ax[3].plot(2 * g[1], mean_leak, 'o', color=op_colors[g[0]],
               markersize=5, label=legend)
    ax[4].plot(2 * g[1], mean_sat, 'o', color=op_colors[g[0]],
               markersize=5, label=legend)
    ax[3].errorbar(2 * g[1], mean_leak, sem_leak, linestyle='none',
                   color=op_colors[g[0]])
    ax[4].errorbar(2 * g[1], mean_sat, sem_sat, linestyle='none',
                   color=op_colors[g[0]])

# Add labels and format axes.
ylabels = ['leakiness', 'saturation', 'dynamic range']
for i in range(3):
    ax[i+3].set_xlabel('number of repressors', fontsize=15)
    ax[i+3].set_xscale('log')
    ax[i+3].set_ylabel(ylabels[i], fontsize=15)
ax[3].set_yscale('log')

for i, a in enumerate(ax):
    if i < 3:
       a.set_xlim([-5E-9, 1E-2])
       a.set_xticks([0, 1E-6, 1E-4, 1E-2])
    else:
       a.set_xlim([1, 1E4])
    a.tick_params(labelsize=14)
    a.set_ylim([-0.01, 1.1])
l = ax[3].legend(loc='lower left', title='binding\n energy ($k_BT$)')
plt.setp(l.get_title(), multialignment='center')
ax[0].legend(loc='upper left', title='rep. / cell')
# add plot letter labels
plt.figtext(0., .96, 'A', fontsize=20)
plt.figtext(0.35, .96, 'B', fontsize=20)
plt.figtext(0.66, .96, 'C', fontsize=20)
plt.figtext(0.0, .46, 'D', fontsize=20)
plt.figtext(0.35, .46, 'E', fontsize=20)
plt.figtext(0.66, .46, 'F', fontsize=20)
plt.tight_layout()
# plt.savefig(output + '/fig5.pdf',
        # bbox_inches='tight')
plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/theory_v_data.pdf', bbox_inches='tight')

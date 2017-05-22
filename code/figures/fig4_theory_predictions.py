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

#===============================================================================
# Set output directory based on the graphicspath.tex file to print in dropbox
#=============================================================================== # dropbox = open('../../doc/induction_paper/graphicspath.tex')
# output = dropbox.read()
# output = re.sub('\\graphicspath{{', '', output)
# output = output[1::]
# output = re.sub('}}\n', '', output)

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

ka_fc = np.exp(-gauss_flatchain[:,0])
ki_fc = np.exp(-gauss_flatchain[:,1])
#===============================================================================
# Plot the theory vs data for all 4 operators with the credible region
#===============================================================================

# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-7, -2, 100)
IPTG_lin = np.array([0, 1E-7])


# Set the colors for the strains
colors = sns.color_palette(n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]


# Define the operators and their respective energies
operators = ['O1', 'O2', 'O3']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize the figure.
fig, ax = plt.subplots(2, 3, figsize=(12, 8))
ax = ax.ravel()


# Plot the predictions.
for i, op in enumerate(operators):
    print(op)
    data = df[df.operator == op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # plot the theory using the parameters from the fit.
        if (op == 'O2') & (rbs == 'RBS1027'):
            label=None
        else:
            label=df[df.rbs==rbs].repressors.unique()[0] * 2
        # Log scale
        ax[i].plot(IPTG, mwc.fold_change_log(IPTG * 1E6,
            ea=ea, ei=ei, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op]),
            color=colors[j], label=label)
        # Linear scale
        ax[i].plot(IPTG_lin, mwc.fold_change_log(IPTG_lin * 1E6,
            ea=ea, ei=ei, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op]),
            color=colors[j], linestyle='--', label=None)

        # plot 95% HPD region using the variability in the MWC parameters
        cred_region = mwc.mcmc_cred_region(IPTG * 1e6,
            gauss_flatchain, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op])
        ax[i].fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                        alpha=0.3, color=colors[j]) # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())

        # plot the experimental data
        # Distinguish between the fit data and the predictions
        if (op == 'O2') & (rbs == 'RBS1027'):
            ax[i].errorbar(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                    fc_mean, yerr=fc_err, linestyle='none', color=colors[j],
                    label=None)
            ax[i].plot(np.sort(data[data.rbs==rbs].IPTG_uM.unique()) / 1E6,
                       fc_mean, marker='o', markersize=5, linestyle='none',
                       markeredgewidth=1, markeredgecolor=colors[j],
                       markerfacecolor='w',
                       label=df[df.rbs=='RBS1027'].repressors.unique()[0] * 2)

    # Add operator and binding energy labels.
    ax[i].set_title(r'%s  $\Delta\varepsilon_{RA} = %s\, k_BT$' %(op, energies[op]), backgroundcolor='#ffedce', fontsize=14, y=1.03)
    # ax[i].text(0.75, 0.09, r'{0}'.format(op), transform=ax[i].transAxes,
            # fontsize=13, backgroundcolor='#ffedce')
    # ax[i].text(0.6, 0.02,
            # r'$\Delta\varepsilon_{RA} = %s\,k_BT$' %energies[op],
            # transform=ax[i].transAxes, fontsize=13, backgroundcolor='#ffedce')
    ax[i].set_xscale('symlog', linthreshx=1E-7, linscalex=0.5)
    ax[i].set_xlabel('IPTG (M)', fontsize=15)
    ax[i].set_ylabel('fold-change', fontsize=16)
    ax[i].set_ylim([-0.01, 1.1])
    ax[i].set_xlim([-5E-9, 1E-2])
    ax[i].set_xticks([0, 1E-6, 1E-4, 1E-2])
    ax[i].tick_params(labelsize=14)


# Plot the properties
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

def dyn_range(num_rep, ep_r, ka_ki, ep_ai=4.5, n_sites=2, n_ns=4.6E6):
    pact_leak = 1 / (1 + np.exp(-ep_ai))
    pact_sat = 1 / (1 + np.exp(-ep_ai) * (ka_ki)**n_sites)
    leak = (1 + pact_leak * (num_rep / n_ns) * np.exp(-ep_r))**-1
    sat = (1 + pact_sat * (num_rep / n_ns) * np.exp(-ep_r))**-1
    return sat - leak

def dyn_cred_region(num_rep, ka_flatchain, ki_flatchain, ep_r, mass_frac=0.95, epsilon=4.5):
    cred_region = np.zeros([2, len(num_rep)])
    ka_ki = ka_flatchain / ki_flatchain
    for i, R in enumerate(num_rep):
        drng = dyn_range(R, ep_r, ka_ki, ep_ai=epsilon)
        cred_region[:, i] = mwc.hpd(drng, mass_frac)
    return cred_region

    # Loop through each repressor copy number and compute the fold-changes
    # for each concentration.
    ka_ki = ka_flatchain / ki_flatchain
    for i, R in enumerate(num_rep):
        drng = dyn_range(R, ep_r, ka_ki, ep_ai=epsilon)
        cred_region[:, i] = mwc.hpd(drng, mass_frac)
    return cred_region
rep_range = np.logspace(0, 4, 200)
ka_ki = np.exp(-ea) / np.exp(-ei)
en_colors = sns.color_palette('viridis', n_colors=len(operators))
titles = ['leakiness', 'saturation', 'dynamic range']
for i, op in enumerate(operators):
    # Compute the properties
    leak = leakiness(rep_range, energies[op], ep_ai=4.5)
    sat = saturation(rep_range, energies[op], 4.5, np.exp(-ea)/np.exp(-ei))
    dyn_rng = dyn_range(rep_range, energies[op], ka_ki)

    ax[3].plot(rep_range, leak, color=en_colors[i], label=energies[op])
    ax[4].plot(rep_range, sat, color=en_colors[i], label=energies[op])
    ax[5].plot(rep_range, dyn_rng, color=en_colors[i], label=energies[op])
    ax[i+3].set_xlabel('number of repressors', fontsize=14)
    ax[i+3].set_ylabel(titles[i], fontsize=15)

    # Plot the credible regions
    dyn_cred = dyn_cred_region(rep_range,
         ka_fc, ki_fc, epsilon=4.5,
         ep_r=energies[op])
    sat_cred = saturation_cred_region(rep_range, energies[op], 4.5, ka_fc, ki_fc)
    ax[5].fill_between(rep_range, dyn_cred[0,:], dyn_cred[1,:],
                    alpha=0.3, color=en_colors[i])
    ax[4].fill_between(rep_range, sat_cred[0,:], sat_cred[1,:],
                    alpha=0.3, color=en_colors[i])
    ax[i+3].set_xlim([1, 1E4])

ax[0].legend(loc='upper left', title='rep. / cell')
ax[3].legend(title='   binding\n energy ($k_BT$)', loc='lower left')
ax[3].set_yscale('log')
# ax[4].set_yscale('log')
# ax[5].set_yscale('log')

for i in range(3, len(ax)):
    ax[i].set_xscale('log')


# ax[0].set_axis_off()
# ax[1].set_axis_off()
# Add plot letter label
plt.figtext(0.01, 0.96, 'C', fontsize=20)
plt.figtext(0.35, 0.96, 'D', fontsize=20)
plt.figtext(0.66, 0.96, 'E', fontsize=20)
plt.figtext(0.01, 0.47, 'F', fontsize=20)
plt.figtext(0.35, 0.47, 'G', fontsize=20)
plt.figtext(0.66, 0.47, 'H', fontsize=20)

plt.tight_layout()
plt.show()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/theory_predictions.pdf')
# plt.savefig(output + '/fig4.pdf', bbox_inches='tight')
# plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission_figures/fig_theory_predictions.pdf', bbox_inches='tight')


# Generate the jointplot to insert into the figure via illustrator.
lab = ['$K_A\,(\mu\mathrm{M})$', '$K_I\,(\mu\mathrm{M})$']
ka_ki_df = pd.DataFrame(np.array([ka_fc, ki_fc]).T, columns=lab)
inds = np.arange(0, len(ka_fc), 1)
np.random.seed(666)

# Calculate the point density

# choices = np.random.choice(inds, size=1E4, replace=False)
plt.close('all')
g = sns.JointGrid(lab[0], lab[1], ka_ki_df, xlim=(100, 200), ylim=(0.45, 0.625), space=0.05)
g.ax_joint.plot(ka_fc, ki_fc, '.', color='#937D69', ms=2, alpha=0.05, rasterized=True, zorder=1)
g.plot_joint(sns.kdeplot, cmap=sns.cubehelix_palette(n_colors=10, as_cmap=True, reverse=True), zorder=10, linewidth=1, n_levels=5, shade=True, alpha=0.5, shade_lowest=False)


# Plot the mode and HPD on the marginals.
ind = np.where(gauss_flatlnprobability==gauss_flatlnprobability.max())[0]
ka_mode = ka_fc[ind][0]
ki_mode = ki_fc[ind][0]
ka_cred = mwc.hpd(ka_fc, mass_frac=0.95)
ki_cred = mwc.hpd(ki_fc, mass_frac=0.95)

g.ax_marg_y.plot(6, ki_mode, 'o', color=colors[4])
g.ax_marg_y.vlines(6, ki_cred[0], ki_cred[1], color=colors[4])
g.ax_marg_x.plot(ka_mode, 0.015, 'o', color=colors[4])
g.ax_marg_x.hlines(0.015, ka_cred[0], ka_cred[1], color=colors[4])

g.ax_joint.set_xlim([100, 230])
g.ax_joint.set_ylim([0.45, 0.65])
g.plot_marginals(sns.kdeplot, shade=True, color=colors[4], zorder=1, linewidth=1)

# # Plot the mode and HPD for each marginal distribution
g.fig.set_figwidth(5.75)
g.fig.set_figheight(3.25)
#


# Save it.

# plt.tight_layout()
# plt.savefig('/Users/gchure/Desktop/ka_ki_distplot.svg', bbox_inches='tight')
# plt.savefig('/Users/gchure/Desktop/test.svg', bbox_inches='tight')
plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/ka_ki_distribution.pdf', bbox_inches='tight')
plt.show()

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
import matplotlib.patches as pch

# Seaborn, useful for graphics
import seaborn as sns

mwc.set_plotting_style()

#===============================================================================
# Set output directory based on the graphicspath.tex file to print in dropbox
#===============================================================================
#dropbox = open('../../doc/induction_paper/graphicspath.tex')
#output = dropbox.read()
#output = re.sub('\\graphicspath{{', '', output)
#output = output[1::]
#output = re.sub('}}\n', '', output)
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
with open('../../data/mcmc/main_text_KaKi.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]

# Compute the Bohr parameter
df['bohr_1027'] = mwc.bohr_fn(df, ea, ei)

#===============================================================================
# Plot the theory vs data for all 4 operators with the credible region
#===============================================================================

# Given this result let's plot all the curves using this parameters.
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
# Define the operators to use in the plot
operators = ['O1', 'O2', 'O3']
markers = ['o', 'D', 's']
F = np.linspace(-10, 10, 200)
plt.figure(figsize=(8, 6))
plt.plot(F, 1 / (1 + np.exp(-F)), '-', color='black')

# Instantiate the legend.
label_col = ['rep./cell', 1740, 1220, 260, 124, 60, 22]
label_O1 = ['O1']
label_O2 = ['O2']
label_O3 = ['O3']
label_empty = ['']
handles = []
for i, operator in enumerate(operators):
    data = df[df.operator==operator]
    # plt.errorbar([], [], label=operator, color=colors[i], fmt='o')
    for j, rbs in enumerate(df.rbs.unique()):
        # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())
        bohr = data[data.rbs==rbs].groupby('IPTG_uM').bohr_1027.mean()
        plt.errorbar(bohr, fc_mean, yerr=fc_err, fmt=markers[i],
                    color=colors[j], label=None, markersize=8, alpha=0.75,
                    capsize=0)
        _p, = plt.plot(bohr, fc_mean, linestyle='none', marker=markers[i],
                      markeredgewidth=2, markeredgecolor=colors[j],
                      markerfacecolor='w', alpha=0.75)

        handles.append(_p)
plt.xlabel(r'free energy ($k_BT$ units)', fontsize=18.5)
plt.ylabel('fold-change', fontsize=18.5)
plt.ylim([-0.01, 1.1])

# Generate the legend handles. Extra is empty space.
extra = pch.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                      linewidth=0)
leg_handles = [extra, extra, extra, extra, extra, extra, extra]
slc = [0, 6, 12, 18]
for i in range(len(slc) - 1):
    leg_handles.append(extra)
    sel_handles = handles[slc[i]:slc[i + 1]]
    for j in range(len(sel_handles)):
        leg_handles.append(sel_handles[j])
labels = np.concatenate([label_col, label_O1, label_empty * 6, label_O2, label_empty * 6, label_O3, label_empty * 6])
plt.legend(leg_handles, labels, loc='upper left', ncol=4, fontsize=13,
           handletextpad=-1.5)
plt.tight_layout()
plt.tick_params(labelsize=17)
output = '/Users/gchure/Dropbox/mwc_induction'
# plt.savefig(output + '/fig_data_collapse_O2_RBS1027_fit.pdf')


# Perform the collapse on the difference in free energy
def ssr_bohr(num_rep, ep_r, c, k_a=139E-6, k_i=0.53E-6, ep_ai=4.5, n_ns=4.6E6, n_sites=2):
    """
    Computes the Bohr parameter for the simple repression architecture.
    """
    pact = mwc.pact_log(c, -np.log(k_a), -np.log(k_i), epsilon=ep_ai, n=2)
    return -np.log(1 + pact * (num_rep / n_ns) * np.exp(-ep_r))

# Instantiate the legend.
label_col = ['rep./cell', 1740, 1220, 260, 124, 60, 22]
label_O1 = ['O1']
label_O2 = ['O2']
label_O3 = ['O3']
label_empty = ['']
handles = []
df['delta_F'] = ssr_bohr(df.repressors * 2, df.binding_energy, df.IPTG_uM/1E6)
plt.figure()

for i, operator in enumerate(operators):
    data = df[df.operator==operator]
    # plt.errorbar([], [], label=operator, color=colors[i], fmt='o')
    for j, rbs in enumerate(df.rbs.unique()):
        # compute the mean value for each concentration
        fc_mean = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs==rbs].groupby('IPTG_uM').fold_change_A.std() / \
        np.sqrt(data[data.rbs==rbs].groupby('IPTG_uM').size())
        delta_F = data[data.rbs==rbs].groupby('IPTG_uM').delta_F.mean()
        plt.errorbar(delta_F, fc_mean, yerr=fc_err, fmt=markers[i],
                    color=colors[j], label=None, markersize=8, alpha=0.75,
                    capsize=0, markerfacecolor='w', markeredgecolor=colors[j],
                    markeredgewidth=2)
        _p, = plt.plot(delta_F, fc_mean, linestyle='none', marker=markers[i],
                      markeredgewidth=2, markeredgecolor=colors[j],
                      markerfacecolor='w', alpha=0.75, zorder=10)
        handles.append(_p)
delta_F_range = np.linspace(-8, 0, 500)
fc_df = np.exp(delta_F_range)
plt.plot(delta_F_range, fc_df, 'k-', zorder=1000)

# Generate the legend handles. Extra is empty space.
extra = pch.Rectangle((0, 0), 1, 1, fill=False, edgecolor='none',
                      linewidth=0)
leg_handles = [extra, extra, extra, extra, extra, extra, extra]
slc = [0, 6, 12, 18]
for i in range(len(slc) - 1):
    leg_handles.append(extra)
    sel_handles = handles[slc[i]:slc[i + 1]]
    for j in range(len(sel_handles)):
        leg_handles.append(sel_handles[j])
labels = np.concatenate([label_col, label_O1, label_empty * 6, label_O2, label_empty * 6, label_O3, label_empty * 6])
plt.legend(leg_handles, labels, loc='upper left', ncol=4, fontsize=13,
           handletextpad=-1.5)
plt.tight_layout()
plt.tick_params(labelsize=17)
plt.xlabel('$F_\mathrm{bound} - F_{\mathrm{bound}}^{(R = 0)}\,\, (k_BT)$', fontsize=18.5)
plt.ylabel('fold-change', fontsize=18.5)
plt.ylim([-0.05, 1.1])
# plt.yscale('log')

# Make an inset of the log scale.
plt.show()
plt.tight_layout()
plt.savefig(output + '/Figures/final_figures/sfig24.pdf', bbox_inches='tight')

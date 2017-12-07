import os
import glob
import pickle
import datetime
# Our numerical workhorses
import numpy as np
import pandas as pd

import mwc_induction_utils as mwc

# Useful plotting libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as pch
import seaborn as sns

mwc.set_plotting_style()

datadir = '../../data/'
df = pd.read_csv(datadir + 'flow_master.csv', comment='#')

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

df.head()

# Load the flat-chain
with open('../../data/mcmc/main_text_KaKi.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    gauss_flatlnprobability = unpickler.load()

# map value of the parameters
max_idx = np.argmax(gauss_flatlnprobability, axis=0)
ea, ei, sigma = gauss_flatchain[max_idx]


def bohr_fn(df, ea, ei, epsilon=4.5):
    '''
    Computes the Bohr parameter for the data in a DataFrame df as a function
    of the MWC parameters ea and ei
    Parameters
    ----------
    df : DataFrame
        Pandas DataFrame containing all the data for which to calculate the bohr
        parameter
    ea, ei : float.
        Minus log of the dissociation constants of the active and the inactive
        states respectively.
    epsilon : float.
        energy difference between the active and the inactive state.

    Returns
    -------
    bohr : array-like.
        Array with all the calculated Bohr parameters.
    '''
    bohr_param = []
    for i in range(len(df)):
        pact = mwc.pact_log(iptg=df.iloc[i].IPTG_uM, ea=ea, ei=ei,
                            epsilon=epsilon)
        F = -1 * (np.log(2 * df.iloc[i].repressors / 4.6E6) + np.log(pact) -
                  df.iloc[i].binding_energy)
        bohr_param.append(F)
    return bohr_param


df['bohr_1027'] = bohr_fn(df, ea, ei)

# Given this result let's plot all the curves using this parameters.
colors = sns.color_palette('colorblind', n_colors=8)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]
# Define the operators to use in the plot
operators = ['O1', 'O2', 'O3']
markers = ['o', 'D', 's']
F = np.linspace(-10, 10, 200)
plt.figure(figsize=(4, 3.5))
plt.plot(F, 1 / (1 + np.exp(-F)), '-', color='black')

# Instantiate the legend.
label_col = ['rep. / cell', 1740, 1220, 260, 124, 60, 22]
label_O1 = ['O1']
label_O2 = ['O2']
label_O3 = ['O3']
label_empty = ['']
handles = []
for i, operator in enumerate(operators):
    data = df[df.operator == operator]
    for j, rbs in enumerate(df.rbs.unique()):
        # compute the mean value for each concentration
        fc_mean = data[data.rbs == rbs].groupby('IPTG_uM').fold_change_A.mean()
        # compute the standard error of the mean
        fc_err = data[data.rbs == rbs].groupby('IPTG_uM').fold_change_A.std() / \
            np.sqrt(data[data.rbs == rbs].groupby('IPTG_uM').size())
        bohr = data[data.rbs == rbs].groupby('IPTG_uM').bohr_1027.mean()
        plt.errorbar(bohr, fc_mean, yerr=fc_err, fmt=None,
                     color=colors[j], label=None, markersize=8, alpha=0.75,
                     capsize=0, linewidth=1)
        _p, = plt.plot(bohr, fc_mean, linestyle='none', marker=markers[i],
                       markeredgewidth=0.5, markeredgecolor=colors[j],
                       markerfacecolor='w', markersize=4, alpha=0.45)

        handles.append(_p)
plt.xlabel(r'Bohr parameter ($k_BT$ units)', fontsize=12)
plt.ylabel('fold-change', fontsize=12)
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
labels = np.concatenate([label_col, label_O1, label_empty * 6,
                         label_O2, label_empty * 6, label_O3, label_empty * 6])
plt.legend(leg_handles, labels, loc='upper left', ncol=4, fontsize=8,
           handletextpad=-3.5)
plt.tight_layout()
plt.tick_params(labelsize=12)
plt.tight_layout()
plt.savefig('../../figures/main_figs/fig7_collapse.svg', bbox_inches='tight')

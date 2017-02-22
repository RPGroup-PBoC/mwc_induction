from Bio import SeqIO
import numpy as np
import matplotlib.pyplot as plt
import mpmath as mp
import math
import itertools
import sys
import os
import seaborn as sns
import pandas as pd
import mwc_induction_utils as mwc
mwc.set_plotting_style()

def fold_change_var_N(Reff, e_AI, N, Op):
    '''
    Computes the fold-change for N >= 1
    Parameters
    ----------
    Reff : array-like
        Array with the values of all repressor copy numbers
    e_AI : float
        Energetic difference between the active and inactive state
    N : float
        Number of operators available for repressor binding
    Op : float
        Binding energy between operator and repressor as inferred in Garcia 2011
    Returns
    -------
    fold-change
    '''

    pA = 1/(1 + np.exp(-e_AI))     #probability that a given repressor is active
    Rtot = Reff/pA
    Op = Op + np.log(pA)    #Convert Hernan energy values to actual energy values
    NNS = 4.6E6    #Number of nonspecific sites
    fc = []      #This will be my fold-change array

    if type(Reff)==np.ndarray:
        for R in Reff: #Here I use a loop to perform the summation and calculate fold-change
            t = 0
            b = 0
            for m in range(0, min(int(mp.floor(R)), int(mp.floor(N)))+1):
                t += mp.fprod([mp.fdiv(mp.factorial(mp.floor(R)), mp.fmul(mp.mpf(NNS)**m, mp.factorial(mp.floor(R - m)))), mp.binomial(mp.floor(N),m), mp.exp(-m*Op), mp.floor(N)-mp.mpf(m)])
                b += mp.fprod([mp.fdiv(mp.factorial(mp.floor(R)), mp.fmul(mp.mpf(NNS)**m, mp.factorial(mp.floor(R - m)))), mp.binomial(mp.floor(N),m), mp.exp(-m*Op)])
            fc.append(float(t/(mp.floor(N)*b)))
    else:
        t = 0
        b = 0
        for m in range(0, min(int(mp.floor(Reff)), int(mp.floor(N)))+1):
            t += mp.fprod([mp.fdiv(mp.factorial(mp.floor(Reff)), mp.fmul(mp.mpf(NNS)**m, mp.factorial(mp.floor(Reff - m)))), mp.binomial(mp.floor(N),m), mp.exp(-m*Op), mp.floor(N)-mp.mpf(m)])
            b += mp.fprod([mp.fdiv(mp.factorial(mp.floor(Reff)), mp.fmul(mp.mpf(NNS)**m, mp.factorial(mp.floor(Reff - m)))), mp.binomial(mp.floor(N),m), mp.exp(-m*Op)])
        fc.append(float(t/(mp.floor(N)*b)))
    return (Rtot, fc)

# Load data from Brewster 2014

data_file = 'C:/Users/Stephanie/Dropbox/mwc_induction/tidy_lacI_multiple_operator_data.csv'
df= pd.read_csv(data_file)

# Establish parameters

Reff = np.arange(0., 1000., 1)
N = 10
N_vals = np.array(df.N.unique(), dtype=float)
e_AI = 4.5
e_AI_array = [-4.0, -2.0, 0, 2.0, 4.0]
O1 = -15.3

# Set color palette
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]


#Plot a) curves showing how e_AI affects curve and b) plots showing fits to Brewster 2014 data

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(11.5, 4))

for i in range(0,len(e_AI_array)):
    fold_change = fold_change_var_N(Reff, e_AI_array[i], N, O1)
    x_coord = N * (1 + np.exp(-e_AI_array[i]))
    x_coord_Reff = x_coord/(1 + np.exp(-e_AI_array[i]))
    point = fold_change_var_N(x_coord_Reff, e_AI_array[i], N, O1)
    ax1.plot(x_coord, point[1], marker='^', fillstyle='full',\
             markerfacecolor='white', markeredgecolor=colors[i], markeredgewidth=2.0, zorder=(i + 5))
    ax1.plot([x_coord, x_coord], [0, point[1][0]], '--', color=colors[i])
    ax1.plot(fold_change[0], fold_change[1], color = colors[i], label = e_AI_array[i])

for j in range(len(N_vals)):
    val = N_vals[j]
    energy = df.energy[df.N==val].unique()[0]
    reps = np.array(df.repressor[df.N==val])
    fc = np.array(df.fold_change[df.N==val])
    ax2.plot(reps, fc, 'o', color=colors[j], label=None)
    ax2.plot(fold_change_var_N(Reff, e_AI, val, energy)[0],\
             fold_change_var_N(Reff, e_AI, val, energy)[1], color=colors[j], label=('%.1f, %.0f' % (energy, val)))

# Make labels

title_dict = {ax1:r'$\Delta \varepsilon_{AI}\ (k_BT)$', ax2:r'$\Delta \varepsilon_{RA}\ (k_BT),\ N$'}
for ax in (ax1, ax2):
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('repressors/cell')
    ax.set_ylabel('fold-change')
    ax.set_xlim(0, 1000)
    ax.set_ylim(1E-4, 1)
    leg = ax.legend(loc='lower left', title=title_dict[ax])
    leg.get_title().set_fontsize(15)

plt.figtext(0.005, 0.95, 'A', fontsize=20)
plt.figtext(0.5, 0.95, 'B', fontsize=20)
plt.tight_layout()
plt.show()

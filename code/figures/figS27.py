import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.patches import Rectangle

import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Functions for calculating leakiness, saturation, and dynamic range

def leakiness(K_A, K_I, e_AI, R, Op):
    '''
    Computes the leakiness of a simple repression construct
    Parameters
    ----------
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
    leakiness
    '''
    return 1 / (1 + 1 / (1 + np.exp(-e_AI)) * R / 5E6 * np.exp(-Op))

def saturation(K_A, K_I, e_AI, R, Op):
    '''
    Computes the saturation of a simple repression construct
    Parameters
    ----------
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
    saturation
    '''
    return 1 / (1 + 1 / (1 + np.exp(-e_AI) * (K_A / K_I)**2) * R / 5E6 *
                np.exp(-Op))


def dynamic_range(K_A, K_I, e_AI, R, Op):
    '''
    Computes the dynamic range of a simple repression construct
    Parameters
    ----------
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
    dynamic range
    '''
    return saturation(K_A, K_I, e_AI, R, Op) - leakiness(K_A, K_I, e_AI, R, Op)


# Establish parameter values
K_A = 139
K_I = 0.53
e_AI = 4.5
Reps = np.array([1740, 1220, 260, 124, 60, 22])
op_array = np.linspace(-18, -8, 100)
O1 = -15.3
O2 = -13.9
O3 = -9.7
ops = [O1, O2, O3]
markers = ['o', 'D', 's']
ops_dict = dict(zip(ops, markers))

# Set color palette
colors = sns.color_palette('colorblind', n_colors=7)
colors[4] = sns.xkcd_palette(['dusty purple'])[0]

# Make plots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2,
                                             figsize=(13, 9.5))
labels = []
for i in range(len(Reps)):
    ax1.plot(op_array, leakiness(K_A, K_I, e_AI, Reps[i], op_array),
    color=colors[i], label=Reps[i])
    ax2.plot(op_array, saturation(K_A, K_I, e_AI, Reps[i], op_array),
    color=colors[i])
    ax3.plot(op_array, dynamic_range(K_A, K_I, e_AI, Reps[i], op_array),
    color=colors[i])
    ax4.axis('off')

    for op in ops:
        ax1.plot(op, leakiness(K_A, K_I, e_AI, Reps[i], op),
                 marker=ops_dict[op],
                 markerfacecolor='white', markeredgecolor=colors[i],
                 markeredgewidth=2)
        ax2.plot(op, saturation(K_A, K_I, e_AI, Reps[i], op),
                 marker=ops_dict[op],
                 markerfacecolor='white', markeredgecolor=colors[i],
                 markeredgewidth=2)
        ax3.plot(op, dynamic_range(K_A, K_I, e_AI, Reps[i], op),
                 marker=ops_dict[op],
                 markerfacecolor='white', markeredgecolor=colors[i],
                 markeredgewidth=2)


# Create legends
empty = [Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)] # List containing one empty rectangle
ims = []
for m in markers:
    for i in range(len(Reps)):
        ims.append(mlines.Line2D([], [], marker=m, markerfacecolor='white',
                   markeredgecolor=colors[i], markeredgewidth=2, linestyle='None'))
legend_handles = empty * 8 + ims[:6] + empty + ims[6:12] + empty + ims[12:]

rep_labels = ['rep./cell', '1740', '1220', '260', '124', '60', '22']
op_labels = [['O1'], ['O2'], ['O3']]
empty_label = ['']
legend_labels = rep_labels + op_labels[0] + empty_label * 6 + op_labels[1] + empty_label * 6 + op_labels[2] + empty_label * 6
ax3.legend(legend_handles, legend_labels, ncol=4, bbox_to_anchor=(2.1, 0.85), handletextpad=-1.65, fontsize=15)

leg = ax1.legend(loc='upper left', title='repressors/cell')
leg.get_title().set_fontsize(15)

# Label axes
labels_dict = {ax1 : {'ylabel' : 'leakiness', 'plotlabel' : '(A)'},
               ax2 : {'ylabel' : 'saturation', 'plotlabel' : '(B)'},
               ax3 : {'ylabel' : 'dynamic range', 'plotlabel' : '(C)'}}
for ax in (ax1, ax2, ax3):
    ax.set_xlabel(r'binding energy $\Delta \varepsilon_{RA}\ (k_BT)$')
    ax.set_ylabel(labels_dict[ax]['ylabel'])
    ax.text(-20.5, 1.02, labels_dict[ax]['plotlabel'], fontsize=20)

plt.savefig('../../figures/SI_figs/figS27.pdf', bbox_inches='tight')

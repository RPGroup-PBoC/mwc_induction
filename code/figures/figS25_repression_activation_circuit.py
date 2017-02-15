"""
Title: figS25_repression_activation.py
Author: Griffin Chure
Last Modified: 2017-02-14
Purpose: To generate the fold-change surfaces shown in figure S25.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set_context('talk')
sns.axes_style('white')


def simple_circuit(c_r, c_a, R, A, ka_r, ki_r, ka_a, ki_a, ep_r, ep_a, ep_ai_a,
            ep_ai_r, ep_ap, nns=4.6E6):
    """
    Computes the fold-change equation for a dual allosteric simple repression
    activation genetic circuit. See SI section "Applications" for details.

    Parameters
    ----------
    c_r, c_a: 2d meshed arrays
        Range of inducer a coactivator concentrations to calculate fold-change
        over. Both should be in the same units as the ka_x and ki_x parameters.
    R, A : ints
        Number of repressors and activators per cell.
    ka_r, ki_r : floats
        Dissociation constants of the inducer to the active and inactive
        repressor respectively. These should be in the same units as c_r.
    ka_a, ki_a : floats
        Dissociation constants of the coactivator to the active and inactive
        activator respecitvely. These should be in the same units as c_a.
    ep_r, ep_a : floats
        Binding energy of the repressor and activator to the DNA respectively.
    ep_ai_a, ep_ai_r: floats
        Allosteric energy between the active and inactive states of both the
        activator and repressor.
    ep_ap : float
        Interaction energy between the activator and polymerase.
    n_ns : int, optional
        Number of non-specific binding sites available to the activator and
        repressor. This function assumes this value is the same for both
        transcription factors. Default value is the length of the E. coli
        genome in units of base pairs.

    Returns
    -------
    fold_change : 2d meshed array
        Fold change values evaluated at each pair of c_r and c_a.
    """
    # Compute the allosteric terms
    pact_a = (1 + c_a / ka_a)**2 / ((1 + c_a/ka_a)**2 + np.exp(ep_ai_a) *
                                    (1 + c_a / ki_a)**2)
    pact_r = (1 + c_r / ka_r)**2 / ((1 + c_r/ka_r)**2 + np.exp(-ep_ai_r) *
                                    (1 + c_r / ki_r)**2)
    # Compute the fold-change.
    numerator = 1 + pact_a * (A / nns)*np.exp(-(ep_a + ep_ap))
    denominator = 1 + pact_a * (A / nns) * np.exp(-ep_a) +\
                 pact_r * (R / nns) * np.exp(-ep_r) +\
                 pact_r * pact_a * (A / nns) * (R / nns) * np.exp(-(ep_a +\
                 ep_r))
    return numerator / denominator


# Define standard parameters.
c = np.logspace(-10, -1, 500)
R = [100, 100, 500, 10]
A = [100, 100, 10, 500]
ep_r = [-15, -10, -12, -12]
ep_a = [-10, -15, -12, -12]
ep_ai_a = 7
ep_ai_r = 7
ep_ap = -3.9
ka_a = 100E-9
ki_a = 500E-6
ka_r = 500E-6
ki_r = 100E-9

# Set up the meshed inducer and coactivator concentrations.
CR, CA = np.meshgrid(c, c)

# Set up the figure canvas.
with sns.axes_style('white'):
    # Compute the fold-change.
    for i in range(4):
        plt.close('all')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        fc = simple_circuit(CR, CA, R[i], A[i], ka_r, ki_r, ka_a, ki_a,
            ep_r[i], ep_a[i], ep_ai_a, ep_ai_r, ep_ap)
        if i < 2:
            cm = 'magma'
        else:
            cm = 'viridis'
        ax.plot_surface(np.log10(CR/ka_r), np.log10(CA/ka_a), np.log10(fc), cmap=cm)
        ax.set_xlabel(r'$\log_{10}\frac{c_R}{K_A^{(R)}}$', labelpad=25, fontsize=25)
        ax.set_ylabel(r'$\log_{10}\frac{c_A}{K_A^{(A)}}$', labelpad=25, fontsize=25)
        ax.set_zlabel(r'$\log_{10}$ fold-change', labelpad=25, fontsize=25)
        ax.set_zlim3d([-2, 2])
        ax.set_xlim3d([-7, 2])
        ax.set_ylim3d([-2, 6])
        ax.set_xticks([-6, -4, -2, 0, 2])
        ax.set_yticks([-2, 0, 2, 4, 6])
        ax.set_zticks([-2, -1, 0, 1, 2])
        ax.view_init(30, 285)
        ax.tick_params(labelsize=18)
        plt.tight_layout()
        plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/surface_%s.svg' %(i+1), bbox_inches='tight')

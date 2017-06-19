# Import numerical and plotting modules/functions
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import fsolve

# Import the project utils
import sys
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc
mwc.set_plotting_style()

colors=sns.color_palette('colorblind').as_hex()
colors[4] = sns.xkcd_palette(['dusty purple']).as_hex()[0]
sns.set_palette(colors)

# Define functions to be used in figure

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

def fugacity(IPTG, R, Ns, e_s, K_A=139E-6, K_I=0.53E-6, e_AI=4.5, Nc=0, e_c=0):
    '''
    Solves for the fugacity of simple repression with
    multiple promoter copies (Ns, with energy e_s) or competitor sites
    (Nc, with energy e_c).
    Parameters
    ----------
    R : float
        Number of repressors per cell
    e_AI : float
        Energetic difference between the active and inactive state
    Ns : float
        Number of specific operators available for repressor binding
    Nc : float
        Number of competitor operators available for repressor binding
    e_s : float
        Binding energy between specific operator and repressor as inferred in Garcia 2011
    e_c : float
        Binding energy between competitor operator and repressor
    K_A : float
        Dissociation constant for active repressor
    K_I : float
        Dissociation constant for inactive repressor
    e_AI : float
        Energetic difference between the active and inactive state
    Returns
    -------
    fugacity at each IPTG concentration
    '''
    NNS = 4.6E6
    lam = []
    func = lambda x: -Reff + Ns*(x * np.exp(-e_s))/(1 + x * np.exp(-e_s)) +\
                         NNS * (x)/(1 + x) + \
                         Nc*(x * np.exp(-e_c))/(1 + x * np.exp(-e_c))
    for c in IPTG:
        Reff = R * pact(c, K_A, K_I, e_AI)
        lam.append(fsolve(func, 0))
    return np.array(lam)

def occupancy(lam, e_s):
    '''
    Computes fold-change for simple repression using the fugacity (lam).
    Parameters
    ----------
    lam : float
        fugacity of system as calculated by fugacity()
    e_s : float
        binding energy of specific operator
    Returns
    -------
    fold-change (occupancy)
    '''
    return 1/(1 + lam * np.exp(-(e_s)))

# Define parameter values
ops = [-15.3, -13.9, -9.7]
op_names = ['O1', 'O2', 'O3']
fig_labels = ['A', 'B', 'C']
reps = [1740, 1220, 260, 124, 60, 22]
Nc = [1, 10, 50, 100, 250, 500]
Ns = [1]
IPTG = np.logspace(-8, -2, 100)
R=260
e_c = -17.0

# Plot figure
fig, ax = plt.subplots(ncols=3,  sharey=False, figsize=(16, 4))

for i, a in enumerate(ax):
    for N in Nc:
        lam_array = fugacity(IPTG, R, Ns=1, e_s=ops[i], Nc=N, e_c=e_c)
        fc = occupancy(lam_array, ops[i])
        a.plot(IPTG, fc, label=N)
    a.set_xscale('log')
    a.set_ylabel('fold-change')
    a.set_xlabel('[IPTG] (M)')
    a.set_ylim(-0.01,1.1)
    a.set_xlim(1E-8, 1E-2)
    a.tick_params(labelsize=14)

    # Add figure text
    a.text(7E-4, 0.1, '%s \n' % op_names[i] + r'$\Delta \varepsilon_{RA}= %0.1f\ k_BT$' % ops[i],\
           ha='center', va='center', fontsize=14)
    a.text(1E-9, 1.1, fig_labels[i], ha='center', va='center', fontsize=24)

# Add legend
leg1 = ax[0].legend(title=r'$N_c$', fontsize=12, loc='center right')
leg1.get_title().set_fontsize(14)

plt.tight_layout()
plt.savefig('fugacity_Nc.pdf', bbox_inches='tight')

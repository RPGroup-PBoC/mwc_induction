import numpy as np
import scipy.special
import scipy.stats as sc

def fold_change_log(IPTG, ea, ei, epsilon, R, epsilon_r):
    '''
    Returns the gene expression fold change according to the thermodynamic model
    with the extension that takes into account the effect of the inducer.
    Parameter
    ---------
    IPTG : array-like.
        concentrations of inducer on which to evaluate the function
    epsilon : float.
        energy difference between the active and the inactive state
    R : array-like.
        repressor copy number for each of the strains. The length of this array
        should be equal to the IPTG array. If only one value of the repressor is
        given it is asssume that all the data points should be evaluated with
        the same repressor copy number
    epsilon_r : array-like
        repressor binding energy. The length of this array
        should be equal to the IPTG array. If only one value of the binding
        energy is given it is asssume that all the data points 
        should be evaluated with the same repressor copy number
        
    Returns
    -------
    fold-change : float.
        gene expression fold change as dictated by the thermodynamic model.
   '''
    return 1.0 / (1.0 + 2.0 * R / 5.0E6 * pact_log(IPTG, ea, ei, epsilon) * \
            (1.0 + np.exp(-epsilon)) * np.exp(-epsilon_r))


    # define a funciton to compute the fold change as a funciton of IPTG
def pact_log(IPTG, ea, ei, epsilon):
    '''
    Returns the probability of a repressor being active as described by the MWC
    model.
    Parameter
    ---------
    IPTG : array-like.
        concentrations of inducer on which to evaluate the function
    epsilon : float.
        energy difference between the active and the inactive state
    Returns
    -------
    pact : float.
        probability of a repressor of being in the active state. Active state is
        defined as the state that can bind to the DNA.
    '''
    pact = (1.0 + IPTG * np.exp(ea))**2.0 / \
    ((1.0 + IPTG * np.exp(ea))**2.0 + np.exp(-epsilon) * (1.0 + IPTG * np.exp(ei))**2.0)
    return pact
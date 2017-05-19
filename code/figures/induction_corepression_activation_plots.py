import numpy as n
import matplotlib.pyplot as plt
import seaborn as sns
import sys
sys.path.insert(0,'../analysis/')
import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Define the parameters for each case.
num_tf = 50
tf_ep = -14  # in units of k_BT
ep_ai = 5
ka = 200E-6  # in units of micromolar
ki = 0.5E-6  # in units of micromolar
ep_ap = -5  # in units of k_BT
n_sites = 2
c_range = np.logspace(-9, -2, 500)

# Compute the fc for each case.
def repression(num_r, ep_r, ep_ai, c, ka, ki, n_sites=2):
    pact_num = (1 + c / ka)**n_sites
    pact_denom = pact_num + np.exp(-ep_ai) * (1 + c / ki)**n_sites
    return (1 + (pact_num / pact_denom) * (num_r / 4.6E6) * np.exp(-ep_r))**-1

def activation(num_act, ep_a, ep_ap, ep_ai, c, ka, ki, n_sites=2):
    pact_num = (1 + c / ka)**n_sites
    pact_denom = pact_num + np.exp(-ep_ai)*(1 + c / ki)**n_sites
    numerator = 1 + (pact_num / pact_denom) * (num_act / 4.6E6) * np.exp(-(ep_a + ep_ap))
    denom = 1 + (pact_num / pact_denom) * (num_act / 4.6E6) * np.exp(-ep_a)
    return numerator / denom

# Compute the foldchange for each case.
ind = repression(num_tf, tf_ep, ep_ai, c_range, ka, ki)
corep = repression(num_tf, tf_ep, ep_ai - 2 * np.log(ka/ki),
                   c_range, ki, ka)
# Make a 3 x 3 plot.
plt.close('all')
plt.figure(figsize=(6,4))
plt.plot(c_range, ind, '-', color='#D56C55', lw=3)
plt.plot(c_range, corep, '-', color='#738FC1', lw=3)
# ax.plot(c_range, act)

plt.xscale('log')
plt.ylim([0, 1.0])
plt.xticks([1E-8, 1E-6, 1E-4, 1E-2], ['', '', '', ''])

plt.ylabel('fold-change', fontsize=15)


plt.savefig('/Users/gchure/Dropbox/mwc_induction/resubmission figures/properties_curves.svg')

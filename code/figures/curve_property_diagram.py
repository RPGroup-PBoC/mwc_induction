import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mwc_induction_utils as mwc
mwc.set_plotting_style()
sns.set_context('talk')

# Define the fold change.
iptg_range = np.logspace(-7, -3, 1000)
ka = 141E-6
ki = 560E-9
ep_ai = 4.5
ep_ra = -9.7

def fold_change(num_rep, iptg_range, ka, ki, ep_ai, ep_ra):
    pact = (1 + iptg_range / ka)**2 / ((1 + iptg_range/ka)**2 +
            np.exp(-ep_ai)*(1 + iptg_range/ki)**2)
    repression = 1 + pact * (num_rep/4.6E6) * np.exp(-ep_ra)
    return 1/repression

p = sns.xkcd_palette(['dusty purple'])[0]
fc = fold_change(1740, iptg_range, ka, ki, ep_ai, ep_ra)
plt.close('all')
plt.figure(figsize=(6, 4))
plt.xlabel('c (M)', fontsize=12)
plt.ylabel('fold-change', fontsize=12)
plt.tick_params(labelsize=10)
plt.margins(0.02)
plt.ylim([0, 1.2])


ax = plt.gca()
plt.text(0.03, 0.82, 'saturation', color='w', backgroundcolor='g',
        fontsize=10, transform=ax.transAxes)
plt.text(0.7, 0.105, 'leakiness', color='w', backgroundcolor='b', fontsize=10,
        transform=ax.transAxes)
plt.hlines(0.565, 1E-7, 1.50217E-5, color='c', alpha=0.45, linewidth=3)
plt.vlines(1.50217E-5, 0, 0.565, color='c', alpha=0.45, linewidth=3)
plt.hlines(np.min(fc), 1E-7, 1E-3, color='b', linewidth=3, alpha=0.45)
plt.hlines(np.max(fc), 1E-7, 1E-3, color='g', linewidth=3, alpha=0.45)
plt.plot(1.50127E-5, 0.565,'o',color='c', alpha=0.85, markersize=12)
plt.text(0.43, 0.05, r'$EC_{50}$', color='w', backgroundcolor='c', fontsize=10,
        transform=ax.transAxes)
plt.plot(iptg_range, fc, 'r-', linewidth=3)
plt.plot(iptg_range, -1.235 + 3.3 * np.linspace(0, 1.0, 1000), 'k--',
        linewidth=2.5,
        alpha=0.75)
plt.text(0.45, 0.9, 'effective Hill coefficient', backgroundcolor='#4b4b4b',
color='w', transform=ax.transAxes, fontsize=10)
plt.vlines(1.5E-3, np.min(fc),np.max(fc), color=p, linewidth=3)
plt.hlines(np.max(fc), 1.2E-3, 1.55E-3, color=p, linewidth=3)
plt.hlines(np.min(fc), 1.2E-3, 1.55E-3, color=p, linewidth=3)
plt.text(0.5E-3, 0.57, 'dynamic range', color='w', backgroundcolor=p,
        fontsize=10)
#plt.hlines(np.max(fc),1, 1.5, color='g', linewidth=2,
#ktransform=ax.transAxes)
plt.xscale('log')
plt.xlim([1E-7, 0.5E-2])
plt.tight_layout()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/Figures/supplementary_figures/titration_properties.pdf',
bbox_inches='tight')

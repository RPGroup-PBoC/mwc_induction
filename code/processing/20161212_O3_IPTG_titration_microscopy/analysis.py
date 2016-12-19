# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy
import os
import glob

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# Seaborn, useful for graphics
import seaborn as sns

# For image processing.
import skimage.io

# Load custom written modules.
import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Set some values.
DATE = 20161212
OPERATOR = 'O3'
STRAINS = ('auto', 'delta', 'RBS1027')
IPTG_RANGE = [0, 0.1, 1, 5, 10, 25, 50, 75, 100, 250, 500, 1000]
# Load the data files.
df = pd.read_csv('output/20161212_O3_IPTG_titration_microscopy.csv')

# Generate a plot of the distributions.
grouped = pd.groupby(df, ['rbs', 'IPTG_uM']).mean_intensity.apply(mwc.ecdf)
fig, ax = plt.subplots(3, 1, sharex=True)
colors = sns.color_palette('PuBu', n_colors=len(IPTG_RANGE))
for i, st in enumerate(STRAINS):
    samp = grouped[st].values
    if i is 0:
        labels = IPTG_RANGE
    else:
        labels = len(IPTG_RANGE) * [None]
    for j, _ in enumerate(IPTG_RANGE):
        ax[i].plot(samp[j][0], samp[j][1], '.', color=colors[j],
                   label=labels[j])
for i, a in enumerate(ax):
    a.margins(0.02)
    a.set_title(STRAINS[i])
    a.set_xscale('log')
ax[1].set_ylabel('ECDF')
ax[2].set_xlabel('mean pixel intensity (A.U.)')
ax[0].legend(title='IPTG (µM)', bbox_to_anchor=(1.18, 1))
plt.savefig('output/' + str(DATE) + '_' + OPERATOR +
            '_intensity_distributions.png', bbox_inches='tight')

# Plot the means
grouped_means = pd.groupby(df, ['rbs', 'IPTG_uM']).mean_intensity.mean()
fig, ax = plt.subplots()
for i, st in enumerate(STRAINS):
    samp = grouped_means[st].values
    ax.plot(np.array(IPTG_RANGE) * 1E-6, samp, 'o--', label=st)
ax.legend(title='strain', loc='center left')
ax.set_xlabel('IPTG (M)')
ax.set_xscale('log')
ax.set_ylabel('mean pixel intensity')
plt.tight_layout()
plt.savefig('output/' + str(DATE) + '_' + OPERATOR +
            '_IPTG_mean_fluoresence.png', bbox_inches='tight')


# Plot the fold change vs the prediction.
epa = -np.log(141E-6)
epi = -np.log(0.56E-6)
epr = -9.7  # In units of kBT
iptg = np.logspace(-9, -2, 1000)
R = np.array([130])  # Number of lac tetramers per cell.
fc = mwc.fold_change_log(iptg, epa, epi, 4.5, R, epr)
# Group the dataframe by IPTG concentration then strain.
fc_grouped = pd.groupby(df, 'IPTG_uM')
fc_exp = []
for group, data in fc_grouped:
    rbs_grp = pd.groupby(data, 'rbs').mean_intensity.mean()
    fc_exp.append((rbs_grp['RBS1027'] - rbs_grp['auto']) /
                  (rbs_grp['delta'] - rbs_grp['auto']))
# Plot the prediction.
plt.figure()
plt.plot(iptg, fc, 'r-', label='prediction')

# Plot the data points.
plt.plot(np.array(IPTG_RANGE)/1E6, fc_exp, 'ro', label='data from microscopy')
plt.legend(title=OPERATOR + ', ' + STRAINS[-1], loc='upper left')
plt.ylim([0, 1.2])
plt.xlabel('IPTG (M)')
plt.ylabel('fold-change')
plt.xscale('log')
plt.tight_layout()

plt.savefig('output/' + str(DATE) + '_' + OPERATOR + '_IPTG_titration.png',
            bbox_inches='tight')

df_IPTG = df.IPTG_uM.unique()

fc_dict = {'date': DATE, 'username': 'gchure', 'operator': OPERATOR,
           'binding_energy': epr, 'rbs': STRAINS[-1],
           'repressors': R[0], 'IPTG_uM': df_IPTG, 'fold_change': fc_exp}
fc_df = pd.DataFrame(fc_dict)
fc_df.to_csv('output/' + str(DATE) + '_' + OPERATOR + '_' + STRAINS[-1] +
             '_IPTG_titration_microscopy_foldchange.csv', index=False)
filenames = ['comments.txt', 'output/' + str(DATE) + '_' + OPERATOR + '_' +
             STRAINS[-1] + '_IPTG_titration_microscopy_foldchange.csv']
with open('../../../data/' + str(DATE) + '_' + OPERATOR +
          '_IPTG_titration_microscopy_foldchange.csv', 'w') as output:
    for fname in filenames:
        with open(fname) as infile:
            output.write(infile.read())

import numpy as np
import matplotlib.pyplot as plt
import mwc_induction_utils as mwc
import pandas as pd
import pickle
mwc.set_plotting_style()


# Define the constants.
R = np.array([130])
inducer = np.logspace(-9, -2, 1000)

# Open the flat chains and data set..
flow_pkl = '../../data/mcmc/SI_C_all_data_O1O2_KaKi_flow_cytometry.pkl'
mic_pkl = '../../data/mcmc/SI_C_all_data_O1O2_KaKi_microscopy.pkl'
with open(flow_pkl, 'rb') as file:
    unpickler = pickle.Unpickler(file)
    flow_gauss_flatchain = unpickler.load()
    flow_gauss_flatlnprobability = unpickler.load()
with open(mic_pkl, 'rb') as file:
    unpickler = pickle.Unpickler(file)
    mic_gauss_flatchain = unpickler.load()
    mic_gauss_flatlnprobability = unpickler.load()

flow_data = pd.read_csv('../../data/flow_master.csv', comment='#')
flow_data = flow_data[((flow_data['operator'] == 'O1') | (
    flow_data['operator'] == 'O2')) & (flow_data['rbs'] == 'RBS1027')]

mic_data = pd.read_csv('../../data/microscopy_master.csv', comment='#')
mic_data = mic_data[((mic_data['operator'] == 'O1') | (
    mic_data['operator'] == 'O2')) & (mic_data['rbs'] == 'RBS1027')]

# Set up a data frame with the chain.
index = ['ka', 'ki', 'sigma']
flow_mcmc = pd.DataFrame(flow_gauss_flatchain, columns=index)
mic_mcmc = pd.DataFrame(mic_gauss_flatchain, columns=index)
flow_mcmc['Ka'] = np.exp(-flow_mcmc['ka'])
flow_mcmc['Ki'] = np.exp(-flow_mcmc['ki'])
mic_mcmc['Ka'] = np.exp(-mic_mcmc['ka'])
mic_mcmc['Ki'] = np.exp(-mic_mcmc['ki'])

index = flow_mcmc.columns

# Extract the mode
flow_max_idx = np.argmax(flow_gauss_flatlnprobability, axis=0)
mic_max_idx = np.argmax(mic_gauss_flatlnprobability, axis=0)
f_ea, f_ei, f_sigma, f_ka, f_ki = flow_mcmc.iloc[flow_max_idx, :]
m_ea, m_ei, m_sigma, m_ka, m_ki = mic_mcmc.iloc[mic_max_idx, :]

# Convert each to molar concentraitons.
f_ka /= 1E6
f_ki /= 1E6
m_ka /= 1E6
m_ki /= 1E6

# Extract the HPD.
flow_ka_hpd = mwc.hpd(flow_mcmc.iloc[:, 3], 0.95)
flow_ki_hpd = mwc.hpd(flow_mcmc.iloc[:, 4], 0.95)
mic_ka_hpd = mwc.hpd(mic_mcmc.iloc[:, 3], 0.95)
mic_ki_hpd = mwc.hpd(mic_mcmc.iloc[:, 4], 0.95)

# Group each DataFrame for the plot.
flow_grouped = flow_data.groupby('operator')
mic_grouped = mic_data.groupby('operator')

# Compute the four theory curves.
flow_O1_theory = mwc.fold_change_log(inducer, -np.log(f_ka), -np.log(f_ki),
                                     4.5, R, epsilon_r=-15.3)
mic_O1_theory = mwc.fold_change_log(inducer, -np.log(m_ka), -np.log(m_ki),
                                    4.5, R, epsilon_r=-15.3)

flow_O2_theory = mwc.fold_change_log(inducer, -np.log(f_ka), -np.log(f_ki),
                                     4.5, R, epsilon_r=-13.9)
mic_O2_theory = mwc.fold_change_log(inducer, -np.log(m_ka), -np.log(m_ki),
                                    4.5, R, epsilon_r=-13.9)

# Compute  the hisograms from the MCMC.
f_ka_hist, f_ka_bins = np.histogram(flow_mcmc['Ka'], bins=100, range=(50, 300))
f_ka_hist = f_ka_hist / np.sum(f_ka_hist)
f_ki_hist, f_ki_bins = np.histogram(
    flow_mcmc['Ki'], bins=100, range=(0.4, 0.9))
f_ki_hist = f_ki_hist / np.sum(f_ki_hist)

m_ka_hist, m_ka_bins = np.histogram(mic_mcmc['Ka'], bins=100, range=(50, 300))
m_ka_hist = m_ka_hist / np.sum(m_ka_hist)
m_ki_hist, m_ki_bins = np.histogram(mic_mcmc['Ki'], bins=100, range=(0.4, 0.9))
m_ki_hist = m_ki_hist / np.sum(m_ki_hist)

# %% Set up the figure axis. This one is a bit complicated.
colors = {'O1': 'b', 'O2': 'r'}
plt.close('all')
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot2grid((4, 5), (0, 0), colspan=3, rowspan=4)
ax2 = plt.subplot2grid((4, 5), (2, 3), colspan=2, rowspan=2)
ax3 = plt.subplot2grid((4, 5), (0, 3), colspan=2, rowspan=2)

# Add the axes labels and scaling.
ax1.set_xlabel('[IPTG] (M)')
ax2.set_xlabel('$K_I\, (\mu\mathrm{M})$')
ax3.set_xlabel('$K_A\, (\mu\mathrm{M})$')
ax1.set_ylabel('fold-change')
ax2.set_ylabel('$P(K_I\, | \, \mathrm{data})$')
ax3.set_ylabel('$P(K_A\, |\,  \mathrm{data})$')
ax1.set_xscale('log')
ax1.set_ylim([0, 1.1])
ax1.set_xlim([1E-9, 1E-2])
fig.text(0, 0.95, '(A)', fontsize=12)
fig.text(0.59, 0.95, '(B)', fontsize=12)


# Plot the theory curves. ORDER HERE MATTERS.
_ = ax1.plot(inducer, flow_O1_theory, 'b--', lw=1.5)
_ = ax1.plot(inducer, mic_O1_theory, 'b-', lw=1.5)

_ = ax1.plot(inducer, flow_O2_theory, 'r--', lw=1.5)
_ = ax1.plot(inducer, mic_O2_theory, 'r-', lw=1.5)

# Add fake entries for clear legends.
_ = ax1.plot([], [], 'o', markerfacecolor='w',
             markeredgecolor='b', markeredgewidth=2)
_ = ax1.plot([], [], 'o', color='b')
_ = ax1.plot([], [], 'o', markerfacecolor='w',
             markeredgecolor='r', markeredgewidth=2)
_ = ax1.plot([], [], 'o', color='r')
labels = ['', '', '', '', 'fit to flow cytometry',
          'fit to microscopy', 'data from microscopy', 'data from flow cytometry']
_ = ax1.legend(labels, loc='upper left', ncol=2, columnspacing=0.1)
t1 = ax1.text(0.04, 0.75, r'$\Delta\varepsilon_{RA} = -15.3\,k_BT$',
              fontsize=8, transform=ax1.transAxes)
t1.set_bbox(dict(color='b', alpha=0.3))
t2 = ax1.text(0.04, 0.68, r'$\Delta\varepsilon_{RA} = -13.9\,k_BT$',
              fontsize=8, transform=ax1.transAxes)
t2.set_bbox(dict(color='r', alpha=0.3))

# Plot the experimental data.
for g, d in flow_grouped:
    _ = ax1.plot(d['IPTG_uM'] / 1E6, d['fold_change_A'], 'o', color=colors[g],
                 alpha=0.5)
for g, d in mic_grouped:
    _ = ax1.plot(d['IPTG_uM'] / 1E6, d['fold_change'], 'o',
                 markerfacecolor='w', markeredgecolor=colors[g],
                 markeredgewidth=1.5)

# Plot the posterior disributions.
_ = ax2.step(f_ki_bins[:-1], f_ki_hist, color='k', alpha=0.7)
_ = ax2.step(m_ki_bins[:-1], m_ki_hist, color='c', alpha=0.7)
_ = ax2.fill_between(f_ki_bins[:-1], f_ki_hist, color='k', alpha=0.5,
                     step='pre')
_ = ax2.fill_between(m_ki_bins[:-1], m_ki_hist,
                     color='c', alpha=0.5, step='pre')
_ = ax3.step(f_ka_bins[:-1], f_ka_hist, color='k',
             alpha=0.7, label='flow\n cytometry')
_ = ax3.step(m_ka_bins[:-1], m_ka_hist, color='c',
             alpha=0.7, label='microscopy')
_ = ax3.fill_between(f_ka_bins[:-1], f_ka_hist,
                     color='k', alpha=0.5, step='pre')
_ = ax3.fill_between(m_ka_bins[:-1], m_ka_hist,
                     color='c', alpha=0.5, step='pre')
_ = ax3.legend(loc='upper right')

mwc.scale_plot(fig, 'two_row')
plt.tight_layout()
plt.savefig('../../figures/SI_figs/figS14.pdf', bbox_inches='tight')

import os
import glob
import pickle
import re

# Our numerical workhorses
import numpy as np
import pandas as pd

# Import the project utils
import sys
sys.path.insert(0, '../analysis/')
import mwc_induction_utils as mwc

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

# favorite Seaborn settings for notebooks
mwc.set_plotting_style()

# Get all of the microscopy Oid data files.
files = glob.glob('../../data/*Oid*microscopy*foldchange.csv')
flow_files = glob.glob('../../data/*Oid*MACSQuant*.csv')
# Merge them into one file.
df = []
flow_df = []
for i, f in enumerate(files):
    df.append(pd.read_csv(f, comment='#'))
for i, f in enumerate(flow_files):
    flow_df.append(pd.read_csv(f, comment='#'))

data = pd.concat(df, axis=0)
flow_data = pd.concat(flow_df, axis=0)
flow_data = flow_data[(flow_data.rbs=='RBS446') | (flow_data.rbs=='RBS1147')]

# Group  the data frame
grouped = pd.groupby(data, 'rbs')

flow_grouped = pd.groupby(flow_data, 'rbs')

# Plot all of the data to see if it makes sense.
colors = ['b', 'g']
plt.figure()
plt.plot([], [], 'd', markeredgecolor='b', markerfacecolor='w', label='RBS1147', markeredgewidth=1)
plt.plot([], [], 'd', markeredgecolor='g', markerfacecolor='w', label='RBS446',
markeredgewidth=1)

# Plot the theory curves.
epa = -np.log(139E-6)
epi = -np.log(0.53E-6)
epr = -17.3  # In units of kBT
iptg = np.logspace(-9, -2, 1000)
R = np.array([30, 62])  # Number of lac tetramers per cell.
for i in range(len(R)):
    fc = mwc.fold_change_log(iptg, epa, epi, 4.5, R[i], epr)
    plt.plot(iptg, fc, '-', color=colors[i], label='R = %s prediction' %(2 * R[i]))

plt.legend()
for g, d in grouped:
    if g == 'RBS1147':
        c = 'b'
    else:
        c = 'g'
    plt.plot(d.IPTG_uM/1E6, d.fold_change, 'd', color=c, alpha=0.75)

for g, d in flow_grouped:
    if g == 'RBS1147':
        c = 'b'
    else:
         c = 'g'
    plt.plot(d.IPTG_uM/1E6, d.fold_change_A, 'o',  color=c, alpha=0.75)


plt.xscale('log')
plt.xlabel('IPTG (M)')
plt.ylabel('fold-change')
plt.show()

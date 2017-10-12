import os
import glob
import tqdm
import numpy as np
import pandas as pd
import scipy

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

# Import the project utils
import sys

import mwc_induction_utils as mwc
mwc.set_plotting_style()

# define variables to use over the script
DATE = 20171011
USERNAME = 'gchure'
IPTG_uM = 50.0

# ---------------------------------------------------------------------------
# Shouldn't need to touch anything below here.
# ---------------------------------------------------------------------------
# %%
# list the directory with the data
datadir = '../../../data/flow/csv/{0}'.format(DATE)

# define the parameter alpha for the automatic gating
alpha = 0.40

# Grab all of the files.
files = glob.glob('{0}*.csv'.format(datadir))

# Make a new DataFrame
cols = ['date', 'username', 'operator',
        'strain', 'repressors', 'IPTG_uM', 'replicate_no',
        'clock_time', 'delta_t', 'FITC-A']
repressors = {'auto': 0, 'delta': 0, 'RBS1027': 130}
df = pd.DataFrame([], columns=cols)

for i, f in enumerate(files):
    # Split the file name.
    date, operator, strain, replicate, _, clock_time, delta_t, _ = f.split(
        '/')[-1].split('_')

    # Load the csv file and gate.
    data = pd.read_csv(f)
    gated = mwc.auto_gauss_gate(data, alpha)
    mean_FITC = gated['FITC-A'].mean()

    # Set the dictionary for the DataFrame.
    df_vars = [date, USERNAME, operator,  strain, repressors[strain], IPTG_uM,
               int(replicate[1:]), clock_time, delta_t, mean_FITC]

    data_dict = {v: df_vars[j] for j, v in enumerate(cols)}
    _df = pd.Series(data_dict)
    df = df.append(_df, ignore_index=True)

# Compute the fold-change for each replicate at each time point.
grouped = df.groupby(['delta_t', 'replicate_no'])
cols.append('fold_change_A')
fc_df = pd.DataFrame([], columns=cols)

for g, d in grouped:
    auto = d[d['strain'] == 'auto']['FITC-A'].values
    delta = d[d['strain'] == 'delta']['FITC-A'].values
    denom = delta - auto
    d['fold_change_A'] = (d['FITC-A'] - auto) / denom
    fc_df = fc_df.append(d, ignore_index=True)

# Save the csv file.
target = './output/{0}_{1}_timeseries_MACSQuant.csv'.format(
    DATE, operator)
if os.path.exists(target) is True:
    os.remove(target)
with open('comments.txt', 'r') as f:
    comments = f.readlines()
with open(target, 'a') as f:
    for line in comments:
        f.write(line)
    fc_df.to_csv(f, index=False)

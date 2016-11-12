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
rc={'lines.linewidth': 3, 
    'axes.labelsize' : 18, 
    'axes.titlesize' : 18,
    'axes.facecolor' : 'F4F3F6',
    'axes.edgecolor' : '000000',
    'axes.linewidth' : 1.2,
    'xtick.labelsize' : 13,
    'ytick.labelsize' : 13,
    'grid.linestyle' : ':',
    'grid.color' : 'a6a6a6'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)
sns.set_palette("deep", color_codes=True)

#=============================================================================== 
# Set output directory based on the graphicspath.tex file to print in dropbox
#=============================================================================== 
dropbox = open('../../doc/induction_paper/graphicspath.tex')
output = dropbox.read()
output = re.sub('\\graphicspath{{', '', output)
output = output[1::]
output = re.sub('}}\n', '', output + '/extra_figures')

#=============================================================================== 
# O2 RBS1027
#=============================================================================== 
# Load the flat-chain
with open('../../data/mcmc/' + '20160815' + \
                  '_gauss_homoscedastic_RBS1027.pkl', 'rb') as file:
    unpickler = pickle.Unpickler(file)
    gauss_flatchain = unpickler.load()
    
# map value of the parameters
ea, ei = np.mean(gauss_flatchain[:, [0, 1]], axis=0)

#=============================================================================== 
# Plot p_act as a function of IPTG with xkcd format
#=============================================================================== 
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-8, -2, 100)

# Set the colors for the strains
colors = sns.color_palette(n_colors=7)

# Initialize plot
plt.xkcd()
fig = plt.figure(figsize=(5,4))
ax = plt.subplot(111)
ax.plot(IPTG, mwc.pact_log(IPTG * 1E6, ea, ei, 4.5))

ax.set_xscale('log')
ax.set_xlabel('IPTG')
ax.set_ylabel(r'p_{act}')
ax.set_ylim([-0.1, 1.1])
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.savefig(output + '/pact_example.pdf')

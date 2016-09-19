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
rc={'lines.linewidth': 2, 
    'axes.labelsize' : 16, 
    'axes.titlesize' : 18,
    'axes.facecolor' : 'F4F3F6',
    'axes.edgecolor' : '000000',
    'axes.linewidth' : 1.2,
    'xtick.labelsize' : 13,
    'ytick.labelsize' : 13,
    'grid.linestyle' : ':',
    'grid.color' : 'a6a6a6'}

pboc_rc = { ##########SIZES################## 
                'lines.linewidth'       : 2, 
                'axes.titlesize'        : 18,
                'axes.labelsize'        : 16,
                'font.family'           : 'Lucida Sans Unicode',
                
                ##########COLORS#################
                'axes.facecolor'        :'#E3DCD0',
                
                #########GRIDS/TICKS############
                'xtick.labelsize'       : 12,
                'ytick.labelsize'       : 12,

                #########LEGEND#################
                'legend.numpoints'      : 1,
                'legend.fontsize'       : 13,
                'legend.loc'            : 'best',
                }
 
#Define the colorscheme. 
# r, b, m, g, orange
pboc  = sns.color_palette(['#d46c55', '#7aa874','#728ec1',
                           '#aa85ab','#e08d14']) 


sns.set_context('notebook', rc=pboc_rc)
sns.set_style('dark', rc=pboc_rc)
sns.set_palette("deep", color_codes=True)

#=============================================================================== 
# Set output directory based on the graphicspath.tex file to print in dropbox
#=============================================================================== 
dropbox = open('../../doc/induction_paper/graphicspath.tex')
output = dropbox.read()
output = re.sub('\\graphicspath{{', '', output)
output = output[1::]
output = re.sub('}}\n', '', output)

#=============================================================================== 
# Read the data
#=============================================================================== 

datadir = '../../data/'
# read the list of data-sets to ignore
data_ignore = pd.read_csv(datadir + 'datasets_ignore.csv', header=None).values
# read the all data sets except for the ones in the ignore list
all_files = glob.glob(datadir + '*' + '_IPTG_titration' + '*csv')
ignore_files = [f for f in all_files for i in data_ignore if i[0] in f]
read_files = [f for f in all_files if f not in ignore_files]
print('Number of unique data-sets: {:d}'.format(len(read_files)))
df = pd.concat(pd.read_csv(f, comment='#') for f in read_files)

# Now we remove the autofluorescence and delta values
df = df[(df.rbs != 'auto') & (df.rbs != 'delta')]

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
# Plot the theory vs data for all 4 operators with the credible region
#=============================================================================== 
# Define the IPTG concentrations to evaluate
IPTG = np.logspace(-8, -2, 100)

# Set the colors for the strains
colors = sns.color_palette(n_colors=7)

# Define the operators and their respective energies
operators = ['O2', 'O1', 'O3'] #, 'Oid']
energies = {'O1': -15.3, 'O2': -13.9, 'O3': -9.7, 'Oid': -17}

# Initialize subplots
fig, ax = plt.subplots(2, 2, figsize=(11, 8))

#Get a list of the axes. 
axes = fig.get_axes()
for axis in axes:
    #Set the ticks to inward facing and white.  
    axis.tick_params(reset=True, axis='both', direction='in', color='white',
    width=1, length=5, top='off', right='off', labelsize=13)

ax = ax.ravel()

#For some horrible and annoying reason, not all parameters can be
#changed through the rc, but have to be changed piecewise. Those are
#here.
plt.rc('legend', fontsize=14) 
#The below is needed for sans serif math display.
plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
plt.rc('mathtext', fontset='stixsans', sf='sans')

# Loop through operators
for i, op in enumerate(operators):
    data = df[df.operator==op]
    # loop through RBS mutants
    for j, rbs in enumerate(df.rbs.unique()):
        # plot the theory using the parameters from the fit.
        ax[i].plot(IPTG, mwc.fold_change_log(IPTG * 1E6, 
            ea=ea, ei=ei, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op]),
            color=colors[j])
        # plot 95% HPD region using the variability in the MWC parameters
        cred_region = mwc.mcmc_cred_region(IPTG * 1E6, 
            gauss_flatchain, epsilon=4.5,
            R=df[(df.rbs == rbs)].repressors.unique(),
            epsilon_r=energies[op])
        ax[i].fill_between(IPTG, cred_region[0,:], cred_region[1,:],
                        alpha=0.3, color=colors[j])
    ax[i].set_xscale('log')
    ax[i].set_xlabel('IPTG (M)')
    ax[i].set_ylabel('fold-change')
    ax[i].set_ylim([-0.01, 1.2])
    ax[i].set_title(op)
ax[0].legend(loc='upper left', title='repressors / cell')

# Remove plot from the Oid placeholder
ax[3].set_axis_off()

plt.tight_layout()
plt.savefig(output + '/fig_theory_predictions_O2_RBS1027_fit.pdf')

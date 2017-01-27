import pandas as pd
import glob
#===============================================================================
# Export flow cytometry data
#===============================================================================

datadir = '../../data/'
# read the list of data-sets to ignore
data_ignore = pd.read_csv(datadir + 'datasets_ignore.csv', header=None).values
# read the all data sets except for the ones in the ignore list
all_files = glob.glob(datadir + '*' + '_IPTG_titration_MACSQuant' + '*csv')
ignore_files = [f for f in all_files for i in data_ignore if i[0] in f]
# Read files that have no dimer data
read_files = [f for f in all_files if f not in ignore_files \
              and 'dimer' not in f and 'Oid' not in f]
# Print number of data sets
print('Number of unique flow-cytometry data sets: {:d}'.format(len(read_files)))
df = pd.concat(pd.read_csv(f, comment='#') for f in read_files)
# Save compiled data frame
df.to_csv(datadir + 'flow_master.csv', index=False)

#===============================================================================
# Export microscopy data
#===============================================================================

datadir = '../../data/'
# read the all data sets except for the ones in the ignore list
all_files = glob.glob(datadir + '*' + '_IPTG_titration_microscopy_foldchange' +\
                      '*csv')
# Read files that have no dimer data
read_files = [f for f in all_files if 'Oid' not in f]
# Print number of data sets
print('Number of unique microscopy data sets: {:d}'.format(len(read_files)))
df = pd.concat(pd.read_csv(f, comment='#') for f in read_files)
# Save compiled data frame
df.to_csv(datadir + 'microscopy_master.csv', index=False)




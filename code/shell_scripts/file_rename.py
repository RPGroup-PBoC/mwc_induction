import os
# Our numerical workhorses
import numpy as np
import pandas as pd

#=============================================================================== 
# Define function to read a csv file containing the standardized file names and
# rename the files accordingly

def file_rename(dirname, csv_filename, file_order='well', file_extension='fcs'):
    '''
    Reads the csv file csv_filename containing all the names of the files to be
    renamed and renames the files according to how the original files were
    named.
    Parameters
    ----------
    dirname : str.
        directory containing the files to be renamed
    csv_filename : str.
        path to csv file containing the standardize filenames to use
    file_order : str.
        Indicator of how the original file names were named. It can take two
        values. 'well' for filenames having row and column in the alphanumeric
        standard format, e.g. A1, B12. 'number' for files that only have the
        number of the well, e.g. 00001, 00012
    '''
    filename = pd.read_csv(csv_filename, header=None)

    # Read the old files that contain the file_extension pattern
    old_files = np.array(os.listdir(dirname))
    csv_bool = np.array([file_extension in f for f in old_files])
    old_files = old_files[np.array(csv_bool)]
    
    if file_order=='number':
        filename = filename.values.flatten(order='F')
        for i, f in enumerate(old_files):
            print(i, f, filename[i] + '.' + file_extension)
            os.rename(dirname + '/' + f, filename[i] + '.' + file_extension)
    

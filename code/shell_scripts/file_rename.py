import os
# Our numerical workhorses
import numpy as np
import pandas as pd

#=============================================================================== 
# Define function to read a csv file containing the standardized file names and
# rename the files accordingly

def file_rename(dirname, pattern, csv_filename, file_order='number', file_extension='fcs'):
    '''
    Reads the csv file csv_filename containing all the names of the files to be
    renamed and renames the files according to how the original files were
    named.
    Parameters
    ----------
    dirname : str.
        directory containing the files to be renamed.
    pattern : str.
        pattern to look for in the old files.
    csv_filename : str.
        path to csv file containing the standardize filenames to use.
        The file should be in the 8x12. 96-well plate format.
    file_order : str.
        Indicator of how the original file names were named. It can take two
        values. 'well' for filenames having row and column in the alphanumeric
        standard format, e.g. A1, B12. 'number' for files that only have the
        number of the well, e.g. 00001, 00012
    '''
    # read the CSV file that contains the new standardized names
    filename = pd.read_csv(csv_filename, header=None)

    # read the old files that contain the file_extension pattern
    old_files = np.array(os.listdir(dirname))
    # find the files that contain the pattern and the file extension
    csv_bool = np.array([pattern in f and file_extension in f for f in old_files])
    old_files = old_files[np.array(csv_bool)]
    
    if file_order=='number':
        # flatten the array into a single column
        filename = filename.values.flatten(order='F')
        # loop through the files and rename them
        for i, f in enumerate(old_files):
            print(i, f, filename[i] + '.' + file_extension)
            # replace on-site the file
            os.replace(dirname + f, filename[i] + '.' + file_extension)
    

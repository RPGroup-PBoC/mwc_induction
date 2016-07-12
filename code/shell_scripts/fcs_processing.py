#!/Users/razo/anaconda/envs/conda_python2.7/bin/python

# import libraries to manipulate the directories
import os
import re

# Our numerical workhorses
import numpy as np
import pandas as pd

# import the Flow Cytometry library
import FlowCytometryTools as facs

# import the library that allows to parse terminal inputs
import optparse
#=============================================================================== 

# Initialize the option parser
parser = optparse.OptionParser()

# add the file option to indicate which file to read
parser.add_option('-f', '--file', dest='filename',
        help='name of single file to be processed', metavar="FILE")

# add the option of receiving a directory such that all the files
# in the directory are exported
parser.add_option('-d', '--directory', dest='inputdir',
        help='input directory')

# add the option of providing a pattern such that only the files with the
# pattern will be processed by the script
parser.add_option('-p', '--pattern', dest='pattern',
        help='pattern to parse files')

# add the output directory
parser.add_option('-o', '--output', dest='outputdir',
        help='output directory')

# add the channels that the user wants to export
parser.add_option('-c', '--channel', action='append', dest='channels',
        help='channels to extract.') 

# get the options and args
options, args = parser.parse_args()
# print options

#=============================================================================== 
# list files
if options.inputdir == None:
    raise ValueError('Please indicate the input directory that contains the \
            fcs files')

# get all the files in the directory
files = np.array(os.listdir(options.inputdir))
# if a pattern is provided select only the files with such a pattern
if options.pattern != None:
    files_idx = np.array([options.pattern in f for f in files])
    files = files[files_idx]

# loop through the files
for f in files: 
    # consider only the fcs files
    if f.endswith('.fcs'):
    # read the file
        fcs_file = facs.FCMeasurement(ID=f, datafile=options.inputdir + '/' + f)
        # if there are not set channels, get all the channels
        if options.channels == None:
            fcs_data = fcs_file.data
        # if channels are indicated, collect only those
        else:
            fcs_data = fcs_file.data[options.channels]

        #parse the file name to change the extension
        filename  = re.sub('.fcs', '.csv', f)
        if options.outputdir == None:
            fcs_data.to_csv(options.inputdir + '/' + filename)
        else:
            fcs_data.to_csv(options.outputdir + '/' + filename)

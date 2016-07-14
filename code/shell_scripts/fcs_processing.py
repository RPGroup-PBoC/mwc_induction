#! /usr/bin/env python2.7
"""
This script parses and cleans up a provided Flow Cytometry Standard (fcs) file
and saves it as a Comma Separated Value (csv).
"""
import os
import re
import numpy as np
import pandas as pd
import optparse
import FlowCytometryTools as flow

# #########################################################################
def main():
    # Initialize the option parser
    parser = optparse.OptionParser()
    
    # add the file option to indicate which file to read
    parser.add_option('-f', '--file', dest='filename', help='name of single\
            file to be processed.', metavar="filename")
    
    # add the option of receiving a directory such that all the files
    # in the directory are exported
    parser.add_option('-d', '--directory', dest='inputdir', help='name of\
            input directory to be processed')
    
    # add the option of providing a pattern such that only the files with the
    # pattern will be processed by the script
    parser.add_option('-p', '--pattern', dest='pattern', help='filename\
            pattern to parse files.')
    
    # add the output directory
    parser.add_option('-o', '--output', dest='outputdir',
            help='name of output directory')
    
    # add the channels that the user wants to export
    parser.add_option('-c', '--channel', action='append', dest='channels',
            help=' individual channels to extract. Each channel must have its\
            own -c flag.') 
    
    # get the options and args
    options, args = parser.parse_args()
    
    # list files
    if (options.inputdir == None) & (options.filename == None):
        raise ValueError('no input directory/file provided! Please indicate\
                the input directory that contains the fcs files')
    
    # get all the files in the directory
    files = []
    if options.inputdir != None:
        usr_files = np.array(os.listdir(options.inputdir))
        # Use the pattern to identify all of the files.
        files_idx = np.array([options.pattern in f for f in usr_files])
        files_names = usr_files[files_idx]
        
        #Add the input directory ahead of each file.
        for f in files_names:
            files.append('%s/%s' %(options.inputdir, f)) 
    else:
        files.append(options.filename)

    # loop through the files
    for f in files: 
        # consider only the fcs files
        if f.endswith('.fcs'):
            # read the file
            fcs_file = flow.FCMeasurement(ID=f, datafile=f)
            # if there are not set channels, get all the channels
            if options.channels == None:
                fcs_data = fcs_file.data
            # if channels are indicated, collect only those
            else:
                fcs_data = fcs_file.data[options.channels]
   
            #parse the file name to change the extension
            filename  = re.sub('.fcs', '.csv', f)
            if options.outputdir == None:
                fcs_data.to_csv(filename)
            else:
                filename = filename.rsplit('/', 1)[1]
                fcs_data.to_csv(options.outputdir + '/' + filename)
                print(filename)
    

if __name__ == '__main__':
    main()
    print 'thank you -- come again'

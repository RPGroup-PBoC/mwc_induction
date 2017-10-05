"""
filename: file_rename.py
date: 2017-10-01
author: Griffin Chure
purpose: Rename FCS files to contain strain and ATC concentration
information
"""
import numpy as np
import pandas as pd
import glob
import os
import shutil

#%% Set the constants.
DATE = '20171003'
INOC_HR = 8
INOC_M = 0.5
OPERATOR = 'O2'
DATADIR = 'data/flow/fcs/'

strain_list = ['auto', 'delta', 'RBS1027']
strain_list = np.array([[s] * 3 for s in strain_list]).flatten()
nums = ['{0:04d}'.format(i) for i in range(1, 10)]
replicates = [j + 1 for i in range(3) for j in range(3)]
strain_dict = {i: j for i, j in zip(nums, strain_list)}
rep_dict = {i: j for i, j in zip(nums, replicates)}
nums
# replicate_no = [(j + 1)  for i in range(3) for j in range(3)]
repressors = [0, 0, 260]


files = np.sort(glob.glob(DATADIR + 'RP2017*.fcs'))
for i, f in enumerate(files):
    # Split to get the time and sample number.
    dir_split = f.split('/')
    time_split = dir_split[-1].split('_')[-1].split('.')
    clock_time = time_split[0]
    samp = time_split[1]
    time_min = int(clock_time[3:-1]) / 60
    time_hr = int(clock_time[:2])
    delta_t = (time_hr - INOC_HR) + (time_min - INOC_M)

    time = time_split[0]
    # Define the new name.
    new_name = '{0}_{1}_{2}_r{3:02d}_clock_{4}_{5:0.2f}_hr.fcs'.format(DATE,
                                                                       OPERATOR,
                                                                       strain_dict[samp],
                                                                       rep_dict[samp],
                                                                       clock_time, delta_t)

    shutil.copy(f, DATADIR + new_name)

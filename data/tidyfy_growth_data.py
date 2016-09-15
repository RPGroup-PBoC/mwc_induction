# -*-coding: utf-8 -*-
"""
Author: Griffin Chure
Creation Date: 2016-08-17

"""
#------------------------------------------------------------------------------- 
# DEPENDENCIES
#------------------------------------------------------------------------------- 
#Allow for python 2.7 compatibility
from __future__ import print_function, unicode_literals
from __future__ import absolute_import, division 

#Packages for numerics
import numpy as np

#File management modules. 
import os
import shutil 
import fnmatch
import glob 

#For managing dataframes.
import pandas as pd 

#Image processing utilities. 
import skimage.io
import skimage.morphology
import skimage.segmentation
import skimage.filters
import skimage.exposure

#Custom-written modules. 
import pboc.filters
import pboc.io
import pboc.segmentation
import pboc.process
import pboc.plot
import pboc.props

#Packages for plotting.
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm #Color map.
import seaborn as sns

rc = {'lines.linewidth': 1.5,
      'axes.labelsize' : 14,
      'axes.titlesize' : 18,
      'axes.facecolor' : 'EBEBEB',
      'axes.edgecolor' : '000000',
      'axes.linewidth' : 0.75,
      'axes.frameon' : True,
      'xtick.labelsize' : 11,
      'ytick.labelsize' : 11,
      'font.family' : 'Lucida Sans Unicode',
      'grid.linestyle' : ':',
      'grid.color' : 'a6a6a6',
      'mathtext.fontset' : 'stixsans',
      'mathtext.sf' : 'sans'}
plt.rc('text.latex', preamble=r'\usepackage{sfmath}')
plt.rc('mathtext', fontset='stixsans', sf='sans') 
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)
sns.set_palette("deep", color_codes=True)
#------------------------------------------------------------------------------- 
data = pd.read_csv('20160816_growth_curve_measurements.csv')


#Separate tube vs plate data.
tube = [data.tube1_od600, data.tube2_od600, data.tube3_od600]
plate = [data.plate1_od600, data.plate2_od600, data.plate3_od600]

tube_df = pd.DataFrame(pd.concat(tube, axis=0))
tube_df.insert(0, 'user', data.username)
tube_df.insert(0, 'sample', 'tube')
plate_df = pd.DataFrame(pd.concat(plate, axis=0))
plate_df.insert(0, 'user', data.username)
plate_df.insert(0, 'sample', 'plate')
df = pd.concat([tube_df, plate_df], axis=0)
df.insert(0, 'time', data.time_hrs)
df.insert(0, 'clock', data.time_clock)
df.columns = 'clock', 'time', 'sample', 'user', 'od_600nm'
df.to_csv('20160816_growth_measurements.csv', index=None)

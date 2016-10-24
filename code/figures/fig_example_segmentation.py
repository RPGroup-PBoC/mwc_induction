"""
Title:
  fig_example_segmentation.py
Creation Date:
  20161024
Last Modified:
  20161024
Author(s):
  Griffin Chure
Purpose:
  Generate a figure of an example segmentation containing an external scale
  bar.
"""

# Import the standard dependencies.
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Image processing utilities.
import skimage.io
import skimage.morphology
import skimage.segmentation
import skimage.filters
import scipy.ndimage

# Set the plotting environment.
rc = {'lines.linewidth': 2,
     'font.family': 'Lucida Sans Unicode',
     'mathtext.fontset': 'stixsans',
     'mathtext.sf': 'sans',
     'legend.frameon': True,
     'legend.fontsize': 13}
sns.set_style('darkgrid', rc=rc)

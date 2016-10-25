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
import glob
import re

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
sns.set_style('dark', rc=rc)
%matplotlib inline

# Set the output folder.
dropbox = open('../../doc/induction_paper/graphicspath.tex')
output = dropbox.read()
output = re.sub('\\graphicspath{{', '', output)
output = output[1::]
output = re.sub('}}\n', '', output + '/extra_figures')

# Load an example image.
data_dir = '../../data/microscopy/20161019/'
im_glob = glob.glob(data_dir + '*auto_0uMIPTG*/*.tif')
ex_im = skimage.io.imread(im_glob[6])
ip_dist = 0.160  # in units of µm/pix.
p = ex_im[:, :, 0]
m = ex_im[:, :, 1]
y = ex_im[:, :, 2]

# Define the segmentation functions.


# #################
def find_zero_crossings(im, selem, thresh):
    """
    This  function computes the gradients in pixel values of an image after
    applying a sobel filter to a given image. This  function is later used in
    the Laplacian of Gaussian cell segmenter (log_segmentation) function. The
    arguments are as follows.

    Parameters
    ----------
    im : 2d-array
        Image to be filtered.
    selem : 2d-array, bool
        Structural element used to compute gradients.
    thresh :  float
        Threshold to define gradients.

    Returns
    -------
    zero_cross : 2d-array
        Image with identified zero-crossings.

    Notes
    -----
    This function as well as `log_segmentation` were written by Justin Bois.
    http://bebi103.caltech.edu/
    """

    # apply a maximum and minimum filter to the image.
    im_max = scipy.ndimage.filters.maximum_filter(im, footprint=selem)
    im_min = scipy.ndimage.filters.minimum_filter(im, footprint=selem)

    # Compute the gradients using a sobel filter.
    im_filt = skimage.filters.sobel(im)

    # Find the zero crossings.
    zero_cross = (((im >= 0) & (im_min < 0)) | ((im <= 0) & (im_max > 0)))\
        & (im_filt >= thresh)

    return zero_cross


# #################
def log_segmentation(im, selem='default', thresh=0.0001, radius=2.0,
                     median_filt=True, clear_border=True, label=False):
    """
    This function computes the Laplacian of a gaussian filtered image and
    detects object edges as regions which cross zero in the derivative.

    Parameters
    ----------
    im :  2d-array
        Image to be processed. Must be a single channel image.
    selem : 2d-array, bool
        Structural element for identifying zero crossings. Default value is
        a 2x2 pixel square.
    radius : float
        Radius for gaussian filter prior to computation of derivatives.
    median_filt : bool
        If True, the input image will be median filtered with a 3x3 structural
        element prior to segmentation.
    selem : 2d-array, bool
        Structural element to be applied for laplacian calculation.
    thresh : float
        Threshold past which
    clear_border : bool
        If True, segmented objects touching the border will be removed.
        Default is True.
    label : bool
        If True, segmented objecs will be labeled. Default is False.

    Returns
    -------
    im_final : 2d-array
        Final segmentation mask. If label==True, the output will be a integer
        labeled image. If label==False, the output will be a bool.

    Notes
    -----
    We thank Justin Bois in his help writing this function.
    https://bebi103.caltech.edu
    """

    # Test that the provided image is only 2-d.
    if len(np.shape(im)) > 2:
        raise ValueError('image must be a single channel!')

    # Determine if the image should be median filtered.
    if median_filt == True:
        selem = skimage.morphology.square(3)
        im_filt = scipy.ndimage.median_filter(im, footprint=selem)
    else:
        im_filt = im
    # Ensure that the provided image is a float.
    if np.max(im) > 1.0:
        im_float = skimage.img_as_float(im_filt)
    else:
        im_float = im_filt

    # Compute the LoG filter of the image.
    im_LoG = scipy.ndimage.filters.gaussian_laplace(im_float, radius)

    # Define the structural element.
    if selem == 'default':
        selem = skimage.morphology.square(3)

    # Using find_zero_crossings, identify the edges of objects.
    edges = find_zero_crossings(im_LoG, selem, thresh)

    # Skeletonize the edges to a line with a single pixel width.
    skel_im = skimage.morphology.skeletonize(edges)

    # Fill the holes to generate binary image.
    im_fill = scipy.ndimage.morphology.binary_fill_holes(skel_im)

    # Remove small objects and objects touching border.
    im_final = skimage.morphology.remove_small_objects(im_fill)
    if clear_border is True:
        im_final = skimage.segmentation.clear_border(im_final, buffer_size=5)

    # Determine if the objects should be labeled.
    if label is True:
        im_final = skimage.measure.label(im_final)

    # Return the labeled image.
    return im_final

# Segment the image and take a look.
im_seg = log_segmentation(m, label=True)

# Apply the area and eccentricity bounds.
props = skimage.measure.regionprops(im_seg)
final_seg = np.zeros_like(im_seg)
for prop in props:
    cell_area = prop.area * ip_dist**2
    if (cell_area > 0.5) & (cell_area < 6) & (prop.eccentricity > 0.8):
        final_seg += im_seg == prop.label

# Highlight the edges of the segmented cells
bounds = skimage.segmentation.find_boundaries(final_seg)
p_float = (p - p.min()) / (p.max() - p.min())
p_copy = np.copy(p_float)
p_copy[bounds] = 1.0
merge = np.dstack((p_copy, p_float, p_float))

# Make the figure with a scale bar.
fig = plt.figure()
ax = plt.gca()
plt.imshow(merge)
ax.set_frame_on(False)
bar_length = 20 / ip_dist
ax.hlines(-20, 0, bar_length, linewidth=2, color='k')
ax.vlines(0, -20, -5, linewidth=1, color='k')
ax.vlines(bar_length, -20, -5, linewidth=1, color='k')
ax.text(0, -22, '20 µm', fontsize=8)


# Turn off the tick marks.
ax.xaxis.set_ticks([])
ax.yaxis.set_ticks([])
plt.tight_layout()
plt.savefig('/Users/gchure/Dropbox/mwc_induction/figures/example_segmentation.pdf', bbox_inches='tight',
            transparent=True)

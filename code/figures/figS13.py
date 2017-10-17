import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.io
import skimage.color
import scipy.ndimage
import pandas as pd
import mwc_induction_utils as mwc
import glob
mwc.set_plotting_style()


def ecdf(data):
    return np.sort(data), np.arange(0, len(data)) / len(data)


# Load an example image.
ex_im = skimage.io.imread('../../data/example_image.ome.tif')
ph = ex_im[:, :, 0]
im_float = ph / ph.max()
mch = ex_im[:, :, 1]

# Segment the image.
seg = mwc.log_segmentation(mch, label=True)
props = skimage.measure.regionprops(seg)
cells = np.zeros_like(seg)
for prop in props:
    area = prop.area * 0.160**2
    ecc = prop.eccentricity
    if (area > 0.5) & (area < 6) & (ecc > 0.8):
        cells += seg == prop.label

# Label the image
cells_lab = skimage.measure.label(cells > 0)

# Merge the segmented regions onto the original image.
ph_float = (ph - ph.min()) / (ph.max() - ph.min())
ph_copy = np.copy(ph_float)
ph_copy[cells_lab > 0] = 0.7
merge = np.dstack((ph_float, ph_copy, ph_copy))

# Load the corresponding csv file with distributions.
mic_data = pd.read_csv('../../data/RBS1027_O2_microscopy_cell_intensities.csv',
                       comment='#')

area_x, area_y = ecdf(mic_data['area'])
ecc_x, ecc_y = ecdf(mic_data['eccentricity'])

# %% Set up the figure canvas. This one is a little complicated.
fig = plt.figure(figsize=(8, 5))
ax1 = plt.subplot2grid((4, 5), (2, 0), colspan=2, rowspan=2)  # Area
ax2 = plt.subplot2grid((4, 5), (0, 0), colspan=2, rowspan=2)  # Eccentricity
ax3 = plt.subplot2grid((4, 5), (0, 2), colspan=3, rowspan=4)  # Segmentation.

# Set the various labels.
ax1.set_xlabel('area (µm$^2$)')
ax1.set_ylabel('ECDF')
ax2.set_xlabel('eccentricity')
ax2.set_ylabel('ECDF')

# Plot the data.
_ = ax1.plot(area_x, area_y, 'k-')
_ = ax2.plot(ecc_x, ecc_y, 'k-')
_ = ax3.imshow(merge)

# Shade the regions of eccentricy and the area that we keep.
ecdf_vec = np.linspace(0, 1.0, 5000)
_ = ax1.fill_betweenx(ecdf_vec, 0.5, 6, color='b', alpha=0.4)
_ = ax2.fill_betweenx(ecdf_vec, 0.8, ax2.get_xlim()[1], color='b', alpha=0.4)

# Add the scale bar for the image.
im_xlim = ax3.get_xlim()
im_ylim = ax3.get_ylim()
bar_length = 10 / 0.160
ax3.set_frame_on(False)
_ = ax3.vlines(im_xlim[1] + 20, 0, bar_length, color='k', linewidth=2)
_ = ax3.hlines(0, im_xlim[1] + 5, im_xlim[1] + 20, linewidth=2, color='k')
_ = ax3.hlines(bar_length, im_xlim[1] + 5, im_xlim[1] + 20, linewidth=2,
               color='k')
_ = ax3.text(im_xlim[1] + 22, 10, '10 µm', rotation=270, fontsize=8)

# Change the axes limits.
ax3.xaxis.set_ticks([])
ax3.yaxis.set_ticks([])
ax1.margins(0.05)
ax2.margins(0.05)


# Add the panel text.
fig.text(0, 0.95, '(A)', fontsize=12)
fig.text(0.438, 0.95, '(B)', fontsize=12)

plt.tight_layout()
mwc.scale_plot(fig, 'two_row')
plt.savefig('../../figures/SI_figs/figS13.pdf', bbox_inches='tight')

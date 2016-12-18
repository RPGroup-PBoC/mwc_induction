import os
import glob

# Our numerical workhorses
import numpy as np
import pandas as pd
import scipy

# Import matplotlib stuff for plotting
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Seaborn, useful for graphics
import seaborn as sns

# For image processing.
import skimage.io

# Load custom written modules.
import mwc_induction_utils as mwc
mwc.set_plotting_style()

# Define the data directory.
data_dir = '../../../data/microscopy/20161214/'

# Set details of the experiment.
DATE = 20161214
USERNAME = 'gchure'
OPERATOR = 'Oid'
BINDING_ENERGY = -17.3
REPRESSORS = (0, 0, 62)
IPDIST = 0.160  # in units of µm per pixel
STRAINS = ['auto', 'delta', 'RBS446']
IPTG_RANGE = (0, 0.1, 5, 10, 25, 50, 75,
              100, 250, 500, 1000, 5000)

# Generate the flatfield images.
dark_glob = glob.glob(data_dir + '*camera_noise*/*tif')
field_glob = glob.glob(data_dir + '*YFP_profile*/*tif')
dark_ims = skimage.io.ImageCollection(dark_glob, conserve_memory=False)
field_ims = skimage.io.ImageCollection(field_glob, conserve_memory=False)
average_field = mwc.average_stack(field_ims)
average_dark = mwc.average_stack(dark_ims)

# Randomly generate an example segmentation.
ex_st = np.random.choice(STRAINS)
ex_iptg = np.random.choice(IPTG_RANGE)

# Iterate through each strain and concentration to make teh dataframes.
dfs = []
for i, st in enumerate(STRAINS):
    for j, iptg in enumerate(IPTG_RANGE):

        # Load the imagesimages = glob.glob
        images = glob.glob(data_dir + '*' + st + '*_' + str(iptg) +
                           'uMIPTG*/*.ome.tif')
        ims = skimage.io.ImageCollection(images)
        for _, x in enumerate(ims):

            print('Processing strain {0} iptg {1} image {2}'.format(st, iptg, _))
            print(images[_])
            _, m, y = mwc.ome_split(x)
            y_flat = mwc.generate_flatfield(y, average_dark, average_field)

            # Segment the mCherry channel.
            m_seg = mwc.log_segmentation(m, label=True)
            if (st == ex_st) & (iptg == ex_iptg):
                ex_seg = m_seg
                ex_phase = _

            # Extract the measurements.
            if np.max(m_seg) > 0:
                im_df = mwc.props_to_df(m_seg, physical_distance=IPDIST,
                                        intensity_image=y_flat)
                # Add strain and  IPTG concentration information.
                im_df.insert(0, 'IPTG_uM', iptg)
                im_df.insert(0, 'repressors', REPRESSORS[i])
                im_df.insert(0, 'rbs', st)
                im_df.insert(0, 'binding_energy', BINDING_ENERGY)
                im_df.insert(0, 'operator', OPERATOR)
                im_df.insert(0, 'username', USERNAME)
                im_df.insert(0, 'date', DATE)

                # Append the dataframe to the global list.
                dfs.append(im_df)

# Concatenate the dataframe
final_df = pd.concat(dfs, axis=0)

# Apply the area and eccentricity bounds.
final_df = final_df[(final_df.area > 0.5) & (final_df.area < 6.0) &
                    final_df.eccentricity > 0.8]
# Add the comments to the header of the data file
final_df.to_csv('output/' + str(DATE) + '_' + OPERATOR +
                '_IPTG_titration_microscopy.csv', index=False)
filenames = ['comments.txt', 'output/' + str(DATE) + '_' + OPERATOR +
             '_IPTG_titration_microscopy.csv']
with open('../../../data/' + str(DATE) + '_' + OPERATOR +
          '_IPTG_titration_microscopy.csv', 'w') as output:
    for fname in filenames:
        with open(fname) as infile:
            output.write(infile.read())

# Save the example segmentation.
lab = skimage.measure.label(ex_seg > 0)
approved_im = np.zeros_like(ex_seg)
props = skimage.measure.regionprops(lab)
for prop in props:
    area = prop.area * IPDIST**2
    if (area > 0.5) & (area < 6) & (prop.eccentricity > 0.8):
        approved_im += lab == prop.label
mask = approved_im > 0
bar_length = 10 / IPDIST
merge = mwc.example_segmentation(mask, ex_phase, bar_length)
skimage.io.imsave('output/' + str(DATE) + '_' + OPERATOR +
                  '_IPTG_titration_microscopy_example_segmentation.png',
                  merge)

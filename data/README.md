# An Allosteric Theory of Transcription Factor Induction

This repository contains all experimental data and programs for the analysis
found in "An Allosteric Theory of Transcription Factor Induction" by Manuel
Razo-Mejia^*^, Stephanie Barnes^*^, Nathan Belliveau^*^, Griffin Chure^*^,
Tal Einav^*^, and Rob Phillips.

An index for the structure of this repository is given below.


## `code`
This folder contains the hand-written Python programs for processing,
analyzing, and displaying the data presented in this work.

* **`notebooks/` \|** Jupyter Notebooks containing Python code and explanation
    for the myriad analyses described in the main text and supplemental
    information.

* **`scripts/` \|** Example Python scripts used for processing and analysis of raw data
    as well as for generating the figures found in the main text and the
    supplemental information.

* **`mwc_induction_utils.py` \|** A Python file containing the homegrown
    functions used in all processing and analysis procedures.


## `data`
This folder contains all experimental data used in the publication. All `csv`
files obey the [tidy data](http://vita.had.co.nz/papers/tidy-data.pdf) guidelines.

* **`mwc_induction_flow_cytometry_fold_change.csv` \|** All experimentally
    measured fold-change values obtained through flow cytometry.

* **`mwc_induction_microscopy_fold_change.csv` \|** All experimentally
    measured fold-change values obtained through single-cell microscopy. Please
    see the supplemental information for a more detailed description.

* **`mwc_induction_mcmc_flatchains.pkl` \|** The result of the markov chain monte carlo sampling used in the Bayesian parameter estimation.

* **`example_fcs_file.fcs` \|** A representative flow cytometry standard (`fcs`)
    file used in the `example_processing.py` Python script. 

* **`example_flow_cytometry_intensity_set.zip \|** A representative flow
    cytometry data set stored as comma separated values (`csv`) used for the
    `example_analysis.py` Python script.

* **`example_microscopy_image_set.zip \|** A representative image set used in
    the `image_processing.ipynb` Jupyter notebook.

* **`example_titration_fold_change_set.csv \|** A representative complete IPTG
    titration data set used for the `bayesian_parameter_estimation.ipynb`
    Jupyter notebook.

* **`comments/` \|** A folder containing all comments for each experimental run
    including flow cytometry and microscopy.

## `other`
This folder contains other various files used in the publication. 

* **`plasmid_maps/` \|** A collection of the plasmid sequences used in this work
    as GenBank (`.gb`) files.

* **`materials_list.txt` \|** A list of the materials used in this work along
    with their catalog numbers.




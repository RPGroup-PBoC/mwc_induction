# An Allosteric Theory of Transcription Factor Induction

This repository contains all experimental data and programs for the analysis
found in "An Allosteric Theory of Transcription Factor Induction" by Manuel
Razo-Mejia^*^, Stephanie Barnes^*^, Nathan Belliveau^*^, Griffin Chure^*^,
Tal Einav^*^, and Rob Phillips.

An index for the structure of this repository is given below.


## `code`
This folder contains the hand-written Python programs for processing,
analyzing, and displaying the data presented in this work.

* **`analysis/` \|** Jupyter Notebooks containing Python code and explanation
    for the myriad analyses described in the main text and supplemental
    information.

* **`examples/` \|** Example Python scripts used for processing and analysis of raw data
    as well as for generating the figures found in the main text and the
    supplemental information.

* **`mwc_induction_utils.py` \|** A Python file containing the homegrown
    functions used in all processing and analysis procedures.


## `data`
This folder contains all experimental data used in the publication. All `csv`
files obey the [tidy data](http://vita.had.co.nz/papers/tidy-data.pdf) guidelines.

* **`flow_master.csv` \|** All experimentally
    measured fold-change values obtained through flow cytometry.

* **`microscopy_master.csv` \|** All experimentally
    measured fold-change values obtained through single-cell microscopy. Please
    see the supplemental information for a more detailed description.

* **`Oid_microscopy_master.csv` \|** All microscopy measurments of the Oid synthetic operator used in the Supplemental Information F.

* **`tidy_lacI_titration_data.csv` \|** The LacI repressor titration data from [Garcia and Phillips 2011](http://www.pnas.org/content/108/29/12173.abstract) and [Brewster *et al.* 2014](ihttp://www.cell.com/abstract/S0092-8674(14)00221-9).

* **`flow_cytometry_comments/` \|** All comments associated with the experimental runs shown in the `flow_master.csv` file.

* **`microscopy_comments/` \|** All comments associated with the experimental runs shown in `microscopy_master.csv` and `Oid_microscopy_master.csv`.

* **`mcmc_flatchains/` \|** All Markov Chain Monte Carlo sampler flatchains used for parameter estimation saved as `.pkl` files.

* **`example_flow_csv/` \|** A collection of representative flow cytometry `.csv` files used for the example `processing.py` file.

## `other`
This folder contains other various files used in the publication.

* **`plasmid_maps/` \|** A collection of the plasmid sequences used in this work
    as GenBank (`.gb`) files.

* **`materials_list.txt` \|** A list of the materials used in this work along
    with their catalog numbers.

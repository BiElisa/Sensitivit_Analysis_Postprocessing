# Sensitivity Analysis - Postprocessing
Script to perform the postprocessing of the sensitivity analysis results obtained with DAKOTA and MAMMA


## Preparation

Be sure that the folder with the current scripts contains the following files:

- dakota_tabular.dat
- dakota_test_parallel.in
- conduit_solver.template

Moreover, you need to have a folder containing all the simulations resulting from the sensitivity analysis. The best structure could be having a folder "workdir" which contains other folders "workdir.1", "workdir.2", etc.

## Execute the extraction of all the data from simulations results, files p.std

Execute the script "extract_allData.py", for example by terminal:

$ python extract_allData.py 

You can also use two input parameters "verbose" and "pause", for example:

$ python extract_allData.py --verbose true --pause false

The script generates, in order, the following 34 files:

* simulations.csv
* data_at_fragmentation_total.csv
* data_at_fragmentation.csv
* data_at_inlet_total.csv
* data_at_inlet.csv
* data_at_vent_total.csv
* data_at_vent.csv
* data_average_total.csv
* data_average.csv
* data_allConcat.csv

* simulations_explosive.csv
* data_at_fragmentation_explosive.csv
* data_at_inlet_explosive.csv
* data_at_vent_explosive.csv
* data_average_explosive.csv
* data_allConcat_explosive.csv

* simulations_notExplosive.csv
* data_at_fragmentation_notExplosive.csv
* data_at_inlet_notExplosive.csv
* data_at_vent_notExplosive.csv
* data_average_notExplosive.csv
* data_allConcat_notExplosive.csv

* simulations_notExplosive_effusive.csv
* data_at_fragmentation_notExplosive_effusive.csv
* data_at_inlet_notExplosive_effusive.csv
* data_at_vent_notExplosive_effusive.csv
* data_average_notExplosive_effusive.csv
* data_allConcat_notExplosive_effusive.csv

* simulations_notExplosive_fountaining.csv
* data_at_fragmentation_notExplosive_fountaining.csv
* data_at_inlet_notExplosive_fountaining.csv
* data_at_vent_notExplosive_fountaining.csv
* data_average_notExplosive_fountaining.csv
* data_allConcat_notExplosive_fountaining.csv

These files are saved in the directory "csv_files". Each column has two rows as header.

## Plot correlations

Execute the script "plot_correlation.py", for example by terminal:

$ python plot_correlation.py 

The script generates several figures which are saved in the directory "plot_correlations". The figure represent the correlation between several variables and their means. The user can modify the script in order to get the personalised correlation plots.

Moreover, the sobol indices are plot, and the user can select which response_fn analyse.

## Plot Sobol indices

Execute the script "plot_sobol.py", for example by terminal:

$ python plot_sobol.py 

The script generates a figure which is saved in the directory "plot_Sobol". The user can modify the script in order to get the personalised plots. For example, the user can select which response_fn analyse.

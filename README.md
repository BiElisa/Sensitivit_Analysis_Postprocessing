# Sensitivity Analysis - Postprocessing
Script to perform the postprocessing of the sensitivity analysis results obtained with DAKOTA and MAMMA


## Preparation

Be sure that the folder with the current scripts contains the following files:

- dakota_tabular.dat
- dakota_test_parallel.in
- conduit_solver.template

The current folder provides examples of those files (dakota_tabular_orig.dat, dakota_test_parallel_orig.in, conduit_solver_orig.template). Rename them by removing "_orig". 

Moreover, you need to have a folder containing all the simulations resulting from the sensitivity analysis. The best structure could be having a folder "workdir" which contains other folders "workdir.1", "workdir.2", etc.

## Execute the extraction of all the data from simulations results, files _p.std

Execute the script "extract_allData.py", for example by terminal:

$ python extract_allData.py 

You can also use two input parameters "verbose" and "pause", for example:

$ python extract_allData.py --verbose true --pause false

The script generates:

1. data_at_fragmentation_total.csv
2. data_at_fragmentation.csv
3. data_at_inlet_total.csv
4. data_at_inlet.csv
5. data_at_vent_total.csv
6. data_at_vent.csv
7. data_average_total.csv
8. data_average.csv

These files are saved in the directory "csv_files".

## Plot correlation plots

Execute the script "plot_correlation.py", for example by terminal:

$ python plot_correlation.py 

The script generates several figures which are saved in the directory "plot_correlations". The figure represent the correlation between several variables and their means. The user can modify the script in order to get the personalised correlation plots.

Moreover, the sobol indices are plot, and the user can select which response_fn analyse.

## Plot Sobol indices

Execute the script "plot_sobol.py", for example by terminal:

$ python plot_sobol.py 

The script generates a figure which is saved in the directory "plot_Sobol". The user can modify the script in order to get the personalised plots. For example, the user can select which response_fn analyse.
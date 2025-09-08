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

This generates four .csv files 

1. data_at_fragmentation_total.csv
2. data_at_inlet_total.csv
3. data_at_vent_total.csv
4. data_average_total.csv
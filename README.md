# Sensitivity Analysis - Postprocessing  

Scripts to perform the postprocessing of sensitivity analysis results obtained with **DAKOTA** and **MAMMA**.  
This repository provides tools to:  
- Extract and organize raw simulation outputs.  
- Generate correlation plots between input parameters and response functions.  
- Plot Sobol indices for variance-based sensitivity analysis.  
- (Planned) Extend functionality to frequency plots and other postprocessing tasks.  

---

## 📑 Table of Contents  

1. [Repository Structure](#-repository-structure)  
2. [Preparation](#-preparation)  
3. [Data Extraction (`extract_allData.py`)](#-data-extraction-extract_alldatapy)  
4. [Correlation Plots (`plot_correlation.py`)](#-correlation-plots-plot_correlationpy)  
5. [Sobol Indices (`plot_sobol.py`)](#-sobol-indices-plot_sobolpy)  
6. [Future Extensions](#-future-extensions)  
7. [Notes](#-notes)  

---

## 📂 Repository Structure  

- **`extract_allData.py`** → Extracts all data from DAKOTA/MAMMA simulations into structured CSV files.  
- **`plot_correlation.py`** → Generates correlation plots between input variables and response functions.  
- **`plot_sobol.py`** → Plots Sobol sensitivity indices.  

---

## ⚙️ Preparation  

Make sure the working directory contains the following files:  

- `dakota_tabular.dat`  
- `dakota_test_parallel.in`  
- `conduit_solver.template`  

Additionally, you need a folder containing all the simulation results from the sensitivity analysis.  
A recommended structure is:  

workdir/
├── workdir.1/
├── workdir.2/
├── ...


---

## 📊 Data Extraction (`extract_allData.py`)  

Run the extraction script to collect all simulation results into CSV files:  

```bash
$ python extract_allData.py

```

Optional arguments:

--verbose true/false → Enable detailed logging.

--pause true/false → Pause execution at key steps.

### Output

The script generates 34 CSV files organized by eruptive style (Explosive, Effusive, Fountaining, etc.) and saves them in the csv_files/ directory.

Examples:

data_allConcat.csv

data_allConcat_explosive.csv

data_allConcat_notExplosive.csv

data_allConcat_notExplosive_effusive.csv

data_allConcat_notExplosive_fountaining.csv

… and others.

Each file uses a two-row header: technical name (x* and response_fn_*) + label (Pressure [Pa], Radius [m], ...).

---

## 📈 Correlation Plots (plot_correlation.py)

Generate correlation plots between input parameters (x1, x2, …) and response functions (response_fn_*).

```bash
$ python plot_correlation.py
```

### Output

Figures saved in plot_correlations/.

Plots show scatter distributions and binned means for comparison.

Multiple eruptive regimes (Explosive, Effusive, Fountaining) can be compared on the same plot.

The script can be easily customized to:

Select specific response_fn_* to analyze.

Apply transformations (e.g., log10, scaling) to axes.

Change labels and plot layouts.

---

## 📉 Sobol Indices (plot_sobol.py)

Plot Sobol sensitivity indices for selected response functions.

```bash
$ python plot_Sobol.py
```

Output

Figures saved in plot_Sobol/.

User can modify the script to choose which response_fn_* to analyze.


## ✨ Notes

All scripts are designed to be modified by the user to adapt to specific workflows.

Figures are saved in .svg format for high-quality vector graphics.

Ensure my_lib_process_utils.py (utility functions) is available in the same directory.
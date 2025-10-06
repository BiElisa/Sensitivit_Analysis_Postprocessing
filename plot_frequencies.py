import pandas as pd
import numpy as np
import os
import sys
import shutil
import subprocess
import matplotlib.pyplot as plt
import math
import warnings
import my_lib_process_utils as utils

def plot_xi_histograms(
        dfs,
        var_specs=None,
        bins=30,
        save_name=None,
        save_dir="plot_histograms",
        fig_num=None
    ):
    """
    Plotta istogrammi per variabili specificate in var_specs, cercandole nei DataFrame.
    
    Parameters
    ----------
    dfs : dict[str, DataFrame] | DataFrame
        Dizionario di DataFrame ({"label": df}) o singolo DataFrame.
        Ogni df con colonne MultiIndex (level 0 = nome tecnico, es. 'x1'; level 1 = label descrittiva).
    var_specs : list[dict]
        Lista di dizionari con specifiche per le variabili da plottare:
        {"col": "x1", "transform": lambda arr: arr/1e6, "label": "Pressure (MPa)", "color": "b"}
    bins : int
        Numero di bins.
    save_name : str
        Nome base del file da salvare (senza estensione).
    save_dir : str
        Cartella di output.
    fig_num : int
        Numero figura matplotlib.
    """

    # Se dfs è un singolo dataframe, convertilo in dict
    if not isinstance(dfs, dict):
        dfs = {"Simulations": dfs}

    # Usa direttamente il primo DataFrame come riferimento (anche se vuoto)
    ref_df = next(iter(dfs.values()))

    # Trova tutte le xi presenti nel DataFrame
    xi_cols = [col for col in ref_df.columns if col[0].startswith('x')]
    if not xi_cols:
        raise ValueError("Nessuna colonna xi trovata nel DataFrame.")

    # Se var_specs è None → genera automaticamente da xi_cols
    if var_specs is None:
        var_specs = []
        for col in xi_cols:
            var_specs.append({
                "col": col[0],
                "label": col[1] if isinstance(col, tuple) else col,
                "color": "b"
            })
    else:
        # Controlla che tutte le xi siano coperte, altrimenti genera un warning
        specified = {spec["col"] for spec in var_specs}
        available = {c[0] for c in xi_cols}
        if not specified.issuperset(available):
            warnings.warn(
                f"⚠️ Le var_specs ({specified}) non coprono tutte le xi disponibili ({available}).",
                UserWarning
            )

    plot_histogram_lists(
        dfs=dfs,
        x_axis=var_specs,
        bins=bins,
        fig_num=fig_num,
        save_name=save_name,
        save_dir=save_dir
    )

def plot_histogram_lists(
        dfs,
        x_axis,
        bins=30,
        fig_num=None,
        save_name=None,
        save_dir="plot_histograms"
    ):
    """
    Funzione generica che stampa istogrammi da una lista di variabili.

    Parameters
    ----------
    dfs : dict[str, DataFrame] | DataFrame
        Dizionario di DataFrame {"label": df} o singolo DataFrame.
    x_axis : list of dict
        Lista di dizionari {"col": ..., "transform": ..., "label": ..., "color": ...}.
    bins : int
        Numero di bins.
    fig_num : int
        Numero figura matplotlib.
    save_name : str
        Nome base per il file da salvare (senza estensione).
    save_dir : str
        Cartella di output.
    """
    if not isinstance(dfs, dict):
        dfs = {"Simulations": dfs}

    num_plots = len(x_axis)
    n_cols = min(3, num_plots)
    n_rows = math.ceil(num_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), num=fig_num)
    axes = axes.flatten()

    for ax, spec in zip(axes, x_axis):
        col_name   = spec["col"]
        transform  = spec.get("transform", None)
        label     = spec.get("label", col_name)
        color      = spec.get("color", "b")

        for df_name, df in dfs.items():
            # Trova la colonna (supporta MultiIndex)
            if isinstance(df.columns, pd.MultiIndex):
                matches = [c for c in df.columns if c[0] == col_name]
                if not matches:
                    continue
                col = matches[0]
            else:
                col = col_name

            data = df[col].dropna().to_numpy()
            if transform:
                data = transform(data)

            ax.hist(
                data,
                bins=bins,
                color=color,
                edgecolor="black",
                alpha=0.5,
                label=df_name
            )

        ax.set_xlabel(label)
        ax.set_ylabel("Frequency")
        ax.legend(fontsize=8)
        ax.grid(False)#True, color='lightgray', linestyle='--', linewidth=0.5)

    # Spegni eventuali assi vuoti
    for ax in axes[num_plots:]:
        ax.axis("off")

    plt.tight_layout()

    if save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{save_name}.svg"))
        print(f"Figura salvata in {save_dir}/{save_name}.svg")

    plt.show()






if __name__ == '__main__':
    """
    Genera grafici di frequenza.
    """
    #region -- Controlliamo che i dati che ci servono siano presenti, altrimenti li andiamo a costruire

    # File richiesti
    mandatory_file = "dakota_test_parallel.in"
    other_files = [
        "data_allConcat.csv",
        "data_allConcat_explosive.csv",
        "data_allConcat_notExplosive.csv",
        "data_allConcat_notExplosive_effusive.csv",
        "data_allConcat_notExplosive_fountaining.csv"
    ]

    # Cartelle
    cwd = os.getcwd()
    csv_dir = os.path.join(cwd, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)

    # --- Controllo mandatory ---
    if not os.path.isfile(os.path.join(cwd, mandatory_file)):
        print(f"Abort: The file '{mandatory_file}' is not in the folder {cwd}.")
        sys.exit(1)

    # --- Controllo altri file ---
    missing = []
    for f in other_files:
        path_current = os.path.join(cwd, f)
        path_csv = os.path.join(csv_dir, f)

        if os.path.isfile(path_current):
            # Se il file è nella cartella corrente, spostalo in csv_files
            shutil.move(path_current, path_csv)
            print(f"Moved '{f}' from current folder to '{csv_dir}'.")
        elif not os.path.isfile(path_csv):
            # Mancante ovunque
            missing.append(f)

    # --- Se mancano file, run extract_allData ---
    if missing:
        print(f"The following files are missing from both current folder and '{csv_dir}': {missing}")
        print("The script 'extract_allData.py' is executed.")
        subprocess.run(["python", "extract_allData.py", "--pause", "false"])

    #endregion

    #region -- Carichiamo tutti i file e dati che ci servono

    read_csv_kwargs = {"header": [0,1], "encoding": "utf-8"}

    csv_dir = "csv_files"  

    df_concat         = pd.read_csv(os.path.join(csv_dir,"data_allConcat.csv"), **read_csv_kwargs)
    df_concat_expl    = pd.read_csv(os.path.join(csv_dir,"data_allConcat_explosive.csv"), **read_csv_kwargs)
    df_concat_notExpl = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive.csv"), **read_csv_kwargs)
    df_concat_eff     = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive_effusive.csv"), **read_csv_kwargs)
    df_concat_fount   = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive_fountaining.csv"), **read_csv_kwargs)

    #df_boundsInfo, input_Min, input_Max = utils.import_dakota_bounds()

    #endregion

    # Cartella in cui salvare i plot di default
    save_dir="plot_frequencies"

    # plot 1 --
#    plot_xi_histograms(
#        dfs=df_concat,
#        save_name="hist_xi"
#    )


    # plot 2 --
    inputs_scaled = [
        {"col": "x1", "transform": lambda x: x/1e6, "label": "Inlet Pressure [MPa]", "color": "b"},
        {"col": "x2", "transform": lambda x: x-273, "label": "Inlet Temperature [°C]", "color": "b"},
        {"col": "x3", "label": "Radius [m]"},
        {"col": "x4", "transform": lambda x: x*100, "label": "Inlet H₂O content [wt.%]", "color": "b"},
        {"col": "x5", "transform": lambda x: x*100, "label": "Inlet CO₂ content [wt.%]", "color": "b"},
        {"col": "x6", "transform": lambda x: x*100, "label": "Inlet phenocryst. content [vol.%]", "color": "b"}
    ]

    plot_xi_histograms(
        dfs = df_concat,
        var_specs = inputs_scaled,
        save_name="hist_inputs_scaled"
    )

    # plot 3 --
    x_axis = [
        {"col":"response_fn_1", "label": "Gas volume fraction"},
        {"col":"response_fn_15", "label": "Fragmentation depth [m]"},
        {"col":"response_fn_12", "transform": np.log10, "label": "Log10(MFR) [kg/s]"},
        {"col":"response_fn_4", "label": "Exit velocity [m/s]"},
        {"col":"response_fn_16", "transform": lambda x: x*100, "label": "Exit crystal content [vol.%]"},
        {"col":"response_fn_18", "label": "Fragmentation criteria"}
    ]

    plot_histogram_lists(
        dfs = df_concat,
        x_axis = x_axis,
        save_name="hist_varie_response_fn"
    )

    """
    plot_xi_histograms(
        dfs={
            "Explosive": df_concat_expl, 
            "Effusive": df_concat_eff
        },
        var_specs=var_specs,
        save_name="hist_selected_xi"
    )
    """


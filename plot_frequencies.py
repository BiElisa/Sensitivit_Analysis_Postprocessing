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
        Lista di dizionari con chiavi:
        {
            "col": str,                      # Nome della colonna da plottare
            "transform": callable | None,    # Funzione di trasformazione
            "label": str | None,             # Etichetta personalizzata asse x
            "color": str | None,             # Colore barre
            "edgecolor": str | None,         # Colore contorno barre
            "xlim": (float, float) | None,   # Limiti asse x
            "xticks": list[float] | None,    # Posizioni ticks
            "xticklabels": list[str] | None  # Etichette ticks
        }
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
        col_name     = spec.get("col")
        transform    = spec.get("transform", None)
        label        = spec.get("label", col_name)
        color        = spec.get("color", "blue")
        edgecolor    = spec.get("edgecolor", "black")
        xlim         = spec.get("xlim")
        xticks       = spec.get("xticks")
        xticklabels  = spec.get("xticklabels")

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
                edgecolor=edgecolor,
                alpha=0.5,
                label=df_name
            )

        ax.set_xlabel(label) #or col)
        
        # Mostra "Frequency" solo nella colonna più a sinistra
        col_idx = ax.get_subplotspec().colspan.start
        if col_idx == 0:
            ax.set_ylabel("Frequency")
        #else:
        #    ax.set_ylabel("")

        ax.legend(fontsize=8)
        ax.grid(False)#True, color='lightgray', linestyle='--', linewidth=0.5)

        #  Imposta eventuali limiti o tick custom
        if xlim is not None:
            ax.set_xlim(xlim)
        if xticks is not None:
            ax.set_xticks(xticks)
        if xticklabels is not None:
            ax.set_xticklabels(xticklabels)

    # Spegni eventuali assi vuoti
    for ax in axes[num_plots:]:
        ax.axis("off")

    plt.tight_layout()

    if save_name:
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, f"{save_name}.svg"))
        print(f"Figura salvata in {save_dir}/{save_name}.svg")

    #plt.show(block=True)
    plt.pause(1)
    plt.close(fig)






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

    # Inizializzo le liste di variabili da plottare 
    inputs_scaled = [
        {"col": "x1", "transform": lambda x: x/1e6, "label": "Inlet Pressure [MPa]", "color": "g"},
        {"col": "x2", "transform": lambda x: x-273, "label": "Inlet Temperature [°C]", "color": "g"},
        {"col": "x3", "label": "Radius [m]", "color": "g"},
        {"col": "x4", "transform": lambda x: x*100, "label": "Inlet H₂O content [wt.%]", "color": "g"},
        {"col": "x5", "transform": lambda x: x*100, "label": "Inlet CO₂ content [wt.%]", "color": "g"},
        {"col": "x6", "transform": lambda x: x*100, "label": "Inlet phenocryst. content [vol.%]", "color": "g"}
    ]

    x_axis_1 = [
        {"col":"response_fn_1", "label": "Gas volume fraction at inlet"},
        {"col":"response_fn_12", "transform": np.log10, "label": "Log10(MFR) [kg/s]"},
        {"col":"response_fn_4", "label": "Exit velocity [m/s]"},       
    ]

    x_axis_2 = [
        {"col":"response_fn_16", "transform": lambda x: x*100, "label": "Exit crystal content [vol.%]"},
        {"col":"response_fn_15", "label": "Fragmentation depth [m]", "color": "red"},
        {"col": "response_fn_18", "label": "Fragm Crit.", "color": "g", "edgecolor": "black", "xlim": (0, 4),
            "xticks": [1, 2, 3], 
            "xticklabels": ["SR", "IN", "BO"]
        },
    ]

    x_axis_3 = [
        {"col":"response_fn_27", "transform": lambda x: x*100, "label": "Gas fraction at fragm. [vol.%]"},
        {"col":"response_fn_28", "label": "Undercooling at fragm. [°C]"},
        {"col":"response_fn_20", "transform": np.log10, "label": "Log10(Viscosity at fragm.) [Pa s]"}
    ]

    #region Plot ALL simulations --------------

#    plot_xi_histograms(
#        dfs={"All simulations":df_concat},
#        save_name="freq_allSim_inputs"
#    )

    """

    plot_xi_histograms(
        dfs = {"All simulations":df_concat},
        var_specs = inputs_scaled,
        save_name="freq_allSim_inputs_scaled"
    )
    
    plot_histogram_lists(
        dfs = {"All simulations":df_concat},
        x_axis = x_axis_1 + x_axis_2,
        save_name="freq_allSim_1"
    )

    plot_histogram_lists(
        dfs = {"All simulations":df_concat},
        x_axis = x_axis_3,
        save_name="freq_allSim_2"
    )
    #endregion

    #region Plot explosive simulations --------------

#    plot_xi_histograms(
#        dfs={"Explosive": df_concat_expl},
#        save_name="freq_expl_inputs"
#    )

    plot_xi_histograms(
        dfs = {"Explosive": df_concat_expl},
        var_specs = inputs_scaled,
        save_name="freq_expl_inputs_scaled"
    )
    
    plot_histogram_lists(
        dfs = {"Explosive": df_concat_expl},
        x_axis = x_axis_1 + x_axis_2,
        save_name="freq_expl_1"
    )

    plot_histogram_lists(
        dfs = {"Explosive": df_concat_expl},
        x_axis = x_axis_3,
        save_name="freq_expl_2"
    )
    #endregion

    #region Plot notExplosive simulations --------------

#    plot_xi_histograms(
#        dfs={"Not explosive": df_concat_notExpl},
#        save_name="freq_notExpl_inputs"
#    )

    plot_xi_histograms(
        dfs = {"Not explosive": df_concat_notExpl},
        var_specs = inputs_scaled,
        save_name="freq_notExpl_inputs_scaled"
    )
    
    plot_histogram_lists(
        dfs = {"Not explosive": df_concat_notExpl},
        x_axis = x_axis_1 + [{"col":"response_fn_19", "transform": np.log10, "label": "Log10(Fountain height) (m)"}],
        save_name="freq_notExpl_1"
    )

    plot_histogram_lists(
        dfs = {"Not explosive": df_concat_notExpl},
        x_axis = x_axis_3,
        save_name="freq_notExpl_2"
    )
    #endregion

    #region Plot notExplosive-Effusive simulations --------------

#    plot_xi_histograms(
#        dfs={"Effusive": df_concat_eff},
#        save_name="freq_eff_inputs"
#    )

    plot_xi_histograms(
        dfs = {"Effusive": df_concat_eff},
        var_specs = inputs_scaled,
        save_name="freq_eff_inputs_scaled"
    )
    
    plot_histogram_lists(
        dfs = {"Effusive": df_concat_eff},
        x_axis = x_axis_1 + [{"col":"response_fn_19", "transform": np.log10, "label": "Log10(Fountain height) (m)"}],
        save_name="freq_eff_1"
    )

    plot_histogram_lists(
        dfs = {"Effusive": df_concat_eff},
        x_axis = x_axis_3,
        save_name="freq_eff_2"
    )
    #endregion

    #region Plot notExplosive-Fountaining simulations --------------

#    plot_xi_histograms(
#        dfs={"Fountaining": df_concat_fount},
#        save_name="freq_fount_inputs"
#    )

    plot_xi_histograms(
        dfs = {"Fountaining": df_concat_fount},
        var_specs = inputs_scaled,
        save_name="freq_fount_inputs_scaled"
    )
    
    plot_histogram_lists(
        dfs = {"Fountaining": df_concat_fount},
        x_axis = x_axis_1 + [{"col":"response_fn_19", "transform": np.log10, "label": "Log10(Fountain height) (m)"}],
        save_name="freq_fount_1"
    )

    plot_histogram_lists(
        dfs = {"Fountaining": df_concat_fount},
        x_axis = x_axis_3,
        save_name="freq_fount_2"
    )
    """

    #region *******************************************************************

    variables_to_plot = [
        {"col": "x2", "transform": lambda x: x-273, "label": "Inlet temperature [°C]", "ylim_max": 22},
        {"col": "x1", "transform": lambda x: x/1e6, "label": "Inlet pressure [MPa]", "ylim_max": 22},
        {"col": "x3", "ylim_max": 22},
        {"col": "x4", "transform": lambda x: x*100, "label": "Inlet H2O content [wt.%]", "ylim_max": 22},
        {"col": "x5", "transform": lambda x: x*100, "label": "Inlet CO2 content [wt.%]", "ylim_max": 22},
        {"col": "x6", "transform": lambda x: x*100, "label": "Inlet phenocrystal content [vol.%]", "ylim_max": 22},
        {"col": "response_fn_12", "transform": np.log10, "yscale": "log", "label": "Log10(MFR) [kg/s]"},
        {"col": "response_fn_4", "transform": np.log10,  "yscale": "log", "label": "Log10(exit velocity) [m/s]"},
        {"col": "response_fn_20", "transform": np.log10,  "yscale": "log", "label": "Log10(Viscosity) [Pa s]"},
        {"col": "response_fn_1", "yscale": "log"},
 #       {"col": "response_fn_15"},
        #{"col": "response_fn_44"},
        #{"col": "response_fn_16"},
        #{"col": "response_fn_20", "transform": np.log10, "label": "Log10(Viscosity at fragm) [Pa s]", "xscale": "log"}
    ]

    N_bins=50

    # Creiamo dei dizionari 

    stats       = {}
    freqs       = {}
    freqs_expl  = {}
    freqs_eff   = {}
    freqs_fount = {}

    for var in variables_to_plot:
        col = var["col"]
        transform = var.get("transform", None)
        label = var.get("label", None)

        # Recuperiamo i valori
        vals       = df_concat      [col].dropna().values  # rimuoviamo eventuali NaN
        vals_expl  = df_concat_expl [col].dropna().values 
        vals_eff   = df_concat_eff  [col].dropna().values 
        vals_fount = df_concat_fount[col].dropna().values 

        # Applica la trasformazione solo se richiesta
        if transform is not None:
            vals       = transform(vals)
            vals_expl  = transform(vals_expl)
            vals_eff   = transform(vals_eff)
            vals_fount = transform(vals_fount)

            # Warning se trasformazione non lineare ma label assente
            if label is None:
                warnings.warn(f"La variabile '{col}' ha una trasformazione ma nessuna label definita!")

        if len(vals) == 0:  # se non ci sono valori, saltare
            continue
        
        vmin, vmax = np.min(vals), np.max(vals)
        bin_edges = np.linspace(vmin, vmax, N_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        freq, _       = np.histogram(vals,       bins=bin_edges)
        freq_expl, _  = np.histogram(vals_expl,  bins=bin_edges)
        freq_eff, _   = np.histogram(vals_eff,   bins=bin_edges)
        freq_fount, _ = np.histogram(vals_fount, bins=bin_edges)
        
        stats[col] = { "min": vmin, "max": vmax, "bin_edges": bin_edges, "bin_centers": bin_centers}

        freqs[col]       = {"frequency": freq }
        freqs_expl[col]  = {"frequency": freq_expl }
        freqs_eff[col]   = {"frequency": freq_eff }
        freqs_fount[col] = {"frequency": freq_fount }


    n_plots = len(variables_to_plot)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.array(axes).flatten() if n_plots > 1 else [axes]  

    # Definiamo marker e colori per i 4 dataset
    markers = ['s', 'o', '>', 'd']
    colors = ['dodgerblue', 'lime', 'red', 'gold']
    labels_legend = ['Effusive','Fountaining','Explosive','Total']

    for i, var in enumerate(variables_to_plot):

        col = var["col"]
        label = var.get("label", None)
        ylim_max = var.get("ylim_max", None)
        xscale = var.get("xscale", None)
        yscale = var.get("yscale", None)

        # --- Controllo presenza colonna ---
        if col not in stats:
            warnings.warn(f"Colonna '{col}' non trovata nei dizionari, salto questo plot.")
            continue

        ax = axes[i]

        # --- Dati da plottare ---
        x = stats[col]["bin_centers"]
        y_total = freqs[col]["frequency"]
        y_eff = freqs_eff[col]["frequency"]
        y_fount = freqs_fount[col]["frequency"]
        y_expl = freqs_expl[col]["frequency"]

        # --- Tracciare le quattro curve ---
        ax.plot(x, y_eff, 'ks', markerfacecolor='b', markersize=5, label='Effusive')
        ax.plot(x, y_fount, 'ko', markerfacecolor='g', markersize=5, label='Fountaining')
        ax.plot(x, y_expl, 'k>', markerfacecolor='r', markersize=5, label='Explosive')
        ax.plot(x, y_total, 'kd', markerfacecolor='y', markersize=5, label='Total')

        # --- Scaling assi ---
        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        # --- Limiti ---
        ax.set_xlim(stats[col]["min"], stats[col]["max"])
        if ylim_max is not None:
            ax.set_ylim(0, ylim_max)
        else:
            ax.set_ylim(0, np.max(y_total) * 1.1)

        # --- Determino la label dell’asse x ---
        if label is None:
            # Prendiamo la seconda riga del MultiIndex (livello 1)
            try:
                col_label = df_concat.columns.get_level_values(1)[
                    df_concat.columns.get_level_values(0).tolist().index(col)
                ]
                xlabel = col_label
            except Exception:
                xlabel = col
        else:
            xlabel = label
        ax.set_xlabel(xlabel)

        if i % n_cols == 0:
            ax.set_ylabel("Frequency of solutions")

        # --- Stile ---
        ax.grid(True, linestyle='--', alpha=0.3)

        # --- Legenda ---
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # --- Layout generale ---
    plt.tight_layout()
    plt.savefig("plot_frequencies_eff_fount_expl.svg")
    plt.show()


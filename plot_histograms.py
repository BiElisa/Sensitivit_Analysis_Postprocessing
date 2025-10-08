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

    utils.check_files_required()

    #region -- Carichiamo tutti i file e dati che ci servono

    read_csv_kwargs = {"header": [0,1], "encoding": "utf-8"}

    csv_dir = "csv_files"  

    df_concat         = pd.read_csv(os.path.join(csv_dir,"data_allConcat.csv"), **read_csv_kwargs)
    df_concat_expl    = pd.read_csv(os.path.join(csv_dir,"data_allConcat_explosive.csv"), **read_csv_kwargs)
    df_concat_notExpl = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive.csv"), **read_csv_kwargs)
    df_concat_eff     = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive_effusive.csv"), **read_csv_kwargs)
    df_concat_fount   = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive_fountaining.csv"), **read_csv_kwargs)

    #endregion

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

    #region -- Plot histograms for ALL simulations 

    plot_xi_histograms(
        dfs={"All simulations":df_concat},
        save_name="freq_allSim_inputs"
    )

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

    #region -- Plot histograms for explosive simulations 

    plot_xi_histograms(
        dfs={"Explosive": df_concat_expl},
        save_name="freq_expl_inputs"
    )

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

    #region -- Plot histograms for notExplosive simulations 

    plot_xi_histograms(
        dfs={"Not explosive": df_concat_notExpl},
        save_name="freq_notExpl_inputs"
    )

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

    #region -- Plot histograms for notExplosive-Effusive simulations 

    plot_xi_histograms(
        dfs={"Effusive": df_concat_eff},
        save_name="freq_eff_inputs"
    )

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

    #region -- Plot histograms for notExplosive-Fountaining simulations 

    plot_xi_histograms(
        dfs={"Fountaining": df_concat_fount},
        save_name="freq_fount_inputs"
    )

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
    #endregion

 
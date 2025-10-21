import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import warnings
import my_lib_process_utils as utils
import user_plot_histograms as user_plt

def plot_xi_histograms(
        df,
        var_specs=None,
        bins=30,
        fig_num=None,
        save_name=None,
        save_dir="plot_histograms"
    ):
    """
    Plotta istogrammi per variabili specificate in var_specs, cercandole nei DataFrame.
    
    Parameters
    ----------
    df : dict[str, DataFrame] | DataFrame
        Dizionario di DataFrame ({"label": df}) o direttamente DataFrame.
        Il dataframe ha colonne MultiIndex (level 0 = nome tecnico, es. 'x1'; level 1 = label descrittiva).
    var_specs : list[dict]
        Lista di dizionari con specifiche per le variabili da plottare:
        {"col": "x1", "transform": lambda x: x/1e6, "label": "Pressure (MPa)", "color": "b"}
    bins : int
        Numero di bins.
    save_name : str
        Nome base del file da salvare (senza estensione).
    save_dir : str
        Cartella di output.
    fig_num : int
        Numero figura matplotlib.
    """

    # Se df è passato come dataframe, convertilo in dict
    if not isinstance(df, dict):
        df = {"Simulations": df}

    # Usa direttamente il primo DataFrame come riferimento (anche se vuoto)
    ref_df = next(iter(df.values()))

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

    # Se save_name non è specificato, generalo automaticamente
    if save_name is None:
        df_key_str = "_".join(df.keys())
        if fig_num is None:
            save_name = f"freq_{df_key_str}_xi"
        else:
            save_name = f"freq_{df_key_str}_xi_fig{fig_num}"
        print(f"Warning: no filename selected for 'save_name'. File saved as '{save_name}'.\n")

    plot_histograms_list(
        df=df,
        x_axis=var_specs,
        bins=bins,
        fig_num=fig_num,
        save_name=save_name,
        save_dir=save_dir
    )

def plot_histograms_list(
        df,
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
    df : dict[str, DataFrame] | DataFrame
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
    if not isinstance(df, dict):
        df = {"Simulations": df}

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

        for df_name, DF in df.items():
            # Trova la colonna (supporta MultiIndex)
            if isinstance(DF.columns, pd.MultiIndex):
                matches = [c for c in DF.columns if c[0] == col_name]
                if not matches:
                    continue
                col = matches[0]
            else:
                col = col_name

            data = DF[col].dropna().to_numpy()
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

        ax.set_xlabel(label) 
        
        # Mostra "Frequency" solo nella colonna più a sinistra
        col_idx = ax.get_subplotspec().colspan.start
        if col_idx == 0:
            ax.set_ylabel("Frequency")

        ax.legend(fontsize=8)
        ax.grid(False)

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

    # Se save_name non è specificato, generalo automaticamente
    if save_name is None:
        df_key_str = "_".join(df.keys())
        if fig_num is None:
            save_name = f"freq_{df_key_str}_my_plot_lists"
        else:
            save_name = f"freq_{df_key_str}_my_plot_lists_fig{fig_num}"
        print(f" * Warning * : no filename selected for 'save_name'. Name autogenerated '{save_name}'.\n")

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

    # Importa ed esegue lo script utente
    user_plt.run_all_plots(
        plot_xi_histograms=plot_xi_histograms,
        plot_histograms_list=plot_histograms_list,
        df_concat=df_concat,
        df_concat_expl=df_concat_expl,
        df_concat_notExpl=df_concat_notExpl,
        df_concat_eff=df_concat_eff,
        df_concat_fount=df_concat_fount
    )

 
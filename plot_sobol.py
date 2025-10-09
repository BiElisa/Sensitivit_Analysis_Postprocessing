import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import my_lib_process_utils as utils

def plot_sobol_indices(
        sobol_indices, 
        xi_labels=None, 
        response_labels=None, 
        fig_num=None,
        save_name=None, 
        save_dir="plot_Sobol"):
    """
    Plotta gli Sobol indices normalizzati per più response_fn come stacked bar plot.

    Parameters
    ----------
    sobol_indices : dict
        Chiavi = nome della response_fn
        Valori = array di Sobol indices normalizzati per ciascun xi
    xi_labels : list of str, optional
        Etichette leggibili per le xi (ex. ['Pressure', 'Temperature']). 
        Devono avere la stessa lunghezza delle xi presenti.
    response_labels : dict, optional
        Etichette leggibili per ciascun response_fn
    save_name : str, optional
        Nome con cui salvare la figura (SVG)
    save_dir : str, optional
        Cartella dove salvare la figura (SVG)
    """
    
    # --- recupera numero totale di xi ---
    first_resp = next(iter(sobol_indices.values()))
    n_xi_total = len(first_resp)
    xi_all = [f"x{i+1}" for i in range(n_xi_total)]

    # --- gestione etichette ---
    if xi_labels is None:
        xi_labels = xi_all
    else:
        if len(xi_labels) != n_xi_total:
            raise ValueError(
                f"xi_labels deve avere la stessa lunghezza delle xi presenti ({n_xi_total}), "
                f"ma ne sono state passate {len(xi_labels)}."
            )

    # --- filtriamo le response_fn da plottare ---
    responses_to_plot = list(response_labels.keys()) if response_labels else list(sobol_indices.keys())
    n_resp = len(responses_to_plot)

    fig, axes = plt.subplots(1, n_resp, figsize=(3*n_resp,5), sharey=True)
    if n_resp == 1:
        axes = [axes]

    color_palette = plt.cm.tab20.colors

    for ax, resp in zip(axes, responses_to_plot):
        indices = np.array(sobol_indices.get(resp, np.zeros(len(xi_labels))))
        bottom = np.zeros(1)
        for i, val in enumerate(indices):
            ax.bar(1, val, bottom=bottom.sum(), width=0.5, color=color_palette[i % len(color_palette)])
            bottom[0] += val

        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([''])
        ax.set_ylim(0,1)
        ax.set_title(response_labels.get(resp, resp) if response_labels else resp)

    # legenda solo nel primo subplot
    axes[0].legend(xi_labels, loc='lower left', fontsize=9)
    fig.suptitle('Sobol indices normalized', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])

    # --- gestione nome file ---
    if save_name is None:
        if fig_num is None:
            save_name = f"sobol_indices_my_plot"
        else:
            save_name = "sobol_indices_my_plot_fig{fig_num}"
        print(f" * Warning * : no filename selected for 'save_name'. File saved as '{save_name}'.\n")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{save_name}.svg"))
    print(f"Figura salvata in {save_dir} as {save_name}")

    #plt.show()
    plt.pause(1.3)
    plt.close(fig)

def compute_sobol_indices(df, stats):
    """
    Calcola gli indici di Sobol di primo ordine normalizzati a partire dai risultati
    di bin_and_average (stats).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame originale con tutte le colonne xi e response_fn.
    stats : dict
        Output di bin_and_average:
        stats[xi][response_fn] = DataFrame con colonne ['mean', 'count', ...]
        stats[xi]["bin_centers"] = array dei centri dei bin

    Returns
    -------
    sobol_indices : dict
        sobol_indices[response_fn] = array normalizzata degli indici di Sobol per ciascun xi
    """

    xi_cols = [col for col in df.columns if col[0].startswith("x")]
    response_cols = [col for col in df.columns if col[0].startswith("response_fn")]

    sobol_indices = {}

    for resp in response_cols:
        total_var = df[resp].var(ddof=0)  # varianza totale
        indices = []

        for xi in xi_cols:
            try:
                bin_means = stats[xi[0]][resp[0]]['mean']  # estrai medie per bin
                var_bin = bin_means.var(ddof=0)           # varianza delle medie
                # protezione divisione per zero
                if total_var == 0:
                    sobol_idx = 0.0
                else:
                    sobol_idx = var_bin / total_var # primo ordine
            except KeyError:
                sobol_idx = 0.0  # o np.nan se vuoi. Questo nel caso la statistica non sia disponibile
            indices.append(sobol_idx)

        indices = np.array(indices)
        # normalizzazione
        sum_indices = np.nansum(indices)
        if sum_indices == 0:
            normalized = indices  # tutti 0
        else:
            normalized = indices / sum_indices

        sobol_indices[resp[0]] = normalized
        
    return sobol_indices

if __name__ == '__main__':
    """
    Genera grafici degli indici di sobol.
    """

    utils.check_files_required()

    #region -- Carichiamo tutti i file e dati che ci servono

    read_csv_kwargs = {"header": [0,1], "encoding": "utf-8"}

    csv_dir = "csv_files"  

    df_concat = pd.read_csv(os.path.join(csv_dir,"data_allConcat.csv"), **read_csv_kwargs)
    df_concat_expl    = pd.read_csv(os.path.join(csv_dir,"data_allConcat_explosive.csv"), **read_csv_kwargs)
    df_concat_notExpl = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive.csv"), **read_csv_kwargs)
    df_concat_eff     = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive_effusive.csv"), **read_csv_kwargs)
    df_concat_fount   = pd.read_csv(os.path.join(csv_dir,"data_allConcat_notExplosive_fountaining.csv"), **read_csv_kwargs)

    df_boundsInfo, input_Min, input_Max = utils.import_dakota_bounds()

    #endregion

    #region -- Calcolo dei dati statistici e degli indici di Sobol

    N_bins = 25

    stats         = utils.bin_and_average(df_concat, N_bins)
    sobol_indices = compute_sobol_indices(df_concat, stats)
    print(f"Elaborati dati statistici e indici di Sobol di 'data_allConcat.csv' con {N_bins} bins.\n")

    stats_expl    = utils.bin_and_average(df_concat_expl, N_bins)
    sobol_indices_expl = compute_sobol_indices(df_concat_expl, stats_expl)
    print(f"Elaborati dati statistici e indici di Sobol di 'data_allConcat_explosive.csv' con {N_bins} bins.\n")

    stats_notExpl = utils.bin_and_average(df_concat_notExpl, N_bins)
    sobol_indices_notExpl = compute_sobol_indices(df_concat_notExpl, stats_notExpl)
    print(f"Elaborati dati statistici e indici di Sobol di 'data_allConcat_notExplosive.csv' con {N_bins} bins.\n")

    stats_eff     = utils.bin_and_average(df_concat_eff, N_bins)
    sobol_indices_eff = compute_sobol_indices(df_concat_eff, stats_eff)
    print(f"Elaborati dati statistici e indici di Sobol di 'data_allConcat_notExplosive_effusive.csv' con {N_bins} bins.\n")

    stats_fount   = utils.bin_and_average(df_concat_fount, N_bins)
    sobol_indices_fount = compute_sobol_indices(df_concat_fount, stats_fount)
    print(f"Elaborati dati statistici e indici di Sobol di 'data_allConcat_notExplosive_fountaining.csv' con {N_bins} bins.\n")

    #endregion

    save_dir="plot_Sobol"

    # chiavi = response_fn, 
    # valori = array di indici normalizzati per ciascun xi
    response_labels_example = {
        'response_fn_1': 'Gas volume fraction',
        'response_fn_15': 'Fragmentation depth',
        'response_fn_12': 'Mass flow rate',
        'response_fn_4': 'Exit velocity',
        'response_fn_16': 'Exit crystal content',
        'response_fn_28': 'Undercooling @Frag'
    }

    plot_sobol_indices(
        sobol_indices, 
        response_labels=response_labels_example, 
        save_name='sobol_indices',
        save_dir=save_dir
    )

    # chiavi = response_fn, 
    # valori = array di indici normalizzati per ciascun xi
    xi_labels = ['Press.','Temp.','Radius','H2O','Crystals', 'CO2']
    response_labels_example = {
        'response_fn_1': 'Gas volume fraction',
        'response_fn_15': 'Fragmentation depth',
        'response_fn_12': 'Mass flow rate',
        'response_fn_4': 'Exit velocity',
        'response_fn_16': 'Exit crystal content',
        'response_fn_28': 'Undercooling @Frag',
        'response_fn_72': 'Avg. viscosity'
    }

    plot_sobol_indices(
        sobol_indices, 
        xi_labels=xi_labels, 
        response_labels=response_labels_example, 
        save_name='sobol_indices_xi_Labels',
        save_dir=save_dir
    )

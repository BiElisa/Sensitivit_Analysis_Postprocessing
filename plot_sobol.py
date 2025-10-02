import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import shutil
import subprocess
import my_lib_process_utils as utils

def plot_sobol_indices(
        sobol_indices, 
        xi_selected=None,
        xi_labels=None, 
        renormalize_subset=False,
        response_labels=None, 
        save_name=None, 
        save_dir="plot_Sobol"):
    """
    Plotta gli Sobol indices normalizzati per più response_fn come stacked bar plot.

    Parameters
    ----------
    sobol_indices : dict
        Chiavi = nome della response_fn
        Valori = array di Sobol indices normalizzati per ciascun xi
    xi_selected :  list of str, optional
        Selezione di variabili xi da plottare (ex. ['x1', 'x3']). Se None, vengono usate tutte le xi disponibili.
    xi_labels : list of str, optional
        Etichette leggibili per le xi selezionate (ex. ['Pressure', 'Temperature']). 
        Se xi_selected e` presente, devono avere la stessa lunghezza.
    renormalize_subset : bool, optional
        Se True e se xi_selected è usato, gli indici vengono ri-normalizzati
        in modo che la somma delle sole variabili selezionate valga 1.
        Se False (default), si mantengono i valori assoluti rispetto al totale
        → il grafico può avere "buchi" (quota residua delle variabili escluse)
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

    # --- Selezione subset di xi ---
    if xi_selected is not None:

        # controlla che tutte le xi scelte esistano
        not_found = [xi for xi in xi_selected if xi not in xi_all]
        if not_found:
            raise ValueError(f"Le variabili {not_found} non sono state trovate (disponibili: {xi_all}).")

        # maschera in base a xi_selected
        mask = [xi in xi_selected for xi in xi_all]

        # filtra sobol_indices
        sobol_indices = {
            resp: np.array(vals)[mask]
            for resp, vals in sobol_indices.items()
        }

        # gestisci etichette
        if xi_labels is None:
            xi_labels = xi_selected
        else:
            if len(xi_labels) != len(xi_selected):
                raise ValueError(
                    f"xi_labels deve avere la stessa lunghezza di xi_selected "
                    f"({len(xi_selected)}), ma ha {len(xi_labels)}"
                )
    else:
        # nessuna selezione → usa tutte
        if xi_labels is None:
            xi_labels = xi_all

    # --- gestione nome file ---
    if save_name is None:
        save_name = "sobol_indices_my_plot"
        print(f" * Warning * : no filename selected for 'save_name'. File saved as '{save_name}'.\n")

    # responses_to_plot = list(sobol_indices.keys()) # questo fa plottare gli indici di tutte le respo_fn
    # filtriamo le response_fn da plottare
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

#        data = indices.reshape(1, -1)
#        bplot = ax.bar(range(1,2), data, stacked=True, width=0.5, color=color_palette[:len(xi_labels)])

        ax.set_xlim(0.5, 1.5)
        ax.set_xticks([1])
        ax.set_xticklabels([''])
        ax.set_ylim(0,1)
        ax.set_title(response_labels.get(resp, resp) if response_labels else resp)

    # legenda solo nel primo subplot
    axes[0].legend(xi_labels, loc='lower left', fontsize=9)
    fig.suptitle('Sobol indices normalized', fontsize=14)
    plt.tight_layout(rect=[0,0,1,0.95])

    if save_name:
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

    plot_sobol_indices(
        sobol_indices, 
        xi_selected = ['x1', 'x3'],
        xi_labels=['a', 'b'], 
        #renormalize_subset=True,
        response_labels=response_labels_example, 
        save_name='sobol_indices_xi_selected',
        save_dir=save_dir
    )




    













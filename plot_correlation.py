import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import my_lib_process_utils as utils
import user_plot_correlation as user_plt

def plot_xi_vs_response_fn( 
        dfs, 
        input_Min,
        input_Max,
        response_col, 
        y_label=None, 
        n_step=3,
        save_name=None,
        save_dir="plot_correlations",
        fig_num=None,
        stats={},
    ):
    """
    Plotta le variabili xi vs una colonna di risposta in un layout dinamico.
    Wrapper che poi richiama plot_correlation_list

    Parameters
    ----------
    dfs : dictionary of DataFrame {"label" : df} or single dataFrame
        DataFrame con MultiIndex sulle colonne (level 0 = nome tecnico, level 1 = label);
        Le colonne sono i parametri di unput {xi} e valori di output {response_fn_i}.
    input_Min : list of float
        Valori minimi (già trasformati) delle xi.
    input_Max : list of float
        Valori massimi (già trasformati) delle xi.
    response_col : str
        Nome della colonna di risposta da plottare (es. 'response_fn_1').
    y_label : str, optional
        Etichetta da mostrare sull'asse Y. Se None, viene usato response_col.
    n_step : int
        Campionamento dei dati per velocizzare il plot (default=1)
    save_name : str, optional
        Nome da usare per il file salvato (senza estensione). Se None, usa y_label o response_col.
    fig_num : int, optional
        Numero della figura (per creare più finestre distinte).
    stats : dict, optional
        Output della funzione bin_and_average. Se fornito, consente di plottare le medie.
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

    # Verifica che la colonna di risposta esista
    matches = [col for col in ref_df.columns if col[0] == response_col]
    if not matches:
        raise ValueError(f"La colonna '{response_col}' non è stata trovata nel DataFrame.")
    if len(matches) > 1:
        print(f"Attenzione: trovate più colonne per '{response_col}', uso la prima.")
    response_tuple = matches[0]

    # Etichetta Y: o quella passata, o quella dal MultiIndex
    y_label = y_label or response_tuple[1]

    # Costruisci le liste per plot_correlation_lists
    x_axis = []
    y_axis = []
    for col in xi_cols:
        # Label della xi (dal MultiIndex se disponibile)
        if isinstance(ref_df.columns, pd.MultiIndex):
            xi_label = ref_df.columns.get_level_values(1)[ref_df.columns.get_loc(col)]
        else:
            xi_label = col[0]

        x_axis.append((col[0], None, xi_label))
        y_axis.append((response_col, None, y_label))

    # Se save_name non è specificato, generalo automaticamente
    if save_name is None:
        dfs_keys_str = "_".join(dfs.keys())
        if fig_num is None:
            save_name = f"corr_{dfs_keys_str}_xi_{response_col}"
        else:
            save_name = f"corr_{dfs_keys_str}_xi_{response_col}_fig{fig_num}"
        print(f"Warning: no filename selected for 'save_name'. File saved as '{save_name}'.\n")

    # Richiama plot_correlation_lists
    plot_correlation_lists(
        dfs=dfs,
        x_axis=x_axis,
        y_axis=y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=n_step,
        fig_num=fig_num,
        save_name=save_name,
        save_dir=save_dir,
        stats=stats,
    )

def plot_correlation_lists(
    dfs,
    x_axis,
    y_axis,
    input_Min,
    input_Max,
    n_step=3,
    fig_num=None,
    save_name=None,
    save_dir="plot_correlations",
    stats={}
):
    """
    Plotta coppie di variabili prese da due liste (x_axis[i], y_axis[i]).

    Parameters
    ----------
    dfs : dictionary of DataFrame {"label" : df} or single dataFrame
        DataFrame con MultiIndex sulle colonne (level 0 = nome tecnico, level 1 = label).
    x_axis, y_axis : list
        Liste di tuple: (col_name, transform_or_factor, label) 
        - col_name : str, es. 'x1' o 'response_fn_55'
        - transform_or_factor : funzione, numero, o None
        - label : str (opzionale)
    input_Min, input_Max : np.array
        Limiti per le variabili x (se col_name inizia con 'x').
    n_step : int
        Passo di campionamento.
    fig_num : int
        Numero figura matplotlib.
    save_name : str
        Base name per salvare le figure.
    stats : dict, optional
        Output della funzione bin_and_average. Se fornito, consente di plottare le medie.
    """

    # Check if "dfs" is a single dataFrame or a dictonary of dataFrame
    if isinstance(dfs, dict):
        df_items = dfs.items()
    else:
        df_items = [("Simulations", dfs)]

    if len(x_axis) != len(y_axis):
        raise ValueError("Le liste x_axis e y_axis devono avere la stessa lunghezza.")
    
    nonlinear_funcs = [np.log10, np.log, np.sqrt, np.exp] # serve per le eventuali trasformazion per x e y

    num_plots = len(x_axis)
    n_cols = min(3, num_plots)
    n_rows = math.ceil(num_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), num=fig_num)
    axes = axes.flatten()

    # Palette e markers per dataset
    colors = ["red", "blue", "limegreen", "orange", "purple", "brown", "gray"]
    markers = ["o", "o", "o", "o"] # ["o", "s", "D", "^", "v", "x"]

    for ax, x_entry, y_entry in zip(axes, x_axis, y_axis):
        x_col, x_transform, x_label = parse_entry(x_entry)
        y_col, y_transform, y_label = parse_entry(y_entry)

        for i, (df_name, df) in enumerate(df_items):
            if df.empty:
                continue  # skip dataset vuoto

            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]

            # Prendi i dati (può essere array vuoto)
            x_vals = df[x_col].to_numpy()[::n_step]
            y_vals = df[y_col].to_numpy()[::n_step]
            if len(x_vals) == 0 or len(y_vals) == 0:
                continue

            # Applica trasformazioni se definite
            if x_transform:
                x_vals = x_transform(x_vals)
            if y_transform:
                y_vals = y_transform(y_vals)

            # Etichette sugli assi
            if not x_label:
                x_label = df.columns.get_level_values(1)[df.columns.get_level_values(0) == x_col][0]
            if not y_label:
                y_label = df.columns.get_level_values(1)[df.columns.get_level_values(0) == y_col][0]

            ax.plot(x_vals, y_vals, marker, color=color, markersize=2, linestyle="none", label=f"Data {df_name}", alpha=0.6)

        # --- Overlay stats se presenti ---

        if stats:
            try:
                bin_centers = stats[x_col]["bin_centers"]

                nonlinear_x = x_transform in nonlinear_funcs
                nonlinear_y = y_transform in nonlinear_funcs

                if nonlinear_x or nonlinear_y:
                    # Ricalcolo binning e medie sui dati trasformati
                    x_vals_full = df[x_col].to_numpy().ravel()
                    y_vals_full = df[y_col].to_numpy().ravel()

                    if x_transform:
                        x_vals_full = x_transform(x_vals_full)
                    if y_transform:
                        y_vals_full = y_transform(y_vals_full)

                    # Ricostruisco i bin sui dati trasformati
                    bins = np.linspace(np.min(x_vals_full), np.max(x_vals_full)*1.00001, len(bin_centers)+1)
                    df_bins = pd.cut(
                        x_vals_full, 
                        bins=bins, 
                        labels=0.5*(bins[:-1]+bins[1:]), 
                        include_lowest=True
                    )

                    tmp = pd.DataFrame({"bin": df_bins, "y": y_vals_full})
                    grouped = tmp.groupby("bin")["y"].mean()

                    bin_centers = grouped.index.to_numpy(dtype=float)
                    resp_means = grouped.to_numpy()

                else:
                    # Uso direttamente le medie pre-calcolate
                    resp_means = stats[x_col][y_col]["mean"].to_numpy()
                    if x_transform:
                        bin_centers = x_transform(bin_centers)
                    if y_transform:
                        resp_means = y_transform(resp_means)

                if len(bin_centers) > 0 and len(resp_means) > 0:
                    ax.plot(bin_centers, resp_means, "k-", lw=2, label="Binned Mean")
            except KeyError:
                pass

        # --- Settaggio limiti se x/y sono xi---
        if x_col.startswith("x"):
            idx = int(x_col[1:]) - 1
            x_min, x_max = input_Min[idx], input_Max[idx]
            if x_transform:
                x_min, x_max = x_transform(np.array([x_min, x_max]))
            ax.set_xlim(x_min, x_max)

        if y_col.startswith("x"):
            idx = int(y_col[1:]) - 1
            y_min, y_max = input_Min[idx], input_Max[idx]
            if y_transform:
                y_min, y_max = y_transform(np.array([y_min, y_max]))
            ax.set_ylim(y_min, y_max)
        # --------------------------------------
        
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend(fontsize=8, loc="best")

    # Spegni eventuali assi vuoti
    for ax in axes[num_plots:]:
        ax.axis("off")

    plt.tight_layout()

    # Se save_name non è specificato, generalo automaticamente
    if save_name is None:
        dfs_keys_str = "_".join(dfs.keys())
        if fig_num is None:
            save_name = f"corr_{dfs_keys_str}_my_plot_lists"
        else:
            save_name = f"corr_{dfs_keys_str}_my_plot_lists_fig{fig_num}"
        print(f"Warning: no filename selected for 'save_name'. File saved as '{save_name}'.\n")

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"{save_name}.svg"))
    print(f"Figura salvata in {save_dir} as {save_name}")

    #plt.show(block=True)
    plt.pause(1.3)
    plt.close(fig)

def parse_entry(entry):

    if isinstance(entry, str):
        # Caso semplice: solo il nome colonna
        return entry, None, None

    elif isinstance(entry, tuple):
        col_name = entry[0]
        transform = entry[1] if len(entry) > 1 else None
        label = entry[2] if len(entry) > 2 else None

        # Caso: numero intero o float
        if isinstance(transform, (int, float)):
            if transform == 0:
                fn = None
            elif transform > 0:
                # Moltiplicatore
                fn = lambda v, f=transform: v * f
            else:
                # Offset additivo (es. -273)
                fn = lambda v, o=transform: v + o

        # Caso: funzione già definita (es. np.log10 o lambda custom)
        elif callable(transform):
            fn = transform

        # Nessuna trasformazione
        elif transform is None:
            fn = None

        else:
            raise ValueError(f"Transform non valido: {transform}")

        return col_name, fn, label

    else:
        raise ValueError(f"Entry non valido: {entry}")


if __name__ == '__main__':
    """
    Genera grafici di correlazione.
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

    df_boundsInfo, input_Min, input_Max = utils.import_dakota_bounds()

    #endregion

    #region -- Calcolo dei dati statistici 

    N_bins = 25

    stats         = utils.bin_and_average(df_concat, N_bins)
    print(f"Elaborati dati statistici di 'data_allConcat.csv' con {N_bins} bins.\n")

    stats_expl    = utils.bin_and_average(df_concat_expl, N_bins)
    print(f"Elaborati dati statistici di 'data_allConcat_explosive.csv' con {N_bins} bins.\n")

    stats_notExpl = utils.bin_and_average(df_concat_notExpl, N_bins)
    print(f"Elaborati dati statistici di 'data_allConcat_notExplosive.csv' con {N_bins} bins.\n")

    stats_eff     = utils.bin_and_average(df_concat_eff, N_bins)
    print(f"Elaborati dati statistici di 'data_allConcat_notExplosive_effusive.csv' con {N_bins} bins.\n")

    stats_fount   = utils.bin_and_average(df_concat_fount, N_bins)
    print(f"Elaborati dati statistici di 'data_allConcat_notExplosive_fountaining.csv' con {N_bins} bins.\n")

    #endregion


    # Importa ed esegue lo script utente
    user_plt.run_all_plots(
        plot_xi_vs_response_fn=plot_xi_vs_response_fn,
        plot_correlation_lists=plot_correlation_lists,
        df_concat=df_concat,
        df_concat_expl=df_concat_expl,
        df_concat_notExpl=df_concat_notExpl,
        df_concat_eff=df_concat_eff,
        df_concat_fount=df_concat_fount,
        input_Min=input_Min,
        input_Max=input_Max,
        stats=stats
    )






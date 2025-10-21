import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import warnings
import my_lib_process_utils as utils
import user_plot_frequencies as user_plt

def plot_frequencies_eruptive_styles(
        variables_to_plot, 
        df_concat, 
        df_concat_expl=None, 
        df_concat_eff=None, 
        df_concat_fount=None,  
        N_bins=50, 
        fig_num=None, 
        save_name=None,
        save_dir="plot_frequencies",
        plot_total=True
    ):
    """
    Plotta le frequenze delle variabili per diversi stili eruttivi.

    Parameters
    ----------
    variables_to_plot : list of dict
        Lista di variabili da plottare con eventuali trasformazioni, label e limiti.
    df_concat : pd.DataFrame
        Dataset totale.
    df_concat_expl : pd.DataFrame, optional
        Dataset esplosivo.
    df_concat_eff : pd.DataFrame, optional
        Dataset effusivo.
    df_concat_fount : pd.DataFrame, optional
        Dataset fountaining.
    N_bins : int, optional
        Numero di bin per l'istogramma, di default 50.
    save_name : str, optional
        Percorso per salvare il grafico, di default None.
    save_dir : str, optional
        Cartella di output.
    """
    
    # Creiamo i dizionari per statistiche e frequenze
    stats = {}
    freqs = {}
    freqs_expl = {}
    freqs_eff = {}
    freqs_fount = {}

    for var in variables_to_plot:
        col = var["col"]
        transform = var.get("transform", None)
        label = var.get("label", None)

        # Recuperiamo i valori
        vals       = df_concat      [col].dropna().values 
        vals_expl  = df_concat_expl [col].dropna().values if df_concat_expl  is not None  else np.array([])
        vals_eff   = df_concat_eff  [col].dropna().values if df_concat_eff   is not None  else np.array([])
        vals_fount = df_concat_fount[col].dropna().values if df_concat_fount is not None  else np.array([])

        # Applica la trasformazione se presente
        if transform is not None:
            vals       = transform(vals)
            if df_concat_expl  is not None: vals_expl  = transform(vals_expl)
            if df_concat_eff   is not None: vals_eff   = transform(vals_eff)
            if df_concat_fount is not None: vals_fount = transform(vals_fount)

        if label is None:
            warnings.warn(f"La variabile '{col}' ha una trasformazione ma nessuna label definita!")

        if len(vals) == 0:
            continue

        vmin, vmax = np.min(vals), np.max(vals)
        bin_edges = np.linspace(vmin, vmax, N_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

        freq, _       = np.histogram(vals,       bins=bin_edges)
        freq_expl, _  = np.histogram(vals_expl,  bins=bin_edges) if len(vals_expl) > 0  else (np.zeros(N_bins, dtype=int), None)
        freq_eff, _   = np.histogram(vals_eff,   bins=bin_edges) if len(vals_eff) > 0   else (np.zeros(N_bins, dtype=int), None)
        freq_fount, _ = np.histogram(vals_fount, bins=bin_edges) if len(vals_fount) > 0 else (np.zeros(N_bins, dtype=int), None)

        stats[col] = {"min": vmin, "max": vmax, "bin_edges": bin_edges, "bin_centers": bin_centers}
        freqs[col] = {"frequency": freq}
        freqs_expl[col] = {"frequency": freq_expl}
        freqs_eff[col] = {"frequency": freq_eff}
        freqs_fount[col] = {"frequency": freq_fount}

    # Preparazione subplot
    n_plots = len(variables_to_plot)
    n_cols = min(4, n_plots)
    n_rows = math.ceil(n_plots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 3*n_rows))
    axes = np.array(axes).flatten() if n_plots > 1 else [axes]

    for i, var in enumerate(variables_to_plot):
        col = var["col"]
        label = var.get("label", None)
        ylim_max = var.get("ylim_max", None)
        xscale = var.get("xscale", None)
        yscale = var.get("yscale", None)

        if col not in stats:
            warnings.warn(f"Colonna '{col}' non trovata nei dizionari, salto questo plot.")
            continue

        ax = axes[i]
        x = stats[col]["bin_centers"]
        y_total = freqs[col]["frequency"]
        y_eff = freqs_eff[col]["frequency"]     if col in freqs_eff else None
        y_fount = freqs_fount[col]["frequency"] if col in freqs_fount else None
        y_expl = freqs_expl[col]["frequency"]   if col in freqs_expl else None

        # Prepara i dati da plottare e la legenda
        plot_data = []
        if y_eff is not None:   plot_data.append((y_eff,   'Effusive', 'ks', 'dodgerblue'))
        if y_fount is not None: plot_data.append((y_fount, 'Fountaining', 'ko', 'lime'))
        if y_expl is not None:  plot_data.append((y_expl,  'Explosive', 'k>', 'red'))
        if plot_total:
            plot_data.append((y_total, 'Total', 'kd', 'gold'))

        for y_vals, label_plot, marker, color in plot_data:
            ax.plot(x, y_vals, marker, markerfacecolor=color, markersize=5, alpha=0.9, label=label_plot)

        if xscale is not None:
            ax.set_xscale(xscale)
        if yscale is not None:
            ax.set_yscale(yscale)

        ax.set_xlim(stats[col]["min"], stats[col]["max"])
        if ylim_max is not None:
            ax.set_ylim(0, ylim_max)
        else:
            ax.set_ylim(0, np.max(y_total) * 1.1)

        if label is None:
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

        ax.grid(True, linestyle='--', alpha=0.3)

        # Legenda solo se ci sono curve
        if i == 0:
            ax.legend(loc='upper right', fontsize=8)

    # Spegni eventuali assi vuoti
    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()

    # Se save_name non è specificato, generalo automaticamente
    if save_name is None:
        if fig_num is None:
            save_name = f"freq_my_plot_lists"
        else:
            save_name = f"freq_my_plot_lists_fig{fig_num}"
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
        plot_frequencies_eruptive_styles=plot_frequencies_eruptive_styles,
        df_concat=df_concat,
        df_concat_expl=df_concat_expl,
        df_concat_notExpl=df_concat_notExpl,
        df_concat_eff=df_concat_eff,
        df_concat_fount=df_concat_fount,
    )
 
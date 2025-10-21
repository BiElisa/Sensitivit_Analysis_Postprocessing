import pandas as pd
import numpy as np
import os
import math
import matplotlib.pyplot as plt
import my_lib_process_utils as utils

def plot_grouped_scatter(
        dfs, 
        group_col, 
        x_col, 
        y_col, 
        n_step=1, 
        save_name=None, 
        save_dir='plot_grouped_scatter'
    ):
    """
    Plot x vs y per uno o più DataFrame, raggruppando i punti in base a group_col
    e collegandoli con linee.

    Parameters
    ----------
    dfs : dict of DataFrame {"label": df} or single DataFrame
        DataFrame con colonne x, y, group.
    x_col : dict
        {"col": str, "transform": func or None, "label": str, "scale": None/'log', "min_val": float or None, "max_val": float or None}
    y_col : dict
        {"col": str, "transform": func or None, "label": str, "scale": None/'log', "min_val": float or None, "max_val": float or None}
    group_col : dict or list of dict
        Colonna da usare per raggruppare i punti e tracciare linee.
        {"col": str, "transform": func or None, "label": str}
    n_step : int
        Passo di campionamento per velocizzare i plot.
    save_name : str, optional
        Nome del file salvato senza estensione. Se None, generato automaticamente.
    save_dir : str
        Cartella dove salvare i plot.
    """

    # --- folder relativo allo script ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir_full = os.path.join(script_dir, save_dir)
    os.makedirs(save_dir_full, exist_ok=True)

    if not isinstance(dfs, dict):
        dfs = {"Simulations": dfs}

    # Se è un singolo gruppo, lo trasformiamo in lista
    if isinstance(group_col, dict):
        group_cols = [group_col]
    elif isinstance(group_col, list):
        group_cols = group_col
    else:
        raise ValueError("group_col deve essere un dict o una lista di dict")
    
    # Funzione interna per estrarre valori con MultiIndex e trasformazioni
    def extract_col_vals(df, col_dict):
        col_name = col_dict["col"]
        if isinstance(df.columns, pd.MultiIndex):
            if col_name in df.columns.get_level_values(0):
                vals = df[col_name].iloc[:,0].to_numpy()
            else:
                vals = df[col_name].to_numpy()
        else:
            vals = df[col_name].to_numpy()
        if col_dict.get("transform"):
            vals = col_dict["transform"](vals)
        return vals

    num_subplots = len(group_cols)
    n_cols = min(2, num_subplots)  # massimo 2 colonne di subplot
    n_rows = math.ceil(num_subplots / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if num_subplots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    colors = ["red", "blue", "limegreen", "orange", "purple", "brown", "gray"]
    markers = ["o", "s", "D", "^", "v", "x"]

    for k, gcol in enumerate(group_cols):
        ax = axes[k]
        for i, (df_label, df) in enumerate(dfs.items()):
            if df.empty:
                continue

            x_vals_full = extract_col_vals(df, x_col)
            y_vals_full = extract_col_vals(df, y_col)
            group_vals  = extract_col_vals(df, gcol)

            df_plot = df.copy()
            df_plot["_x"] = x_vals_full
            df_plot["_y"] = y_vals_full
            df_plot["_group"] = group_vals

            unique_groups = np.sort(df_plot["_group"].unique())
            for j, g in enumerate(unique_groups):
                sub_df = df_plot[df_plot["_group"] == g].sort_values("_x")
                ax.plot(
                    sub_df["_x"][::n_step],
                    sub_df["_y"][::n_step],
                    marker=markers[i % len(markers)],
                    color=colors[i % len(colors)],
                    linestyle='-', 
                    markersize=3,
                    label=f"{df_label} | {gcol.get('label', gcol['col'])}={g}"
                )
        
        ax.set_xlabel(x_col.get("label", x_col["col"]))
        ax.set_ylabel(y_col.get("label", y_col["col"]))
        if x_col.get("scale") == "log":
            ax.set_xscale("log")
        if y_col.get("scale") == "log":
            ax.set_yscale("log")
        if x_col.get("min_val") is not None and x_col.get("max_val") is not None:
            ax.set_xlim(x_col["min_val"], x_col["max_val"])
        if y_col.get("min_val") is not None and y_col.get("max_val") is not None:
            ax.set_ylim(y_col["min_val"], y_col["max_val"])
        ax.grid(True)
        ax.legend(fontsize=7, loc="best")
        ax.set_title(gcol.get("label", gcol["col"]))

    # Spegni eventuali assi vuoti
    for ax in axes[num_subplots:]:
        ax.axis("off")

    plt.tight_layout()

    save_name_full = save_name or f"{y_col['col']}_vs_{x_col['col']}_grouped_subplots"
    save_path = os.path.join(save_dir_full, f"{save_name_full}.svg")
    plt.savefig(save_path)
    plt.show()
    print(f"Figura salvata in {save_path}")



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

    #endregion

    # Cartella in cui salvare i plot di default
    save_dir="plot_grouped_scatter"

    dfs={
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        }
    
    x_col={
            "col": "x2", 
            "transform": lambda v: v-273, 
            "label": "Inlet temperature [°C]", 
        }
    
    y_col={
            "col": "response_fn_4", 
            "transform": np.log10, 
            "label": "log10(MER) [kg/s]", 
        }
    
    plot_grouped_scatter(
        dfs={"All simulations": df_concat},
        x_col=x_col,
        y_col=y_col,
        group_col=[
            {"col": "x1", "transform": lambda v: v/1e6, "label": "Pressure [MPa]"},
            {"col": "x3", "label": "Radius"}
        ],
        n_step=3,
        save_name="prova2"
    )

    




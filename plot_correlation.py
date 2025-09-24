import pandas as pd
import numpy as np
import os
import sys
import shutil
import subprocess
import my_lib_process_utils as utils

if __name__ == '__main__':
    """
    Genera grafici di correlazione.
    """
    #region -- Controlliamo che i dati che ci servono siano presenti, altrimenti li andiamo a costruire

    # File richiesti
    mandatory_file = "dakota_test_parallel.in"
    other_files = [
        "data_allConcat.csv",
        "data_allConcat_explosive.csv",
        "data_allConcat_notExplosive.csv",
        "data_allConcat_notExplosive_effusive.csv",
        "data_allConcat_notExplosive_fountaining.csv",
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

    df_boundsInfo, input_Min, input_Max = utils.import_dakota_bounds()

    #endregion

    N_bins = 25
    stats         = utils.bin_and_average(df_concat, N_bins)
    stats_expl    = utils.bin_and_average(df_concat_expl, N_bins)
    stats_notExpl = utils.bin_and_average(df_concat_notExpl, N_bins)
    stats_eff     = utils.bin_and_average(df_concat_eff, N_bins)
    stats_fount   = utils.bin_and_average(df_concat_fount, N_bins)
    

    # Numero di step per il campionamento
    num_step_campionamento = 3

    # Cartella in cui salvare i plot di default
    save_dir="plot_correlations"

    input('...')

    #region -- Plot di correlazione fra una response_fn e tutti i parametri di input

    utils.plot_xi_vs_response_fn(
        df = df_concat,
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_1',
        y_label='Gas volume fraction',
        n_step=num_step_campionamento, 
        save_name="corr_Gas_volume_fraction",
        save_dir=save_dir,
        fig_num=1,
        stats=stats
    )

    utils.plot_xi_vs_response_fn(
        df = df_concat,
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_15',
        y_label='Fragmentation depth (m)',
        n_step=num_step_campionamento,  
        save_name="corr_Fragmentation_depth",
        save_dir=save_dir,
        fig_num=2,
        stats=stats
    )

    utils.plot_xi_vs_response_fn(
        df = df_concat,
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_12',
        y_label='Mass flow rate (kg/s)',
        n_step=num_step_campionamento,  
        save_name="corr_Mass_flow_rate",
        save_dir=save_dir,
        fig_num=3,
        stats=stats
    )

    utils.plot_xi_vs_response_fn(
        df = df_concat,
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_4',
        y_label='Exit velocity (m/s)',
        n_step=num_step_campionamento,  
        save_name="corr_Exit_velocity",
        save_dir=save_dir,
        fig_num=4,
        stats=stats
    )
    
    utils.plot_xi_vs_response_fn(
        df = df_concat,
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_16',
        #y_label='Exit crystal content (vol.%)',
        n_step=num_step_campionamento,  
        save_name="corr_Exit_crystal_content",
        save_dir=save_dir,
        fig_num=5,
        stats=stats
    )

    #endregion
    
    #region -- plot_correlazioni_con_x4 --
    list_for_x_axis = [
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #1
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #2
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #3
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #4
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #5
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #6
    ]

    list_for_y_axis = [
        ("response_fn_4", None, "Exit velocity [m/s]"), #1
        ("response_fn_15"), #2
        ("response_fn_30", lambda v: v*100, "Diss. H2O content @Frag [wt.%]"), #3
        ("response_fn_25"), #4
        ("response_fn_24"), #5
        ("response_fn_20", np.log10, "Log 10 Viscosity @Frag [Pa s]"), #6
    ]

    utils.plot_lists(
        df=df_concat,
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=3,
        fig_num=7,
        save_name="corr_con_x4",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----

    #region -- plot_correlazioni_con_x5 --
    list_for_x_axis = [
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #1
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #2
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #3
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #4
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #5
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #6
    ]

    list_for_y_axis = [
        ("response_fn_4", None, "Exit velocity [m/s]"), #1
        ("response_fn_15"), #2
        ("response_fn_30", None, "Diss. H2O content @Frag [wt.%]"), #3
        ("response_fn_25"), #4
        ("response_fn_24"), #5
        ("response_fn_20", np.log10, "Log 10 Viscosity @Frag [Pa s]"), #6
    ]

    utils.plot_lists(
        df=df_concat,
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=1,
        fig_num=8,
        save_name="corr_con_x5",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----
    """
    #region -- ESEMPIO DI PLOT VARI CON TRASFORMAZIONI E LABELS --
    list_for_x_axis = [
        ("x1", None, None),
        ("x2", lambda v: v-273, "Temperatura (°C)"),
        ("response_fn_17", None, "Undercooling [K]"),
        ("x1", None, None),
        ("x2", lambda v: v-273, "Temperatura (°C)"),
        ("response_fn_17", None, "Undercooling [K]"),
    ]

    list_for_y_axis = [
        ("response_fn_55", lambda v: v*100, "H₂O Diss. (%)"),
        ("x5", None, None),
        ("response_fn_20", np.log10, "log10(Y)"),
        ("response_fn_55", lambda v: v*100, "H₂O Diss. (%)"),
        ("x5", None, None),
        ("response_fn_20", np.log10, "log10(Y)"),
    ]

    utils.plot_lists(
        df=df_transformed,
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        n_step=1,
        fig_num=30,
        save_name="plot_correlazioni_varie",
        stats=stats
    )
    #endregion
    """





import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        "simulations.csv", 
        "data_at_fragmentation_total.csv",
        "data_at_fragmentation.csv",
        "data_at_inlet_total.csv",
        "data_at_inlet.csv",
        "data_at_vent_total.csv",
        "data_at_vent.csv",
        "data_average_total.csv",
        "data_average.csv"
    ]

    # Cartelle
    cwd = os.getcwd()
    save_dir = os.path.join(cwd, "csv_files")
    os.makedirs(save_dir, exist_ok=True)

    # --- Controllo mandatory ---
    if not os.path.isfile(os.path.join(cwd, mandatory_file)):
        print(f"Abort: The file '{mandatory_file}' is not in the folder {cwd}.")
        sys.exit(1)

    # --- Controllo altri file ---
    missing = []
    for f in other_files:
        path_current = os.path.join(cwd, f)
        path_csv = os.path.join(save_dir, f)

        if os.path.isfile(path_current):
            # Se il file è nella cartella corrente, spostalo in csv_files
            shutil.move(path_current, path_csv)
            print(f"Moved '{f}' from current folder to '{save_dir}'.")
        elif not os.path.isfile(path_csv):
            # Mancante ovunque
            missing.append(f)

    # --- Se mancano file, run extract_allData ---
    if missing:
        print(f"The following files are missing from both current folder and '{save_dir}': {missing}")
        print("The script 'extract_allData.py' is executed.")
        subprocess.run(["python", "extract_allData.py", "--pause", "false"])
    

    #endregion

    #region -- Carichiamo tutti i file e dati che ci servono

    read_csv_kwargs = {"header": [0,1], "encoding": "utf-8"}

    save_dir = "csv_files"  

    df_dakota_output = pd.read_csv(os.path.join(save_dir,"simulations.csv"), **read_csv_kwargs)

    df_fragmentation = pd.read_csv(os.path.join(save_dir,"data_at_fragmentation.csv"), **read_csv_kwargs)

    df_inlet = pd.read_csv(os.path.join(save_dir,"data_at_inlet.csv"), **read_csv_kwargs)

    df_vent = pd.read_csv(os.path.join(save_dir,"data_at_vent.csv"), **read_csv_kwargs)

    df_average = pd.read_csv(os.path.join(save_dir,"data_average.csv"), **read_csv_kwargs)

    df_concat = pd.concat([df_dakota_output, df_fragmentation, df_inlet, df_vent, df_average], axis=1)

    df_transformed = utils.transform_units_of_variables(df_concat)
    #print(df_transformed)
    df_transformed.to_csv(os.path.join(save_dir,'data_allConcat_unitsTransformed.csv'), index=False)

    #endregion

    df_boundsInfo, input_Min, input_Max = utils.import_dakota_bounds()

    # Applica le stesse trasformazioni delle colonne di df_transformed
    adj_input_Min, adj_input_Max = utils.adjust_bounds_with_units(input_Min, input_Max, df_concat)
    print(f'\nadj_input_Min = {adj_input_Min}')
    print(f'adj_input_Max = {adj_input_Max}\n')

    #region -- Plot di correlazione fra una response_fn e tutti i parametri di input

    # Numero di step per il campionamento
    num_step_campionamento = 3

    utils.plot_xi_vs_response_fn(
        df = df_transformed,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        response_col='response_fn_1',
        y_label='Gas volume fraction',
        n_step=num_step_campionamento, 
        save_name="plot_correlazione_Gas_volume_fraction",
        fig_num=1
    )

    utils.plot_xi_vs_response_fn(
        df = df_transformed,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        response_col='response_fn_15',
        y_label='Fragmentation depth (m)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Fragmentation_depth",
        fig_num=2
    )

    utils.plot_xi_vs_response_fn(
        df = df_transformed,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        response_col='response_fn_12',
        y_label='Mass flow rate (kg/s)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Mass_flow_rate",
        fig_num=3
    )

    utils.plot_xi_vs_response_fn(
        df = df_transformed,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        response_col='response_fn_4',
        y_label='Exit velocity (m/s)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Exit_velocity",
        fig_num=4
    )

    utils.plot_xi_vs_response_fn(
        df = df_transformed,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        response_col='response_fn_16',
        y_label='Exit crystal content (vol.%)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Exit_crystal_content",
        fig_num=5
    )

    #endregion

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
        fig_num=6,
        save_name="plot_correlazioni_varie"
    )

    plt.show()

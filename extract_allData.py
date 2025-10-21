import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import sys
import argparse
import my_lib_extract_data as extract
import my_lib_process_utils as utils

def extract_allData (verbose = True, pause = True, save_all_files=False):

    print("\n\n ---- Start the execution of extract_allData ----\n")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    if verbose:
        print(f"The current script is launched from: \n{script_dir}\n")

    #Creiamo la cartella "csv_files" dentro la cartella dello script  
    csv_dir = os.path.join(script_dir, "csv_files")
    os.makedirs(csv_dir, exist_ok=True)

    #region -- Importiamo il tabellone generato da Dakota 

    # Importa i dati tabulari di Dakota per determinare N e memorizzare {xi} e {response_fn_i}
    df_dakota_output = utils.import_dakota_tabular_new()
    if df_dakota_output is None:
        sys.exit()

    number_tot_sim = len(df_dakota_output)
    if verbose:
        print(f"Totale simulazioni trovate: {number_tot_sim}.\n")

    #endregion

    #region -- Aggiunta di nuove colonne al tabellone

    # Total crystal content
    df_dakota_output["response_fn_16"] = (
        df_dakota_output["response_fn_7"] 
        + df_dakota_output["response_fn_8"] 
        + df_dakota_output["response_fn_9"]
    )

    plag_liquidus = -0.64 * (df_dakota_output["response_fn_2"] / 1e6) + 1132 + 273

    # Undercooling
    df_dakota_output["response_fn_17"] = plag_liquidus - df_dakota_output["response_fn_6"] 

    df_dakota_output["response_fn_18"] = df_dakota_output["response_fn_14"] # EB : da eliminare successivamente

    df_dakota_output["response_fn_19"] = np.where(
        df_dakota_output["response_fn_15"] > -0.5,
        236.0 * (df_dakota_output["response_fn_12"] ** 0.25),
        (df_dakota_output["response_fn_4"] ** 2.0) / (2.0 * 9.8)
    )
    #endregion

    #region -- Aggiungiamo una seconda riga allo header del tabellone per avere labels leggibili 
        
    # Trova tutte le colonne che iniziano con 'x'
    xi_cols = [col for col in df_dakota_output.columns if col.startswith('x')]
    n_xi = len(xi_cols)
    if verbose:
        print(f"Trovate {n_xi} variabili di input: {xi_cols}\n")
        
    templateFile_path = os.path.join(script_dir, "conduit_solver.template")

    if not os.path.exists(templateFile_path):
        print("⚠️  Warning: file 'conduit_solver.template' non trovato.")
        print("   Selezionalo manualmente...\n")

        # Apri finestra di dialogo per cercarlo
        root = tk.Tk()
        root.withdraw()
        templateFile_path = filedialog.askopenfilename(
            title="Seleziona il file 'conduit_solver.template'",
            filetypes=[("Template file", "*.template")]
        )

        if not templateFile_path:
            print("❌ Nessun file selezionato. Operazione annullata.\n")
            sys.exit(1)

    else:
        if verbose: print(f"Trovato il file template in: \n{templateFile_path}\n")

    xi_labels = utils.get_xi_labels_from_template(df_dakota_output, templateFile_path)
    if verbose:
        print(f"Assegnate le labels descrittive ai parametri di input: \n{xi_labels}\n")

    response_labels = {
        'response_fn_1': 'Total Gas volume fraction',
        'response_fn_2': 'Pressure 1 [Pa]',
        'response_fn_3': 'Pressure 2 [Pa]',
        'response_fn_4': 'Liquid/particles velocity [m/s]',
        'response_fn_5': 'Gas velocity [m/s]',
        'response_fn_6': 'Exit temperature [K]',
        'response_fn_7': 'Plagioclase crystallinity [-]',
        'response_fn_8': 'Pyroxene crystallinity [-]',
        'response_fn_9': 'Olivine crystallinity [-]',
        'response_fn_10': 'Mach number ',
        'response_fn_11': 'Exit mixture velocity [m/s]',
        'response_fn_12': 'Mass flow rate [kg/s]',
        'response_fn_13': 'Volume flow rate [m3/s]',
        'response_fn_14': 'Fragmentation Code',
        'response_fn_15': 'Fragmentation depth [m]',
        'response_fn_16': 'Total crystal content [vol.]',
        'response_fn_17': 'Undercooling relative to plag [K]',
        'response_fn_18': 'Viscosity at fragmentation [Pa·s]',
        'response_fn_19': 'Fragmentation length scale [m]',
    }

    # Costruisci la riga di etichette
    labels = []
    for col in df_dakota_output.columns:
        if col in xi_labels:
            labels.append(str(xi_labels[col]))
        elif col in response_labels:
            labels.append(str(response_labels[col]))
        else:
            labels.append(str(col))

    # Costruisci DataFrame con doppia riga di header
    df_dakota_output.columns = pd.MultiIndex.from_arrays(
        [df_dakota_output.columns, labels],
        names=["variable", "description"]
    )
    if verbose >1: print(f"df_dakota_output = \n{df_dakota_output}")

    #endregion

    #region -- Recuperiamo il nome dei bak_file ed il percorso delle cartelle workdir

    print("Serve il percorso alle cartelle workdir.")
    print("Seleziona manualmente il 'bak' file di una delle workdir...\n")

    # Apri una finestra di dialogo per selezionare un file .bak
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Seleziona un file '.bak' da una cartella workdir.N",
        filetypes=[("Bak files", "*.bak")],
        initialdir=script_dir # Opens the dialogue window in the script folder
    )

    if not filepath:
        print("❌ Nessun file selezionato. Operazione annullata.\n")
        return

    # Trova la cartella principale risalendo dal percorso del file selezionato
    # Ad esempio, da '.../workdir.1/file.bak' si ottiene '...'
    main_dir = os.path.dirname(os.path.dirname(filepath))

    # Estrai il nome del file di riferimento
    bak_name = os.path.basename(filepath)
        
    if verbose:
        print(f"Cartella principale delle 'workdir' identificata: \n{main_dir}\n")
        print(f"Nome del file '.bak' di riferimento: {bak_name}\n")

    #endregion

    #region -- Estraz. dati @ framm./inlet/vent/average. e concatenazione col tabellone in "data_allConcat.csv"

    filename_simulations = "data_allConcat"
    filename_at_frag  = "data_at_fragmentation"
    filename_at_inlet = "data_at_inlet"
    filename_at_vent  = "data_at_vent"
    filename_average  = "data_average"

    if pause: 
        input("Cerchiamo oppure creiamo le informazioni alla frammentazione...")
    df_fragmentation = extract.extract_data_at_frag(main_dir, bak_name, number_tot_sim, csv_dir, filename_at_frag)
    if verbose >1: print(f"df_fragmentation=\n{df_fragmentation}")
    if save_all_files:
        df_fragmentation.to_csv(os.path.join(csv_dir,filename_at_frag+'_total.csv'), index=False, encoding="utf-8", float_format="%.6f")

    if pause:
        input("Cerchiamo oppure creiamo le informazioni all'inlet...")
    df_inlet = extract.extract_data_at_inlet(main_dir, bak_name, number_tot_sim, csv_dir, filename_at_inlet)
    if verbose >1: print(f"df_inlet=\n{df_inlet}")
    if save_all_files:
        df_inlet.to_csv(os.path.join(csv_dir,filename_at_inlet+'_total.csv'), index=False, encoding="utf-8", float_format="%.6f")

    if pause:
        input("Cerchiamo oppure creiamo le informazioni al vent...")
    df_vent = extract.extract_data_at_vent(main_dir, bak_name, number_tot_sim, csv_dir, filename_at_vent)
    if verbose >1: print(f"df_vent=\n{df_vent}")
    if save_all_files:
        df_vent.to_csv(os.path.join(csv_dir,filename_at_vent+'_total.csv'), index=False, encoding="utf-8", float_format="%.6f")

    if pause:
        input("Cerchiamo oppure creiamo le informazioni average...")
    df_average = extract.extract_data_average(main_dir, bak_name, number_tot_sim, csv_dir, filename_average)
    if verbose >1: print(f"df_average=\n{df_average}")
    if save_all_files:
        df_average.to_csv(os.path.join(csv_dir,filename_average+'_total.csv'), index=False, encoding="utf-8", float_format="%.6f")

    df_concat = pd.concat([df_dakota_output, df_fragmentation, df_inlet, df_vent, df_average], axis=1) 
    if verbose >1: print(f"df_concat=\n{df_concat}")

    #endregion

    #region -- Salvataggio del tabellone cosi` composto come CSV file

    if save_all_files:
        df_concat.to_csv(os.path.join(csv_dir,'data_allConcat_total.csv'), index=False, encoding="utf-8", float_format="%.6f")
        if verbose:
            print(f"\nI dati ottenuti vengono concatenati e salvati in 'data_allConcat_total.csv' in:\n{csv_dir}\n")

    #endregion

    #region -- Puliamo il tabellone generato da dakota dalle simulazioni nulle 

    mask_invalid = (
        (df_concat[("response_fn_1", "Total Gas volume fraction")] == -1) | 
        (df_concat[("response_fn_12", "Mass flow rate [kg/s]")] > 5e9)
    )

    df_concat_clean = df_concat.loc[~mask_invalid].reset_index(drop=True)
    df_concat_clean.to_csv(os.path.join(csv_dir,'data_allConcat.csv'), index=False, encoding="utf-8", float_format="%.6f")

    print(f"\nEliminate {mask_invalid.sum()} eliminate perche` nulle. \nSalvate le {number_tot_sim - mask_invalid.sum()} simulazioni valide come 'data_allConcat.csv' in:\n{csv_dir}\n")

    #endregion

    #region -- Selezioniamo solo le simulazioni esplosive

    mask_explosive = df_concat_clean[("response_fn_15", "Fragmentation depth [m]")] > -0.5

    df_concat_clean_expl  = df_concat_clean[mask_explosive]

    suffix = "_explosive.csv"

    df_concat_clean_expl.to_csv(os.path.join(csv_dir, filename_simulations + suffix), index=False, encoding="utf-8", float_format="%.6f")

    print(f"Estratte le {mask_explosive.sum()} simulazioni esplosive e salvate come '{filename_simulations}{suffix}' in:\n{csv_dir}\n")

    #endregion

    #region -- Selezioniamo solo le simulazioni NON esplosive

    mask_notExplosive = (df_concat_clean[("response_fn_15", "Fragmentation depth [m]")] < 0) | (df_concat_clean[("response_fn_1", "Total Gas volume fraction")] < 0.6)

    df_concat_clean_notExpl  = df_concat_clean[mask_notExplosive]

    suffix = "_notExplosive.csv"

    df_concat_clean_notExpl.to_csv(os.path.join(csv_dir, filename_simulations + suffix), index=False, encoding="utf-8", float_format="%.6f")

    print(f"Estratte le {mask_notExplosive.sum()} simulazioni non esplosive e salvate come '{filename_simulations}{suffix}' in:\n{csv_dir}\n ")

    #endregion

    #region -- Selezioniamo solo le simulazioni effusive (fra le NON esplosive)

    mask_effusive = df_concat_clean_notExpl[("response_fn_19", "Fragmentation length scale [m]")] <= 0.1

    df_concat_clean_eff  = df_concat_clean_notExpl[mask_effusive]

    suffix = "_notExplosive_effusive.csv"

    df_concat_clean_eff.to_csv(os.path.join(csv_dir, filename_simulations + suffix), index=False, encoding="utf-8", float_format="%.6f")

    print(f"Estratte le {mask_effusive.sum()} simulazioni effusive e salvate come '{filename_simulations}{suffix}' in:\n{csv_dir}\n")

    #endregion

    #region -- Selezioniamo solo le simulazioni fontanamento (fra le NON esplosive)

    mask_fountaining = df_concat_clean_notExpl[("response_fn_19", "Fragmentation length scale [m]")] > 0.1

    df_concat_clean_fount  = df_concat_clean_notExpl[mask_fountaining]

    suffix = "_notExplosive_fountaining.csv"

    df_concat_clean_fount.to_csv(os.path.join(csv_dir, filename_simulations + suffix), index=False, encoding="utf-8", float_format="%.6f")

    print(f"Estratte le {mask_fountaining.sum()} simulazioni di fontanamento e salvate come '{filename_simulations}{suffix}' in:\n{csv_dir}\n  ")

    #endregion


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Esempio con booleani verbose e pause")
    
    parser.add_argument(
        "--verbose",
        type=int,
        choices=range(0, 2),  # consente solo 0, 1, 2, 3
        default=1,
        help=(
            "Livello di verbosità: "
            "0 = silenzioso, "
            "1 = base, "
            "2 = dettagliato, "
#            "3 = debug (default: 1)"
        )
    )
    parser.add_argument(
        "--pause", 
        type=lambda v: v.lower() in ("true", "1", "yes"), 
        default=True, 
        help="Pausa all'esecuzione (default True)"
    )
    parser.add_argument(
        "--save_all_files", 
        type=lambda v: v.lower() in ("true", "1", "yes"), 
        default=False, 
        help="Salva tutti i file CSV intermedi ai calcoli (default False)"
    )
    
    args = parser.parse_args()
    
    extract_allData(args.verbose, args.pause, args.save_all_files)





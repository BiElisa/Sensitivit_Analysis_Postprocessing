import os
import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import sys
import argparse
import my_lib_extract_data as extract
import my_lib_process_utils as utils
import my_lib_remove_simulations_from_csv as rm

def extract_allData (verbose = True, pause = True):

    print("\n\n ---- Start the execution of extract_allData ----\n")

    # Apri una finestra di dialogo per selezionare un file .bak
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Seleziona un file '.bak' da una cartella workdir.N",
        filetypes=[("Bak files", "*.bak")]
    )


    if not filepath:
        print("Nessun file selezionato. Operazione annullata.")
    else:

        #region -- Importiamo il tabellone generato da Dakota --
        # Trova la cartella principale risalendo dal percorso del file selezionato
        # Ad esempio, da '.../workdir.1/file.bak' si ottiene '...'
        main_dir = os.path.dirname(os.path.dirname(filepath))

        # Estrai il nome del file di riferimento
        bak_name = os.path.basename(filepath)
        
        if verbose:
            print(f"Cartella principale identificata: {main_dir}")
            print(f"Nome del file '.bak' di riferimento: {bak_name}")

        # Importa i dati tabulari di Dakota per determinare N e memorizzare {xi} e {response_fn_i}
        df_dakota_output = utils.import_dakota_tabular_new()
        if df_dakota_output is None:
            sys.exit()

        N = len(df_dakota_output)
        if verbose:
            print(f"Totale simulazioni trovate {N}.")

        #endregion

        # -- Puliamo il tabellone generato da dakota dalle simulazioni nulle --
        if verbose:
            print("Andiamo ad eliminare le simulazioni nulle.")
        df_dakota_clean, number_null_sim = rm.remove_null_simulations(df_dakota_output, "simulations.csv")

        #region -- Aggiunta di nuove colonne a "simulations.csv--

        # Total crystal content
        df_dakota_clean["response_fn_16"] = (
            df_dakota_clean["response_fn_7"] 
            + df_dakota_clean["response_fn_8"] 
            + df_dakota_clean["response_fn_9"]
        )

        plag_liquidus = -0.64 * (df_dakota_clean["response_fn_2"] / 1e6) + 1132 + 273

        # Undercooling
        df_dakota_clean["response_fn_17"] = plag_liquidus - df_dakota_clean["response_fn_6"] 

        df_dakota_clean["response_fn_18"] = df_dakota_clean["response_fn_14"] # EB : da eliminare successivamente

        df_dakota_clean["response_fn_19"] = np.where(
            df_dakota_clean["response_fn_15"] > -0.5,
            236.0 * (df_dakota_clean["response_fn_12"] ** 0.25),
            (df_dakota_clean["response_fn_4"] ** 2.0) / (2.0 * 9.8)
        )
        #endregion

        #region -- Aggiungiamo una seconda riga al tabellone per avere labels leggibili --
        
        # Trova tutte le colonne che iniziano con 'x'
        if verbose:
            xi_cols = [col for col in df_dakota_output.columns if col.startswith('x')]
            n_xi = len(xi_cols)
            if verbose:
                print(f"Trovate {n_xi} variabili di input xi: {xi_cols}")

        if not os.path.exists("conduit_solver.template"):
            print("Abort: The file 'conduit_solver.template' is not in the folder.")
            sys.exit(1)

        xi_labels = utils.get_xi_labels_from_template(df_dakota_output, "conduit_solver.template",)
        if verbose:
            print(f"xi_labels = \n{xi_labels}")

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

        # Costruisci seconda riga di etichette
        labels_row = []
        for col in df_dakota_clean.columns:
            if col in xi_labels:
                labels_row.append(str(xi_labels[col]))
            elif col in response_labels:
                labels_row.append(str(response_labels[col]))
            else:
                labels_row.append(str(col))

        # -- Salvataggio automatico CSV ---
        csv_dir = "csv_files"
        os.makedirs(csv_dir, exist_ok=True)

        filename_simulations = "simulations.csv"
        save_path = os.path.join(csv_dir, filename_simulations)

        # Salva con doppia intestazione
        with open(save_path, "w", encoding="utf-8", newline='') as f:
            f.write(",".join(df_dakota_clean.columns) + "\n")
            f.write(",".join(labels_row) + "\n")
            df_dakota_clean.to_csv(f, index=False, encoding="utf-8") #  header=False,

        #endregion
        
        if verbose:
            print(f"I dati delle simulazioni avvenute con successo vengono salvati in {save_path},\ndove ogni colonna ha doppia intestazione.")
            print("Passiamo ad estrapolare maggiori informazioni da questi dati.")

        # Chiama le funzioni di estrazione, passando il percorso della cartella principale

        read_csv_kwargs = {"header": [0,1], "encoding": "utf-8"}

        df_dakota_output = pd.read_csv(os.path.join(csv_dir, filename_simulations), **read_csv_kwargs)

        filename_at_frag = "data_at_fragmentation"
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni alla frammentazione...")
        extract.extract_data_at_frag(main_dir, bak_name, N, csv_dir, filename_at_frag)
        df_fragmentation = pd.read_csv(os.path.join(csv_dir, filename_at_frag + ".csv"), **read_csv_kwargs)
        
        filename_at_inlet = "data_at_inlet"
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni all'inlet...")
        extract.extract_data_at_inlet(main_dir, bak_name, N, csv_dir, filename_at_inlet)
        df_inlet = pd.read_csv(os.path.join(csv_dir, filename_at_inlet + ".csv"), **read_csv_kwargs)
        
        filename_at_vent = "data_at_vent"
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni al vent...")
        extract.extract_data_at_vent(main_dir, bak_name, N, csv_dir, filename_at_vent)
        df_vent = pd.read_csv(os.path.join(csv_dir, filename_at_vent + ".csv"), **read_csv_kwargs)

        filename_average = "data_average"
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni averaged...")
        extract.extract_data_average(main_dir, bak_name, N, csv_dir, filename_average)
        df_average = pd.read_csv(os.path.join(csv_dir,filename_average + ".csv"), **read_csv_kwargs)


        df_concat = pd.concat([df_dakota_output, df_fragmentation, df_inlet, df_vent, df_average], axis=1)

        #print(df_concat)
        df_concat.to_csv(os.path.join(csv_dir,'data_allConcat.csv'), index=False)

        if verbose:
            print(f"\nI dati estratti fin'ora vengono tutti concatenati in 'data_allConcat.csv',\nun data frame dove ogni colonna ha doppia intestazione.")




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Esempio con booleani verbose e pause")
    
    parser.add_argument(
        "--verbose", 
        type=lambda v: v.lower() in ("true", "1", "yes"), 
        default=True, 
        help="Attiva modalità verbose (default True)"
    )
    parser.add_argument(
        "--pause", 
        type=lambda v: v.lower() in ("true", "1", "yes"), 
        default=True, 
        help="Pausa all'esecuzione (default True)"
    )
    
    args = parser.parse_args()
    
    extract_allData(args.verbose, args.pause)





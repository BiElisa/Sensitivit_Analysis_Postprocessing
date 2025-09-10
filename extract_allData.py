import os
import tkinter as tk
from tkinter import filedialog
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

        #region -- Importiamo il tabellne generato da Dakota --
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
        xi_cols = [col for col in df_dakota_output.columns if col.startswith('x')]
        n_xi = len(xi_cols)
        if verbose:
            print(f"Trovate {n_xi} variabili di input xi: {xi_cols}")

        # Converti dinamicamente in array numpy
        xi_arrays = {col: df_dakota_output[col].to_numpy() for col in xi_cols}

        xi_labels, xi_transforms, input_Min, input_Max = utils.get_xi_labels_from_template(df_dakota_output, "conduit_solver.template")
        #print(xi_labels)

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
            "response_fn_16": "Total crystal content [vol.%]",
            "response_fn_17": "Undercooling relative to plag [K]",
            "response_fn_18": "Viscosity at fragmentation [Pa·s]",
            "response_fn_19": "Fragmentation length scale [m]",
        }

        # Costruisci seconda riga di etichette
        labels_row = []
        for col in df_dakota_output.columns:
            if col in xi_labels:
                labels_row.append(xi_labels[col])
            elif col in response_labels:
                labels_row.append(response_labels[col])
            else:
                labels_row.append(col)

        # Salva con doppia intestazione
        filename_with_labels = "simulations.csv"
        with open(filename_with_labels, "w") as f:
            f.write(",".join(df_dakota_clean.columns) + "\n")
            f.write(",".join(labels_row) + "\n")
            df_dakota_clean.to_csv(f, index=False, header=False)

        #endregion
        
        if verbose:
            print(f"I dati delle simulazioni avvenute con successo vengono salvati in {filename_with_labels}, dove ogni colonna ha doppia intestazione.")
            print("Passiamo ad estrapolare maggiori informazioni da questi dati.")

        # Chiama le funzioni di estrazione, passando il percorso della cartella principale
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni alla frammentazione...")
        extract.extract_data_at_frag(main_dir, bak_name, N)
        
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni all'inlet...")
        extract.extract_data_at_inlet(main_dir, bak_name, N)
        
        if pause:
            input("\nCerchiamo oppure creiamo le informazioni al vent...")
        extract.extract_data_at_vent(main_dir, bak_name, N)

        if pause:
            input("\nCerchiamo oppure creiamo le informazioni averaged...")
        extract.extract_data_average(main_dir, bak_name, N)




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





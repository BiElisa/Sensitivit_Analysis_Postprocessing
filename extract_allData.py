import os
import tkinter as tk
from tkinter import filedialog
import my_lib_extract_data
import sys
import my_lib_process_utils 


if __name__ == '__main__':
    # Apri una finestra di dialogo per selezionare un file .bak
    root = tk.Tk()
    root.withdraw()
    filepath = filedialog.askopenfilename(
        title="Seleziona un file .bak da una cartella workdir.N",
        filetypes=[("Bak files", "*.bak")]
    )

    if not filepath:
        print("Nessun file selezionato. Operazione annullata.")
    else:
        # Trova la cartella principale risalendo dal percorso del file selezionato
        # Ad esempio, da '.../workdir.1/file.bak' si ottiene '...'
        main_dir = os.path.dirname(os.path.dirname(filepath))

        # Estrai il nome del file di riferimento
        bak_name = os.path.basename(filepath)
        
        print(f"Cartella principale identificata: {main_dir}")
        print(f"Nome del file .bak di riferimento: {bak_name}")

        # Importa i dati tabulari di Dakota per determinare N e memorizzare {xi} e {response_fn_i}
        df_dakota = my_lib_process_utils.import_dakota_tabular_new()
        if df_dakota is None:
            sys.exit()

        N = len(df_dakota)
        print(f"Trovate {N} simulazioni da analizzare.")

        # Chiama le funzioni di estrazione, passando il percorso della cartella principale
        my_lib_extract_data.extract_data_at_frag(main_dir, bak_name, N)
        my_lib_extract_data.extract_data_at_inlet(main_dir, bak_name, N)
        my_lib_extract_data.extract_data_at_vent(main_dir, bak_name, N)
        my_lib_extract_data.extract_data_average(main_dir, bak_name, N)

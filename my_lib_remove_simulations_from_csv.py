import pandas as pd
import os
import my_lib_process_utils as utils

def remove_null_simulations(total_input, output_filename):
    """
    Rimuove le simulazioni nulle dal dataset 'total_filename'.
    Condizioni di rimozione:
      - response_fn_1 == -1
      - response_fn_12 > 5e9

    Args:
        total_input (str): può essere un path (str) o un DataFrame già caricato da pulire.
        output_filename (str): nome del file CSV pulito da salvare.

    Returns:
        pd.DataFrame: DataFrame filtrato
        int: numero di simulazioni eliminate
    """

    # 1. Carica o legge il CSV totale
    if isinstance(total_input, str):
        if not os.path.exists(total_input):
            raise FileNotFoundError(f"Il file '{total_input}' non esiste. Devi crearlo.")
        df = utils.read_csv_with_labels(total_input)
    else:
        df = total_input.copy()  # se è già un DataFrame

    # 2. Leggi dakota_tabular_dat
    df_dakota = utils.import_dakota_tabular_new()
    if df_dakota is None:
        raise RuntimeError("Errore nella lettura di dakota_tabular_dat")

    # 3. Maschera delle righe nulle
    mask_invalid = (df_dakota["response_fn_1"] == -1) | (df_dakota["response_fn_12"] > 5e9)
    number_null_sim = mask_invalid.sum()

    # 4. Applica maschera
    df_clean = df.loc[~mask_invalid].reset_index(drop=True)

    # 5. Salva CSV con dati puliti
        # 5. Salva CSV con dati puliti **e etichette leggibili**
    # Leggiamo la seconda riga originale
    if isinstance(total_input, str):
        with open(total_input, "r") as f:
            f.readline()  # skip prima riga
            second_line = f.readline().strip()
        with open(output_filename, "w") as f_out:
            f_out.write(",".join(df.columns) + "\n")  # header tecnico
            f_out.write(second_line + "\n")           # etichette leggibili
            df_clean.to_csv(f_out, index=False, header=False)
    else:
        df_clean.to_csv(output_filename, index=False)
    print(f"Salvato '{output_filename}' ({len(df_clean)} simulazioni valide, {number_null_sim} eliminate).")

    return df_clean, number_null_sim


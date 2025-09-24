import pandas as pd
import os
import my_lib_process_utils as utils

def remove_null_simulations(input_filename, output_filename, save_dir="csv_files"):
    """
    Rimuove le simulazioni nulle dal dataset 'input_filename'.
    Condizioni di rimozione:
      - response_fn_1 == -1
      - response_fn_12 > 5e9

    Args:
        input_filename (str | pd.DataFrame): può essere un path (str) o un DataFrame già caricato da pulire.
        output_filename (str): nome del file CSV pulito da salvare.
        save_dir (str): directory in cui leggere/salvare i file (default "csv_files").

    Returns:
        pd.DataFrame: DataFrame filtrato
        int: numero di simulazioni eliminate
    """

    os.makedirs(save_dir, exist_ok=True)

    # 1. Carica o legge il CSV totale
    if isinstance(input_filename, str):
        input_path = os.path.join(save_dir, input_filename)
        output_path = os.path.join(save_dir, output_filename)

        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Il file '{save_dir}/{input_filename}' non esiste. Devi crearlo.")
        df = utils.read_csv_with_labels(input_path)
    else:
        df = input_filename.copy()  # se è già un DataFrame
        output_path = os.path.join(save_dir, output_filename)

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
    # Leggiamo la seconda riga originale
    if isinstance(input_filename, str):
        with open(input_path, "r", encoding="utf-8") as f:
            f.readline()  # skip prima riga
            second_line = f.readline().strip()

        with open(output_path, "w", encoding="utf-8") as f_out:
            f_out.write(",".join(df.columns) + "\n")  # header tecnico
            f_out.write(second_line + "\n")           # etichette leggibili
            df_clean.to_csv(f_out, index=False, header=False)
    else:
        df_clean.to_csv(output_path, index=False)
    print(f"Salvato '{output_path}' ({len(df_clean)} simulazioni valide, {number_null_sim} eliminate perche` nulle).\n")

    return df_clean, number_null_sim


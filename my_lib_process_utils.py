import os  # standard library to interact with the operative system (paths, directories, etc)
import re  # standard library for regular expressions
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

def read_bak(pathname, filename):
    """
    Legge un file di tipo .bak e ne estrae i parametri, salvandoli in un dizionario.
    
    Args:
        pathname (str): Il percorso della cartella del file.
        filename (str): Il nome del file da leggere.
    
    Returns:
        dict: Un dizionario contenente i parametri estratti e i loro valori.
    """
    
    variables = {}
    filepath = os.path.join(pathname, filename)
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Errore: Il file {filepath} non è stato trovato.")
        return None

    # Sostituisce i segni di uguale per facilitare il parsing e gestisce le stringhe tra virgolette
    content = content.replace('=', ' = ')
    #content = re.sub(r'(\w+)\s*=\s*(".*?")', r'\1 = \2', content)
    #content = re.sub(r'(\w+)\s*=\s*(\d+\.?\d*[Ee]?[+-]?\d*)', r'\1 = \2', content)

    # Usa un'espressione regolare per trovare tutte le coppie chiave-valore
    # Questa regex cerca una parola (la chiave) seguita da ' = ' e da un valore.
    # Il valore può essere un numero (con notazione scientifica D o E), una stringa, o una lista
    # La parte `|` gestisce la tua logica di assegnazione `X, Y, Z`
    matches = re.findall(r'(\w+)\s*=\s*(".*?"|\'.*?\'|[+-]?\d*\.?\d*[EeDd]?[+-]?\d*|[+-]?\d*\.?\d*[EeDd]?[+-]?\d*\s*,\s*[+-]?\d*\.?\d*[EeDd]?[+-]?\d*)', content)

    for key, value in matches:
        key = key.strip().lower()
        value = value.strip().replace('"', '').replace("'", "")
        
        # Gestisce i casi con multiple assegnazioni sulla stessa riga, come 'X, Y, Z'
        if ',' in value:
            val_list = [v.strip() for v in value.split(',')]
            val_converted = []
            for v in val_list:
                try:
                    val_converted.append(float(v.replace('D', 'e')))
                except ValueError:
                    val_converted.append(v)
            variables[key] = val_converted
        else:
            try:
                # Prova a convertire il valore in un numero (intero o float)
                if 'D' in value or 'E' in value:
                    value = float(value.replace('D', 'e'))
                else:
                    value = int(value) if value.isdigit() else float(value)
                variables[key] = value
            except ValueError:
                # Se la conversione fallisce, mantiene il valore come stringa
                variables[key] = value.strip() # e strip() rimuove gli spazi che seguono

    return variables

def import_dakota_tabular_new(filename='dakota_tabular.dat'):
    """
    Importa dati dal file dakota_tabular.dat utilizzando pandas.
    
    Args:
        filename (str): Il nome del file da importare.
    
    Returns:
        df := pandas.DataFrame: Il DataFrame contenente i dati del file.
    """
    if not os.path.exists(filename):
        print(f"Errore: Il file '{filename}' non è stato trovato all'interno della cartella corrente.")
        return None
    
    # Legge il file, saltando la prima riga di intestazione.
    # Questo approccio è più robusto e garantisce che vengano lette tutte le righe.
    df = pd.read_csv(filename, sep='\s+', skiprows=1, header=None)

    # Legge la riga di intestazione separatamente per assegnare i nomi delle colonne
    with open(filename, 'r') as f:
        header_line = f.readline().strip()

    # Pulisce la riga di intestazione dai caratteri di commento e la divide
    column_names = header_line.replace('%', '').split()

    # Assegna i nomi delle colonne al DataFrame
    df.columns = column_names

    # Rinomina la prima colonna per coerenza
    df.rename(columns={'%eval_id': 'eval_id'}, inplace=True)

    #print(f'df = {df}')
    
    return df

def search_and_read_std_file(main_dir, bak_name, index_simul):
    """
    Cerca il file .std per una data simulazione e se lo trova lo legge.

    Args:
        main_dir (str): Percorso alla directory principale.
        index_simul (int): Indice della simulazione (da 0 a N-1).
        bak_name (str): Nome del file .bak di riferimento.

    Returns:
        tuple: Una tupla contenente (std_data, bak_data) se i file sono letti con successo,
               altrimenti (None, None).
    """
    
    # Costruisci il percorso alla cartella di lavoro
    workdir_path = os.path.join(main_dir, f'workdir.{index_simul + 1}')

    # Andiamo a cercare il file .std
    std_filename = bak_name.replace('.bak', '_p.std')
    std_filepath = os.path.join(workdir_path, std_filename)

    # Se il file .std non c'e` allora terminiamo la procedura
    if not os.path.exists(std_filepath) :
        return None, None

    try:
        # Leggiamo il file .bak e memoriziamo i parametri ivi contenuti
        bak_path = os.path.join(workdir_path, bak_name)
        bak_data = read_bak(os.path.dirname(bak_path), os.path.basename(bak_path))
        
        # Apri il file per leggere riga per riga e gestire la struttura specifica
        with open(std_filepath, 'r') as f:
            
            # Legge il raggio dalla prima riga
            radius_line = f.readline()
            radius = float(radius_line.strip()) # <-- vedi se spostarlo altrove
            
            # Legge il numero di celle dalla seconda riga
            comp_cells_line = f.readline()
            comp_cells = int(comp_cells_line.strip())
            
            # Legge il resto del file come un'unica stringa
            rest_of_file = f.read()

        # Estrae tutti i numeri (interi e float) dalla stringa usando una regex
        all_numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eEdD][-+]?\d+)?', rest_of_file)
        std_data = np.array([float(x.replace('D', 'e')) for x in all_numbers])

        return std_data, bak_data
    
    except Exception as e:
        print(f"Errore nella lettura dei file per la simulazione {index_simul + 1}: {e}")
        return None, None

def process_simulation_data(std_data, bak_data):
    """
    Elabora i dati grezzi da un file .std e i parametri da un .bak.
    Restituisce un dizionario contenente tutte le variabili fisiche calcolate.
    
    Args:
        std_data (np.ndarray): Array di dati grezzi dal file .std.
        bak_data (dict): Dizionario di parametri dal file .bak.
        radius (float): Raggio dal file .std.
        
    Returns:
        dict: Un dizionario di tutte le variabili calcolate.
    """
    
    # Estrai le costanti necessarie dal file .bak
    Z0 = bak_data.get('z0', 0)
    ZN = bak_data.get('zn', 0)
    RADIUS = bak_data.get('radius', 1)
    N_CRY = bak_data.get('n_cry', 0)
    N_GAS = bak_data.get('n_gas', 0)
    P_OUT = bak_data.get('p_out', 101300.0)
    MAX_PACKING_FRACTION = bak_data.get('max_packing_fraction', 100000000000.0)
    SURFACE_TENSION = bak_data.get('surface_tension', 0.05)
    ELASTIC_MODULUS = bak_data.get('elastic_modulus', 10000000000.0)
    STRAIN_RATE_VISCOSITY = bak_data.get('strain_rate_viscosity', 'Mixture')
    RE_BUBBLY_MAGMA_EXP_THR = bak_data.get('re_bubbly_magma_exp_thr', 1.0)
    LOG10_BUBBLE_NUMBER_DENSITY = bak_data.get('log10_bubble_number_density', -1)
    LOG10_FRICTION_COEFFICIENT = bak_data.get('log10_friction_coefficient', -5)
    THROAT_BUBBLE_RATIO = bak_data.get('throat_bubble_ratio', 1)
    TORTUOSITY_FACTOR = bak_data.get('tortuosity_factor', 0.5)
    VISC_2 = bak_data.get('visc_2', 0)

    # Calcola il numero di righe per il reshape
    num_rows = 8 + 2 * N_CRY + 4 * N_GAS + 3

    ## Reshape dei dati 
    #data_reshaped = np.reshape(std_data, (num_rows, -1), order='F')
    # Reshape dei dati (saltando le prime 2 righe che contengono radius e comp_cells)
    data_reshaped = np.reshape(std_data[:], (num_rows, -1), order='F')
    
    zeta_grid = data_reshaped[0, :]
    zeta_grid_reverse = ZN - zeta_grid + Z0
    
    # Estrazione e calcolo delle variabili fisiche
    alfa_2 = data_reshaped[1 : 1 + N_GAS, :]
    alfa_1 = 1.0 - np.sum(alfa_2, axis=0)
    p_1 = data_reshaped[1 + N_GAS + 1 - 1, :]
    u_1 = data_reshaped[1 + N_GAS + 3 - 1, :]
    u_2 = data_reshaped[1 + N_GAS + 4 - 1, :]
    T = data_reshaped[1 + N_GAS + 5 - 1, :]
    #print(f'alfa_2[0] = {alfa_2[0]}')
    #print(f'alfa_2 dimensions {alfa_2.shape[0]}x{alfa_2.shape[1]}')
    #print(f'p_1[0] = {p_1[0]}')
    #print(f'p_1 dimensions {p_1.shape[0]}')
    #print(f'u_1[0] = {u_1[0]}')
    #print(f'T[0] = {T[0]}')
    
    beta_start_index = 1 + N_GAS + 5
    beta = data_reshaped[beta_start_index : beta_start_index + N_CRY, :]
    beta_tot = np.sum(beta, axis=0)
    
    x_d_start_index = 1 + N_GAS + 5 + N_CRY
    x_d = data_reshaped[x_d_start_index : x_d_start_index + N_GAS, :]
    
    rho_1 = data_reshaped[1 + N_GAS + 5 + N_CRY + N_GAS + 1 - 1, :]

    rho_2_start_index = 1 + N_GAS + 5 + N_CRY + N_GAS + 1
    rho_2 = data_reshaped[rho_2_start_index : rho_2_start_index + N_GAS, :]

    rho_mix = alfa_1 * rho_1 + np.sum(alfa_2 * rho_2, axis=0)
    c_1 = alfa_1 * rho_1 / rho_mix
    c_2 = 1.0 - c_1
    u_mix = c_1 * u_1 + c_2 * u_2
    u_rel = u_2 - u_1

    visc = data_reshaped[num_rows - 3 -1, :]
    visc_melt = data_reshaped[num_rows - 2 -1, :]
    
    # Calcolo del tasso di deformazione elongazionale
    elongational_strain_rate = np.zeros_like(u_1)
    elongational_strain_rate[1:] = (u_1[1:] - u_1[:-1]) / (zeta_grid[1:] - zeta_grid[:-1])
    elongational_strain_rate[0] = elongational_strain_rate[1]
    
    # Calcolo del numero di Deborah
    deborah_number = np.zeros_like(u_1)
    deborah_threshold = 0.01 + u_1 * 0.0

    if STRAIN_RATE_VISCOSITY == 'Mixture':
        deborah_number = elongational_strain_rate * visc / \
                        (np.maximum(1.0 - beta_tot / MAX_PACKING_FRACTION, 1.0e-14) * ELASTIC_MODULUS)
    else:
        deborah_number = elongational_strain_rate * visc_melt / \
                        (np.maximum(1.0 - beta_tot / MAX_PACKING_FRACTION, 1.0e-14) * ELASTIC_MODULUS)
    
    deborah_number = np.minimum(deborah_number, deborah_threshold)
    
    # Raccogli tutte le variabili in un dizionario
    processed_data = {
        # Dati dal file .bak
        'Z0': Z0, 'ZN': ZN, 
        'RADIUS': RADIUS, 
        'N_GAS': N_GAS, 
        'N_CRY': N_CRY,  
        'P_OUT': P_OUT, 
        'VISC_2': VISC_2,
        'ELASTIC_MODULUS': ELASTIC_MODULUS,
        'MAX_PACKING_FRACTION': MAX_PACKING_FRACTION,
        'STRAIN_RATE_VISCOSITY': STRAIN_RATE_VISCOSITY,
        'RE_BUBBLY_MAGMA_EXP_THR': RE_BUBBLY_MAGMA_EXP_THR,
        'LOG10_BUBBLE_NUMBER_DENSITY': LOG10_BUBBLE_NUMBER_DENSITY,
        'TORTUOSITY_FACTOR': TORTUOSITY_FACTOR,
        'THROAT_BUBBLE_RATIO': THROAT_BUBBLE_RATIO,
        'LOG10_FRICTION_COEFFICIENT': LOG10_FRICTION_COEFFICIENT,
        'SURFACE_TENSION': SURFACE_TENSION, 

        # Dati dal file .std
        'zeta_grid': zeta_grid,
        'alfa_1': alfa_1, 'alfa_2': alfa_2, 
        'p_1': p_1, 'u_1': u_1, 'u_2': u_2, 'T': T,
        'beta': beta, 'beta_tot': beta_tot, 'x_d': x_d, 
        'rho_1': rho_1, 'rho_2': rho_2, 'rho_mix': rho_mix,
        'visc': visc, 'visc_melt': visc_melt,
        'elongational_strain_rate': elongational_strain_rate,
        'deborah_number': deborah_number, 
        'deborah_threshold': deborah_threshold,
        'u_mix': u_mix, 'u_rel': u_rel
    }
    
    return processed_data

def import_dakota_bounds():
    filename = "dakota_test_parallel.in"
    
    # Controlla se il file esiste
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"File '{filename}' non trovato nella directory {os.getcwd()}")
    
    bounds_data = {}
    
    with open(filename, "r") as f:
        for line in f:
            line_stripped = line.strip()
            
            # Cerca lower_bounds o upper_bounds
            if line_stripped.startswith("lower_bounds") or line_stripped.startswith("upper_bounds"):
                key = line_stripped.split("=")[0].strip()
                
                # Prendi tutto quello dopo "="
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", line_stripped)
                values = [float(x) for x in nums]
                
                bounds_data[key] = {
                    "values": values,
                    "count": len(values)
                }
    
    # Crea DataFrame con i dati
    rows = []
    for bound_type, info in bounds_data.items():
        row = {
            "type": bound_type,
            "count": info["count"],
            **{f"x{j+1}": v for j, v in enumerate(info["values"])}
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Se ci sono entrambe le righe
    input_Min = bounds_data["lower_bounds"]["values"] if "lower_bounds" in bounds_data else None
    input_Max = bounds_data["upper_bounds"]["values"] if "upper_bounds" in bounds_data else None
    
    print("Bounds trovati:")
    print(df)
    print(f'input_Min = {input_Min}, \ninput_Max = {input_Max}')
    
    return df, input_Min, input_Max

def adjust_bounds_with_units(input_Min, input_Max, df):
    """
    Applica le stesse trasformazioni delle colonne df ai limiti input_Min e input_Max.

    Parameters
    ----------
    input_Min : list of float
        Valori minimi originali delle xi (in unità del file Dakota).
    input_Max : list of float
        Valori massimi originali delle xi.
    df : pandas.DataFrame
        DataFrame trasformato con MultiIndex nelle colonne.
        Serve per leggere le unità correnti (seconda riga di intestazione).

    Returns
    -------
    adj_Min, adj_Max : list of float
        Liste di limiti trasformati, coerenti con i dati di df.
    """

    adj_Min, adj_Max = [], []

    for i, (low, high) in enumerate(zip(input_Min, input_Max)):
        # Trova il nome colonna corrispondente (es. 'x1', 'x2'...)
        col = f"x{i+1}"
        if col not in df.columns.get_level_values(0):
            # se non è presente, lascia invariato
            adj_Min.append(low)
            adj_Max.append(high)
            continue

        # prendi la label descrittiva con unità
        label = df.loc[:, col].columns[0]  # seconda riga (unità)

        if "[Pa]" in label:
            adj_Min.append(low / 1e6)
            adj_Max.append(high / 1e6)
        elif "[K]" in label:
            adj_Min.append(low - 273)
            adj_Max.append(high - 273)
        elif "[vol.]" in label:
            adj_Min.append(low * 100)
            adj_Max.append(high * 100)
        elif "[wt.]" in label:
            adj_Min.append(low * 100)
            adj_Max.append(high * 100)
        else:
            adj_Min.append(low)
            adj_Max.append(high)

    return adj_Min, adj_Max

def read_csv_with_labels(path):
    """
    Legge un CSV che può avere:
    - Nessuna riga di intestazione -> usa header=0
    - Una sola riga di intestazione (nomi colonne standard)
    - Due righe: la prima con i nomi tecnici (x1, response_fn_1, ecc.)
                 la seconda con etichette leggibili (da ignorare)
    
    Restituisce un DataFrame con le colonne standard (prima riga).
    """
    # Leggi le prime due righe come testo
    with open(path, "r") as f:
        first_line = f.readline().strip().split(",")
        second_line = f.readline().strip().split(",")
    
    # Caso 1: il file ha una sola riga di header (nessuna seconda riga)
    if not second_line or len(second_line) != len(first_line):
        return pd.read_csv(path, header=0)

    # Caso 2: controlliamo se la seconda riga è "etichette leggibili"
    # Regola semplice: se almeno una colonna non è numerica pura → è label
    def looks_like_label(s):
        try:
            float(s)  # se si converte a float → numero
            return False
        except ValueError:
            return True

    if any(looks_like_label(cell) for cell in second_line):
        # Due righe di intestazione: saltiamo la seconda
        return pd.read_csv(path, header=0, skiprows=[1])
    else:
        # In realtà la seconda riga contiene già dati → normale read_csv
        return pd.read_csv(path, header=0)

def load_or_process_csv(filename, process_func=None, dependencies=None):
    """
    Carica un CSV se esiste, altrimenti esegue una funzione di processo
    e ricarica i CSV dipendenti.
    
    Parameters
    ----------
    filename : str
        Nome del file CSV da caricare
    process_func : callable, optional
        Funzione da eseguire se il file non esiste (es. remove_null_simulations_at_frag)
    dependencies : list of str, optional
        Lista di file CSV da caricare dopo il process_func
    
    Returns
    -------
    df : pandas.DataFrame
    """
    if os.path.isfile(filename):
        print(f"Carico {filename}")
        return read_csv_with_labels(filename)
    else:
        print(f"{filename} non trovato, eseguo {process_func.__name__ if process_func else 'nessuna funzione'}...")
        if process_func:
            process_func()
        dfs = {}
        if dependencies:
            for dep in dependencies:
                if os.path.isfile(dep):
                    dfs[dep] = pd.read_csv(dep)
                else:
                    print(f"Attenzione: file {dep} mancante")
        if os.path.isfile(filename):
            return read_csv_with_labels(filename)
        return dfs if dfs else None
    
def get_xi_labels_from_template(df, template_file):
    """
    Legge il file conduit_solver.template e restituisce un dizionario che
    mappa ogni xi (es. 'x1') al nome della variabile associata. 

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenente le colonne xi
    template_file : str
        Percorso al file conduit_solver.template
    header_rows_to_skip : int
        Numero di righe di intestazione extra da saltare nel DataFrame

    Returns
    -------
    xi_labels : dict
        Dizionario { 'x1': 'Inlet temp. [°C]', ... }
    """

    # Converte automaticamente le colonne xi in numerico
    xi_cols = [col for col in df.columns if col.startswith('x')]
    for xi in xi_cols:
        df[xi] = pd.to_numeric(df[xi], errors='coerce')

    xi_labels = {}

    with open(template_file, 'r') as f:
        for line in f:
            matches = re.findall(r"\{(x\d+)\}", line)
            if matches:
                var_name = line.split('=')[0].strip()

                # Assegna label e trasformazione secondo regole
                for xi in matches:
                    if var_name == 'RADIUS':
                        xi_labels[xi] = 'Radius [m]'
                    elif var_name == 'T_IN':
                        xi_labels[xi] = 'Inlet temp. [K]'
                    elif var_name == 'P1_IN':
                        xi_labels[xi] = 'Inlet press. [Pa]'
                    elif var_name == 'BETA_C0':
                        xi_labels[xi] = 'Phenocryst. content [vol.]'
                    elif var_name == 'X_EX_DIS_IN':
                        if len(matches) == 2:
                            xi_labels[matches[0]] = 'Inlet H2O content [wt.]'
                            xi_labels[matches[1]] = 'Inlet CO2 content [wt.]'
                        elif len(matches) == 1:
                            xi_labels[xi] = 'Inlet H2O content [wt.]'
                    else:
                        xi_labels[xi] = var_name

    # Trova tutte le colonne xi presenti nel DataFrame
    xi_cols = [col for col in df.columns if col.startswith('x')]

    return xi_labels

def transform_units_of_variables(df):
    """
    Applica trasformazioni di unità di misura al DataFrame in base
    alle etichette di unità presenti nella seconda riga di intestazione.
    
    Regole:
    - [Pa]   → diviso per 1e6, sostituito con [MPa]
    - [K]    → meno 273, sostituito con [°C]
    - [vol.] → moltiplicato per 100, sostituito con [vol.%]
    - [wt.]  → moltiplicato per 100, sostituito con [wt.%]

    Se il DataFrame non ha MultiIndex nelle colonne,
    prova a costruirlo automaticamente con fix_headers().

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame con due righe di intestazione:
        - la prima contiene i nomi "x1, x2..."
        - la seconda contiene le descrizioni con unità di misura.

    Returns
    -------
    df : pd.DataFrame
        DataFrame con colonne trasformate e unità aggiornate.
    """
    df = df.copy()

    new_columns = []
    for head1, head2 in df.columns:
        new_label = head2
        if "[Pa]" in head2:
            df[(head1, head2)] = pd.to_numeric(df[(head1, head2)], errors="coerce") / 1e6
            new_label = head2.replace("[Pa]", "[MPa]")
        elif "[K]" in head2:
            df[(head1, head2)] = pd.to_numeric(df[(head1, head2)], errors="coerce") - 273
            new_label = head2.replace("[K]", "[°C]")
        elif "[vol.]" in head2:
            df[(head1, head2)] = pd.to_numeric(df[(head1, head2)], errors="coerce") * 100
            new_label = head2.replace("[vol.]", "[vol.%]")
        elif "[wt.]" in head2:
            df[(head1, head2)] = pd.to_numeric(df[(head1, head2)], errors="coerce") * 100
            new_label = head2.replace("[wt.]", "[wt.%]")

        new_columns.append((head1, new_label))

    # Aggiorna le intestazioni con i nuovi label
    df.columns= pd.MultiIndex.from_tuples(new_columns)

    return df

def plot_xi_vs_response_fn( 
        df, 
        input_Min,
        input_Max,
        response_col, 
        y_label=None, 
        n_step=1,
        save_name=None,
        fig_num=None
    ):
    """
    Plotta le variabili xi vs una colonna di risposta in un layout dinamico.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame con MultiIndex (nome tecnico, label descrittiva)
        con tutte le colonne {xi} e le colonne di risposta {response_fn_i}.
    input_Min : list of float
        Valori minimi (già trasformati) delle xi.
    input_Max : list of float
        Valori massimi (già trasformati) delle xi.
    response_col : str
        Nome della colonna di risposta da plottare (es. 'response_fn_1').
    y_label : str, optional
        Etichetta da mostrare sull'asse Y. Se None, viene usato response_col.
    n_step : int
        Campionamento dei dati per velocizzare il plot (default=1)
    save_name : str, optional
        Nome da usare per il file salvato (senza estensione). Se None, usa y_label o response_col.
    fig_num : int, optional
        Numero della figura (per creare più finestre distinte).
    """

    # Trova tutte le xi presenti nel DataFrame
    xi_cols = [col for col in df.columns if col[0].startswith('x')]
    n_xi = len(xi_cols)

    # Trova la colonna corrispondente nel MultiIndex
    matches = [col for col in df.columns if col[0] == response_col]
    if not matches:
        raise ValueError(f"La colonna '{response_col}' non è stata trovata nel DataFrame.")
    if len(matches) > 1:
        print(f"Attenzione: trovate più colonne per '{response_col}', uso la prima.")
    response_tuple = matches[0]  # (response_fn_x, label descrittiva)

    y_label = y_label or response_col

    # Estrai le label dal secondo livello del MultiIndex se presente
    xi_labels = {}
    for col in xi_cols:
        if isinstance(df.columns, pd.MultiIndex):
            xi_labels[col] = df.columns.get_level_values(1)[df.columns.get_loc(col)]
        else:
            xi_labels[col] = col

    # Layout subplot dinamico
    n_cols = min(3, n_xi)
    n_rows = math.ceil(n_xi / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), num=fig_num)
    axes = axes.flatten() if n_xi > 1 else [axes]

    for i, xi in enumerate(xi_cols):
        x_values =df[xi].to_numpy()[::n_step] 
        y_values = df[response_col].to_numpy()[::n_step]


        axes[i].plot(x_values, y_values, 'rs', markerfacecolor='r', markersize=2)
        axes[i].set_xlim(input_Min[i], input_Max[i])
        axes[i].set_xlabel(xi_labels[xi][1])
        axes[i].set_ylabel(y_label)

        # Mostra numeri sugli assi e tick automatici
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        axes[i].tick_params(axis='both', which='major', labelsize=10)

    # Nascondi assi vuoti se n_xi < n_rows*n_cols
    for j in range(i+1, n_rows*n_cols):
        axes[j].axis('off')

    plt.tight_layout()

    # -- Salvataggio automatico ---
    save_dir = "plot_correlations"
    os.makedirs(save_dir, exist_ok=True)

    save_label = save_name or y_label.replace(" ", "_")
    save_path = os.path.join(save_dir, f"{save_label}.png")

    plt.savefig(save_path, dpi=300)
    print(f"Figura salvata in {save_path}")

    # Mostra senza bloccare
    plt.show(block=False)
    plt.pause(0.1)

def plot_multiple_y_vs_single_x(
    df, 
    x_col, 
    y_cols, 
    xi_labels,
    xi_transforms,
    y_labels, 
    x_transform=None, 
    y_transforms=None, 
    n_step=1,
    save_name=None, 
    fig_num=None
):
    """
    Plotta più colonne Y vs un'unica colonna X in subplot.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame contenente le colonne X e Y
    x_col : str
        Nome della colonna da usare come asse X
    y_cols : list of str
        Lista di colonne da plottare come Y
    y_labels : list of str
        Etichette da mettere sugli assi Y
    x_transform : callable, optional
        Funzione di trasformazione da applicare a X (es. lambda x: x*100)
    y_transforms : list of callable, optional
        Lista di funzioni da applicare alle colonne Y
    n_step : int
        Campionamento per velocizzare il plot
    save_name : str, optional
        Nome del file da salvare
    fig_num : int, optional
        Numero della figura
    """

    import math
    import matplotlib.pyplot as plt
    import os

    n_plots = len(y_cols)
    n_cols = min(3, n_plots)
    n_rows = math.ceil(n_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), num=fig_num)
    axes = axes.flatten() if n_plots > 1 else [axes]

    x_vals = df[x_col].to_numpy()[::n_step]
    if x_transform:
        x_vals = x_transform(x_vals)

    if y_transforms is None:
        y_transforms = [lambda y: y]*n_plots

    for i, (y_col, y_label, y_trans) in enumerate(zip(y_cols, y_labels, y_transforms)):
        y_vals = y_trans(df[y_col].to_numpy()[::n_step])
        axes[i].plot(x_vals, y_vals, 'rs', markerfacecolor='r', markersize=2)
        axes[i].set_xlabel('H2O content (wt.%)')
        axes[i].set_ylabel(y_label)
        axes[i].xaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        axes[i].yaxis.set_major_locator(plt.MaxNLocator(nbins=6))
        axes[i].tick_params(axis='both', which='major', labelsize=10)
        axes[i].grid(True)

    # Nascondi eventuali subplot vuoti
    for j in range(i+1, n_rows*n_cols):
        axes[j].axis('off')

    plt.tight_layout()

    # Salvataggio
    if save_name:
        save_dir = "plot_correlations"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{save_name}.png")
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved in {save_dir} as {save_name}.png")

    plt.show(block=False)
    plt.pause(0.1)

def plot_x_fixed_yi_change(
    df,
    x_col,
    input_Min,
    input_Max,
    response_cols,
    n_step=1,
    fig_num=None,
    save_name=None
):
    """
    General function to plot multiple response variables vs a fixed input variable.
    Layout adjusts automatically based on number of plots.

    Parameters
    ----------
    df : pd.DataFrame
        Contains `x_col` and response variables.
    x_col : str
        Input variable for the horizontal axis.
    input_Min, input_Max : np.array
        Min/max bounds for xi (used for plotting limits).
    response_cols : list of tuples
        Each tuple: (response_col, transform_fn_or_None, ylabel).
    n_step : int
        Sampling step for speed.
    fig_num : int, optional
        If you want multiple figure windows.
    save_name : str, optional
        Base name for saving. Saves into 'plot_correlations/' if provided.
    """
    num_plots = len(response_cols)
    n_cols = int(math.sqrt(num_plots)) + 1 # oppure : min(3, num_plots)
    n_rows = math.ceil(num_plots / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), num=fig_num)
    axes = axes.flatten()

    xi_index = int(x_col[1:]) - 1  # x4 → index 3
    x_vals = df[x_col].to_numpy()[::n_step]
    
    x_min = input_Min[xi_index]
    x_max = input_Max[xi_index]

    for ax, (resp_col, transform, y_label_custom) in zip(axes, response_cols):
        # Controlliamo che df sia multiIndice
        if isinstance(df.columns, pd.MultiIndex):
            matches = df.columns[df.columns.get_level_values(0) == resp_col]
            if len(matches) == 0:
                raise KeyError(
                    f"Colonna {resp_col} non trovata. Disponibili: {set(df.columns.get_level_values(0))}"
                )
            col_tuple = matches[0]  # colonna completa
            y = df[col_tuple].to_numpy()[::n_step]
        else:
            if resp_col not in df.columns:
                raise KeyError(f"Colonna {resp_col} non trovata. Disponibili: {set(df.columns)}")
            y = df[resp_col].to_numpy()[::n_step]
        
        # Applica la trasformazione se presente
        if transform:
            y = transform(y)

        ax.plot(x_vals, y, 'rs', markerfacecolor='r', markersize=2)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_label)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True)

        # Estrai valori
        y_vals = df[(resp_col, resp_label)].to_numpy()[::n_step]
        if transform:
            y_vals = transform(y_vals)

        # Etichetta asse Y: se hai passato un custom lo usa, altrimenti usa il label del file
        ylabel = y_label_custom or resp_label

        ax.plot(x_vals, y_vals, 'rs', markerfacecolor='r', markersize=2)
        ax.set_xlim(x_min, x_max)
        ax.set_xlabel(df.columns.get_level_values(1)[df.columns.get_level_values(0) == x_col][0])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='both', which='major', labelsize=9)
        ax.grid(True)

    for ax in axes[num_plots:]:
        ax.axis('off')

    plt.tight_layout()

    if save_name:
        save_dir = "plot_correlations"
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{save_name}.png")
        plt.savefig(path, dpi=300)
        print(f"Saved plot at: {path}")

    plt.show(block=False)
    plt.pause(0.1)
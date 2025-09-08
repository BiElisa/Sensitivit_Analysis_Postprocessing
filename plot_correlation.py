import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import my_lib_process_utils as utils
import my_lib_remove_simulations_from_csv as rm

if __name__ == '__main__':
    """
    Esegue calcoli, pulizia dei dati e genera grafici di correlazione.
    """

    filename = 'simulations.csv'

    # Carica (o crea) i dati dal file 'simulations.csv'
    if os.path.exists(filename):
        df_dakota_output = pd.read_csv(filename)

    else:
        print(f"Errore: Il file '{filename}' non è stato trovato. Lo vado a costruire.")

        # Cerchiamo i limiti inferiori e superiori assegnati ai parametri di input
        df_dakota_input, dakota_input_Min, dakota_input_Max  = utils.import_dakota_bounds()
        if df_dakota_input is None:
            sys.exit()

        df_dakota_output = utils.import_dakota_tabular_new() #contiene {xi} e {response_fn_i}, i=1,...,15
        if df_dakota_output is None:
            sys.exit()
            
        df_dakota_output, number_null_sim = rm.remove_null_simulations(df_dakota_output, "simulations.csv")

        # Aggiunta di nuove colonne

        # Total crystal content
        df_dakota_output["response_fn_16"] = df_dakota_output["response_fn_7"] + df_dakota_output["response_fn_8"] + df_dakota_output["response_fn_9"]

        plag_liquidus = -0.64 * (df_dakota_output["response_fn_2"] / 1e6) + 1132 + 273
        # Undercooling
        df_dakota_output["response_fn_17"] = plag_liquidus - df_dakota_output["response_fn_6"] 

        df_dakota_output["response_fn_18"] = df_dakota_output["response_fn_14"] # EB : da eliminare successivamente

        df_dakota_output["response_fn_19"] = np.where(
            df_dakota_output["response_fn_15"] > -0.5,
            236.0 * (df_dakota_output["response_fn_12"] ** 0.25),
            (df_dakota_output["response_fn_4"] ** 2.0) / (2.0 * 9.8)
        )

        # Aggiungiamo una seconda riga al sile per avere delle labels leggibili
        
        # Trova tutte le colonne che iniziano con 'x'
        xi_cols = [col for col in df_dakota_output.columns if col.startswith('x')]
        n_xi = len(xi_cols)
        print(f"Trovate {n_xi} variabili xi: {xi_cols}")

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
            f.write(",".join(df_dakota_output.columns) + "\n")
            f.write(",".join(labels_row) + "\n")
            df_dakota_output.to_csv(f, index=False, header=False)
        
        print(f"Simulazioni salvate in {filename_with_labels} ({len(df_dakota_output)} righe).")

    # data_at_fragmentation.csv
    df_fragmentation = utils.load_or_process_csv(
        "data_at_fragmentation.csv",
        process_func=lambda : rm.remove_null_simulations(
            "data_at_fragmentation_total.csv",
            "data_at_fragmentation.csv"
        ),
        dependencies=["simulations.csv"]
    )
    # EB : aggiungere la response_fn_28

    input('AAA...')

    # data_at_inlet.csv
    df_inlet = utils.load_or_process_csv(
        "data_at_inlet.csv",
        process_func=lambda : rm.remove_null_simulations(
            "data_at_inlet_total.csv",
            "data_at_inlet.csv"
        ),
        dependencies=["simulations.csv", "data_at_fragmentation.csv"]
    )

    # data_at_vent.csv
    df_vent = utils.load_or_process_csv(
        "data_at_vent.csv",
        process_func=lambda : rm.remove_null_simulations(
            "data_at_vent_total.csv",
            "data_at_vent.csv"
        ),
        dependencies=["simulations.csv", "data_at_fragmentation.csv", "data_at_inlet.csv"]
    )

    # data_average.csv
    df_average = utils.load_or_process_csv(
        "data_average.csv",
        process_func=lambda : rm.remove_null_simulations(
            "data_average_total.csv",
            "data_average.csv"
        ),
        dependencies=["simulations.csv", "data_at_fragmentation.csv", "data_at_inlet.csv", "data_at_vent.csv"]
    )
    

    df_concat = pd.concat([df_dakota_output, df_fragmentation, df_inlet, df_vent, df_average], axis=1)
    #print(f'df = {df}')
    #input('...')


    ## Iniziamo a lavorare per i plot

    # Numero di step per il campionamento
    num_step_campionamento = 3

    input ('...')

    utils.plot_xi_vs_response_fn(
        xi_labels=xi_labels,
        xi_transforms=xi_transforms,
        input_Min=input_Min,
        input_Max=input_Max,
        df = df_concat,
        response_col='response_fn_1',
        y_label='Gas volume fraction',
        n_step=num_step_campionamento,  # campiona ogni tot valori per velocizzare il plot
        save_name="plot_correlazione_Gas_volume_fraction",
        fig_num=1
    )

    utils.plot_xi_vs_response_fn(
        xi_labels=xi_labels,
        xi_transforms=xi_transforms,
        input_Min=input_Min,
        input_Max=input_Max,
        df = df_concat,
        response_col='response_fn_15',
        y_label='Fragmentation depth (m)',
        n_step=num_step_campionamento, 
        save_name="plot_correlazione_Fragmentation_depth",
        fig_num=2
    )

    utils.plot_xi_vs_response_fn(
        xi_labels=xi_labels,
        xi_transforms=xi_transforms,
        input_Min=input_Min,
        input_Max=input_Max,
        df = df_concat,
        response_col='response_fn_12',
        y_label='Mass flow rate (kg/s)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Mass_flow_rate",
        fig_num=3
    )

    utils.plot_xi_vs_response_fn(
        xi_labels=xi_labels,
        xi_transforms=xi_transforms,
        input_Min=input_Min,
        input_Max=input_Max,
        df = df_concat,
        response_col='response_fn_4',
        y_label='Exit velocity (m/s)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Exit_velocity",
        fig_num=4
    )

    # Crea colonna temporanea con valori moltiplicati per 100
    df_concat["response_fn_16_scaled"] = df_concat["response_fn_16"] * 100

    utils.plot_xi_vs_response_fn(
        xi_labels=xi_labels,
        xi_transforms=xi_transforms,
        input_Min=input_Min,
        input_Max=input_Max,
        df = df_concat,
        response_col='response_fn_16_scaled',
        y_label='Exit crystal content (vol.%)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Exit_crystal_content",
        fig_num=5
    )
    """
    utils.plot_xi_vs_response_fn(
        xi_labels=xi_labels,
        xi_transforms=xi_transforms,
        input_Min=input_Min,
        input_Max=input_Max,
        df = df_concat,
        response_col='response_fn_28',
        y_label='Undercooling @Frag (°C)',
        n_step=num_step_campionamento,  
        save_name="plot_correlazione_Undercooling_at_frag",
        fig_num=6
    )
    """

    # Prepariamo la lista delle variabili da plottare sulle 'y'
    response_defs = [
        ("response_fn_4",  None,            "Exit velocity (m/s)"),
        ("response_fn_15", None,            "Fragmentation depth (m)"),
        ("response_fn_30", lambda y: y*100, "Diss. H₂O content @Frag (wt.%)"),
        ("response_fn_25", lambda y: y-273, "Temperature @Frag (°C)"),
        ("response_fn_24", lambda y: y*100, "Crystal content @Frag (vol.%)"),
        ("response_fn_20", np.log10,        "Log10(viscosity) @Frag"),
    ]

    utils.plot_x_fixed_yi_change(
        df=df_concat,
        x_col="x4",
        input_Min=input_Min,
        input_Max=input_Max,
        response_defs=response_defs,
        n_step=3,
        fig_num=11,
        save_name="plot_correlazioni_H2O"
    )




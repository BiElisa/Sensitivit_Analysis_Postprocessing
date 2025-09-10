import numpy as np
import pandas as pd
import os
import my_lib_process_utils as utils
import my_lib_remove_simulations_from_csv as rm

def extract_data_at_frag(main_dir, bak_name, N):
    """
    Estrae i dati da tutti i file .bak e .std contenuti nelle cartelle workdir.N.
    
    Args:
        main_dir (str): Il percorso della cartella principale contenente le cartelle workdir.
        bak_name (str): Il nome del file .bak di riferimento (es. Piton_Emb2.bak).
    """

    if os.path.exists('data_at_fragmentation_total.csv'):
        print(f"'data_at_fragmentation_total.csv' esiste già")
        
        if os.path.exists('data_at_fragmentation.csv'):
            print("'data_at_fragmentation.csv' esiste già")
            print("La funzione extract_data_at_frag non verrà eseguita.")
            return
        
        print("'data_at_fragmentation.csv' invece va creato")
        rm.remove_null_simulations("data_at_fragmentation_total.csv", "data_at_fragmentation.csv")
        return

    # 1. Inizializza i dizionari per i dati di risposta
    response_data = {f'response_fn_{i}': np.full(N, np.nan) for i in range(20, 39)}
    
    # 2. Loop sulle simulazioni
    for index_simul in range(N):
        print(f'\r{round((index_simul + 1)/N *100, 1)}% of extract_data_at_frag completed', end='', flush=True) 

        #Chiamo funzione che cerca e legge .std file e .bak file
        std_data, bak_data = utils.search_and_read_std_file( main_dir, bak_name, index_simul)       

        # 3. Elaborazione dei dati se i file sono disponibili
        if (std_data is not None and bak_data is not None):

            data = utils.process_simulation_data(std_data, bak_data)

            # Per leggibilita` del seguito, copio localmente alcune variabili
            radius = data['RADIUS']
            N_CRY=data['N_CRY']
            N_GAS=data['N_GAS']
            VISC_2 = data['VISC_2']
            ELASTIC_MODULUS = data['ELASTIC_MODULUS']
            MAX_PACKING_FRACTION = data['MAX_PACKING_FRACTION']
            STRAIN_RATE_VISCOSITY = data['STRAIN_RATE_VISCOSITY']
            RE_BUBBLY_MAGMA_EXP_THR = data['RE_BUBBLY_MAGMA_EXP_THR']
            LOG10_BUBBLE_NUMBER_DENSITY = data['LOG10_BUBBLE_NUMBER_DENSITY']
            TORTUOSITY_FACTOR = data['TORTUOSITY_FACTOR']
            THROAT_BUBBLE_RATIO = data['THROAT_BUBBLE_RATIO']

            zeta_grid = data['zeta_grid']
            T=data['T']
            p_1= data['p_1']
            alfa_1 = data['alfa_1']
            x_d = data['x_d']
            beta = data['beta']
            rho_1 = data['rho_1']
            rho_mix = data['rho_mix']
            u_1 = data['u_1']
            u_mix = data['u_mix']
            u_rel = data['u_rel']
            visc = data['visc']
            visc_melt = data['visc_melt']
            elongational_strain_rate = data['elongational_strain_rate']
            deborah_number = data['deborah_number']
            deborah_threshold = data['deborah_threshold']

            # Cerchiamo le variabili specifiche solo dei dati a frammentazione
            bubble_exp_vel = np.zeros_like(rho_1)
            bubble_exp_vel_per_ul = np.zeros_like(rho_1)
            re_bubble_exp = np.zeros_like(rho_1)
            expansion_strain_rate = np.zeros_like(rho_1)
            
            for i in range(1, len(rho_1)): #  per i da 2 a length(rho_1)

                dh_magma = zeta_grid[i] - zeta_grid[i-1]
                volume_magma = dh_magma * np.pi * radius * radius
                mass_magma = rho_mix[i] * volume_magma
                volume_magma_old = mass_magma / rho_mix[i-1]
                dh_magma_old = volume_magma_old / (np.pi * radius * radius)
                u_mix_old =u_mix[i-1]
                bubble_exp_vel[i] = (dh_magma - dh_magma_old) * u_mix_old / dh_magma # velocita` di espansione del magma`
                bubble_exp_vel_per_ul[i] = bubble_exp_vel[i] / dh_magma # tasso di esapansione
                re_bubble_exp[i] = rho_mix[i] * bubble_exp_vel_per_ul[i] / visc[i]
                if i == 1:
                    # Per il primo elemento del ciclo, bubble_exp_vel_per_ul_old = 0
                    expansion_strain_rate[i] = bubble_exp_vel_per_ul[i] / dh_magma
                else:
                    expansion_strain_rate[i] = (bubble_exp_vel_per_ul[i] - bubble_exp_vel_per_ul[i-1]) / dh_magma

            #print(f'expansion_strain_rate = {expansion_strain_rate}')
            #input('...')

            re_bubble_exp[0] = re_bubble_exp[1]
            bubble_exp_vel[0] = bubble_exp_vel[1]
            bubble_exp_vel_per_ul[0] = bubble_exp_vel_per_ul[1]
            re_bubble_threshold = re_bubble_exp * 0.0 + RE_BUBBLY_MAGMA_EXP_THR

            # (3) Ciclo per il flag di frammentazione

            fragmentation_flag = 0

            visc_at_frag, visc_melt_at_frag = 0.0, 0.0
            beta_at_frag = np.zeros(3)
            beta_tot_at_frag, T_at_frag, P_at_frag = 0.0, 0.0, 0.0
            vesic_at_frag, rho_1_at_frag, rho_mix_at_frag = 0.0, 0.0, 0.0
            u_mix_at_frag, u_rel_at_frag = 0.0, 0.0
            x_d_at_frag = np.zeros(2)
            elong_strain_rate_at_frag, deborah_number_at_frag = 0.0, 0.0
            re_bubble_exp_at_frag, ratio_deborah_reynolds_exp_at_frag = 0.0, 0.0
            
            for i in range(len(u_1)):
                if (((deborah_number[i] >= deborah_threshold[i]) or \
                    ((re_bubble_exp[i] >= re_bubble_threshold[i]) and \
                    (1.0 - alfa_1[i] > 1e-2))) and (fragmentation_flag == 0)):
                    # se siamo nelle condizioni di frammentare
                    # e siamo al limite della frammentazione

                    #input("siamo alla frammentazione!!!")

                    # Stampa la condizione di frammentazione soddisfatta
                    #if deborah_number[i] >= deborah_threshold[i]:
                    #    print(f"La frammentazione è avvenuta a causa del numero di Deborah.")
                    #    print(f"infatti deborah_number[i] = {deborah_number[i]} e` >= deborah_threshold[i]= {deborah_threshold[i]}")
                    #if (re_bubble_exp[i] >= re_bubble_threshold[i]) and (1.0 - alfa_1[i] > 1e-2):
                    #    print(f"La frammentazione è avvenuta a causa del numero di Reynolds per l'espansione delle bolle.")
                    #    print(f"infatti re_bubble_exp[i]= {re_bubble_exp[i]} e` >= re_bubble_threshold[i] = {re_bubble_threshold[i]}")
                    #    print(f"e inoltre 1.0 - alfa_1[i] = {1.0-alfa_1[i]}> 1e-2")

                    
                    fragmentation_flag = 1 
                    visc_at_frag = visc[i]
                    visc_melt_at_frag = visc_melt[i]

                    #print(f'visc_at_frag = {visc_at_frag}')
                    #print(f'visc_melt_at_frag = {visc_melt_at_frag}')
                    
                    if (N_CRY == 3):
                        beta_at_frag[0] = beta[0, i]
                        beta_at_frag[1] = beta[1, i]
                        beta_at_frag[2] = beta[2, i]
                    elif (N_CRY == 1):
                        beta_at_frag[0] = beta[0, i]
                        beta_at_frag[1] = 0.0
                        beta_at_frag[2] = 0.0
                        
                    beta_tot_at_frag = beta_at_frag[0] + beta_at_frag[1] + beta_at_frag[2]
                    #print(f'beta_tot_at_frag = {beta_tot_at_frag}')
                    
                    T_at_frag = T[i]
                    P_at_frag = p_1[i]
                    vesic_at_frag = 1.0 - alfa_1[i]
                    
                    rho_1_at_frag = rho_1[i]
                    rho_mix_at_frag = rho_mix[i]
                    u_mix_at_frag =u_mix[i]
                    u_rel_at_frag = u_rel[i]
                    
                    if (N_GAS == 2):
                        x_d_at_frag[0] = x_d[0, i]
                        x_d_at_frag[1] = x_d[1, i]
                    elif (N_GAS == 1):
                        x_d_at_frag[0] = x_d[0, i]
                        x_d_at_frag[1] = 0.0
                        
                    elong_strain_rate_at_frag = elongational_strain_rate[i]
                    #print(f'elong_strain_rate_at_frag = {elong_strain_rate_at_frag}')
                    #print(f'siamo alla posizione i = {i}')
                    
                    deborah_number_at_frag = deborah_number[i]
                    
                    re_bubble_exp_at_frag = re_bubble_exp[i]
                    
                    ratio_deborah_reynolds_exp_at_frag = deborah_number_at_frag / re_bubble_exp_at_frag
                    #print(f"deborah_number_at_frag = {deborah_number_at_frag}")
                    #print(f"re_bubble_exp_at_frag = {re_bubble_exp_at_frag}")
                    #print(f"ratio_deborah_reynolds_exp_at_frag = {ratio_deborah_reynolds_exp_at_frag}")
                    #input('...')
                                        
            # (4) Caso in cui la frammentazione non avviene
            if (fragmentation_flag == 0):
                #input('No frammentazione in questo punto ...')
                visc_at_frag = visc[-1]
                visc_melt_at_frag = visc_melt[-1]
                
                if (N_CRY == 3):
                    beta_at_frag[0] = beta[0, -1]
                    beta_at_frag[1] = beta[1, -1]
                    beta_at_frag[2] = beta[2, -1]
                elif (N_CRY == 1):
                    beta_at_frag[0] = beta[0, -1]
                    beta_at_frag[1] = 0.0
                    beta_at_frag[2] = 0.0
                    
                beta_tot_at_frag = beta_at_frag[0] + beta_at_frag[1] + beta_at_frag[2]
                

                T_at_frag = T[-1]
                P_at_frag = p_1[-1]
                vesic_at_frag = 1.0 - alfa_1[-1]
                
                rho_1_at_frag = rho_1[-1]
                rho_mix_at_frag = rho_mix[-1]
                u_mix_at_frag = u_mix[-1]
                u_rel_at_frag = u_rel[-1]
                
                if (N_GAS == 2):
                    x_d_at_frag[0] = x_d[0, -1]
                    x_d_at_frag[1] = x_d[1, -1]
                elif (N_GAS == 1):
                    x_d_at_frag[0] = x_d[0, -1]
                    x_d_at_frag[1] = 0.0
                    
                elong_strain_rate_at_frag = elongational_strain_rate[-1]
                
                if STRAIN_RATE_VISCOSITY == 'Mixture':
                    deborah_number_at_frag = visc_at_frag * elong_strain_rate_at_frag / \
                                            ((1.0 - beta_tot_at_frag / MAX_PACKING_FRACTION) * ELASTIC_MODULUS)
                else:
                    deborah_number_at_frag = visc_melt_at_frag * elong_strain_rate_at_frag / \
                                            ((1.0 - beta_tot_at_frag / MAX_PACKING_FRACTION) * ELASTIC_MODULUS)
                
                re_bubble_exp_at_frag = re_bubble_exp[-1]
                ratio_deborah_reynolds_exp_at_frag = deborah_number_at_frag / re_bubble_exp_at_frag

                #print(f"deborah_number_at_frag = {deborah_number_at_frag}")
                #print(f"re_bubble_exp_at_frag = {re_bubble_exp_at_frag}")
                #print(f"ratio_deborah_reynolds_exp_at_frag = {ratio_deborah_reynolds_exp_at_frag}")

            # (5) Calcoli finali
            bubble_number_density = 10**LOG10_BUBBLE_NUMBER_DENSITY
            
            # r_b := raggio medio delle bolle (nota che (1.0 - vesic_at_frag) rappresenta la fraz. volum. del liquido)
            # Si prende il volume totale occupato dalle bolle (vesic_at_frag), lo si divide per il numero totale di bolle
            # per ottenere il volume di una singola bolla, e poi si calcola il raggio prendendo la radice cubica.
            r_b_at_frag = (vesic_at_frag / (4.0/3.0 * np.pi * bubble_number_density * (1.0 - vesic_at_frag)))**(1.0/3.0)
            
            # k1 := termine nella formula di Darcy per flusso in mezzo poroso
            # E` una permeabilità modificata che tiene conto sia della dimensione 
            # delle bolle che della frazione di gas e della tortuosità del percorso.
            k1_at_frag = ((THROAT_BUBBLE_RATIO * r_b_at_frag)**2.0 * vesic_at_frag**TORTUOSITY_FACTOR)/8.0
            
            # Numero di Stokes aiuta a capire il grado di accoppiamento tra la fase gassosa e quella liquida
            stokes_at_frag = (rho_1_at_frag * k1_at_frag / VISC_2) / (radius / u_mix_at_frag)
            
            reynolds_at_frag = (rho_mix_at_frag * u_mix_at_frag * radius * 2)/visc_at_frag
            
            ratio_u_mix_rel_at_frag = u_mix_at_frag / u_rel_at_frag
            
            # (6) Salvataggio dei risultati nel dizionario
            response_data['response_fn_20'][index_simul] = visc_at_frag
            response_data['response_fn_21'][index_simul] = beta_at_frag[0]
            response_data['response_fn_22'][index_simul] = beta_at_frag[1]
            response_data['response_fn_23'][index_simul] = beta_at_frag[2]
            response_data['response_fn_24'][index_simul] = beta_tot_at_frag
            response_data['response_fn_25'][index_simul] = T_at_frag
            response_data['response_fn_26'][index_simul] = P_at_frag
            response_data['response_fn_27'][index_simul] = vesic_at_frag
            response_data['response_fn_30'][index_simul] = x_d_at_frag[0]
            response_data['response_fn_31'][index_simul] = x_d_at_frag[1]
            response_data['response_fn_32'][index_simul] = elong_strain_rate_at_frag
            response_data['response_fn_33'][index_simul] = deborah_number_at_frag
            response_data['response_fn_34'][index_simul] = re_bubble_exp_at_frag
            response_data['response_fn_35'][index_simul] = ratio_deborah_reynolds_exp_at_frag
            response_data['response_fn_36'][index_simul] = visc_melt_at_frag
            response_data['response_fn_37'][index_simul] = reynolds_at_frag
            response_data['response_fn_38'][index_simul] = ratio_u_mix_rel_at_frag

    # 4. Salva i risultati in un file
    output_df = pd.DataFrame(response_data)

    # Dizionario con le descrizioni leggibili
    labels = {
        'response_fn_20': "Viscosity at fragmentation",
        'response_fn_21': "Fraz. cry_1 at fragmentation",
        'response_fn_22': "Fraz. cry_2 at fragmentation",
        'response_fn_23': "Fraz. cry_3 at fragmentation",
        'response_fn_24': "Total cry. at fragmentation",
        'response_fn_25': "Temperature at fragmentation",
        'response_fn_26': "Pressure at fragmentation",
        'response_fn_27': "Vesicularity at fragmentation",
        'response_fn_30': "Gas_1 diss. at fragmentation",
        'response_fn_31': "Gas_2 diss. at fragmentation",
        'response_fn_32': "Elongational strain rate at frag",
        'response_fn_33': "Deborah number at fragmentation",
        'response_fn_34': "Reynolds bubble expansion at frag",
        'response_fn_35': "Deborah/Reynolds ratio at frag",
        'response_fn_36': "Melt viscosity at fragmentation",
        'response_fn_37': "Reynolds number at fragmentation",
        'response_fn_38': "u_mix / u_rel at fragmentation"
    }

    # Creiamo un DataFrame con una sola riga contenente le descrizioni
    labels_df = pd.DataFrame([labels])

    # Concatenazione: prima riga descrizioni, poi dati veri
    final_df = pd.concat([labels_df, output_df], ignore_index=True)

    # Salvataggio in csv
    final_df.to_csv('data_at_fragmentation_total.csv', index=False)
    print("\nEstrazione completata. Dati salvati in 'data_at_fragmentation_total.csv'.")

    # Rimuoviamo le simulazini nulle
    rm.remove_null_simulations("data_at_fragmentation_total.csv", "data_at_fragmentation.csv")


def extract_data_at_inlet(main_dir, bak_name, N):
    """
    Estrae i dati relativi all'inlet da tutti i file .bak e .std. nelle cartelle workdir.N.

    Args:
        main_dir (str): Il percorso della cartella principale contenente le cartelle workdir.
        bak_name (str): Il nome del file .bak di riferimento.
    """

    # Controllo se il file esiste già
    if os.path.exists('data_at_inlet_total.csv'):
        print(f"'data_at_inlet_total.csv' esiste già")
        
        if os.path.exists('data_at_inlet.csv'):
            print("'data_at_inlet.csv' esiste già")
            print("La funzione extract_data_at_inlet non verrà eseguita.")
            return
        
        print("'data_at_inlet.csv' invece va creato")
        rm.remove_null_simulations("data_at_inlet_total.csv", "data_at_inlet.csv")
        return

    # 1. Inizializza i dizionari per i dati di risposta
    response_data = {f'response_fn_{i}': np.full(N, np.nan) for i in range(40, 50)}
    
    # 2. Loop sulle simulazioni
    for index_simul in range(N):
        print(f'\r{round((index_simul + 1)/N *100, 1)}% of extract_data_at_inlet completed', end='', flush=True) 

        #Chiamo funzione che cerca e legge .std file e .bak file
        std_data, bak_data = utils.search_and_read_std_file( main_dir, bak_name, index_simul)

        # 3. Elaborazione dei dati se i file sono disponibili
        if (std_data is not None and bak_data is not None):

            data = utils.process_simulation_data(std_data, bak_data)

            # Per leggibilita` del seguito, copio localmente alcune variabili
            #radius = data['RADIUS']
            N_CRY=data['N_CRY']
            N_GAS=data['N_GAS']

            #zeta_grid = data['zeta_grid']
            #T=data['T']
            #p_1= data['p_1']
            #alfa_1 = data['alfa_1']
            alfa_2 = data['alfa_2']
            x_d = data['x_d']
            beta = data['beta']
            #rho_1 = data['rho_1']
            #rho_mix = data['rho_mix']
            u_1 = data['u_1']
            #u_mix = data['u_mix']
            #u_rel = data['u_rel']
            visc = data['visc']
            #visc_melt = data['visc_melt']
            #elongational_strain_rate = data['elongational_strain_rate']
            #deborah_number = data['deborah_number']
            #deborah_threshold = data['deborah_threshold']

            # Estrazione dei valori all'inlet (indice 0 in Python)
            visc_at_inlet = visc[0]

            if (N_CRY == 3):
                beta_at_inlet = [beta[0, 0], beta[1, 0], beta[2, 0]]
            elif (N_CRY == 1):
                beta_at_inlet = [beta[0, 0], 0.0, 0.0]
            else:
                beta_at_inlet = [0.0, 0.0, 0.0]
            beta_tot_at_inlet = sum(beta_at_inlet)

            #T_at_inlet = T[0]
            #P_at_inlet = p_1[0]
            u_1_at_inlet = u_1[0]
            #vesic_at_inlet = 1.0 - alfa_1[0]
            #elong_strain_rate_at_inlet = elongational_strain_rate[0]
            
            if (N_GAS == 2):
                alfa_2_inlet = [alfa_2[0, 0], alfa_2[1, 0]]
                x_d_at_inlet = [x_d[0, 0], x_d[1, 0]]
            elif (N_GAS == 1):
                alfa_2_inlet = [alfa_2[0, 0], 0.0]
                x_d_at_inlet = [x_d[0, 0], 0.0]
            else:
                alfa_2_inlet = [0.0, 0.0]
                x_d_at_inlet = [0.0, 0.0]
            
            # Salvataggio dei risultati nel dizionario
            response_data['response_fn_40'][index_simul] = visc_at_inlet
            response_data['response_fn_41'][index_simul] = beta_at_inlet[0]
            response_data['response_fn_42'][index_simul] = beta_at_inlet[1]
            response_data['response_fn_43'][index_simul] = beta_at_inlet[2]
            response_data['response_fn_44'][index_simul] = beta_tot_at_inlet
            response_data['response_fn_45'][index_simul] = alfa_2_inlet[0]
            response_data['response_fn_46'][index_simul] = alfa_2_inlet[1]
            response_data['response_fn_47'][index_simul] = x_d_at_inlet[0]
            response_data['response_fn_48'][index_simul] = x_d_at_inlet[1]
            response_data['response_fn_49'][index_simul] = u_1_at_inlet


    # 4. Salva i risultati in un file CSV
    output_df = pd.DataFrame(response_data)

    # Dizionario con le descrizioni leggibili
    labels = {
        'response_fn_40': 'Viscosity at inlet',
        'response_fn_41': 'Fraz. cry_1 at inlet',
        'response_fn_42': 'Fraz. cry_2 at inlet',
        'response_fn_43': 'Fraz. cry_3 at inlet',
        'response_fn_44': 'Total cry. fract. at inlet',
        'response_fn_45': 'Gas_1 exs. at inlet',
        'response_fn_46': 'Gas_2 exs. at inlet',
        'response_fn_47': 'Gas_1 diss. at inlet',
        'response_fn_48': 'Gas_2 diss. at inlet',
        'response_fn_49': 'u_1 at inlet'
    }

    # Creiamo un DataFrame con una sola riga contenente le descrizioni
    labels_df = pd.DataFrame([labels])

    # Concatenazione: prima riga descrizioni, poi dati veri
    final_df = pd.concat([labels_df, output_df], ignore_index=True)

    # Salvataggio in csv
    final_df.to_csv('data_at_inlet_total.csv', index=False)

    #output_df.to_csv('data_at_inlet_total.csv', index=False)
    print("\nEstrazione completata. Dati salvati in 'data_at_inlet_total.csv'.")

    # Rimuoviamo le simulazini nulle
    rm.remove_null_simulations("data_at_inlet_total.csv", "data_at_inlet.csv")


def extract_data_at_vent (main_dir, bak_name, N):
    """
    Estrae i dati da tutti i file .bak e .std contenuti nelle cartelle workdir.N.
    
    Args:
        main_dir (str): Il percorso della cartella principale contenente le cartelle workdir.
        bak_name (str): Il nome del file .bak di riferimento (es. Piton_Emb2.bak).
    """
    
    if os.path.exists('data_at_vent_total.csv'):
        print(f"'data_at_vent_total.csv' esiste già")
        
        if os.path.exists('data_at_vent.csv'):
            print("'data_at_vent.csv' esiste già")
            print("La funzione extract_data_at_vent non verrà eseguita.")
            return
        
        print("'data_at_vent.csv' invece va creato")
        rm.remove_null_simulations("data_at_vent_total.csv", "data_at_vent.csv")
        return

    # 1. Inizializza i dizionari per i dati di risposta
    response_data = {f'response_fn_{i}': np.full(N, np.nan) for i in range(50, 61)}
    
    # 2. Loop sulle simulazioni
    for index_simul in range(N):
        print(f'\r{round((index_simul + 1)/N *100, 1)}% of extract_data_at_vent completed', end='', flush=True) 

        #Chiamo funzione che cerca e legge .std file e .bak file
        std_data, bak_data = utils.search_and_read_std_file( main_dir, bak_name, index_simul)       

        # 3. Elaborazione dei dati se i file sono disponibili
        if (std_data is not None and bak_data is not None):

            data = utils.process_simulation_data(std_data, bak_data)

            # Per leggibilita` del seguito, copio localmente alcune variabili
            radius = data['RADIUS']
            N_CRY = data['N_CRY']
            N_GAS = data['N_GAS']
            #P_OUT = data['P_OUT']
            #VISC_2 = data['VISC_2']
            #ELASTIC_MODULUS = data['ELASTIC_MODULUS']
            #MAX_PACKING_FRACTION = data['MAX_PACKING_FRACTION']
            #STRAIN_RATE_VISCOSITY = data['STRAIN_RATE_VISCOSITY']
            #RE_BUBBLY_MAGMA_EXP_THR = data['RE_BUBBLY_MAGMA_EXP_THR']
            #LOG10_BUBBLE_NUMBER_DENSITY = data['LOG10_BUBBLE_NUMBER_DENSITY']
            #TORTUOSITY_FACTOR = data['TORTUOSITY_FACTOR']
            #THROAT_BUBBLE_RATIO = data['THROAT_BUBBLE_RATIO']
            SURFACE_TENSION = data['SURFACE_TENSION']

            #zeta_grid = data['zeta_grid']
            #T=data['T']
            #p_1= data['p_1']
            alfa_1 = data['alfa_1']
            alfa_2 = data['alfa_2']
            x_d = data['x_d']
            beta = data['beta']
            rho_1 = data['rho_1']
            rho_mix = data['rho_mix']
            u_1 = data['u_1']
            u_mix = data['u_mix']
            u_rel = data['u_rel']
            visc = data['visc']
            #visc_melt = data['visc_melt']
            #elongational_strain_rate = data['elongational_strain_rate']
            #deborah_number = data['deborah_number']
            #deborah_threshold = data['deborah_threshold']

            # Estrazione dei valori al vent (indice -1 in Python)

            visc_at_vent = visc[-1]

            if (N_CRY == 3):
                beta_at_vent = [beta[0, -1], beta[1, -1], beta[2, -1]]
            elif (N_CRY == 1):
                beta_at_vent = [beta[0, -1], 0.0, 0.0]
            else:
                beta_at_vent = [0.0, 0.0, 0.0]
            beta_tot_at_vent = sum(beta_at_vent)

            #T_at_vent = T[-1]
            #P_at_vent = p_1[-1]
            #u_1_at_vent = u_1[-1]
            #vesic_at_vent = 1.0 - alfa_1[-1]
            
            if (N_GAS == 2):
                alfa_2_vent = [alfa_2[0, -1], alfa_2[1, -1]]
                x_d_at_vent = [x_d[0, -1], x_d[1, -1]]
            elif (N_GAS == 1):
                alfa_2_vent = [alfa_2[0, -1], 0.-1]
                x_d_at_vent = [x_d[0, -1], 0.-1]
            else:
                alfa_2_vent = [0.0, 0.0]
                x_d_at_vent = [0.0, 0.0]

            
            rho_g = (rho_mix - alfa_1 * rho_1) / (1.0 - alfa_1)
            rho_l = rho_1
            density_l = rho_l[-1] # density of the liquid, kg/m3
            density_g = rho_g[-1] # density of the gas, kg/m3
            #P_infty = P_OUT #Pressure at infinity, Pa
            #P_r = p_1[-1] #Pressure of the bubble, Pa 
            mu_l = visc[-1] # viscosity of the liquid, Pa s (EB: ma non si dovrebbe usare "visc_melt"?)
            gas_melt_rel_velocity = u_rel[-1] # m/s
            #gas_volume_fraction = 1.0 - alfa_1[-1]
            #bubble_number_density = 10^LOG10_BUBBLE_NUMBER_DENSITY
            
            filament_diam0 = 0.05 # m
            filament_diam = np.logspace(-6, -1, 1000)

            lambda_cap = np.sqrt(density_l * filament_diam**3 / SURFACE_TENSION)
            lambda_vis = mu_l * filament_diam / SURFACE_TENSION


            Oh_number = lambda_vis / lambda_cap
            extension_rate = gas_melt_rel_velocity / filament_diam0 * np.sqrt(density_g / density_l)
            lambda_def = 1.0 / extension_rate

            fluid_breakup = 0.0 
            filament_breakup = 0.0
            De_star = 0.0

            # Ciclo for per la logica condizionale
            # Usiamo 'enumerate' per ottenere sia l'indice che il valore
            for i in range(len(filament_diam)):
                if Oh_number[i-1] > 1:
                    lambda_inst = lambda_vis[i-1]
                else:
                    lambda_inst = lambda_cap[i-1]

                De_star = lambda_inst / lambda_def

                if De_star < 1:
                    fluid_breakup = 1.0
                    filament_breakup = filament_diam[i-1]
                else:
                    break # Ferma il ciclo alla prima condizione non soddisfatta



            # (6) Salvataggio dei risultati nel dizionario
            response_data['response_fn_50'][index_simul] = visc_at_vent
            response_data['response_fn_51'][index_simul] = beta_at_vent[0]
            response_data['response_fn_52'][index_simul] = beta_at_vent[1]
            response_data['response_fn_53'][index_simul] = beta_at_vent[2]
            response_data['response_fn_54'][index_simul] = beta_tot_at_vent
            response_data['response_fn_55'][index_simul] = alfa_2_vent[0]
            response_data['response_fn_56'][index_simul] = alfa_2_vent[1]
            response_data['response_fn_57'][index_simul] = x_d_at_vent[0]
            response_data['response_fn_58'][index_simul] = x_d_at_vent[1]
            response_data['response_fn_59'][index_simul] = fluid_breakup
            response_data['response_fn_60'][index_simul] = filament_breakup


    # 4. Salva i risultati in un file
    output_df = pd.DataFrame(response_data)

    # Dizionario con le descrizioni leggibili
    labels = {
        'response_fn_50': 'Viscosity at vent',
        'response_fn_51': 'Fraz. crystal 1 at vent',
        'response_fn_52': 'Fraz. crystal 2 at vent',
        'response_fn_53': 'Fraz. crystal 3 at vent',
        'response_fn_54': 'Total cry. fract. at vent',
        'response_fn_55': 'Gas_1 exs. fraction at vent',
        'response_fn_56': 'Gas_1 exs. fraction at vent',
        'response_fn_57': 'Dissolved gas_1 fraction at vent',
        'response_fn_58': 'Dissolved gas_2 fraction at vent',
        'response_fn_59': 'Fluid breakup flag at vent',
        'response_fn_60': 'Filament breakup diameter at vent'
    }

    # Creiamo un DataFrame con una sola riga contenente le descrizioni
    labels_df = pd.DataFrame([labels])

    # Concatenazione: prima riga descrizioni, poi dati veri
    final_df = pd.concat([labels_df, output_df], ignore_index=True)

    # Salvataggio in csv
    final_df.to_csv('data_at_vent_total.csv', index=False)
    #output_df.to_csv('data_at_vent_total.csv', index=False)
    print("\nEstrazione completata. Dati salvati in 'data_at_vent_total.csv'.")

    # Rimuoviamo le simulazini nulle
    rm.remove_null_simulations("data_at_vent_total.csv", "data_at_vent.csv")


def extract_data_average (main_dir, bak_name, N):
    """
    Estrae i dati da tutti i file .bak e .std contenuti nelle cartelle workdir.N.
    
    Args:
        main_dir (str): Il percorso della cartella principale contenente le cartelle workdir.
        bak_name (str): Il nome del file .bak di riferimento (es. Piton_Emb2.bak).
    """

    if os.path.exists('data_average_total.csv'):
        print(f"'data_average_total.csv' esiste già")
        
        if os.path.exists('data_average.csv'):
            print("'data_average.csv' esiste già")
            print("La funzione extract_data_average non verrà eseguita.")
            return
        
        print("'data_average.csv' invece va creato")
        rm.remove_null_simulations("data_average_total.csv", "data_average.csv")
        return

    # 1. Inizializza i dizionari per i dati di risposta
    response_data = {f'response_fn_{i}': np.full(N, np.nan) for i in range(70, 76)}
    
    # 2. Loop sulle simulazioni
    for index_simul in range(N):
        print(f'\r{round((index_simul + 1)/N *100, 1)}% of extract_data_average completed', end='', flush=True) 

        #Chiamo funzione che cerca e legge .std file e .bak file
        std_data, bak_data = utils.search_and_read_std_file( main_dir, bak_name, index_simul)       

        # 3. Elaborazione dei dati se i file sono disponibili
        if (std_data is not None and bak_data is not None):

            data = utils.process_simulation_data(std_data, bak_data)

            # Per leggibilita` del seguito, copio localmente alcune variabili
            Z0 = data['Z0']
            ZN = data['ZN']
            RADIUS = data['RADIUS']
            #N_CRY = data['N_CRY']
            #N_GAS = data['N_GAS']
            #P_OUT = data['P_OUT']
            VISC_2 = data['VISC_2']
            #ELASTIC_MODULUS = data['ELASTIC_MODULUS']
            #MAX_PACKING_FRACTION = data['MAX_PACKING_FRACTION']
            #STRAIN_RATE_VISCOSITY = data['STRAIN_RATE_VISCOSITY']
            #RE_BUBBLY_MAGMA_EXP_THR = data['RE_BUBBLY_MAGMA_EXP_THR']
            LOG10_BUBBLE_NUMBER_DENSITY = data['LOG10_BUBBLE_NUMBER_DENSITY']
            TORTUOSITY_FACTOR = data['TORTUOSITY_FACTOR']
            THROAT_BUBBLE_RATIO = data['THROAT_BUBBLE_RATIO']
            LOG10_FRICTION_COEFFICIENT = data['LOG10_FRICTION_COEFFICIENT']
            #SURFACE_TENSION = data['SURFACE_TENSION']

            zeta_grid = data['zeta_grid']
            T=data['T']
            p_1= data['p_1']
            alfa_1 = data['alfa_1']
            #alfa_2 = data['alfa_2']
            #x_d = data['x_d']
            #beta = data['beta']
            rho_1 = data['rho_1']
            rho_mix = data['rho_mix']
            u_1 = data['u_1']
            u_2 = data['u_2']
            u_mix = data['u_mix']
            #u_rel = data['u_rel']
            visc = data['visc']
            #visc_melt = data['visc_melt']
            #elongational_strain_rate = data['elongational_strain_rate']
            #deborah_number = data['deborah_number']
            #deborah_threshold = data['deborah_threshold']

            # Estrazione dei valori averaged 

            delta_time = np.diff(zeta_grid) / ( 0.5*(u_1[:-1] + u_1[1:]));
            time = np.zeros(len(delta_time) + 1)
            time[1:] = np.cumsum(delta_time)

            cooling_rate = np.zeros_like(u_mix)
            cooling_rate[1:] = (T[:-1] - T[1:]) / (time[1:] - time[:-1])
                    
            u_mix_avg = np.sum(0.5 * (u_mix[1:] + u_mix[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)

            #u_1_avg = np.sum(0.5 * (u_1[1:] + u_1[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)

            #u_2_avg = np.sum(0.5 * (u_2[1:] + u_2[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)

            rho_mix_avg = np.sum(0.5 * (rho_mix[1:] + rho_mix[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)
            
            rho_1_avg = np.sum(0.5 * (rho_1[1:] + rho_1[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)

            alfa_1_avg = np.sum(0.5 * (alfa_1[1:] + alfa_1[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)

            alfa_2_avg = 1.0 - alfa_1_avg

            visc_avg = np.sum(0.5 * (visc[1:] + visc[:-1]) * np.diff(zeta_grid)) / (ZN - Z0)

            Reynolds_avg = (rho_mix_avg * u_mix_avg * RADIUS * 2)/visc_avg

            BUBBLE_NUMBER_DENSITY = 10**LOG10_BUBBLE_NUMBER_DENSITY
            
            FRICTION_COEFFICIENT = 10**LOG10_FRICTION_COEFFICIENT

            r_b_avg = (alfa_2_avg / (4.0/3.0 * np.pi * BUBBLE_NUMBER_DENSITY * (1.0 - alfa_2_avg) ) )**(1/3)

            k1_avg = ((THROAT_BUBBLE_RATIO * r_b_avg)**2.0 * alfa_2_avg ** TORTUOSITY_FACTOR) / 8.0

            #k2_avg = ((THROAT_BUBBLE_RATIO * r_b_avg) * alfa_2_avg ** (0.5 + 1.5*TORTUOSITY_FACTOR)) / FRICTION_COEFFICIENT

            #Stokes_avg = (rho_1_avg * k1_avg / VISC_2 ) / (RADIUS / u_mix_avg)
            
            T_min = np.min(T)
            index_T_min = np.argmin(T) # np.argmin() restituisce l'indice del valore minimo

            cooling_rate_avg = (T[0] - T_min)/(time[index_T_min]);
            
            decompression_rate_avg = (p_1[0] - p_1[-1]) / 1e6 / time[-1]


            # (6) Salvataggio dei risultati nel dizionario
            response_data['response_fn_70'][index_simul] = u_mix_avg
            response_data['response_fn_71'][index_simul] = rho_mix_avg
            response_data['response_fn_72'][index_simul] = visc_avg
            response_data['response_fn_73'][index_simul] = Reynolds_avg
            response_data['response_fn_74'][index_simul] = cooling_rate_avg
            response_data['response_fn_75'][index_simul] = decompression_rate_avg


    # 4. Salva i risultati in un file
    output_df = pd.DataFrame(response_data)

    # Dizionario con le descrizioni leggibili
    labels = {
        'response_fn_70': 'Average magma velocity',
        'response_fn_71': 'Average mixture density',
        'response_fn_72': 'Average viscosity',
        'response_fn_73': 'Average Reynolds number',
        'response_fn_74': 'Average cooling rate',
        'response_fn_75': 'Average decompression rate'
    }

    # Creiamo un DataFrame con una sola riga contenente le descrizioni
    labels_df = pd.DataFrame([labels])

    # Concatenazione: prima riga descrizioni, poi dati veri
    final_df = pd.concat([labels_df, output_df], ignore_index=True)

    # Salvataggio in csv
    final_df.to_csv('data_average_total.csv', index=False)
    #output_df.to_csv('data_average_total.csv', index=False)
    print("\nEstrazione completata. Dati salvati in 'data_average_total.csv'.")

    # Rimuoviamo le simulazini nulle
    rm.remove_null_simulations("data_average_total.csv", "data_average.csv")

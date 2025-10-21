import numpy as np

def run_all_plots(
    plot_sobol_indices,
    sobol_indices
):
    """
    Script utente per generare i plot desiderati.
    Tutto quello che sta qui può essere modificato liberamente.
    """

    save_dir="plot_Sobol"

    # plot 1

    response_labels_example = {
        'response_fn_1': 'Gas volume fraction',
        'response_fn_15': 'Fragmentation depth',
        'response_fn_12': 'Mass flow rate',
        'response_fn_4': 'Exit velocity',
        'response_fn_16': 'Exit crystal content',
        'response_fn_28': 'Undercooling @Frag'
    }

    plot_sobol_indices(
        sobol_indices, 
        response_labels=response_labels_example, 
        save_name='sobol_indices',
        save_dir=save_dir
    )

    # plot 2

    xi_labels = ['Press.','Temp.','Radius','H2O','CO2','Crystals']
    response_labels_example = {
        'response_fn_1': 'Gas volume fraction',
        'response_fn_15': 'Fragmentation depth',
        'response_fn_12': 'Mass flow rate',
        'response_fn_4': 'Exit velocity',
        'response_fn_16': 'Exit crystal content',
        'response_fn_28': 'Undercooling @Frag',
        'response_fn_72': 'Avg. viscosity'
    }

    plot_sobol_indices(
        sobol_indices, 
        xi_labels=xi_labels, 
        response_labels=response_labels_example, 
        save_name='sobol_indices_xi_Labels',
        save_dir=save_dir
    )

import numpy as np

def run_all_plots(
    plot_frequencies_eruptive_styles,
    df_concat,
    df_concat_expl,
    df_concat_notExpl,
    df_concat_eff,
    df_concat_fount
):

    """
    Script utente per generare i plot desiderati.
    Tutto quello che sta qui può essere modificato liberamente.
    """

    save_dir="plot_frequencies"

    #region -- Plot frequencies to compare the different eruptive styles

    N_bins=40

    variables_to_plot = [
        {"col": "x2", "transform": lambda x: x-273, "label": "Inlet temperature [°C]"},
        {"col": "x1", "transform": lambda x: x/1e6, "label": "Inlet pressure [MPa]"},
        {"col": "x3"},
        {"col": "x4", "transform": lambda x: x*100, "label": "Inlet H2O content [wt.%]"},
        {"col": "x5", "transform": lambda x: x*100, "label": "Inlet CO2 content [wt.%]"},
        {"col": "x6", "transform": lambda x: x*100, "label": "Inlet phenocrystal content [vol.%]"}
    ]

    plot_frequencies_eruptive_styles(
        variables_to_plot,
        df_concat=df_concat, 
        df_concat_expl=df_concat_expl, 
        df_concat_eff=df_concat_eff, 
        df_concat_fount=df_concat_fount, 
        N_bins=N_bins, 
        save_name=f"freq_input_parameters_{N_bins}bins"
    )

    variables_to_plot = [
        {"col": "response_fn_12", "transform": np.log10, "yscale": "log", "label": "Log10(MFR) [kg/s]"},
        {"col": "response_fn_4", "transform": np.log10,  "yscale": "log", "label": "Log10(exit velocity) [m/s]"},
        {"col": "response_fn_20", "transform": np.log10,  "yscale": "log", "label": "Log10(Viscosity) [Pa s]"},
        {"col": "response_fn_1", "yscale": "log"},
 #       {"col": "response_fn_15"},
        #{"col": "response_fn_44"},
        #{"col": "response_fn_16"},
        #{"col": "response_fn_20", "transform": np.log10, "label": "Log10(Viscosity at fragm) [Pa s]", "xscale": "log"}
    ]

    plot_frequencies_eruptive_styles(
        variables_to_plot,
        df_concat, 
        df_concat_expl=df_concat_expl, 
        df_concat_eff=df_concat_eff, 
        df_concat_fount=df_concat_fount, 
        N_bins=N_bins, 
        save_name=f"freq_output_parameters_{N_bins}bins",
        plot_total=False
    )
    #endregion

import numpy as np

def run_all_plots(
    plot_xi_histograms,
    plot_histograms_list,
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

    save_dir="plot_histograms"

    # Inizializzo le liste di variabili da plottare 
    inputs_scaled = [
        {"col": "x1", "transform": lambda x: x/1e6, "label": "Inlet Pressure [MPa]", "color": "g"},
        {"col": "x2", "transform": lambda x: x-273, "label": "Inlet Temperature [°C]", "color": "g"},
        {"col": "x3", "label": "Radius [m]", "color": "g"},
        {"col": "x4", "transform": lambda x: x*100, "label": "Inlet H₂O content [wt.%]", "color": "g"},
        {"col": "x5", "transform": lambda x: x*100, "label": "Inlet CO₂ content [wt.%]", "color": "g"},
        {"col": "x6", "transform": lambda x: x*100, "label": "Inlet phenocryst. content [vol.%]", "color": "g"}
    ]

    x_axis_1 = [
        {"col":"response_fn_1", "label": "Gas volume fraction at inlet"},
        {"col":"response_fn_12", "transform": np.log10, "label": "Log10(MFR) [kg/s]"},
        {"col":"response_fn_4", "label": "Exit velocity [m/s]"},       
    ]

    x_axis_2 = [
        {"col":"response_fn_16", "transform": lambda x: x*100, "label": "Exit crystal content [vol.%]"},
        {"col":"response_fn_15", "label": "Fragmentation depth [m]", "color": "red"},
        {"col": "response_fn_18", "label": "Fragm Crit.", "color": "g", "edgecolor": "black", "xlim": (0, 4),
            "xticks": [1, 2, 3], 
            "xticklabels": ["SR", "IN", "BO"]
        },
    ]

    x_axis_3 = [
        {"col":"response_fn_27", "transform": lambda x: x*100, "label": "Gas fraction at fragm. [vol.%]"},
        {"col":"response_fn_28", "label": "Undercooling at fragm. [°C]"},
        {"col":"response_fn_20", "transform": np.log10, "label": "Log10(Viscosity at fragm.) [Pa s]"}
    ]

    #region -- Plot histograms for ALL simulations 

    plot_xi_histograms(
        df={"All simulations":df_concat},
        save_name="freq_allSim_inputs"
    )

    plot_xi_histograms(
        df = {"All simulations":df_concat},
        var_specs = inputs_scaled,
        save_name="freq_allSim_inputs_scaled"
    )
    
    plot_histograms_list(
        df = {"All simulations":df_concat},
        x_axis = x_axis_1 + x_axis_2,
        save_name="freq_allSim_1"
    )

    plot_histograms_list(
        df = {"All simulations":df_concat},
        x_axis = x_axis_3,
        save_name="freq_allSim_2"
    )
    #endregion

    #region -- Plot histograms for explosive simulations 

    plot_xi_histograms(
        df={"Explosive": df_concat_expl},
        save_name="freq_expl_inputs"
    )

    plot_xi_histograms(
        df = {"Explosive": df_concat_expl},
        var_specs = inputs_scaled,
        save_name="freq_expl_inputs_scaled"
    )
    
    plot_histograms_list(
        df = {"Explosive": df_concat_expl},
        x_axis = x_axis_1 + x_axis_2,
        save_name="freq_expl_1"
    )

    plot_histograms_list(
        df = {"Explosive": df_concat_expl},
        x_axis = x_axis_3,
        save_name="freq_expl_2"
    )
    #endregion

    #region -- Plot histograms for notExplosive simulations 

    plot_xi_histograms(
        df={"Not explosive": df_concat_notExpl},
        save_name="freq_notExpl_inputs"
    )

    plot_xi_histograms(
        df = {"Not explosive": df_concat_notExpl},
        var_specs = inputs_scaled,
        save_name="freq_notExpl_inputs_scaled"
    )
    
    plot_histograms_list(
        df = {"Not explosive": df_concat_notExpl},
        x_axis = x_axis_1 + [{"col":"response_fn_19", "transform": np.log10, "label": "Log10(Fountain height) (m)"}],
        save_name="freq_notExpl_1"
    )

    plot_histograms_list(
        df = {"Not explosive": df_concat_notExpl},
        x_axis = x_axis_3,
        save_name="freq_notExpl_2"
    )
    #endregion

    #region -- Plot histograms for notExplosive-Effusive simulations 

    plot_xi_histograms(
        df={"Effusive": df_concat_eff},
        save_name="freq_eff_inputs"
    )

    plot_xi_histograms(
        df = {"Effusive": df_concat_eff},
        var_specs = inputs_scaled,
        save_name="freq_eff_inputs_scaled"
    )
    
    plot_histograms_list(
        df = {"Effusive": df_concat_eff},
        x_axis = x_axis_1 + [{"col":"response_fn_19", "transform": np.log10, "label": "Log10(Fountain height) (m)"}],
        save_name="freq_eff_1"
    )

    plot_histograms_list(
        df = {"Effusive": df_concat_eff},
        x_axis = x_axis_3,
        save_name="freq_eff_2"
    )
    #endregion

    #region -- Plot histograms for notExplosive-Fountaining simulations 

    plot_xi_histograms(
        df={"Fountaining": df_concat_fount},
        save_name="freq_fount_inputs"
    )

    plot_xi_histograms(
        df = {"Fountaining": df_concat_fount},
        var_specs = inputs_scaled,
        save_name="freq_fount_inputs_scaled"
    )
    
    plot_histograms_list(
        df = {"Fountaining": df_concat_fount},
        x_axis = x_axis_1 + [{"col":"response_fn_19", "transform": np.log10, "label": "Log10(Fountain height) (m)"}],
        save_name="freq_fount_1"
    )

    plot_histograms_list(
        df = {"Fountaining": df_concat_fount},
        x_axis = x_axis_3,
        save_name="freq_fount_2"
    )
    #endregion
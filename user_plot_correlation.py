import numpy as np

def run_all_plots(
    plot_xi_vs_response_fn,
    plot_correlation_lists,
    df_concat,
    df_concat_expl,
    df_concat_notExpl,
    df_concat_eff,
    df_concat_fount,
    input_Min,
    input_Max,
    stats
):
    """
    Script utente per generare i plot desiderati.
    Tutto quello che sta qui può essere modificato liberamente.
    """

    save_dir="plot_correlations"

    #region -- plot correlazioni Expl/Eff/Fount con response_fn_1 
    plot_xi_vs_response_fn(
        dfs = {
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_1',
        n_step=1, 
        save_name="corrExpEffFount_xi_vs_resp_fn_1_GasFraction",
        save_dir=save_dir,
        stats=stats
    )
    #endregion 

    #region -- plot correlazioni Expl con response_fn_15 
    plot_xi_vs_response_fn(
        dfs = {"Explosive": df_concat_expl},
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_15',
        n_step=1, 
        save_name="corrExp_xi_vs_resp_fn_15_FragmDepth",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----

    #region -- plot correlazioni Expl/Eff/Fount con response_fn_12 
    list_for_x_axis = [
        ("x1",lambda v: v/1e6, "Inlet pressure [MPa]"), #1
        ("x2",lambda v: v-273, "Inlet temperature [°C]"), #2
        ("x3",), #3
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #4
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #5
        ("x6",lambda v: v*100, "Inlet phenocryst. content [vol.%]"), #6
    ]

    list_for_y_axis = [
        ("response_fn_12", np.log10, "log10(MER) [kg/s]"), #1
        ("response_fn_12", np.log10, "log10(MER) [kg/s]"), #2
        ("response_fn_12", np.log10, "log10(MER) [kg/s]"), #3
        ("response_fn_12", np.log10, "log10(MER) [kg/s]"), #4
        ("response_fn_12", np.log10, "log10(MER) [kg/s]"), #5
        ("response_fn_12", np.log10, "log10(MER) [kg/s]"), #6
    ]

    plot_correlation_lists(
        dfs={
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=1,
        save_name="corrExpEffFount_xi_vs_resp_fn_12_log10MER",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----

    #region -- plot correlazioni Expl/Eff/Fount con response_fn_4 
    list_for_x_axis = [
        ("x1",lambda v: v/1e6, "Inlet pressure [MPa]"), #1
        ("x2",lambda v: v-273, "Inlet temperature [°C]"), #2
        ("x3",), #3
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #4
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #5
        ("x6",lambda v: v*100, "Inlet phenocryst. content [vol.%]"), #6
    ]

    list_for_y_axis = [
        ("response_fn_4", np.log10), #1
        ("response_fn_4", np.log10), #2
        ("response_fn_4", np.log10), #3
        ("response_fn_4", np.log10), #4
        ("response_fn_4", np.log10), #5
        ("response_fn_4", np.log10), #6
    ]

    plot_correlation_lists(
        dfs={
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=1,
        save_name="corrExpEffFount_xi_vs_resp_fn_4_log10ExitVelocity",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----

    #region -- Plot correlazioni Expl/Eff/Fount con response_fn_16 
    plot_xi_vs_response_fn(
        dfs = {
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_16',
        n_step=1,
        save_name="corrExpEffFount_xi_vs_resp_fn_16_ExitCrystalContent",
        save_dir=save_dir,
        stats=stats
    )
    #endregion

    #region -- Plot correlazioni Expl/Eff/Fount con response_fn_28 
    plot_xi_vs_response_fn(
        dfs = {
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        input_Min=input_Min,
        input_Max=input_Max,
        response_col='response_fn_28',
        n_step=1,
        save_name="corrExpEffFount_xi_vs_resp_fn_28_Undercooling@Fragm",
        save_dir=save_dir,
        stats=stats
    )
    #endregion
    
    #region -- Plot correlazioni Expl/Eff/Fount con x4 
    list_for_x_axis = [
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #1
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #2
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #3
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #4
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #5
        ("x4",lambda v: v*100, "Inlet H₂O content [wt.%]"), #6
    ]

    list_for_y_axis = [
        ("response_fn_4", None, "Exit velocity [m/s]"), #1
        ("response_fn_15"), #2
        ("response_fn_30", lambda v: v*100, "Diss. H2O content @Frag [wt.%]"), #3
        ("response_fn_25"), #4
        ("response_fn_24"), #5
        ("response_fn_20", np.log10, "Log 10 Viscosity @Frag [Pa s]"), #6
    ]

    plot_correlation_lists(
        #dfs=df_concat,
        dfs = {
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=1,
        save_name="corrExpEffFount_x4",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----

    #region -- Plot correlazioni Expl/Eff/Fount con x5 
    list_for_x_axis = [
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #1
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #2
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #3
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #4
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #5
        ("x5",lambda v: v*100, "Inlet CO₂ content [wt.%]"), #6
    ]

    list_for_y_axis = [
        ("response_fn_4", None, "Exit velocity [m/s]"), #1
        ("response_fn_15"), #2
        ("response_fn_30", None, "Diss. H2O content @Frag [wt.%]"), #3
        ("response_fn_25"), #4
        ("response_fn_24"), #5
        ("response_fn_20", np.log10, "Log 10 Viscosity @Frag [Pa s]"), #6
    ]

    plot_correlation_lists(
        #dfs=df_concat,
        dfs = {
            "Explosive"   : df_concat_expl,
            "Effusive"    : df_concat_eff,
            "Fountaining" : df_concat_fount
        },
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=input_Min,
        input_Max=input_Max,
        n_step=1,
        save_name="corrExpEffFount_x5",
        save_dir=save_dir,
        stats=stats
    )
    #endregion ----

    """
    #region -- ESEMPIO DI PLOT VARI CON TRASFORMAZIONI E LABELS --
    list_for_x_axis = [
        ("x1", None, None),
        ("x2", lambda v: v-273, "Temperatura (°C)"),
        ("response_fn_17", None, "Undercooling [K]"),
        ("x1", None, None),
        ("x2", lambda v: v-273, "Temperatura (°C)"),
        ("response_fn_17", None, "Undercooling [K]"),
    ]

    list_for_y_axis = [
        ("response_fn_55", lambda v: v*100, "H₂O Diss. (%)"),
        ("x5", None, None),
        ("response_fn_20", np.log10, "log10(Y)"),
        ("response_fn_55", lambda v: v*100, "H₂O Diss. (%)"),
        ("x5", None, None),
        ("response_fn_20", np.log10, "log10(Y)"),
    ]

    plot_correlation_lists(
        df=df_concat,
        x_axis=list_for_x_axis,
        y_axis=list_for_y_axis,
        input_Min=adj_input_Min,
        input_Max=adj_input_Max,
        n_step=1,
        save_name="plot_correlazioni_varie",
        stats=stats
    )
    #endregion
    """

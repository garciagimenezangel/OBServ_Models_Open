
# Model names
di_model_name = "Lonsdorf.optimizedGA_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult"
dict_mm_models = dict()
dict_mm_models["Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult"] = "Baseline"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open1_forEd1_crEd1_div1_ins1max_dist1_suitmult"] = "All modules"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open1_forEd0_crEd0_div0_ins0max_dist0_suitmult"] = "Open forest"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open0_forEd1_crEd0_div0_ins0max_dist0_suitmult"] = "Edge forest"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd1_div0_ins0max_dist0_suitmult"] = "Edge crop fields"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div1_ins0max_dist0_suitmult"] = "Landsc complexity"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins1max_dist0_suitmult"] = "Pollinators activity"
dict_mm_models["Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist1_suitmult"] = "Dist to seminatural"
dict_mm_models["Lonsdorf.ESTIMAP_lcCont0_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult"] = "Discrete LC types"
dict_ml_models = dict()
dict_ml_models["pred_svr"] = "SVR"
dict_ml_models["pred_br"] = "BayR"
dict_ml_models["pred_gbr"] = "GBR"
di_model = 'Lonsdorf.optimizedGA_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'

# mm_model = 'Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins1max_dist0_suitmult'
mm_model_name = "Pollinators activity"
mm_model_shortname = "Poll.Activ."
# di_model = 'Lonsdorf.optimizedGA_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'
ml_model = 'pred_br'
ml_model_name = 'BayR'

# Datasets
dataset_mm_local = 'lons_studywise'
dataset_ml_test = 'ml_test'

# Output figures
dir_fig = "C:/Users/Angel/Downloads"
eps_fig_landsc_var = 'local_analysis_landsc_var.eps'

import pandas as pd
import metrics
import settings as sett


def run():
    df_stats = pd.DataFrame()
    df_stats, df_data_mm_small, df_data_mm_large = compute_mm_stats(df_stats)
    df_stats, df_data_di_small, df_data_di_large = compute_di_stats(df_stats)
    df_stats, df_data_ml = compute_ml_stats(df_stats)
    return df_stats, df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml


def compute_ml_stats(df_stats):
    # Machine Learning models
    for model in ['pred_svr', 'pred_gbr', 'pred_br']:
        df_data = metrics.get_data(model, "All", sett.dataset_ml_test)
        df_global = metrics.get_metrics_global(df_data, 'All')
        df_global['model'] = "ML ({})".format(sett.dict_ml_models[model])
        df_global['guild'] = 'Bumblebees+Other wildbees'
        df_stats = pd.concat([df_stats, df_global], axis=0)
        if model == sett.sel_ml_model:
            df_selected_model = df_data
    return df_stats, df_selected_model


def compute_di_stats(df_stats):
    # Data-informed model metrics.
    df_data_di_small = metrics.get_data(sett.di_model_name, "Small", sett.dataset_ml_test)
    df_data_di_large = metrics.get_data(sett.di_model_name, "Large", sett.dataset_ml_test)
    df_di_small = metrics.get_metrics_global(df_data_di_small, "Small")
    df_di_large = metrics.get_metrics_global(df_data_di_large, 'Large')
    df_di_small['model'] = "DI-MM"
    df_di_small['guild'] = 'Other wildbees'
    df_di_large['model'] = 'DI-MM'
    df_di_large['guild'] = 'Bumblebees'
    df_stats = pd.concat([df_stats, df_di_small], axis=0)
    df_stats = pd.concat([df_stats, df_di_large], axis=0)
    return df_stats, df_data_di_small, df_data_di_large


def compute_mm_stats(df_stats):
    # Mechanistic models
    for model in sett.dict_mm_models.keys():
        df_data_mm_small = metrics.get_data(model, "Small", sett.dataset_mm_global)
        df_data_mm_large = metrics.get_data(model, "Large", sett.dataset_mm_global)
        df_mm_small = metrics.get_metrics_global(df_data_mm_small, "Small")
        df_mm_large = metrics.get_metrics_global(df_data_mm_large, 'Large')
        df_mm_small['model'] = "MM ({})".format(sett.dict_mm_models[model])
        df_mm_small['guild'] = 'Other wildbees'
        df_mm_large['model'] = "MM ({})".format(sett.dict_mm_models[model])
        df_mm_large['guild'] = 'Bumblebees'
        df_stats = pd.concat([df_stats, df_mm_small], axis=0)
        df_stats = pd.concat([df_stats, df_mm_large], axis=0)
        if model == sett.sel_mm_model:
            df_small_selected = df_data_mm_small
            df_large_selected = df_data_mm_large
    return df_stats, df_small_selected, df_large_selected

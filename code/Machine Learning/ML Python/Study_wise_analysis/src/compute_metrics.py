import pandas as pd
import numpy as np
import metrics
import settings as sett
import pollinators_dependency as poll_dep


def run():
    df_stats = pd.DataFrame()
    df_stats, df_data_mm_small, df_data_mm_large = compute_mm_stats(df_stats)
    df_stats, df_data_di_small, df_data_di_large = compute_di_stats(df_stats)
    df_stats, df_data_ml = compute_ml_stats(df_stats)
    dict_study_poll_dep = compute_poll_dep()
    dict_management = compute_management()
    df_stats = df_stats.reset_index().drop(columns='level_1')
    df_stats['poll_dep'] = [dict_study_poll_dep[x] if x in dict_study_poll_dep.keys() else 'unknown' for x in df_stats['study_id']]
    df_stats['management'] = [set_management(x, dict_management) for x in df_stats['study_id']]
    return df_stats, df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml


def set_management(key, dict_management):
    default_val = 'conventional'
    if key in dict_management.keys():
        value = dict_management[key]
        if type(value) is str:
            return value
        else:
            return default_val
    else:
        return default_val


def compute_ml_stats(df_stats):
    # ML models
    for model in sett.dict_ml_models.keys():
        df_data = metrics.get_data(model, "All", sett.dataset_ml_test)
        df_ml = metrics.get_metrics_by_study(df_data, 'All')
        df_ml['model'] = "ML ({})".format(sett.dict_ml_models[model])
        df_ml['guild'] = 'Bumblebees+Wildbees'
        df_stats = pd.concat([df_stats, df_ml], axis=0)
    return df_stats, df_data


def compute_di_stats(df_stats):
    # Data-informed model metrics
    df_data_di_small = metrics.get_data(sett.di_model, "Small", sett.dataset_ml_test)
    df_data_di_large = metrics.get_data(sett.di_model, "Large", sett.dataset_ml_test)
    df_di_small = metrics.get_metrics_by_study(df_data_di_small, "Small")
    df_di_large = metrics.get_metrics_by_study(df_data_di_large, 'Large')
    df_di_small['model'] = "DI-MM"
    df_di_small['guild'] = 'Wildbees'
    df_di_large['model'] = 'DI-MM'
    df_di_large['guild'] = 'Bumblebees'
    df_stats = pd.concat([df_stats, df_di_small], axis=0)
    df_stats = pd.concat([df_stats, df_di_large], axis=0)
    return df_stats, df_data_di_small, df_data_di_large


def compute_mm_stats(df_stats):
    # Mechanistic models
    for model in sett.dict_mm_models.keys():
        df_data_mm_small = metrics.get_data(model, "Small", sett.dataset_mm_local)
        df_data_mm_large = metrics.get_data(model, "Large", sett.dataset_mm_local)
        df_data_mm_small = df_data_mm_small.copy().loc[np.invert(df_data_mm_small.model.apply(np.isnan))]
        df_data_mm_large = df_data_mm_large.copy().loc[np.invert(df_data_mm_large.model.apply(np.isnan))]
        df_data_mm_small['log_vr_small'] = np.log(df_data_mm_small['ab_wildbees'] + 1)
        df_data_mm_large['log_vr_large'] = np.log(df_data_mm_large['ab_bombus'] + 1)
        df_mm_small = metrics.get_metrics_by_study(df_data_mm_small, "Small")
        df_mm_large = metrics.get_metrics_by_study(df_data_mm_large, 'Large')
        df_data_mm_small.rename(columns={"log_vr_small": "log_ab_small"}, inplace=True)
        df_data_mm_large.rename(columns={"log_vr_large": "log_ab_large"}, inplace=True)
        df_mm_small['model'] = "MM ({})".format(sett.dict_mm_models[model])
        df_mm_small['guild'] = 'Wildbees'
        df_mm_large['model'] = "MM ({})".format(sett.dict_mm_models[model])
        df_mm_large['guild'] = 'Bumblebees'
        df_stats = pd.concat([df_stats, df_mm_small], axis=0)
        df_stats = pd.concat([df_stats, df_mm_large], axis=0)
    return df_stats, df_data_mm_small, df_data_mm_large


def compute_poll_dep():
    df_field = metrics.get_field_data(sett.dataset_mm_local)
    df_field['poll_dep'] = df_field['crop'].map(poll_dep.dep)
    dict_study_poll_dep = dict(zip(df_field.study_id, df_field.poll_dep))
    return dict_study_poll_dep


def compute_management():
    df_field = metrics.get_field_data(sett.dataset_mm_local)
    dict_study_management = dict(zip(df_field.study_id, df_field.management))
    return dict_study_management

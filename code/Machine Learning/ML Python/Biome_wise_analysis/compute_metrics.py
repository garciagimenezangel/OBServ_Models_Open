import pandas as pd
import numpy as np
import metrics
import biome_dict
import settings as sett


def run():
    df_stats = pd.DataFrame()
    df_stats, df_data_mm_small, df_data_mm_large = compute_mm_stats(df_stats)
    df_stats, df_data_di_small, df_data_di_large = compute_di_stats(df_stats)
    df_stats, df_data_ml = compute_ml_stats(df_stats)
    df_stats = df_stats.reset_index().drop(columns='level_1')
    df_stats['Biome'] = df_stats.biome_num.replace(biome_dict.biome_names)
    return df_stats, df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml


def compute_ml_stats(df_stats):
    # ML model
    df_biomes = pd.read_csv(sett.dataset_biomes)
    df_data = metrics.get_data(sett.ml_model, "All", sett.dataset_ml_test)
    df_data = df_data.merge(df_biomes, on=['site_id', 'study_id'])
    df_global = metrics.get_metrics_by_biome(df_data, 'All')
    df_global['model'] = "ML ({})".format(sett.ml_model_name)
    df_global['guild'] = 'Bumblebees+Wildbees'
    df_stats = pd.concat([df_stats, df_global], axis=0)
    return df_stats, df_data


def compute_di_stats(df_stats):
    # Data-informed model metrics
    df_biomes = pd.read_csv(sett.dataset_biomes)
    df_data_di_small = metrics.get_data(sett.di_model, "Small", sett.dataset_ml_test)
    df_data_di_large = metrics.get_data(sett.di_model, "Large", sett.dataset_ml_test)
    df_data_di_small = df_data_di_small.merge(df_biomes, on=['site_id', 'study_id'])
    df_data_di_large = df_data_di_large.merge(df_biomes, on=['site_id', 'study_id'])
    df_di_small = metrics.get_metrics_by_biome(df_data_di_small, "Small")
    df_di_large = metrics.get_metrics_by_biome(df_data_di_large, 'Large')
    df_di_small['model'] = "DI-MM"
    df_di_small['guild'] = 'Wildbees'
    df_di_large['model'] = 'DI-MM'
    df_di_large['guild'] = 'Bumblebees'
    df_stats = pd.concat([df_stats, df_di_small], axis=0)
    df_stats = pd.concat([df_stats, df_di_large], axis=0)
    return df_stats, df_data_di_small, df_data_di_large


def compute_mm_stats(df_stats):
    # Mechanistic model
    df_biomes = pd.read_csv(sett.dataset_biomes)
    df_data_mm_small = metrics.get_data(sett.mm_model, "Small", sett.dataset_mm_global)
    df_data_mm_large = metrics.get_data(sett.mm_model, "Large", sett.dataset_mm_global)
    df_data_mm_small = df_data_mm_small.copy().loc[np.invert(df_data_mm_small.model.apply(np.isnan))]
    df_data_mm_large = df_data_mm_large.copy().loc[np.invert(df_data_mm_large.model.apply(np.isnan))]
    df_data_mm_small = df_data_mm_small.merge(df_biomes, on=['site_id', 'study_id'])
    df_data_mm_large = df_data_mm_large.merge(df_biomes, on=['site_id', 'study_id'])
    df_mm_small = metrics.get_metrics_by_biome(df_data_mm_small, "Small")
    df_mm_large = metrics.get_metrics_by_biome(df_data_mm_large, 'Large')
    df_mm_small['model'] = "MM ({})".format(sett.mm_model_name)
    df_mm_small['guild'] = 'Wildbees'
    df_mm_large['model'] = "MM ({})".format(sett.mm_model_name)
    df_mm_large['guild'] = 'Bumblebees'
    df_stats = pd.concat([df_stats, df_mm_small], axis=0)
    df_stats = pd.concat([df_stats, df_mm_large], axis=0)
    return df_stats, df_data_mm_small, df_data_mm_large

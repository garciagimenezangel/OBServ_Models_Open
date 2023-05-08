import pandas as pd
import numpy as np
import settings as sett


def get_feature_data():
    df_features = pd.read_csv(sett.csv_features)
    # Set biome as categorical and replace NA by unknown
    df_features['biome_num'] = df_features.biome_num.astype('object')
    df_features['biome_num'] = df_features.biome_num.replace(np.nan, "unknown")
    df_features.drop(columns=['system:index', '.geo', 'refYear'], inplace=True)
    # Average columns related to land cover, computed with different radius for small and large guilds
    cols_to_avg = [col.split('_small')[0] for col in df_features.columns if 'small' in col]
    for col in cols_to_avg:
        col_small = col + '_small'
        col_large = col + '_large'
        df_features[col] = (df_features[col_small] + df_features[col_large]) / 2
        df_features.drop(columns=[col_small, col_large], inplace=True)
    df_features.rename(columns={'crop': 'cropland'}, inplace=True)
    return df_features


def get_field_data():
    df_field = pd.read_csv(sett.csv_field)
    return df_field[['site_id', 'study_id', 'latitude', 'longitude', 'crop', 'management', 'ab_wildbees', 'ab_bombus', 'total_sampled_time', 'sampling_year', 'sampling_abundance']]

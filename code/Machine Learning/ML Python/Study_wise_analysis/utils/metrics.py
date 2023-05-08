import os.path

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import pollinators_dependency as poll_dep, pollinators_dependency_num as poll_dep_num
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Get metrics from the computed model values, under different configurations, splitting by study.
# Metrics are computed from a direct comparison model vs observations and must include:
# - From a linear regression: R2, slope, p-value
# - From a rank correlation coefficient e.g. Spearman:  coefficient, p-value
# Register also number of sites in the study, crop, management, country,
# (std/mean)^2 (->coef. of variation ^2) (abundance and model, see Lonsdorf et al 2009)


########################################


############# FUNCTIONS #################
def get_model_data(modelname, guild='All'):
    # Get model data.
    # There are three options for guild:
    # 1)'All': From the two guilds (small and large)
    # 2&3) From guilds "Small"/"Large"
    # 2 options for types of model: lonsdorf or ML
    data_dir = "C:/Users/Angel/git/Observ_models/data/"
    if 'Lonsdorf' in modelname:
        if guild == 'All':
            model_file = 'model_data_lite.csv'
        elif guild == 'Small':
            model_file = 'model_data_lite_small.csv'
        elif guild == 'Large':
            model_file = 'model_data_lite_large.csv'
        df_model = pd.read_csv(data_dir + model_file).drop(columns=['sampling_year', 'management'])
    elif 'pred_' in modelname:
        model_file = 'ML/Regression/tables/prediction_by_site.csv'
        df_model = pd.read_csv(data_dir + model_file)
    df_model = df_model[['study_id', 'site_id', modelname]]
    df_model.columns = ['study_id', 'site_id', 'model']
    return df_model


def get_field_data(dataset):
    data_dir = r'C:\Users\Angel\git\Observ_models\data\Prepared Datasets'
    df_field = pd.read_csv(os.path.join(data_dir, dataset + '.csv'))
    return df_field


def set_pollinator_dependency(data):
    data['poll_dep'] = data.crop.map(poll_dep.dep)
    data['poll_dep_num'] = data.poll_dep.map(poll_dep_num.dep_num)
    return data


def get_proportion_of_significant_fits(df_model, option='Spearman'):
    thr = 0.05
    if (option == 'linreg'):
        df_significant = df_model.loc[(df_model.p_val < thr) & (df_model.slope > 0)]
    elif (option == 'Spearman'):
        df_significant = df_model.loc[(df_model.Spearman_p < thr) & (df_model.Spearman_coef > 0)]
    elif (option == 'Kendall'):
        df_significant = df_model.loc[(df_model.Kendall_p < thr) & (df_model.Kendall_coef > 0)]
    elif (option == 'conf_int'):
        df_significant = df_model.loc[(df_model.conf_int_min > 0)]
    return len(df_significant.index) / len(df_model.index)


# Linear regression parameters by study
def get_linreg_params(df, target_col='log_ab'):
    if (len(df.model.unique()) > 1):  # condition: not every model value is the same (for GBR(), it might be the case)
        observed = df[target_col]
        X_constant = sm.add_constant(df['model'])
        lin_reg = sm.OLS(observed, X_constant).fit()
        p_val = lin_reg.pvalues[1]
        slope = lin_reg.params[1]
        conf_int_min = lin_reg.conf_int(alpha=0.05)[0][1]
        conf_int_max = lin_reg.conf_int(alpha=0.05)[1][1]
        r2 = lin_reg.rsquared
        r2_adj = lin_reg.rsquared_adj
    else:
        p_val, conf_int_min, conf_int_max, slope, r2, r2_adj = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    return pd.DataFrame({'p_val': [p_val],
                         'conf_int_min': [conf_int_min],
                         'conf_int_max': [conf_int_max],
                         'slope': [slope],
                         'r2': [r2],
                         'r2_adj': [r2_adj]})


def get_rank_correlation(df, target_col='log_ab'):
    if (len(df.model.unique()) > 1):  # condition: not every model value is the same (for GBR(), it might be the case)
        observed = df[target_col]
        model = df['model']
        sp_coef, sp_p = stats.spearmanr(model, observed)
        ke_coef, ke_p = stats.kendalltau(model, observed)
    else:
        sp_coef, sp_p, ke_coef, ke_p = np.nan, np.nan, np.nan, np.nan
    return pd.DataFrame({'Spearman_coef': [sp_coef], 'Spearman_p': [sp_p],
                         'Kendall_coef': [ke_coef], 'Kendall_p': [ke_p]})


def get_mae(df, target_col = 'log_ab'):
    if (len(df.model.unique()) > 1):  # condition: not every model value is the same (for GBR(), it might be the case)
        observed = df[target_col]
        model = df['model']
        mae = mean_absolute_error(model, observed)
    else:
        mae = np.nan
    return pd.DataFrame({'MAE': [mae]})


def get_rmse(df, target_col = 'log_ab'):
    if (len(df.model.unique()) > 1):  # condition: not every model value is the same (for GBR(), it might be the case)
        observed = df[target_col]
        model = df['model']
        rmse = mean_squared_error(model, observed, squared=False)
    else:
        rmse = np.nan
    return pd.DataFrame({'RMSE': [rmse]})


def get_metadata(df_study):
    n_sites = len(df_study)
    mean_ab = np.mean(df_study.abund)
    poll_dep = df_study.iloc[0].poll_dep
    poll_dep_num = df_study.iloc[0].poll_dep_num
    biome = df_study.iloc[0].biome_num
    full_crop = df_study.iloc[0].crop
    n_max_char = np.min([len(full_crop), 5])
    crop = full_crop[0:n_max_char]
    management = get_management(df_study)
    full_country = df_study.iloc[0].country
    n_max_char = np.min([len(full_country), 5])
    country = full_country[0:n_max_char]
    full_sampling_method = df_study.iloc[0].sampling_abundance
    n_max_char = np.min([len(full_sampling_method), 15])
    sampling_method = full_sampling_method[0:n_max_char]
    return pd.DataFrame({'n_sites': [n_sites], 'crop': [crop], 'management': [management], 'country': [country], 'sampling_method': [sampling_method],
                         'poll_dep': [poll_dep], 'poll_dep_num': [poll_dep_num], 'biome': [biome], 'mean_ab': [mean_ab],
                         'full_crop_name': [full_crop], 'full_country': [full_country], 'full_sampling': [full_sampling_method]})


def get_management(df_study):
    df_study['management'] = df_study.management.fillna("conventional")
    conventional = df_study.loc[df_study.management == "conventional"]
    ipm = df_study.loc[df_study.management == "IPM"]
    unmanaged = df_study.loc[df_study.management == "unmanaged"]
    organic = df_study.loc[df_study.management == "organic"]
    n_convention = len(conventional)
    n_ipm = len(ipm)
    n_unmanaged = len(unmanaged)
    n_organic = len(organic)
    n_array = np.array([n_convention, n_ipm, n_unmanaged, n_organic])
    manag_array = np.array(["conventional", "IPM", "unmanaged", "organic"])
    i_max = np.where(n_array == np.amax(n_array))
    return (manag_array[i_max][0])


def get_landscape_standardized_variance(df_study):
    try:
        return pd.DataFrame({'landsc_var': [np.power((np.std(df_study.model) / np.mean(df_study.model)), 2)]})
    except ZeroDivisionError as ex:
        study_id = df_study['study_id'].unique()
        print(f"Found ZeroDivisionError exception in study {study_id} when computing landscape standardized variance. Returned NaN value. Exception {ex}")
        return pd.DataFrame({'landsc_var': [np.nan]})


def get_data(modelname, guild, dataset):
    """
    Get data corresponding to a model name, guild and dataset (see options for each in the description of each parameter)
    :param modelname: model options are 'ml' and eve
    :param guild: 'Large' or 'Small' or 'All'
    :param dataset: 'lons_global', 'lons_studywise', 'ml_global', 'ml_train', 'ml_test'
    :return:
    """
    # Get data
    df_model = get_model_data(modelname, guild)
    df_field = get_field_data(dataset)
    df_data = df_field.merge(df_model)
    return df_data


def get_metrics_by_study(df_data, guild):
    target_col = get_target_column(guild)
    # Linear regression by study
    df_linreg = df_data.groupby('study_id').apply(get_linreg_params, target_col=target_col)
    # Rank correlation by study
    df_rank = df_data.groupby('study_id').apply(get_rank_correlation, target_col=target_col)
    # Metadata by study
    df_metadata = df_data.groupby('study_id').apply(get_mae, target_col=target_col)
    # MAE
    df_mae = df_data.groupby('study_id').apply(get_mae, target_col=target_col)
    # RMSE
    df_rmse =  df_data.groupby('study_id').apply(get_rmse, target_col=target_col)
    # Coefficient of variation by study
    df_variation = df_data.groupby('study_id').apply(get_landscape_standardized_variance)
    # Merge dataframes
    df_merged = df_linreg \
        .merge(df_rank, left_index=True, right_index=True) \
        .merge(df_mae, left_index=True, right_index=True) \
        .merge(df_rmse, left_index=True, right_index=True) \
        .merge(df_variation, left_index=True, right_index=True)
    return df_merged


def get_metrics_by_biome(df_data, guild):
    target_col = get_target_column(guild)
    # Linear regression
    df_linreg = df_data.groupby('biome_num').apply(get_linreg_params, target_col=target_col)
    # Rank correlation
    df_rank = df_data.groupby('biome_num').apply(get_rank_correlation, target_col=target_col)
    # MAE
    df_mae = df_data.groupby('biome_num').apply(get_mae, target_col=target_col)
    # RMSE
    df_rmse =  df_data.groupby('biome_num').apply(get_rmse, target_col=target_col)
    # Coefficient of variation by study
    df_variation = df_data.groupby('biome_num').apply(get_landscape_standardized_variance)
    # Number of sites
    n_sites = df_data.groupby('biome_num').count().site_id
    df_rank['n_sites'] = n_sites.values
    # Merge dataframes
    df_merged = df_linreg \
        .merge(df_rank, left_index=True, right_index=True) \
        .merge(df_mae, left_index=True, right_index=True) \
        .merge(df_rmse, left_index=True, right_index=True) \
        .merge(df_variation, left_index=True, right_index=True)
    return df_merged


def get_metrics_global(df_data, guild):
    target_col = get_target_column(guild)
    df_aux = df_data[['model', target_col]].dropna()
    # Linear regression
    df_linreg = get_linreg_params(df_aux, target_col=target_col)
    # Rank correlation
    df_rank = get_rank_correlation(df_aux, target_col=target_col)
    # MAE
    df_mae = get_mae(df_aux, target_col=target_col)
    # RMSE
    df_rmse = get_rmse(df_aux, target_col=target_col)
    # Merge dataframes
    df_merged = df_linreg.merge(df_rank, left_index=True, right_index=True)
    df_merged = df_merged.merge(df_mae, left_index=True, right_index=True)
    df_merged = df_merged.merge(df_rmse, left_index=True, right_index=True)
    return df_merged


def get_target_column(guild):
    if guild == 'Small':
        target_col = 'log_vr_small'
    elif guild == 'Large':
        target_col = 'log_vr_large'
    else:
        target_col = 'log_visit_rate'
    return target_col


def format_with_statistical_significance(value, pval):
    num_str = '{:.2f}'.format(value)
    if (pval < 0.001):
        num_str = num_str + "***"
    elif (pval < 0.01):
        num_str = num_str + "**"
    elif (pval < 0.05):
        num_str = num_str + "*"
    return num_str


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from scipy import stats
from utils import pollinators_dependency as poll_dep
from utils import crop_families as crop_fam
import warnings
warnings.filterwarnings('ignore')

from utils import define_root_folder
root_folder = define_root_folder.root_folder

def get_feature_data():
    featuresDir = root_folder + "data/features/"
    df_features = pd.read_csv(featuresDir + 'Features.csv')
    # Set biome as categorical and replace NA by unknown
    df_features['biome_num'] = df_features.biome_num.astype('object')
    df_features['biome_num'] = df_features.biome_num.replace(np.nan,"unknown")
    df_features.drop(columns=['system:index', '.geo', 'refYear'], inplace=True)
    cols_to_avg = [col.split('_small')[0] for col in df_features.columns if 'small' in col]
    for col in cols_to_avg:
        col_small = col+'_small'
        col_large = col+'_large'
        df_features[col] = (df_features[col_small] + df_features[col_large])/2
        df_features.drop(columns=[col_small, col_large], inplace=True)
    df_features.rename(columns={'crop':'cropland'}, inplace=True)
    return df_features

def get_field_data(coords=False):
    df_field     = pd.read_csv(root_folder+'data/CropPol/CropPol_field_level_data.csv')
    if coords:
        return df_field[['site_id', 'study_id', 'latitude', 'longitude', 'crop', 'management',
                        'ab_wildbees', 'ab_syrphids', 'ab_bombus',
                        'total_sampled_time', 'sampling_year', 'sampling_abundance']]
    else:
        return df_field[['site_id', 'study_id', 'crop', 'management',
                        'ab_wildbees', 'ab_syrphids', 'ab_bombus',
                        'total_sampled_time', 'sampling_year', 'sampling_abundance']]

def compute_visit_rate(data):
    data['visit_rate_wb_bmb_syr'] = (data['ab_wildbees'] + data['ab_syrphids'] + data['ab_bombus']) / data['total_sampled_time']
    data['log_visit_rate']        = np.log(data['visit_rate_wb_bmb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'ab_bombus', 'total_sampled_time', 'visit_rate_wb_bmb_syr'], inplace=True)
    return data

def compute_visit_rate_small(data):
    data['visit_rate_wb_syr'] = (data['ab_wildbees']+ data['ab_syrphids']+1) / data['total_sampled_time']
    data['log_vr_small']      = np.log(data['visit_rate_wb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'visit_rate_wb_syr'], inplace=True)
    return data

def compute_visit_rate_large(data):
    # Compute comparable abundance
    data['visit_rate_bmb'] = (data['ab_bombus']+1) / data['total_sampled_time']
    data['log_vr_large']   = np.log(data['visit_rate_bmb'])
    data.drop(columns=['ab_bombus', 'visit_rate_bmb'], inplace=True)
    return data

def fill_missing_abundances(data):
    data.loc[data['ab_bombus'].isna(), 'ab_bombus']     = 0
    data.loc[data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    data.loc[data['ab_syrphids'].isna(), 'ab_syrphids'] = 0
    return data

def fill_biome(x, data):
    data_study_id = data.loc[data.study_id == x, ]
    return data_study_id.biome_num.mode().iloc[0]

def fill_missing_biomes(data):
    missing_biome = data.loc[data.biome_num == 'unknown',]
    new_biome     = [fill_biome(x, data) for x in missing_biome.study_id]
    data.loc[data.biome_num == 'unknown', 'biome_num'] = new_biome
    return data

def remap_crops(data, option):
    if (option == 'dependency'):
        data['crop'] = data['crop'].map(poll_dep.dep)
    elif (option == 'family'):
        data['crop'] = data['crop'].map(crop_fam.family)
    return data

def check_normality(data, column):
    sns.distplot(data[column])
    # skewness and kurtosis
    print("Skewness: %f" % data[column].skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % data[column].kurt()) # Kurtosis: -0.168611
    # Check normality log_visit_rate
    sns.distplot(data[column], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data[column], plot=plt)

def boxplot(data, x, y, ymin=-5, ymax=2):
    fig = sns.boxplot(x=x, y=y, data=data)
    fig.axis(ymin=ymin, ymax=ymax)

def is_sampling_method_accepted(x):
    cond1 = 'pan trap' not in x
    cond2 = x != "nan"
    return (cond1 & cond2)

def is_one_guild_measured(x,y,z, thresh):
    return ( (~np.isnan(x) & (x>thresh)) | (~np.isnan(y) & (y>thresh)) | (~np.isnan(z) & (z>thresh)) )

def are_abundances_integer(study_data): # do not exclude NAs (filtered or transformed in other steps)
    tol = 0.05
    cond_wb  = ((study_data['ab_wildbees'] % 1) < tol) | ((study_data['ab_wildbees'] % 1) > (1-tol)) | study_data['ab_wildbees'].isna()
    cond_syr = ((study_data['ab_syrphids'] % 1) < tol) | ((study_data['ab_syrphids'] % 1) > (1-tol)) | study_data['ab_syrphids'].isna()
    cond_bmb = ((study_data['ab_bombus'] % 1) < tol)   | ((study_data['ab_bombus'] % 1) > (1-tol))   | study_data['ab_bombus'].isna()
    cond = cond_wb & cond_syr & cond_bmb
    return all(cond)

def are_abundances_integer(study_data): # do not exclude NAs (filtered or transformed in other steps)
    tol = 0.05
    cond_wb  = ((study_data['ab_wildbees'] % 1) < tol) | ((study_data['ab_wildbees'] % 1) > (1-tol)) | study_data['ab_wildbees'].isna()
    cond_syr = ((study_data['ab_syrphids'] % 1) < tol) | ((study_data['ab_syrphids'] % 1) > (1-tol)) | study_data['ab_syrphids'].isna()
    cond_bmb = ((study_data['ab_bombus'] % 1) < tol)   | ((study_data['ab_bombus'] % 1) > (1-tol))   | study_data['ab_bombus'].isna()
    cond = cond_wb & cond_syr & cond_bmb
    return all(cond)

def apply_conditions(data, thresh_ab=0):
    # 1. Abundances of all sites in the study must be integer numbers (tolerance of 0.05)
    abs_integer = data.groupby('study_id').apply(are_abundances_integer)
    sel_studies = abs_integer.index[abs_integer]
    cond1       = data['study_id'].isin(sel_studies)
    print("1. Studies with all abundances integer:")
    print(abs_integer.describe())
    print("1b: Sites")
    print(cond1.describe())

    # 2. At least one guild measured with abundance > thresh_ab
    cond2 = pd.Series([is_one_guild_measured(x,y,z,thresh_ab) for (x,y,z) in zip(data['ab_wildbees'], data['ab_syrphids'], data['ab_bombus'])])
    print("2. At least one guild measured with abundance > "+str(thresh_ab)+":")
    print(cond2.describe())

    # 3. Set temporal threshold (sampling year >= 1992). This removes years 1990, 1991, that show not-very-healthy values of "comparable abundance"
    refYear = data['sampling_year'].str[:4].astype('int')
    cond3 = (refYear >= 1992)
    print("3. Ref year >=1992:")
    print(cond3.describe())

    # 4. Sampling method != pan trap
    cond4 = pd.Series([ is_sampling_method_accepted(x) for x in data['sampling_abundance'].astype('str') ])
    print("4. Sampling method not pan trap:")
    print(cond4.describe())
    data.drop(columns=['sampling_abundance'], inplace=True)

    # 5. Total sampled time != NA
    cond5 = ~data['total_sampled_time'].isna()
    print("5. Defined sampled time:")
    print(cond5.describe())

    # 6. Remove rows with 7 or more NaN values
    cond6 = (data.isnull().sum(axis=1) < 7)
    print("6. Less than 7 NAs per row:")
    print(cond6.describe())

    # Filter by conditions
    all_cond = (cond1 & cond2 & cond3 & cond4 & cond5 & cond6)
    print("ALL:")
    print(all_cond.describe())
    return data[ all_cond ]

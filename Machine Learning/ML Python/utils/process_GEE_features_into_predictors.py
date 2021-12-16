import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from utils import data_preparation as dp
import pickle
import warnings
warnings.filterwarnings('ignore')

root_folder = "C:/Users/angel/git/OBServ_Models_Open/Machine Learning/"

#######################################
# Get
#######################################
featuresDir = root_folder + "data/features/"
df_features = pd.read_csv(featuresDir + 'Features_for_predictions.csv')
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

num_pipeline = Pipeline([
    ('num_imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])
x_transformed = num_pipeline.fit_transform(df_features.drop(columns=['study_id','site_id']))

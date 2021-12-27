import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from utils import data_preparation as dp
import warnings
warnings.filterwarnings('ignore')

from utils import define_root_folder
root_folder = define_root_folder.root_folder

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
df_features = dp.fill_missing_biomes(df_features)

# Apply transformations (fill values, standardize, one-hot encoding)
# First, replace numeric by mean, grouped by study_id (if all sites have NAs, then replace by dataset mean later in the imputer)
pred_num     = df_features.select_dtypes('number')
n_nas        = pred_num.isna().sum().sum()
pred_num['study_id'] = df_features.study_id
pred_num = pred_num.groupby('study_id').transform(lambda x: x.fillna(x.mean()))
print("NA'S before transformation: " + str(n_nas))
print("Total numeric values: " + str(pred_num.size))
print("Percentage: " + str(n_nas*100/pred_num.size))

# Define pipleline
numeric_col = list(pred_num)
onehot_col  = ["biome_num"]
num_pipeline = Pipeline([
    ('num_imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])
onehot_pipeline = Pipeline([
    ('onehot_encoder', OneHotEncoder())
])
X = onehot_pipeline.fit(df_features[onehot_col])
onehot_encoder_names = X.named_steps['onehot_encoder'].get_feature_names()
full_pipeline = ColumnTransformer([
    ("numeric", num_pipeline, numeric_col),
    ("onehot",  onehot_pipeline, onehot_col )
])

#######################################
# Transform
#######################################
x_transformed = full_pipeline.fit_transform(df_features)

# Convert into data frame
numeric_col = np.array(pred_num.columns)
onehot_col  = np.array(onehot_encoder_names)
feature_names = np.concatenate( (numeric_col, onehot_col), axis=0)
predictors_prepared = pd.DataFrame(x_transformed, columns=feature_names, index=df_features.index)

# Save
predictors_prepared.to_csv(root_folder+"/data/predict_here/features_prepared.csv", index=False)


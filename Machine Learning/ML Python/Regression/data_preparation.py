"""
Script to get and prepare the data for the subsequent steps in the machine learning pipeline.
The preparation of data consists of:
1) Collect data from the features extracted in GEE and CropPol, apply filters, and harmonize visitation rate values.
2) Transform data using a specific pipeline for data preparation. This step includes standardizing numeric predictors
and transforming categorical variables into numeric.
3) Stratified split training and test (split by author ID). Split data based on the author ID, and create an iterator
based on the study ID, for any cross-validation in the next steps. See more details in the reference article of this
work.

This is the first step of a process that includes the following operations:
1) Prepare data (data_preparation.py)
2) Select a model to use as baseline for the selection of features (model_selection.py)
3) Select features based on collinearity (feature_collinearity.py)
4) Model selection and hyper-parameter tuning (model_selection.py)
5) Generate a trained model
6) Compute predictions (predict.py and/or prediction_stats.py)
"""

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

#######################################
# Get
#######################################
df_features = dp.get_feature_data()
df_field    = dp.get_field_data()
data = df_features.merge(df_field, on=['study_id', 'site_id'])
data = dp.apply_conditions(data)
data = dp.fill_missing_biomes(data)
# data = remap_crops(data, option="family") # remap crops to poll. dependency or family?
data = dp.fill_missing_abundances(data)
data = dp.compute_visit_rate(data)
data['author_id'] = [study.split("_",2)[0] + study.split("_",2)[1] for study in data.study_id]
# data = dp.add_landcover_diversity(data) # In the article, land cover diversity is extracted from a different script in GEE. TODO: include the extraction of land cover diversity in the general script of GEE 'features', and implement dp.add_landcover_diversity(data)

# Separate predictors and labels
predictors = data.drop("log_visit_rate", axis=1)
labels     = data['log_visit_rate'].copy()

# (Set biome as categorical)
predictors['biome_num'] = predictors.biome_num.astype('object')

#######################################
# Pipeline
#######################################
# Apply transformations (fill values, standardize, one-hot encoding)
# First, replace numeric by mean, grouped by study_id (if all sites have NAs, then replace by dataset mean later in the imputer)
pred_num     = predictors.select_dtypes('number')
n_nas        = pred_num.isna().sum().sum()
pred_num['study_id'] = data.study_id
pred_num = pred_num.groupby('study_id').transform(lambda x: x.fillna(x.mean()))
print("NA'S before transformation: " + str(n_nas))
print("Total numeric values: " + str(pred_num.size))
print("Percentage: " + str(n_nas*100/pred_num.size))

# Define pipleline
numeric_col = list(pred_num)
onehot_col  = ["biome_num", "crop"]
ordinal_col = ["management"]
dummy_col   = ["study_id","site_id","author_id"] # keep this to use later (e.g. create custom cross validation iterator)
num_pipeline = Pipeline([
    ('num_imputer', SimpleImputer(strategy="mean")),
    ('std_scaler', StandardScaler())
])
ordinal_pipeline = Pipeline([
     ('manag_imputer', SimpleImputer(strategy="constant", fill_value="conventional")),
     ('ordinal_encoder', OrdinalEncoder(categories=[['conventional','IPM','unmanaged','organic']]))
])
onehot_pipeline = Pipeline([
    ('onehot_encoder', OneHotEncoder())
])
dummy_pipeline = Pipeline([('dummy_imputer', SimpleImputer(strategy="constant", fill_value=""))])
X = onehot_pipeline.fit(predictors[onehot_col])
onehot_encoder_names = X.named_steps['onehot_encoder'].get_feature_names()
full_pipeline = ColumnTransformer([
    ("numeric", num_pipeline, numeric_col),
    ("ordinal", ordinal_pipeline, ordinal_col),
    ("dummy", dummy_pipeline, dummy_col),
    ("onehot",  onehot_pipeline, onehot_col )
])

#######################################
# Transform
#######################################
x_transformed = full_pipeline.fit_transform(predictors)

# Convert into data frame
numeric_col = np.array(pred_num.columns)
dummy_col = np.array(["study_id","site_id","author_id"])
onehot_col  = np.array(onehot_encoder_names)
feature_names = np.concatenate( (numeric_col, ordinal_col, dummy_col, onehot_col), axis=0)
predictors_prepared = pd.DataFrame(x_transformed, columns=feature_names, index=predictors.index)
dataset_prepared = predictors_prepared.copy()
dataset_prepared['log_visit_rate'] = labels

# Reset indices
data.reset_index(inplace=True, drop=True)
dataset_prepared.reset_index(inplace=True, drop=True)

#############################################################
# Stratified split training and test (split by author ID)
#############################################################
df_authors = data.groupby('author_id', as_index=False).first()[['author_id','biome_num']]
# For the training set, take biomes with more than one count (otherwise I get an error in train_test_split.
# They are added in the test set later, to keep all data
has_more_one     = df_authors.groupby('biome_num').count().author_id > 1
df_authors_split = df_authors.loc[has_more_one[df_authors.biome_num].reset_index().author_id,]
strata           = df_authors_split.biome_num.astype('category')

x_train, x_test, y_train, y_test = train_test_split(df_authors_split, strata, stratify=strata, test_size=0.3, random_state=4)
authors_train   = x_train.author_id
train_selection = [ (x_train.author_id == x).any() for x in data.author_id ]
df_train = dataset_prepared[train_selection].reset_index(drop=True)
df_test  = dataset_prepared[[~x for x in train_selection]].reset_index(drop=True)

# Save predictors and labels (train and set), removing study_id
df_train.drop(columns=['study_id', 'site_id', 'author_id']).to_csv(path_or_buf= dp.root_folder + 'data/train/data_prepared.csv', index=False)
df_test.drop(columns=['study_id', 'site_id', 'author_id']).to_csv(path_or_buf= dp.root_folder + 'data/test/data_prepared.csv', index=False)

# Save data (not processed by pipeline) including study_id and site_id
train_withIDs = data[train_selection].copy().reset_index(drop=True)
test_withIDs  = data[[~x for x in train_selection]].copy().reset_index(drop=True)
train_withIDs.to_csv(path_or_buf= dp.root_folder + 'data/train/data_prepared_withIDs.csv', index=False)
test_withIDs.to_csv(path_or_buf= dp.root_folder + 'data/test/data_prepared_withIDs.csv', index=False)

# Save custom cross validation iterator
df_studies = data[train_selection].reset_index(drop=True).groupby('study_id', as_index=False).first()[['study_id', 'biome_num']]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=135)
target = df_studies.loc[:, 'biome_num'].astype(int)
df_studies['fold'] = -1
n_fold = 0
for train_index, test_index in skf.split(df_studies, target):
    df_studies.loc[test_index,'fold'] = n_fold
    n_fold = n_fold+1
df_studies.drop(columns=['biome_num'], inplace=True)
dict_folds = df_studies.set_index('study_id').T.to_dict('records')[0]
df_train.replace(to_replace=dict_folds, inplace=True)
myCViterator = []
for i in range(0,5):
    trainIndices = df_train[df_train['study_id'] != i].index.values.astype(int)
    testIndices = df_train[df_train['study_id'] == i].index.values.astype(int)
    myCViterator.append((trainIndices, testIndices))
with open(dp.root_folder + 'data/train/myCViterator.pkl', 'wb') as f:
    pickle.dump(myCViterator, f)

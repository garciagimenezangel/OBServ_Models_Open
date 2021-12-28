"""
Script to compute predictions, based on already trained models.
Note: the script 'utils/process_GEE_features_into_predictors.py' can be used to generate the csv file that is needed
here to compute predictions

This is the last step of a process that includes the following operations:
1) Prepare data (data_preparation.py)
2) Select a model to use as baseline for the selection of features (model_selection.py)
3) Select features based on collinearity (feature_collinearity.py)
4) Model selection and hyper-parameter tuning (model_selection.py)
5) Generate a trained model
6) Compute predictions (predict.py and/or prediction_stats.py)
"""

import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from utils import define_root_folder
root_folder = define_root_folder.root_folder

def fillEmptyColumns(ref_columns, data):
    data_filled = data.copy()
    for col in ref_columns:
        if not (col in data.columns):
            data_filled[col] = 0
    return data_filled

n_features = 49
model = pickle.load(open(root_folder + 'data/trained_models/nusvr_full.pkl', 'rb')) # full: model trained using the entire database; training: model trained using only the training subset

# TEST 1: compute predictions from a dataset where all the predictors are present and standardized
# NOTE: if the 'test' dataset does not have a column used as a predictor in the model (for example, it might be the case
# of types of crop), we fill it with zeros
test_prepared = pd.read_csv(root_folder+'data/predict_here/data_reduced49.csv')
predictors_test = test_prepared.iloc[:, :-1]
labels_test = np.array(test_prepared.iloc[:, -1:]).flatten()
yhat = model.predict(predictors_test)
pd_result =  pd.DataFrame({})
pd_result['observations'] = labels_test
pd_result['predictions'] = yhat
pd_result.to_csv(root_folder+'data/predictions/test1.csv', index=False)

# TEST 2: compute predictions from a 'features' file computed using the GEE script available in this repository
features_prepared = pd.read_csv(root_folder+'data/predict_here/features_prepared.csv')
test_prepared = fillEmptyColumns(model.feature_names_in_, features_prepared)
test_prepared = test_prepared[model.feature_names_in_]
yhat = model.predict(test_prepared)
pd_result =  pd.DataFrame({'predictions':yhat})
pd_result.to_csv(root_folder+'data/predictions/test2.csv', index=False)

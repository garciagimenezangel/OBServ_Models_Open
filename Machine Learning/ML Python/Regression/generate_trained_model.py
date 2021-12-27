"""
Script to train and save ML models, to be used later for predicting visitation rate.
We save trained models using the entire dataset, and using only the training subset.

This is the fifth step of a process that includes the following operations:
1) Prepare data (data_preparation.py)
2) Select a model to use as baseline for the selection of features (model_selection.py)
3) Select features based on collinearity (feature_collinearity.py)
4) Model selection and hyper-parameter tuning (model_selection.py)
5) Generate a trained model
6) Compute predictions (predict.py and/or prediction_stats.py)
"""

import pandas as pd
import numpy as np
import warnings
from utils import generate_trained_model as gtm
warnings.filterwarnings('ignore')

n_features=49
train_prepared   = gtm.get_train_data_reduced(n_features)
test_prepared    = gtm.get_test_data_reduced(n_features)
# train_prepared   = gtm.get_train_data_full()
# test_prepared    = gtm.get_test_data_full()
predictors_train = train_prepared.iloc[:,:-1]
labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
predictors_test  = test_prepared.iloc[:,:-1]
labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()

# GENERATE TRAINED MODELS (two outputs: train using training set, or using the entire dataset training+test)
entire_set           = pd.concat([train_prepared,test_prepared])
svr_model_training   = gtm.compute_svr_model(train_prepared, n_features)
nusvr_model_training = gtm.compute_nusvr_model(train_prepared, n_features)
gbr_model_training   = gtm.compute_gbr_model(train_prepared, n_features)
svr_model_full       = gtm.compute_svr_model(entire_set, n_features)
nusvr_model_full     = gtm.compute_nusvr_model(entire_set, n_features)
gbr_model_full       = gtm.compute_gbr_model(entire_set, n_features)
gtm.save_model(svr_model_training, "svr_training")
gtm.save_model(nusvr_model_training, "nusvr_training")
gtm.save_model(gbr_model_training, "gbr_training")
gtm.save_model(svr_model_full, "svr_full")
gtm.save_model(nusvr_model_full, "nusvr_full")
gtm.save_model(gbr_model_full, "gbr_full")




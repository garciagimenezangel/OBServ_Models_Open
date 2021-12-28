"""
Ancillary functions for the generation of trained models
"""

import ast
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from scipy import stats
from sklearn.svm import SVR, NuSVR
import seaborn as sns
from scipy.stats import norm
warnings.filterwarnings('ignore')
import pickle

from utils import define_root_folder
root_folder = define_root_folder.root_folder

def check_normality(array):
    sns.distplot(array)
    # skewness and kurtosis
    print("Skewness: %f" % array.skew())
    print("Kurtosis: %f" % array.kurt())
    # Check normality log_visit_rate
    sns.distplot(array, fit=norm)
    fig = plt.figure()
    res = stats.probplot(array, plot=plt)

def get_train_data_reduced(n_features):
    return pd.read_csv(root_folder+'data/train/data_reduced_'+str(n_features)+'.csv')

def get_test_data_reduced(n_features):
    return pd.read_csv(root_folder+'data/test/data_reduced_'+str(n_features)+'.csv')

def get_train_data_full():
    return pd.read_csv(root_folder+'data/train/data_prepared.csv')

def get_test_data_full():
    return pd.read_csv(root_folder+'data/test/data_prepared.csv')

def get_train_data_withIDs():
    return pd.read_csv(root_folder+'data/train/data_prepared_withIDs.csv')

def get_test_data_withIDs():
    return pd.read_csv(root_folder+'data/test/data_prepared_withIDs.csv')

def get_best_models(n_features=0):
    data_dir = root_folder + "data/hyperparameters/"
    if n_features>0:
        return pd.read_csv(data_dir + 'best_scores_'+str(n_features)+'.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores.csv')

def compute_gbr_model(train_prepared, n_features):
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "GradientBoostingRegressor()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = GradientBoostingRegressor(loss=d['loss'], learning_rate=d['learning_rate'], n_estimators=d['n_estimators'], subsample=d['subsample'],
                                      min_samples_split=d['min_samples_split'], min_samples_leaf=d['min_samples_leaf'], min_weight_fraction_leaf=d['min_weight_fraction_leaf'],
                                      max_depth=d['max_depth'], min_impurity_decrease=d['min_impurity_decrease'], max_features=d['max_features'], alpha=d['alpha'],
                                      max_leaf_nodes=d['max_leaf_nodes'], ccp_alpha=d['ccp_alpha'], random_state=135)
    model.fit(predictors_train, labels_train)
    return model

def compute_gbr_predictions(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    model = compute_gbr_model(train_prepared, n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    yhat  = model.predict(predictors_test)
    return yhat, labels_test

def compute_svr_model(train_prepared, n_features):
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    return model

def compute_svr_predictions(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    model            = compute_svr_model(train_prepared, n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    yhat  = model.predict(predictors_test)
    return yhat, labels_test

def compute_nusvr_model(train_prepared, n_features):
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    return model

def compute_nusvr_predictions(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    model            = compute_nusvr_model(train_prepared, n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    yhat  = model.predict(predictors_test)
    return yhat, labels_test

def compute_gbr_stats(n_features):
    yhat, labels_test = compute_gbr_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae   = mean_absolute_error(X_reg, y_reg)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2    = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    sp_coef, sp_p = stats.spearmanr(yhat, labels_test)
    return pd.DataFrame({
        'model': "GBR",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'sp_coef': sp_coef
    }, index=[0])

def compute_svr_stats(n_features):
    yhat, labels_test = compute_svr_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae   = mean_absolute_error(X_reg, y_reg)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2    = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    sp_coef, sp_p = stats.spearmanr(yhat, labels_test)
    return pd.DataFrame({
        'model': "SVR",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'sp_coef': sp_coef
    }, index=[0])

def compute_nusvr_stats(n_features):
    yhat, labels_test = compute_nusvr_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae   = mean_absolute_error(X_reg, y_reg)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2    = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    sp_coef, sp_p = stats.spearmanr(yhat, labels_test)
    return pd.DataFrame({
        'model': "NuSVR",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'sp_coef': sp_coef
    }, index=[0])

def save_model(model, model_name):
    pickle.dump(model, open(root_folder+'data/trained_models/'+model_name+'.pkl', 'wb'))

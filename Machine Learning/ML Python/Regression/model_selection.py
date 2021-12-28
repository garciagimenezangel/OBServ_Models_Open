"""
Script to select candidate models among the set of estimators available in scikit-learn, and tune their parameters
with a randomized search

This script is used in the second and fourth steps of a process that includes the following operations:
1) Prepare data (data_preparation.py)
2) Select a model to use as baseline for the selection of features (model_selection.py)
3) Select features based on collinearity (feature_collinearity.py)
4) Model selection and hyper-parameter tuning (model_selection.py)
5) Generate a trained model
6) Compute predictions (predict.py and/or prediction_stats.py)
"""

import numpy as np
import pandas as pd
import warnings
import pickle
import matplotlib
matplotlib.use('TkAgg')
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.utils import all_estimators
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform
warnings.filterwarnings('ignore')

from utils import define_root_folder
root_folder = define_root_folder.root_folder

def get_data_prepared():
    return pd.read_csv(root_folder+'data/train/data_prepared.csv')

def get_data_reduced(n_features):
    return pd.read_csv(root_folder+'data/train/data_reduced_'+str(n_features)+'.csv')

reduced_n_feat = True # 1st run False, 2nd (after feature collinearity) True
n_features = 49

if reduced_n_feat:
    data_prepared = get_data_reduced(n_features)
else:
    data_prepared = get_data_prepared()

predictors    = data_prepared.iloc[:,:-1]
labels        = np.array(data_prepared.iloc[:,-1:]).flatten()

# Load custom cross validation
with open(root_folder+'data/train/myCViterator.pkl', 'rb') as file:
    myCViterator = pickle.load(file)

########################
# Model selection
########################
# LIST OF ESTIMATORS OF TYPE "REGRESSOR" (TRY ALL)
estimators = all_estimators(type_filter='regressor')
results = []
for name, RegressorClass in estimators:
    try:
        print('Regressor: ', name)
        reg = RegressorClass()
        reg.fit(predictors, labels)
        abundance_predictions = reg.predict(predictors)
        mae = mean_absolute_error(labels, abundance_predictions)
        print('MAE all: ', mae)
        scores = cross_val_score(reg, predictors, labels, scoring="neg_mean_absolute_error", cv=myCViterator)
        mae_scores = -scores
        print("Mean:", mae_scores.mean())
        print("Std:", mae_scores.std())
        results.append({'reg':reg, 'MAE all':mae, 'mean':mae_scores.mean(), 'std':mae_scores.std()})
    except Exception as e:
        print(e)
df_results = pd.DataFrame(results)
df_results_sorted = df_results.sort_values(by=['mean'], ascending=True)
if reduced_n_feat:
    df_results_sorted.to_csv(path_or_buf=root_folder+'data/hyperparameters/model_selection_'+str(n_features)+'.csv', index=False)
else:
    df_results_sorted.to_csv(path_or_buf=root_folder + 'data/hyperparameters/model_selection.csv', index=False)

########################
# Shortlist: check df_results and see which show low 'mean' and not-too-low 'MAE all' (sign of possible overfitting)
#######################
# Selected estimators (no particular order):
# 1 NuSVR
# 2 RandomForestRegressor
# 3 SVR
# 4 GradientBoostingRegressor

########################
# Hyperparameter tuning
# I use randomized search over a (small) set of parameters, to get the best score. I repeat the process several
# times, using a parameter space "surrounding" the best parameters in the previous step
########################
results=[]
# Note: BayesSearchCV in currently latest version of scikit-optimize not compatible with scikit-learn 0.24.1
# When scikit-optimize version 0.9.0 is available (currently in development), use: BayesSearchCV(model,params,cv=5)
# NuSVR
model = NuSVR()
# define search space
params = dict()
params['kernel']  = ['linear', 'rbf', 'sigmoid']
params['nu']      = uniform(loc=0, scale=1)
params['C']       = uniform(loc=0, scale=4)
params['gamma']   = uniform(loc=0, scale=0.5)
params['coef0']   = uniform(loc=-1, scale=1)
params['shrinking'] = [False, True]
# define the search
search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                            verbose=2, random_state=135, n_jobs=6)
search.fit(predictors, labels)
cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

# SVR
model = SVR()
# define search space
params = dict()
params['kernel']  = ['linear', 'rbf', 'sigmoid']
params['epsilon']      = uniform(loc=0, scale=1)
params['C']       = uniform(loc=0, scale=4)
params['gamma']   = uniform(loc=0, scale=0.5)
params['coef0']   = uniform(loc=-1, scale=1)
params['shrinking'] = [False, True]
# define the search
search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                            verbose=2, random_state=135, n_jobs=6)
search.fit(predictors, labels)
cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

# GradientBoostingRegressor
model = GradientBoostingRegressor()
# define search space
params = dict()
params['loss'] = ['ls', 'lad', 'huber', 'quantile']
params['learning_rate'] = uniform(loc=0, scale=1)
params['n_estimators'] = [50, 100, 200, 400, 600]
params['subsample'] = uniform(loc=0, scale=1)
params['min_samples_split'] = uniform(loc=0, scale=1)
params['min_samples_leaf'] = uniform(loc=0, scale=0.5)
params['min_weight_fraction_leaf'] = uniform(loc=0, scale=0.5)
params['max_depth'] = [2, 4, 8, 16, 32]
params['min_impurity_decrease'] = uniform(loc=0, scale=1)
params['max_features'] = uniform(loc=0, scale=1)
params['alpha'] = uniform(loc=0, scale=1)
params['max_leaf_nodes'] = [8, 16, 32, 64]
params['ccp_alpha'] = uniform(loc=0, scale=1)
# define the search
search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                            verbose=2, random_state=135, n_jobs=6)
search.fit(predictors, labels)
cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

# RandomForestRegressor
model = RandomForestRegressor()
# define search space
params = dict()
params['n_estimators'] = [50, 100, 200, 400, 600]
params['criterion'] = ['mse','mae']
params['max_depth'] = [2, 4, 8, 16, 32]
params['min_samples_split'] = uniform(loc=0, scale=1)
params['min_samples_leaf'] = uniform(loc=0, scale=0.5)
params['min_weight_fraction_leaf'] = uniform(loc=0, scale=0.5)
params['min_impurity_decrease'] = uniform(loc=0, scale=1)
params['max_features'] = uniform(loc=0, scale=1)
params['max_leaf_nodes'] = [8, 16, 32, 64]
params['ccp_alpha'] = uniform(loc=0, scale=1)
# define the search
search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                            verbose=2, random_state=135, n_jobs=6)
search.fit(predictors, labels)
cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

df_best_scores = pd.DataFrame(results)
df_best_scores_sorted = df_best_scores.sort_values(by=['best_score'], ascending=False)
if reduced_n_feat:
    df_best_scores_sorted.to_csv(path_or_buf=root_folder + 'data/hyperparameters/best_scores_'+str(n_features)+'.csv', index=False)
else:
    df_best_scores_sorted.to_csv(path_or_buf=root_folder + 'data/hyperparameters/best_scores.csv', index=False)

# OTHER REGRESSORS...
#
# # TweedieRegressor
# model = TweedieRegressor(max_iter=10000)
# # define search space
# params = dict()
# params['power'] = [0,2,3]
# params['alpha'] = uniform(loc=0, scale=3)
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})
#

# # KNeighborsRegressor
# model = KNeighborsRegressor()
# # define search space
# params = dict()
# params['n_neighbors']  = [3,4,5,6,8,10]
# params['weights']  = ['uniform','distance']
# params['leaf_size'] = [10,20,30,40,60,100]
# params['p']   = [1,2,4]
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})
#
#
# # MLPRegressor
# model = MLPRegressor(max_iter=10000, solver='sgd')
# # define search space
# params = dict()
# params['hidden_layer_sizes'] = [(20,),(50,),(100,),(200,)]
# params['activation'] = ['identity', 'logistic', 'tanh', 'relu']
# params['alpha'] = uniform(loc=0, scale=0.1)
# params['learning_rate'] = ['constant', 'invscaling', 'adaptive']
# params['learning_rate_init'] = uniform(loc=0, scale=0.1)
# params['power_t'] = uniform(loc=0, scale=1)
# params['momentum'] = uniform(loc=0, scale=1)
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

#
# # AdaBoostRegressor
# model = AdaBoostRegressor()
# # define search space
# params = dict()
# params['n_estimators'] = [10,40,80,120,160,200]
# params['loss'] = ['linear', 'square', 'exponential']
# params['learning_rate'] = uniform(loc=0, scale=1)
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

# LassoCV
# model = LassoCV(max_iter=10000)
# # define search space
# params = dict()
# params['n_alphas'] = [40,80,120,160,200]
# params['fit_intercept'] = [False, True]
# params['eps'] = uniform(loc=0, scale=0.1)
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})
#
# # HistGradientBoostingRegressor
# model = HistGradientBoostingRegressor(max_iter=1000)
# # define search space
# params = dict()
# params['loss'] = ['least_squares', 'least_absolute_deviation']
# params['learning_rate'] = uniform(loc=0, scale=1)
# params['max_depth'] = [2, 4, 8, 16, 32]
# params['min_samples_leaf'] = [4, 8, 16, 32,64]
# params['max_leaf_nodes'] = [8, 16, 32, 64]
# params['max_bins'] = [32, 64, 128, 255]
# params['l2_regularization'] = uniform(loc=0, scale=1)
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=200,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

# # BayesianRidge
# model = BayesianRidge(n_iter=10000)
# params = dict()
# params['alpha_1'] = uniform(loc=0, scale=0.5)
# params['alpha_2'] = uniform(loc=0, scale=0.5)
# params['lambda_1'] = uniform(loc=0, scale=0.5)
# params['lambda_2'] = uniform(loc=0, scale=0.5)
# params['fit_intercept'] = [False, True]
# params['normalize'] = [False, True]
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})

# # PLSRegression
# model = PLSRegression(max_iter=10000)
# # define search space
# params = dict()
# params['n_components'] = [1,2,4,8,16,32]
# # define the search
# search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
#                             verbose=2, random_state=135, n_jobs=6)
# search.fit(predictors, labels)
# cvres = pd.DataFrame(search.cv_results_).sort_values(by=['mean_test_score'], ascending=False)
# for i in range(0,min(10, len(cvres))): results.append({'model': model, 'best_params': cvres.iloc[i].params, 'best_score': cvres.iloc[i].mean_test_score})
#
#


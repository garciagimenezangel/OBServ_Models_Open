
import pickle
from functools import reduce
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.feature_selection import RFE, RFECV
import warnings
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.inspection import permutation_importance
import datetime
import ast
from utils import feature_aliases
warnings.filterwarnings('ignore')
models_repo = "C:/Users/Angel/git/Observ_models/"
root_dir    = models_repo + "data/Prepared Datasets/"


def get_train_data_prepared(n_feat=0):
    if n_feat:
        return pd.read_csv(root_dir+'ml_train_reduced_{}.csv'.format(str(n_feat)))
    else:
        return pd.read_csv(root_dir+'ml_train.csv')


def get_test_data_prepared(n_feat=0):
    if n_feat:
        return pd.read_csv(root_dir+'ml_test_reduced_{}.csv'.format(str(n_feat)))
    else:
        return pd.read_csv(root_dir+'ml_test.csv')


def get_best_models(n_feat=0):
    data_dir = models_repo + "data/ML/Regression/hyperparameters/"
    if n_feat:
        return pd.read_csv(data_dir + 'best_scores_' + str(n_feat) + '.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores_all_predictors.csv')



if __name__ == '__main__':

    # Get best models
    n_features = 22
    df_best_models = get_best_models(n_features)

    # Load custom cross validation
    with open(root_dir + 'myCViterator.pkl', 'rb') as file:
        myCViterator = pickle.load(file)

    # Get df_data
    train_prepared = get_train_data_prepared(n_features)
    if n_features:
        predictors_train = train_prepared.drop(columns=['log_visit_rate'])
    else:  # use all predictors
        predictors_train = train_prepared.drop(columns=['study_id', 'site_id', 'author_id', 'log_vr_small', 'log_vr_large', 'log_visit_rate'])
    labels_train = np.array(train_prepared['log_visit_rate'])
    test_prepared = get_test_data_prepared(n_features)

    #######################################
    # Feature importance with Gradient Boosting Regressor
    #######################################
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "GradientBoostingRegressor()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = GradientBoostingRegressor(loss=d['loss'], learning_rate=d['learning_rate'], n_estimators=d['n_estimators'], subsample=d['subsample'],
                                      min_samples_split=d['min_samples_split'], min_samples_leaf=d['min_samples_leaf'], min_weight_fraction_leaf=d['min_weight_fraction_leaf'],
                                      max_depth=d['max_depth'], min_impurity_decrease=d['min_impurity_decrease'], max_features=d['max_features'], alpha=d['alpha'],
                                      max_leaf_nodes=d['max_leaf_nodes'], ccp_alpha=d['ccp_alpha'], random_state=135)
    model.fit(predictors_train, labels_train)
    df_feat_imp_gbr = pd.DataFrame(sorted(zip(model.feature_importances_, predictors_train.columns), reverse=True))
    df_feat_imp_gbr['feature'] = df_feat_imp_gbr[1]
    df_feat_imp_gbr['GBR']     = df_feat_imp_gbr[0]
    df_feat_imp_gbr = df_feat_imp_gbr[['feature','GBR']]

    # #######################################
    # # Permutation importance
    # #######################################
    # Bayesian Ridge
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "BayesianRidge(n_iter=10000)"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = BayesianRidge(n_iter=1000, alpha_1=d['alpha_1'], alpha_2=d['alpha_2'], fit_intercept=d['fit_intercept'], lambda_1=d['lambda_1'], lambda_2=d['lambda_2'],
                          normalize=d['normalize'])
    model.fit(predictors_train, labels_train)
    perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
    imp_values = perm_importance.importances_mean / np.sum(perm_importance.importances_mean)
    df_feat_imp_bayr = pd.DataFrame({'feature':predictors_train.columns, 'BayR':imp_values})


    # # NuSVR
    # best_model       = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
    # d     = ast.literal_eval(best_model.best_params)
    # model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    # model.fit(predictors_train, labels_train)
    # perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
    # imp_values = perm_importance.importances_mean / np.sum(perm_importance.importances_mean)
    # df_feat_imp_nusvr = pd.DataFrame({'feature':predictors_train.columns, 'NuSVR':imp_values})
    #
    # SVR
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
    imp_values = perm_importance.importances_mean / np.sum(perm_importance.importances_mean)
    df_feat_imp_svr = pd.DataFrame({'feature':predictors_train.columns, 'SVR':imp_values})

    # Merge feature importance for different models
    df_feat_imp = reduce(lambda left, right: pd.merge(left, right, on=['feature']), [df_feat_imp_bayr, df_feat_imp_gbr, df_feat_imp_svr])

    # Collapse rows of biomes and crops into one single row (each)
    df_biome_rows = df_feat_imp[df_feat_imp.feature.str.contains('x0')]
    sum_biomes    = df_biome_rows.sum()
    sum_biomes.values[0] = 'Sum(biomes)'
    # df_crop_rows  = df_feat_imp[df_feat_imp.feature.str.contains('x1')]
    # sum_crops     = df_crop_rows.sum()
    # sum_crops.values[0]  = 'Sum(crops)'
    df_other_rows = df_feat_imp[(~df_feat_imp.feature.str.contains('x0')) & (~df_feat_imp.feature.str.contains('x1'))]
    df_imp_summ   = df_other_rows.append(sum_biomes, ignore_index=True)
    # df_imp_summ   = df_imp_summ.append(sum_crops, ignore_index=True)
    df_imp_summ["feature"].replace(feature_aliases.feat_dict, inplace=True)

    # Sort values
    model_to_sort = 'BayR'
    df_imp_summ.sort_values(by=[model_to_sort], ascending=False, inplace=True)

    # Plot
    labels = df_imp_summ.feature
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, df_imp_summ.BayR, width, label='BayR')
    rects2 = ax.bar(x, df_imp_summ.GBR, width, label='GBR')
    rects3 = ax.bar(x + width, df_imp_summ.SVR, width, label='SVR')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Importance (0-1)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=80)
    ax.legend(bbox_to_anchor=(1.25, 1), loc=1, borderaxespad=0.)
    fig.tight_layout()
    plt.show()
    plt.savefig(f'C:/Users/Angel/git/Observ_models/data/ML/Regression/plots/feature_importance_feat{n_features}.eps', format='eps', bbox_inches='tight')

    # #######################################
    # # Select important variables?
    # #######################################
    # df_sel_feat = df_feat_imp[df_feat_imp.importance > 0]
    # data_reduced_train = train_prepared[ np.append(np.array(df_sel_feat.variable),['log_visit_rate']) ]
    # data_reduced_test  = test_prepared[ np.append(np.array(df_sel_feat.variable),['log_visit_rate']) ]
    # data_reduced_train.to_csv('C:/Users/Angel/git/Observ_models/data/ML/Regression/train/data_reduced_13.csv', index=False)
    # data_reduced_test.to_csv('C:/Users/Angel/git/Observ_models/data/ML/Regression/test/data_reduced_13.csv', index=False)

import ast
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from scipy.stats import norm

warnings.filterwarnings('ignore')
models_repo = "C:/Users/Angel/git/Observ_models/"
prepared_data_folder = models_repo + "data/Prepared Datasets/"
ml_folder = models_repo + "data/ML/Regression/"


# def check_normality(array):
#     sns.distplot(array)
#     # skewness and kurtosis
#     print("Skewness: %f" % array.skew())
#     print("Kurtosis: %f" % array.kurt())
#     # Check normality log_visit_rate
#     sns.distplot(array, fit=norm)
#     fig = plt.figure()
#     res = stats.probplot(array, plot=plt)


def get_lonsdorf_prediction_files():
    path = models_repo + 'data/Lonsdorf evaluation/Model predictions/'
    return [i for i in os.listdir(path) if os.path.isfile(os.path.join(path, i)) and 'lm ' in i]


def get_lonsdorf_predictions(file='lm pred Open forest.csv'):
    return pd.read_csv(models_repo + 'data/Lonsdorf evaluation/Model predictions/' + file)


def get_train_data_reduced(n_features):
    return pd.read_csv(prepared_data_folder + 'ml_train_reduced_{}.csv'.format(n_features))


def get_test_data_reduced(n_features):
    return pd.read_csv(prepared_data_folder + 'ml_test_reduced_{}.csv'.format(n_features))


def get_train_data_full():
    return pd.read_csv(prepared_data_folder + 'ml_train.csv').drop(columns=['study_id', 'site_id', 'author_id', 'log_vr_small', 'log_vr_large'])


def get_test_data_full():
    return pd.read_csv(prepared_data_folder + 'ml_test.csv').drop(columns=['study_id', 'site_id', 'author_id', 'log_vr_small', 'log_vr_large'])


def get_train_data_withIDs():
    return pd.read_csv(prepared_data_folder + 'ml_train.csv').drop(columns=['log_vr_small', 'log_vr_large'])


def get_test_data_withIDs():
    return pd.read_csv(prepared_data_folder + 'ml_test.csv').drop(columns=['log_vr_small', 'log_vr_large'])


def get_best_models(n_features=0):
    data_dir = models_repo + "data/ML/Regression/hyperparameters/"
    if n_features > 0:
        return pd.read_csv(data_dir + 'best_scores_' + str(n_features) + '.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores_all_predictors.csv')
#
#
# def check_normality(data, column):
#     sns.distplot(data[column])
#     # skewness and kurtosis
#     print("Skewness: %f" % data[column].skew())  # Skewness: -0.220768
#     print("Kurtosis: %f" % data[column].kurt())  # Kurtosis: -0.168611
#     # Check normality log_visit_rate
#     sns.distplot(data[column], fit=norm)
#     fig = plt.figure()
#     res = stats.probplot(data[column], plot=plt)


def compute_gbr_predictions(n_features):
    model = compute_gbr_model(n_features)
    yhat = model.predict(predictors_test)
    return yhat, labels_test


def compute_gbr_model(n_features):
    train_prepared = get_train_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:, :-1]
    labels_train = np.array(train_prepared.iloc[:, -1:]).flatten()
    df_best_models = get_best_models(n_features)
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "GradientBoostingRegressor()"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = GradientBoostingRegressor(loss=d['loss'], learning_rate=d['learning_rate'], n_estimators=d['n_estimators'], subsample=d['subsample'],
                                      min_samples_split=d['min_samples_split'], min_samples_leaf=d['min_samples_leaf'], min_weight_fraction_leaf=d['min_weight_fraction_leaf'],
                                      max_depth=d['max_depth'], min_impurity_decrease=d['min_impurity_decrease'], max_features=d['max_features'], alpha=d['alpha'],
                                      max_leaf_nodes=d['max_leaf_nodes'], ccp_alpha=d['ccp_alpha'], random_state=135)
    model.fit(predictors_train, labels_train)
    return model


def compute_svr_model(n_features):
    train_prepared = get_train_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:, :-1]
    labels_train = np.array(train_prepared.iloc[:, -1:]).flatten()
    df_best_models = get_best_models(n_features)
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    return model


def compute_nusvr_model(n_features):
    train_prepared = get_train_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:, :-1]
    labels_train = np.array(train_prepared.iloc[:, -1:]).flatten()
    df_best_models = get_best_models(n_features)
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    return model


def compute_bayesian_ridge_model(n_features):
    train_prepared = get_train_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:, :-1]
    labels_train = np.array(train_prepared.iloc[:, -1:]).flatten()
    df_best_models = get_best_models(n_features)
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "BayesianRidge(n_iter=10000)"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = BayesianRidge(n_iter=1000, alpha_1=d['alpha_1'], alpha_2=d['alpha_2'], fit_intercept=d['fit_intercept'], lambda_1=d['lambda_1'], lambda_2=d['lambda_2'], normalize=d['normalize'])
    model.fit(predictors_train, labels_train)
    return model


def compute_bayesian_ridge_predictions(n_features):
    model = compute_bayesian_ridge_model(n_features)
    yhat = model.predict(predictors_test)
    return yhat, labels_test


def compute_svr_predictions(n_features):
    model = compute_svr_model(n_features)
    yhat = model.predict(predictors_test)
    return yhat, labels_test


def compute_nusvr_predictions(n_features):
    model = compute_nusvr_model(n_features)
    yhat = model.predict(predictors_test)
    return yhat, labels_test


def compute_mlp_predictions(n_features):
    train_prepared = get_train_data_reduced(n_features)
    test_prepared = get_test_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:, :-1]
    labels_train = np.array(train_prepared.iloc[:, -1:]).flatten()
    predictors_test = test_prepared.iloc[:, :-1]
    labels_test = np.array(test_prepared.iloc[:, -1:]).flatten()
    df_best_models = get_best_models(n_features)
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "MLPRegressor(max_iter=10000, solver='sgd')"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = MLPRegressor(activation=d['activation'], alpha=d['alpha'], hidden_layer_sizes=d['hidden_layer_sizes'],
                         learning_rate=d['learning_rate'], learning_rate_init=d['learning_rate_init'], momentum=d['momentum'],
                         power_t=d['power_t'], max_iter=10000, solver='sgd', random_state=135)
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)
    return yhat, labels_test


def compute_bayesian_ridge_stats(n_features):
    yhat, labels_test = compute_bayesian_ridge_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg, y_reg)
    r2 = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    sp_coef, sp_p = stats.spearmanr(yhat, labels_test)
    return pd.DataFrame({
        'model': "BR",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'sp_coef': sp_coef
    }, index=[0])


def compute_gbr_stats(n_features):
    yhat, labels_test = compute_gbr_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg, y_reg)
    r2 = reg.score(X_reg, y_reg)
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
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg, y_reg)
    r2 = reg.score(X_reg, y_reg)
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
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg, y_reg)
    r2 = reg.score(X_reg, y_reg)
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


def compute_mlp_stats(n_features):
    yhat, labels_test = compute_mlp_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg, y_reg)
    r2 = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    sp_coef, sp_p = stats.spearmanr(yhat, labels_test)
    return pd.DataFrame({
        'model': "MLP",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'sp_coef': sp_coef
    }, index=[0])


def compute_lons_stats():
    results = pd.DataFrame(columns=['model', 'mae', 'r2', 'slope'])
    files = get_lonsdorf_prediction_files()
    for file in files:
        df_lons = get_lonsdorf_predictions(file)
        X_reg, y_reg = np.array(df_lons.lm_predicted), np.array(df_lons.log_visit_rate)
        mae = mean_absolute_error(X_reg, y_reg)
        reg = LinearRegression().fit(X_reg.reshape(-1, 1), y_reg.reshape(-1, 1))
        r2 = reg.score(X_reg.reshape(-1, 1), y_reg.reshape(-1, 1))
        slope = reg.coef_[0][0]
        sp_coef, sp_p = stats.spearmanr(df_lons.lm_predicted, df_lons.log_visit_rate)
        model = df_lons.iloc[0].model
        model_res = pd.DataFrame({'model': model, 'mae': mae, 'r2': r2, 'slope': slope, 'sp_coef': sp_coef}, index=[0])
        results = pd.concat([results, model_res], axis=0, ignore_index=True)
    return results


def compute_combined_stats(n_features):
    df_lons = get_lonsdorf_predictions()
    yhat, labels_test = compute_gbr_predictions(n_features)
    test_withIDs = get_test_data_withIDs()
    df_ml = pd.DataFrame({'obs': labels_test, 'pred': yhat, 'study_id': test_withIDs.study_id, 'site_id': test_withIDs.site_id})
    df_combined = pd.merge(df_lons, df_ml, on=['study_id', 'site_id'])
    df_combined['yhat'] = df_combined[['pred', 'lm_predicted']].mean(axis=1)
    X_reg, y_reg = np.array(df_combined.yhat), np.array(df_combined.log_visit_rate)
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg.reshape(-1, 1), y_reg.reshape(-1, 1))
    r2 = reg.score(X_reg.reshape(-1, 1), y_reg.reshape(-1, 1))
    slope = reg.coef_[0][0]
    sp_coef, sp_p = stats.spearmanr(df_combined.yhat, df_combined.log_visit_rate)
    model = "Average(" + df_lons.iloc[0].model + ", GBR)"
    return pd.DataFrame({
        'model': model,
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope,
        'sp_coef': sp_coef
    }, index=[0])


def get_mechanistic_values(model_name):
    data_dir = "C:/Users/Angel/git/Observ_models/data/"
    return pd.read_csv(data_dir + 'model_data_lite.csv')[['site_id', 'study_id', model_name]]


# def compute_ml_with_lons(n_features, model_name='Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'):
#     train_prepared   = get_train_data_reduced(n_features)
#     test_prepared    = get_test_data_reduced(n_features)
#     predictors_train = train_prepared.iloc[:,:-1]
#     labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
#     predictors_test  = test_prepared.iloc[:,:-1]
#     labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
#     train_with_IDs   = get_train_data_withIDs()
#     train_prepared['study_id'] = train_with_IDs.study_id
#     train_prepared['site_id']  = train_with_IDs.site_id
#     train_mech_values= get_mechanistic_values(model_name)
#     train_prepared   = pd.merge(train_prepared, train_mech_values, on=['study_id','site_id'])
#     train_prepared.drop(columns=['study_id','site_id'], inplace=True)
#
#
#     return df_data.merge(model_data, on=['study_id', 'site_id'])
#
#
#     df_best_models   = get_best_models(n_features)
#     best_model       = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
#     d     = ast.literal_eval(best_model.best_params)
#     model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
#     model.fit(predictors_train, labels_train)
#     yhat  = model.predict(predictors_test)

if __name__ == '__main__':
    n_features = 22  # use n_features = 0 to use every predictor
    if n_features:
        train_prepared = get_train_data_reduced(n_features)
        test_prepared = get_test_data_reduced(n_features)
    else:
        train_prepared   = get_train_data_full()
        test_prepared    = get_test_data_full()
    predictors_train = train_prepared.drop(columns=['log_visit_rate'])
    labels_train = np.array(train_prepared['log_visit_rate'])
    predictors_test = test_prepared.drop(columns=['log_visit_rate'])
    labels_test = np.array(test_prepared['log_visit_rate'])

    # Stats ( MAE, R2, Slope, Sp.coef.: for a few ml and all mechanistic configurations )
    svr_stats = compute_svr_stats(n_features)
    svr_stats['type'] = "ML"
    # nusvr_stats = compute_nusvr_stats(n_features)
    # nusvr_stats['type'] = "ML"
    br_stats = compute_bayesian_ridge_stats(n_features)
    br_stats['type'] = "ML"
    gbr_stats = compute_gbr_stats(n_features)
    gbr_stats['type'] = "ML"
    # mlp_stats   = compute_mlp_stats(n_features)
    # mlp_stats['type'] = "ML"
    # comb_stats = compute_combined_stats(n_features)
    # comb_stats['type'] = "Combination"
    # lons_stats = compute_lons_stats()
    # lons_stats['type'] = "Mechanistic"
    all_stats = pd.concat([svr_stats, br_stats, gbr_stats], axis=0, ignore_index=True).drop(columns=['n_features'])
    cols = all_stats.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    all_stats = all_stats[cols]
    print(all_stats.to_latex(index=False, float_format='%.2f'))
    all_stats.to_csv(ml_folder + '/prediction_stats.csv', index=False)

    # Plots
    yhat, labels_test = compute_bayesian_ridge_predictions(n_features)
    # Observed versus predicted
    fig, ax = plt.subplots()
    df_ml = pd.DataFrame({'obs': labels_test, 'pred': yhat})
    df_ml['source'] = 'ML'
    # df_lons = get_lonsdorf_predictions()[['log_visit_rate', 'lm_predicted']]
    # df_lons['source'] = 'Mechanistic'
    # df_lons.columns = df_ml.columns
    # test_withIDs = get_test_data_withIDs()
    # df_ml_id = pd.DataFrame({'obs':labels_test, 'yhat':yhat, 'study_id':test_withIDs.study_id, 'site_id':test_withIDs.site_id})
    # df_lons_id  = get_lonsdorf_predictions()
    # df_comb = pd.merge(df_lons_id, df_ml_id, on=['study_id', 'site_id'])
    # df_comb['pred'] = (df_comb.yhat + df_comb.lm_predicted)/2
    # df_comb = df_comb[['obs','pred']]
    limits_obs = np.array([np.min(df_ml['obs']) - 0.2, np.max(df_ml['obs']) + 0.5])
    # limits_pred = np.array([np.min(df_lons['pred']) - 0.2, np.max(df_lons['pred']) + 0.5])
    m_ml, b_ml = np.polyfit(df_ml.pred, df_ml.obs, 1)
    # m_lons, b_lons = np.polyfit(df_lons.pred, df_lons.obs, 1)
    # m_comb, b_comb    = np.polyfit(df_comb.pred, df_comb.obs, 1)
    # ax.scatter(df_lons['pred'], df_lons['obs'], color='green', alpha=0.5, label="Mechanistic")  # predictions mechanistic
    ax.scatter(df_ml['pred'], df_ml['obs'], color='red', alpha=0.5, label="Machine Learning")  # predictions ml
    # ax.scatter(df_comb['pred'], df_comb['obs'],  color='blue',  alpha=0.5, label="Combined")  # predictions combined
    # ax.plot(limits_obs, limits_obs, alpha=0.5, color='orange', label='observed=prediction')  # obs=pred
    # plt.plot(limits_obs, m_lons * limits_obs + b_lons, color='green')  # linear reg mechanistic
    plt.plot(limits_obs, m_ml * limits_obs + b_ml, color='red')  # linear reg ml
    # plt.plot(limits_obs, m_comb * limits_obs + b_comb, color='blue')    # linear reg combined
    ax.set_xlim(limits_obs[0], limits_obs[1])
    ax.set_ylim(limits_obs[0], limits_obs[1])
    ax.set_xlabel("Prediction", fontsize=14)
    ax.set_ylabel("log(Visitation Rate)", fontsize=14)
    ax.legend(loc='best', fontsize=14)
    plt.show()  # Save from window to adjust size
    # plt.savefig('C:/Users/Angel/git/Observ_models/data/ML/Regression/plots/predictions.eps', format='eps')

    # Save ML predictions
    df_test = get_test_data_withIDs()
    yhat_svr, labels_svr = compute_svr_predictions(n_features)
    yhat_br, labels_br = compute_bayesian_ridge_predictions(n_features)
    yhat_gbr, labels_gbr = compute_gbr_predictions(n_features)
    df_svr_pred = pd.DataFrame({'pred_svr': yhat_svr, 'study_id': df_test.study_id, 'site_id': df_test.site_id})
    df_br_pred = pd.DataFrame({'pred_br': yhat_br, 'study_id': df_test.study_id, 'site_id': df_test.site_id})
    df_gbr_pred = pd.DataFrame({'pred_gbr': yhat_gbr, 'study_id': df_test.study_id, 'site_id': df_test.site_id})
    df_predictions = pd.merge(df_svr_pred, df_br_pred, on=['study_id', 'site_id'])
    df_predictions = pd.merge(df_predictions, df_gbr_pred, on=['study_id', 'site_id'])
    df_predictions.to_csv(ml_folder + '/tables/prediction_by_site.csv', index=False)

    # # Density difference (observed-predicted), organic vs not-organic
    # test_management = get_test_data_full()
    # kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 1})
    # plt.figure()
    # df = pd.DataFrame({'obs': labels_test, 'pred': yhat, 'is_organic': [x == 3 for x in test_management.management]})
    # df_org = df[df.is_organic]
    # df_noorg = df[[(x == False) for x in df.is_organic]]
    # diff_org = df_org.obs - df_org.pred
    # diff_noorg = df_noorg.obs - df_noorg.pred
    # sns.distplot(diff_org, color="green", label="Organic farming", **kwargs)
    # sns.distplot(diff_noorg, color="red", label="Not organic", **kwargs)
    # plt.xlabel("(Observed - Predicted)", fontsize=14)
    # plt.ylabel("Density", fontsize=14)
    # plt.legend()
    #
    # # Density difference (observed-predicted), ML vs mechanistic
    # kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 1})
    # plt.figure()
    # df_ml = pd.DataFrame({'obs': labels_test, 'pred': yhat})
    # df_ml['source'] = 'ML'
    # df_lons = get_lonsdorf_predictions()
    # df_lons['source'] = 'Mechanistic'
    # df_lons.columns = df_ml.columns
    # diff_ml = df_ml.obs - df_ml.pred
    # diff_lons = df_lons.obs - df_lons.pred
    # sns.distplot(diff_lons, color="green", label="Mechanistic", **kwargs)
    # sns.distplot(diff_ml, color="red", label="ML", **kwargs)
    # plt.xlabel("(Observed - Predicted)", fontsize=14)
    # plt.ylabel("Density", fontsize=14)
    # plt.legend()

    # # Linear regression
    # X_reg, y_reg = np.array(df_lons.pred).reshape(-1, 1), np.array(df_lons.obs).reshape(-1, 1)
    # reg = LinearRegression().fit(X_reg, y_reg)
    # reg.score(X_reg, y_reg)
    # X_reg, y_reg = np.array(df_ml.pred).reshape(-1, 1), np.array(df_ml.obs).reshape(-1, 1)
    # reg = LinearRegression().fit(X_reg, y_reg)
    # reg.score(X_reg, y_reg)
    #
    # # Scatter plot organic vs not-organic
    # test_management = get_test_data_full()
    # fig, ax = plt.subplots()
    # df = pd.DataFrame({'obs': labels_test, 'pred': yhat, 'is_organic': [x == 3 for x in test_management.management]})
    # df_org = df[df.is_organic]
    # df_noorg = df[[(x == False) for x in df.is_organic]]
    # ax.scatter(df_org['pred'], df_org['obs'], color='green', alpha=0.5, label='Organic farming')
    # ax.scatter(df_noorg['pred'], df_noorg['obs'], color='red', alpha=0.5, label='Not organic')
    # ax.plot(yhat, yhat, alpha=0.5, color='orange', label='y=prediction ML')
    # ax.set_xlim(-5.5, 0)
    # ax.set_xlabel("Prediction ML", fontsize=14)
    # ax.set_ylabel("log(Visitation Rate)", fontsize=14)
    # ax.legend()
    # plt.show()
    #
    # # Interactive plot - organic
    # check_data = get_test_data_withIDs()
    # test_management = get_test_data_full()
    # is_organic = (test_management.management == 3)
    # check_data['is_organic'] = is_organic
    # df = pd.concat([check_data, pd.DataFrame(yhat, columns=['predicted'])], axis=1)
    # # fig = px.scatter(df, x="vr_pred", y="vr_obs", hover_data=df.columns, color="is_organic", trendline="ols")
    # # fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, color="is_organic", trendline="ols")
    # fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, trendline="ols")
    # fig.show()
    #
    # # Interactive plot - lonsdorf
    # check_data = get_test_data_withIDs(n_features)
    # df_ml = pd.DataFrame({'obs': labels_test, 'pred': yhat})
    # df_ml['source'] = 'ML'
    # df_ml = pd.concat([df_ml, check_data], axis=1)
    # df_lons = get_lonsdorf_predictions()
    # df_lons['source'] = 'Mechanistic'
    # df_lons = pd.concat([df_lons, check_data], axis=1)
    # df_lons.columns = df_ml.columns
    # df = pd.concat([df_ml, df_lons], axis=0)
    # fig = px.scatter(df, x="pred", y="obs", hover_data=df.columns, color="source", trendline="ols")
    # fig.show()

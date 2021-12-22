
import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import generate_trained_model as gtm
warnings.filterwarnings('ignore')
from scipy import stats

root_folder = "C:/Users/angel/git/OBServ_Models_Open/Machine Learning/"
n_features = 49

# Check stats model predictions
X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
mae = gtm.mean_absolute_error(X_reg, y_reg)
reg = LinearRegression().fit(X_reg, y_reg)
r2 = reg.score(X_reg, y_reg)
slope = reg.coef_[0][0]
sp_coef, sp_p = stats.spearmanr(yhat, labels_test)
a = pd.DataFrame({
    'model': "NuSVR",
    'n_features': n_features,
    'mae': mae,
    'r2': r2,
    'slope': slope,
    'sp_coef': sp_coef
}, index=[0])

# Stats ( MAE, R2, Slope, Sp.coef.: for a few ml and all mechanistic configurations )
svr_stats   = gtm.compute_svr_stats(n_features)
svr_stats['type'] = "ML"
nusvr_stats = gtm.compute_nusvr_stats(n_features)
nusvr_stats['type'] = "ML"
gbr_stats = gtm.compute_gbr_stats(n_features)
gbr_stats['type'] = "ML"
# mlp_stats   = compute_mlp_stats(n_features)
# mlp_stats['type'] = "ML"
comb_stats  = gtm.compute_combined_stats(n_features)
comb_stats['type'] = "Combination"
lons_stats  = gtm.compute_lons_stats()
lons_stats['type']  = "Mechanistic"
all_stats   = pd.concat([svr_stats, nusvr_stats, gbr_stats, lons_stats, comb_stats], axis=0, ignore_index=True).drop(columns=['n_features'])
cols = all_stats.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_stats = all_stats[cols]
print(all_stats.to_latex(index=False, float_format='%.2f'))
all_stats.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/tables/prediction_stats.csv', index=False)

# Plots
yhat, labels_test = gtm.compute_gbr_predictions(n_features)
# Observed versus predicted
fig, ax           = plt.subplots()
df_ml             = pd.DataFrame({'obs':labels_test, 'pred':yhat})
df_ml['source']   = 'ML'
df_lons           = gtm.get_lonsdorf_predictions()[['log_visit_rate','lm_predicted']]
df_lons['source'] = 'Mechanistic'
df_lons.columns   = df_ml.columns
# test_withIDs = get_test_data_withIDs()
# df_ml_id = pd.DataFrame({'obs':labels_test, 'yhat':yhat, 'study_id':test_withIDs.study_id, 'site_id':test_withIDs.site_id})
# df_lons_id  = get_lonsdorf_predictions()
# df_comb = pd.merge(df_lons_id, df_ml_id, on=['study_id', 'site_id'])
# df_comb['pred'] = (df_comb.yhat + df_comb.lm_predicted)/2
# df_comb = df_comb[['obs','pred']]
limits_obs        = np.array([np.min(df_lons['obs']) - 0.2, np.max(df_lons['obs']) + 0.5])
limits_pred       = np.array([np.min(df_lons['pred']) - 0.2, np.max(df_lons['pred']) + 0.5])
m_ml, b_ml        = np.polyfit(df_ml.pred  , df_ml.obs  , 1)
m_lons, b_lons    = np.polyfit(df_lons.pred, df_lons.obs, 1)
# m_comb, b_comb    = np.polyfit(df_comb.pred, df_comb.obs, 1)
ax.scatter(df_lons['pred'], df_lons['obs'],  color='green', alpha=0.5, label="Mechanistic")       # predictions mechanistic
ax.scatter(df_ml['pred'],   df_ml['obs'],    color='red',   alpha=0.5, label="Machine Learning")  # predictions ml
# ax.scatter(df_comb['pred'], df_comb['obs'],  color='blue',  alpha=0.5, label="Combined")  # predictions combined
ax.plot(limits_obs, limits_obs, alpha=0.5, color='orange',label='observed=prediction')            # obs=pred
plt.plot(limits_obs, m_lons * limits_obs + b_lons, color='green')   # linear reg mechanistic
plt.plot(limits_obs, m_ml   * limits_obs + b_ml, color='red')       # linear reg ml
# plt.plot(limits_obs, m_comb * limits_obs + b_comb, color='blue')    # linear reg combined
ax.set_xlim(limits_obs[0], limits_obs[1])
ax.set_ylim(limits_obs[0], limits_obs[1])
ax.set_xlabel("Prediction", fontsize=14)
ax.set_ylabel("log(Visitation Rate)", fontsize=14)
ax.legend(loc='best', fontsize=14)
plt.show() # Save from window to adjust size
# plt.savefig('C:/Users/angel/git/Observ_models/data/ML/Regression/plots/predictions.eps', format='eps')

# Save ML predictions
df_test = gtm.get_test_data_withIDs()
yhat_svr,   labels_svr   = gtm.compute_svr_predictions(n_features)
yhat_nusvr, labels_nusvr = gtm.compute_nusvr_predictions(n_features)
yhat_gbr,   labels_gbr   = gtm.compute_gbr_predictions(n_features)
df_svr_pred              = pd.DataFrame( {'pred_svr':   yhat_svr,   'study_id': df_test.study_id, 'site_id':df_test.site_id })
df_nusvr_pred            = pd.DataFrame( {'pred_nusvr': yhat_nusvr, 'study_id': df_test.study_id, 'site_id':df_test.site_id })
df_gbr_pred              = pd.DataFrame( {'pred_gbr':   yhat_gbr,   'study_id': df_test.study_id, 'site_id':df_test.site_id })
df_predictions           = pd.merge(df_svr_pred, df_nusvr_pred,  on=['study_id', 'site_id'])
df_predictions           = pd.merge(df_predictions, df_gbr_pred, on=['study_id', 'site_id'])
df_predictions.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/tables/prediction_by_site.csv', index=False)

# Density difference (observed-predicted), organic vs not-organic
test_management = gtm.get_test_data_full()
kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 1})
plt.figure()
df = pd.DataFrame({'obs':labels_test, 'pred':yhat, 'is_organic':[ x == 3 for x in test_management.management ]})
df_org     = df[ df.is_organic ]
df_noorg   = df[ [(x==False) for x in df.is_organic] ]
diff_org   = df_org.obs   - df_org.pred
diff_noorg = df_noorg.obs - df_noorg.pred
sns.distplot(diff_org, color="green", label="Organic farming", **kwargs)
sns.distplot(diff_noorg, color="red", label="Not organic", **kwargs)
plt.xlabel("(Observed - Predicted)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend()

# Density difference (observed-predicted), ML vs mechanistic
kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 1})
plt.figure()
df_ml       = pd.DataFrame({'obs':labels_test, 'pred':yhat})
df_ml['source'] = 'ML'
df_lons = gtm.get_lonsdorf_predictions()
df_lons['source'] = 'Mechanistic'
df_lons.columns = df_ml.columns
diff_ml   = df_ml.obs   - df_ml.pred
diff_lons = df_lons.obs - df_lons.pred
sns.distplot(diff_lons, color="green", label="Mechanistic", **kwargs)
sns.distplot(diff_ml,   color="red",   label="ML", **kwargs)
plt.xlabel("(Observed - Predicted)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend()

# Linear regression
X_reg, y_reg = np.array(df_lons.pred).reshape(-1, 1), np.array(df_lons.obs).reshape(-1, 1)
reg = LinearRegression().fit(X_reg, y_reg)
reg.score(X_reg, y_reg)
X_reg, y_reg = np.array(df_ml.pred).reshape(-1, 1), np.array(df_ml.obs).reshape(-1, 1)
reg = LinearRegression().fit(X_reg, y_reg)
reg.score(X_reg, y_reg)

# Scatter plot organic vs not-organic
test_management = gtm.get_test_data_full()
fig, ax = plt.subplots()
df = pd.DataFrame({'obs':labels_test, 'pred':yhat, 'is_organic':[ x == 3 for x in test_management.management ]})
df_org     = df[ df.is_organic ]
df_noorg   = df[ [(x==False) for x in df.is_organic] ]
ax.scatter(df_org['pred'],   df_org['obs'],   color='green', alpha=0.5, label='Organic farming')
ax.scatter(df_noorg['pred'], df_noorg['obs'], color='red',   alpha=0.5, label='Not organic')
ax.plot(yhat,yhat, alpha=0.5, color='orange',label='y=prediction ML')
ax.set_xlim(-5.5,0)
ax.set_xlabel("Prediction ML", fontsize=14)
ax.set_ylabel("log(Visitation Rate)", fontsize=14)
ax.legend()
plt.show()

# Interactive plot - organic
check_data = gtm.get_test_data_withIDs()
test_management = gtm.get_test_data_full()
is_organic = (test_management.management == 3)
check_data['is_organic'] = is_organic
df = pd.concat([ check_data, pd.DataFrame(yhat, columns=['predicted']) ], axis=1)
# fig = px.scatter(df, x="vr_pred", y="vr_obs", hover_data=df.columns, color="is_organic", trendline="ols")
# fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, color="is_organic", trendline="ols")
fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, trendline="ols")
fig.show()

# Interactive plot - lonsdorf
check_data  = gtm.get_test_data_withIDs(n_features)
df_ml       = pd.DataFrame({'obs':labels_test, 'pred':yhat})
df_ml['source'] = 'ML'
df_ml = pd.concat([df_ml, check_data], axis=1)
df_lons = gtm.get_lonsdorf_predictions()
df_lons['source'] = 'Mechanistic'
df_lons = pd.concat([df_lons, check_data], axis=1)
df_lons.columns = df_ml.columns
df = pd.concat([ df_ml, df_lons ], axis=0)
fig = px.scatter(df, x="pred", y="obs", hover_data=df.columns, color="source", trendline="ols")
fig.show()







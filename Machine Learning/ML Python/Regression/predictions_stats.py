
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import generate_trained_model as gtm
warnings.filterwarnings('ignore')

from utils import define_root_folder
root_folder = define_root_folder.root_folder
n_features = 49

# Stats ( MAE, R2, Slope, Sp.coef.: for a few ml and all mechanistic configurations )
svr_stats   = gtm.compute_svr_stats(n_features)
svr_stats['type'] = "ML"
nusvr_stats = gtm.compute_nusvr_stats(n_features)
nusvr_stats['type'] = "ML"
gbr_stats = gtm.compute_gbr_stats(n_features)
gbr_stats['type'] = "ML"
# mlp_stats   = compute_mlp_stats(n_features)
# mlp_stats['type'] = "ML"
all_stats   = pd.concat([svr_stats, nusvr_stats, gbr_stats], axis=0, ignore_index=True).drop(columns=['n_features'])
cols = all_stats.columns.tolist()
cols = cols[-1:] + cols[:-1]
all_stats = all_stats[cols]
print(all_stats.to_latex(index=False, float_format='%.2f'))
all_stats.to_csv(root_folder+'data/tables/prediction_stats.csv', index=False)

# Plots
yhat, labels_test = gtm.compute_gbr_predictions(n_features)
# Observed versus predicted
fig, ax           = plt.subplots()
df_ml             = pd.DataFrame({'obs':labels_test, 'pred':yhat})
df_ml['source']   = 'ML'
limits_obs        = np.array([np.min(df_ml['obs']) - 0.2, np.max(df_ml['obs']) + 0.5])
limits_pred       = np.array([np.min(df_ml['pred']) - 0.2, np.max(df_ml['pred']) + 0.5])
m_ml, b_ml        = np.polyfit(df_ml.pred  , df_ml.obs  , 1)
ax.scatter(df_ml['pred'],   df_ml['obs'],   alpha=0.5, label="Machine Learning")  # predictions ml
ax.plot(limits_obs, m_ml   * limits_obs + b_ml, label='linear regression')       # linear reg ml
ax.plot(limits_obs, limits_obs, alpha=0.5, label='observed=prediction')            # obs=pred
ax.set_xlim(limits_obs[0], limits_obs[1])
ax.set_ylim(limits_obs[0], limits_obs[1])
ax.set_xlabel("Prediction", fontsize=14)
ax.set_ylabel("log(Visitation Rate)", fontsize=14)
ax.legend(loc='best', fontsize=14)
plt.show() # Save from window to adjust size
# plt.savefig(root_folder+'data/plots/predictions.eps', format='eps')

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
df_predictions.to_csv(root_folder+'data/tables/prediction_by_site.csv', index=False)

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
diff_ml   = df_ml.obs   - df_ml.pred
sns.distplot(diff_ml,   color="red",   label="ML", **kwargs)
plt.xlabel("(Observed - Predicted)", fontsize=14)
plt.ylabel("Density", fontsize=14)
plt.legend()

# Linear regression score
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







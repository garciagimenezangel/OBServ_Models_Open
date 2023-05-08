
import pandas as pd
from functools import reduce
from utils import metrics, model_dict
import warnings
warnings.filterwarnings('ignore')

# Create comparative table between models, with a summary of the full set of stats
df_stats = pd.DataFrame()
for model in model_dict.model_labels.keys():
    if ("Lonsdorf" in model):
        df_data = metrics.get_data(model, metrics.guild)
        df_studywise = metrics.get_metrics_by_study(df_data)
        df_studywise['model'] = model_dict.model_labels[model]
        df_stats = pd.concat([df_stats, df_studywise], axis=0)
        df_stats['study_id'] = [x[0] for x in df_stats.index]
        df_studywise.to_csv("C:/Users/Angel/git/Observ_models/data/Lonsdorf evaluation/Study metrics/"+model+".csv", index=False)

mean_slope = df_stats[['model','slope']]        .groupby(['model'], sort=False, as_index=False).mean().rename(columns={'slope': 'mean slope'})
mean_r2    = df_stats[['model','r2']]           .groupby(['model'], sort=False, as_index=False).mean().rename(columns={'r2': 'mean r2'})
mean_spear = df_stats[['model','Spearman_coef']].groupby(['model'], sort=False, as_index=False).mean().rename(columns={'Spearman_coef': 'mean Spearman coef'})
std_slope  = df_stats[['model','slope']]        .groupby(['model'], sort=False, as_index=False).std().rename(columns={'slope': 'std slope'})
std_r2     = df_stats[['model','r2']]           .groupby(['model'], sort=False, as_index=False).std().rename(columns={'r2': 'std r2'})
std_spear  = df_stats[['model','Spearman_coef']].groupby(['model'], sort=False, as_index=False).std().rename(columns={'Spearman_coef': 'std Spearman coef'})
prop_sign  = df_stats.groupby('model').apply(metrics.get_proportion_of_significant_fits, option='Spearman')
prop_sign  = pd.DataFrame({'model':prop_sign.index, 'significant':prop_sign.values})
df_summary = reduce(lambda left,right: pd.merge(left,right,on=['model']), [mean_slope, std_slope, mean_r2, std_r2, mean_spear, std_spear, prop_sign])
df_summary.to_csv("C:/Users/Angel/git/Observ_models/data/Lonsdorf evaluation/Study metrics/summary.csv", index=False)

# # Correlations
# X_reg, y_reg = np.array(df_studywise.var_model).reshape(-1, 1), np.array(df_studywise.Spearman_coef).reshape(-1, 1)
# reg = LinearRegression().fit(X_reg, y_reg)
# reg.score(X_reg, y_reg)

# model_name = 'Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'
# df_study = df_data.loc[0:15]
# observed = df_study[['log_ab']]
# X_constant = sm.add_constant(df_study[[model_name]])
# lin_reg = sm.OLS(observed, X_constant).fit()
# p_val = lin_reg.pvalues[1]
#
# model = df_study[[model_name]]
# scatter(model, observed)
# plt.xlabel("Prediction ML")
# plt.ylabel("log(Observed abundance)")
# plot(model, lin_reg.params[1] * model + lin_reg.params[0])


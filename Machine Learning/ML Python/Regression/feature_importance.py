"""
Script to compute the relative importance of the features in the ML models
"""

import pickle
from functools import reduce
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import warnings
from sklearn.svm import SVR, NuSVR
from sklearn.inspection import permutation_importance
import ast
from utils import feature_aliases
warnings.filterwarnings('ignore')

from utils import define_root_folder
root_folder = define_root_folder.root_folder

def get_train_data_reduced(n_features):
    return pd.read_csv(root_folder+'data/train/data_reduced_'+str(n_features)+'.csv')

def get_test_data_reduced(n_features):
    return pd.read_csv(root_folder+'data/test/data_reduced_'+str(n_features)+'.csv')

def get_train_data_prepared():
    return pd.read_csv(root_folder+'data/train/data_prepared.csv')

def get_test_data_prepared():
    return pd.read_csv(root_folder+'data/test/data_prepared.csv')

def get_best_models(n_features=0):
    data_dir = root_folder+'data/hyperparameters/'
    if n_features>0:
        return pd.read_csv(data_dir + 'best_scores_'+str(n_features)+'.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores.csv')


# Get best models
n_features=49
df_best_models = get_best_models(n_features)

# Load custom cross validation
with open(root_folder+'data/train/myCViterator.pkl', 'rb') as file:
    myCViterator = pickle.load(file)

# Get data
train_prepared   = get_train_data_reduced(n_features)
test_prepared    = get_test_data_reduced(n_features)
predictors_train = train_prepared.iloc[:,:-1]
labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()

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

# ##########################################
# # Permutation importance for other models
# ##########################################
# NuSVR
best_model       = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
d     = ast.literal_eval(best_model.best_params)
model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
model.fit(predictors_train, labels_train)
perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
imp_values = perm_importance.importances_mean / np.sum(perm_importance.importances_mean)
df_feat_imp_nusvr = pd.DataFrame({'feature':predictors_train.columns, 'NuSVR':imp_values})

# SVR
best_model       = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
d     = ast.literal_eval(best_model.best_params)
model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
model.fit(predictors_train, labels_train)
perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
imp_values = perm_importance.importances_mean / np.sum(perm_importance.importances_mean)
df_feat_imp_svr = pd.DataFrame({'feature':predictors_train.columns, 'SVR':imp_values})

# Merge feature importance for different models
df_feat_imp = reduce(lambda left, right: pd.merge(left, right, on=['feature']), [df_feat_imp_gbr, df_feat_imp_nusvr, df_feat_imp_svr])

# Collapse rows of biomes and crops into one single row (each)
df_biome_rows = df_feat_imp[df_feat_imp.feature.str.contains('x0')]
sum_biomes    = df_biome_rows.sum()
sum_biomes.values[0] = 'Sum(biomes)'
df_crop_rows  = df_feat_imp[df_feat_imp.feature.str.contains('x1')]
sum_crops     = df_crop_rows.sum()
sum_crops.values[0]  = 'Sum(crops)'
df_other_rows = df_feat_imp[(~df_feat_imp.feature.str.contains('x0')) & (~df_feat_imp.feature.str.contains('x1'))]
df_imp_summ   = df_other_rows.append(sum_biomes, ignore_index=True)
df_imp_summ   = df_imp_summ.append(sum_crops, ignore_index=True)
df_imp_summ["feature"].replace(feature_aliases.feat_dict, inplace=True)

# Sort values
model_to_sort = 'NuSVR'
df_imp_summ.sort_values(by=[model_to_sort], ascending=False, inplace=True)

# Plot
labels = df_imp_summ.feature
x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x + width / 2, df_imp_summ.NuSVR, width, label='NuSVR')
rects2 = ax.bar(x - width / 2, df_imp_summ.GBR, width, label='GBR')
# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Importance (0-1)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=80)
ax.legend(bbox_to_anchor=(1.25, 1), loc=1, borderaxespad=0.)
fig.tight_layout()
plt.show()
plt.savefig(root_folder+'data/plots/feature_importance.eps', format='eps', bbox_inches='tight')



# Because this dataset contains multicollinear features, the permutation importance will show that none of the features
# are important. One approach to handling multicollinearity is by performing hierarchical clustering on the features’
# Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
import pickle
from collections import defaultdict
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import NuSVR
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform
from scipy import stats
from utils import feature_aliases
warnings.filterwarnings('ignore')

root_folder = "C:/Users/angel/git/OBServ_Models_Open/Machine Learning/"

def get_train_data_prepared():
    data_dir   = root_folder + "data/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_prepared():
    data_dir   = root_folder + "data/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def compute_vif(X):
    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return(vif)

############################################
# The analysis of collinearity and subsequent selection of features based on such analysis, must be done using the
# training set only, because selection of the features affect the final predictions on the test set, so if used in this
# process, those predictions would be biased to better results.
train_prepared = get_train_data_prepared()
predictors_train = train_prepared.iloc[:, :-1]
labels_train = np.array(train_prepared.iloc[:, -1:]).flatten()
test_prepared   = get_test_data_prepared()
predictors_test = test_prepared.iloc[:,:-1]

# Load custom cross validation
with open(root_folder + 'data/train/myCViterator.pkl', 'rb') as file:
    myCViterator = pickle.load(file)

# This analysis is performed only on originally-numeric columns
num_cols = list(predictors_train.columns[:56])
cat_cols = list(predictors_train.columns[56:])
cat_cols.remove('x1_Brassica rapa')         # Crops species that are not present in the training set (if crop species used)
cat_cols.remove('x1_Citrus limon')
cat_cols.remove('x1_Cucurbita pepo')
cat_cols.remove('x1_Gossypium hirsutum')
# cat_cols.remove('x1_Rutaceae')            # Crops families that are not present in the training set (if crop families used)
cat_cols.remove('x0_6.0')                   # Biomes that are not present in the training set
train_num = train_prepared[num_cols]
predictors_train = predictors_train[num_cols+cat_cols]

# Define model
model = NuSVR() # Any model (loop over several?)

################################################
# Plot a heatmap of the correlated features:
################################################
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
corr = spearmanr(train_num).correlation
# Ensure the correlation matrix is symmetric
corr = (corr + corr.T)/2
np.fill_diagonal(corr, 1)
# We convert the correlation matrix to a distance matrix before performing
# hierarchical clustering using Ward's linkage.
distance_matrix = 1 - np.abs(corr)
dist_linkage = hierarchy.ward(squareform(distance_matrix))
dendro = hierarchy.dendrogram(dist_linkage, orientation='left', labels=[feature_aliases.feat_dict[x] for x in train_num.columns])
dendro_idx = np.arange(0, len(dendro['ivl']))

# ax2.imshow(corr[dendro['leaves'], :][:, dendro['leaves']])
# ax2.set_xticks(dendro_idx)
# ax2.set_yticks(dendro_idx)
# ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
# ax2.set_yticklabels(dendro['ivl'])
# fig.tight_layout()
# plt.show()
plt.savefig(root_folder + 'plots/dendrogram.eps', format='eps', bbox_inches='tight')

# Cross-validation to find the best threshold
results = []
thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
for t in thresholds:
    # Select features for threshold=t
    cluster_ids = hierarchy.fcluster(dist_linkage, t, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_columns  = [v[0] for v in cluster_id_to_feature_ids.values()]
    selected_features = train_num.columns[selected_columns]
    print("N selected features: " + str(np.size(selected_features)))

    # Compute indicators
    mae_t = []
    r2_t  = []
    sp_t  = []
    slope_t = []
    for ind_train, ind_test in myCViterator:
        features_train = predictors_train.iloc[ind_train]
        features_test  = predictors_train.iloc[ind_test]
        target_train   = np.array(train_prepared.iloc[ind_train, -1:]).flatten()
        target_test    = np.array(train_prepared.iloc[ind_test, -1:]).flatten()
        X_train_sel    = features_train[selected_features]
        X_test_sel     = features_test[selected_features]

        # Predict with the selected features
        model.fit(X_train_sel, target_train)
        yhat    = model.predict(X_test_sel)
        X_reg, y_reg = yhat.reshape(-1, 1), target_test.reshape(-1, 1)
        reg     = LinearRegression().fit(X_reg, y_reg)
        mae_t   = np.concatenate([mae_t,   [mean_absolute_error(X_reg, y_reg)]])
        r2_t    = np.concatenate([r2_t,    [reg.score(X_reg, y_reg)]])
        slope_t = np.concatenate([slope_t, [reg.coef_[0][0]]])
        sp_coef = stats.spearmanr(yhat, target_test)[0]
        sp_t = np.concatenate([sp_t, [sp_coef]])

    results.append({
        'threshold': t,
        'N_features': np.size(selected_features),
        'mae_mean': np.mean(mae_t),
        'mae_std':  np.std(mae_t),
        'r2_mean': np.mean(r2_t),
        'r2_std':  np.std(r2_t),
        'sp_mean': np.mean(sp_t),
        'sp_std': np.std(sp_t),
        'slope_mean': np.mean(slope_t),
        'slope_std': np.std(slope_t),
    })

    # Plot results
    df_results = pd.DataFrame(results)
    fig = plt.figure()
    axes = plt.subplot(111)
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.75, box.height])
    axes.set_xlabel("Threshold")
    axes.set_ylabel("Metrics")
    axes.plot(df_results.threshold, df_results.mae_mean, 'o-', color='r', label="MAE")
    axes.plot(df_results.threshold, df_results.r2_mean, 'o-', color='g', label="R2")
    axes.plot(df_results.threshold, df_results.sp_mean, 'o-', color='b', label="Spearman Coef.")
    axes.fill_between(df_results.threshold, df_results.mae_mean - df_results.mae_std, df_results.mae_mean + df_results.mae_std, color='r', alpha=0.1)
    axes.fill_between(df_results.threshold, df_results.r2_mean - df_results.r2_std, df_results.r2_mean + df_results.r2_std, color='g', alpha=0.1)
    axes.fill_between(df_results.threshold, df_results.sp_mean - df_results.sp_std, df_results.sp_mean + df_results.sp_std, color='b', alpha=0.1)
    axes.legend(bbox_to_anchor=(1.4, 1), loc=1, borderaxespad=0.)
    df_results.to_csv(root_folder + 'data/tables/feature_collinearity.csv', index=False)
    plt.savefig(root_folder + 'data/plots/feature_collinearity.tiff', format='tiff', bbox_inches='tight')

# Select features with the best threshold
best_t = 0.75
cluster_ids = hierarchy.fcluster(dist_linkage, best_t, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_columns = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features = train_num.columns[selected_columns].to_list()
print("N selected numeric features: " + str(np.size(selected_features)))
predictors_reduced_train = predictors_train[selected_features]

# Variance Inflation Factor: remove, one by one, the variables with the highest value, until every predictor
# shows VIF<10 (ref: https://onlinelibrary.wiley.com/doi/epdf/10.1111/j.1600-0587.2012.07348.x)
vif = compute_vif(predictors_reduced_train)
# selected_features = ...
# predictors_reduced_train = predictors_train[selected_features]
# ...repeat vif until all variables show VIF<10...
vif.to_csv(root_folder + 'data/tables/vif.csv', index=False)

# Save reduced datasets
predictors_reduced_train = predictors_train[selected_features+cat_cols]
predictors_reduced_test  = predictors_test[selected_features+cat_cols]
data_reduced_train       = pd.concat([predictors_reduced_train, train_prepared.iloc[:,-1:]], axis=1)
data_reduced_test        = pd.concat([predictors_reduced_test,  test_prepared.iloc[:,-1:]], axis=1)

data_reduced_train.to_csv(root_folder + 'data/train/data_reduced_'+
                         str(np.size(selected_features+cat_cols)) +'.csv', index=False)
data_reduced_test.to_csv(root_folder + 'data/test/data_reduced_'+
                         str(np.size(selected_features+cat_cols)) +'.csv', index=False)

# Save table for clusters
clusters = []
for values in cluster_id_to_feature_ids.values():
    proxy    = feature_aliases.feat_dict[train_num.columns[values[0]]]
    features = [feature_aliases.feat_dict[x] for x in train_num.columns[values[0:]]]
    clusters.append({'Proxy': proxy, 'Features': ', '.join(features)})
df_clusters = pd.DataFrame(clusters)
df_clusters.to_csv(root_folder + 'data/tables/feature_clusters.csv', index=False)

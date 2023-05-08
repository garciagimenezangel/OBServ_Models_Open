
import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVC, SVR, NuSVR
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import pandas as pd
models_repo = "C:/Users/Angel/git/Observ_models/"
root_dir    = models_repo + "data/Prepared Datasets/"
out_dir = models_repo + "data/ML/Regression/plots/"

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

def get_best_models(n_features=0):
    data_dir = models_repo + "data/ML/Regression/hyperparameters/"
    if n_features > 0:
        return pd.read_csv(data_dir + 'best_scores_' + str(n_features) + '.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores_all_predictors.csv')

def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : estimator instance
        An estimator instance implementing `fit` and `predict` methods which
        will be cloned for each validation.

    title : str
        Title for the chart.

    X : array-like of shape (n_samples, n_features)
        Training vector, where ``n_samples`` is the number of samples and
        ``n_features`` is the number of features.

    y : array-like of shape (n_samples) or (n_samples, n_features)
        Target relative to ``X`` for classification or regression;
        None for unsupervised learning.

    axes : array-like of shape (3,), default=None
        Axes to use for plotting the curves.

    ylim : tuple of shape (2,), default=None
        Defines minimum and maximum y-values plotted, e.g. (ymin, ymax).

    cv : int, cross-validation generator or an iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, default=None
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like of shape (n_ticks,)
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the ``dtype`` is float, it is regarded
        as a fraction of the maximum size of the training set (that is
        determined by the selected validation method), i.e. it has to be within
        (0, 1]. Otherwise it is interpreted as absolute sizes of the training
        sets. Note that for classification the number of samples usually have
        to be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    fig = plt.figure()
    axes = plt.subplot(111)
    box = axes.get_position()
    axes.set_position([box.x0, box.y0, box.width * 0.75, box.height])

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("MAE")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True,
                       scoring="neg_mean_absolute_error")
    train_scores_mean = np.mean(-train_scores, axis=1)
    train_scores_std = np.std(-train_scores, axis=1)
    test_scores_mean = np.mean(-test_scores, axis=1)
    test_scores_std = np.std(-test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation")
    axes.legend(bbox_to_anchor=(1.5, 1), loc=1, borderaxespad=0.)

    return plt


# Load df_data
n_features = 47  # use n_features = 0 to specify "All"
if n_features:  # use only N predictors (selected in a previous step such as feature collinearity analysis)
    data_prepared = get_train_data_prepared(n_features)
    predictors = data_prepared.drop(columns=['log_visit_rate'])
else:  # use all predictors
    data_prepared = get_train_data_prepared()
    predictors = data_prepared.drop(columns=['study_id', 'site_id', 'author_id', 'log_vr_small', 'log_vr_large', 'log_visit_rate'])
labels = np.array(data_prepared['log_visit_rate'])

# Load custom cross validation
with open(root_dir + 'myCViterator.pkl', 'rb') as file:
    myCViterator = pickle.load(file)

# Plot learning curve
title = "Learning Curves"
# best_model = df_best_models.loc[df_best_models.model.astype(str) == "GradientBoostingRegressor()"].iloc[0]
# d = ast.literal_eval(best_model.best_params)
# model = GradientBoostingRegressor(loss=d['loss'], learning_rate=d['learning_rate'], n_estimators=d['n_estimators'],
#                                   subsample=d['subsample'],
#                                   min_samples_split=d['min_samples_split'], min_samples_leaf=d['min_samples_leaf'],
#                                   min_weight_fraction_leaf=d['min_weight_fraction_leaf'],
#                                   max_depth=d['max_depth'], min_impurity_decrease=d['min_impurity_decrease'],
#                                   max_features=d['max_features'], alpha=d['alpha'],
#                                   max_leaf_nodes=d['max_leaf_nodes'], ccp_alpha=d['ccp_alpha'], random_state=135)
df_best_models = get_best_models(n_features)
use_model = 'BayR'
if use_model == 'BayR':
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "BayesianRidge(n_iter=10000)"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = BayesianRidge(n_iter=1000, alpha_1=d['alpha_1'], alpha_2=d['alpha_2'], fit_intercept=d['fit_intercept'], lambda_1=d['lambda_1'], lambda_2=d['lambda_2'],
                          normalize=d['normalize'])
elif use_model == 'SVR':
    best_model = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
    d = ast.literal_eval(best_model.best_params)
    model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
plot_learning_curve(model, title, predictors, labels, cv=5, n_jobs=6)
plt.savefig('C:/Users/Angel/git/Observ_models/data/ML/Regression/plots/learning_curve_{}feat_{}.tiff'.format(str(n_features), use_model), format='tiff')



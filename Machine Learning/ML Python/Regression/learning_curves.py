"""
Script to visualize the learning curves of the models
"""

import ast
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC, SVR, NuSVR
from sklearn.model_selection import learning_curve
import pandas as pd

from utils import define_root_folder
root_folder = define_root_folder.root_folder

def get_data_prepared():
    data_dir   = root_folder + "data/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_data_reduced(n_features):
    data_dir   = root_folder + "data/train/"
    return pd.read_csv(data_dir+'data_reduced_'+str(n_features)+'.csv')

def get_best_models(n_features=0):
    data_dir = root_folder + "data/hyperparameters/"
    if n_features>0:
        return pd.read_csv(data_dir + 'best_scores_'+str(n_features)+'.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores.csv')

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
    axes.legend(bbox_to_anchor=(1.4, 1), loc=1, borderaxespad=0.)

    return plt


# Load data
data_prepared = get_data_reduced(49)
df_best_models = get_best_models(49)
predictors = data_prepared.iloc[:, :-1]
labels = np.array(data_prepared.iloc[:, -1:]).flatten()
# Load custom cross validation
with open(root_folder+'/data/train/myCViterator.pkl', 'rb') as file:
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
best_model = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
d = ast.literal_eval(best_model.best_params)
model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
plot_learning_curve(model, title, predictors, labels, cv=5, n_jobs=6)
plt.savefig(root_folder+'/data/plots/learning_curve_49feat.tiff', format='tiff')



import pandas as pd
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
import pollinators_dependency as poll_dep
import logger


def compute_visit_rate(df_data):
    df_data['visit_rate_wb_bmb'] = (df_data['ab_wildbees'] + df_data['ab_bombus']) / df_data['total_sampled_time']
    df_data['log_visit_rate'] = np.log(df_data['visit_rate_wb_bmb'])
    df_data.drop(columns=['visit_rate_wb_bmb'], inplace=True)
    return df_data


def compute_visit_rate_small(df_data):
    df_data['visit_rate_wb'] = (df_data['ab_wildbees'] + 1) / df_data['total_sampled_time']
    df_data['log_vr_small'] = np.log(df_data['visit_rate_wb'])
    df_data.drop(columns=['visit_rate_wb'], inplace=True)
    return df_data


def compute_visit_rate_large(df_data):
    df_data['visit_rate_bmb'] = (df_data['ab_bombus'] + 1) / df_data['total_sampled_time']
    df_data['log_vr_large'] = np.log(df_data['visit_rate_bmb'])
    df_data.drop(columns=['visit_rate_bmb'], inplace=True)
    return df_data


def fill_missing_abundances(df_data):
    df_data.loc[df_data['ab_bombus'].isna(), 'ab_bombus'] = 0
    df_data.loc[df_data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    return df_data


def fill_biome(x, df_data):
    data_study_id = df_data.loc[df_data.study_id == x, ]
    return data_study_id.biome_num.mode().iloc[0]


def fill_missing_biomes(df_data):
    missing_biome = df_data.loc[df_data.biome_num == 'unknown', ]
    new_biome = [fill_biome(x, df_data) for x in missing_biome.study_id]
    df_data.loc[df_data.biome_num == 'unknown', 'biome_num'] = new_biome
    return df_data


def add_landcover_diversity(data):
    data_dir = "C:/Users/Angel/git/Observ_models/data/"
    model_data = pd.read_csv(data_dir + 'model_data.csv')[['site_id', 'study_id', 'geodata.landCoverDiversity']]
    model_data.rename(columns={'geodata.landCoverDiversity': 'lc_div'}, inplace=True)
    return data.merge(model_data, on=['study_id', 'site_id'])


def run_pipeline(df_data):
    """
    Run pipeline to prepare training and test datasets
    :param df_data:
    :return:
    """
    df_data = fill_missing_biomes(df_data)
    df_data = fill_missing_abundances(df_data)
    df_data = compute_visit_rate(df_data)
    df_data = compute_visit_rate_small(df_data)
    df_data = compute_visit_rate_large(df_data)
    df_data['author_id'] = [study.split("_", 2)[0] + study.split("_", 2)[1] for study in df_data.study_id]
    df_data = add_landcover_diversity(df_data)

    # Separate predictors and labels
    predictors = df_data.drop(columns=["log_visit_rate", "log_vr_small", "log_vr_large"], axis=1)
    labels = df_data['log_visit_rate'].copy()
    labels_small = df_data['log_vr_small'].copy()
    labels_large = df_data['log_vr_large'].copy()

    # (Set biome as categorical)
    predictors['biome_num'] = predictors.biome_num.astype('object')

    # Drop not-predictor columns
    not_predictor_columns = ['latitude', 'longitude', 'ab_wildbees', 'ab_bombus', 'total_sampled_time', 'sampling_year', 'sampling_abundance']
    dummy_col = ["study_id", "site_id", "author_id"]  # keep this to use later (e.g. create custom cross validation iterator)
    df_not_predictors = predictors.copy()[not_predictor_columns]
    predictors.drop(columns=not_predictor_columns, inplace=True)

    #######################################
    # Pipeline
    #######################################
    # Apply transformations (fill values, standardize, one-hot encoding)
    # First, replace numeric by mean, grouped by study_id (if all sites have NAs, then replace by dataset mean later in the imputer)
    pred_num = predictors.select_dtypes('number')
    n_nas = pred_num.isna().sum().sum()
    pred_num['study_id'] = df_data.study_id
    pred_num = pred_num.groupby('study_id').transform(lambda x: x.fillna(x.mean()))
    logger.logging.info("NA'S before transformation: " + str(n_nas))
    logger.logging.info("Total numeric values: " + str(pred_num.size))
    logger.logging.info("Percentage: " + str(n_nas * 100 / pred_num.size))

    # Define pipeline
    numeric_col = list(pred_num)
    onehot_col = ["biome_num", "crop"]
    ordinal_col = ["management"]
    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler())
    ])
    ordinal_pipeline = Pipeline([
        ('manag_imputer', SimpleImputer(strategy="constant", fill_value="conventional")),
        ('ordinal_encoder', OrdinalEncoder(categories=[['conventional', 'IPM', 'unmanaged', 'organic']]))
    ])
    onehot_pipeline = Pipeline([
        ('onehot_encoder', OneHotEncoder())
    ])
    dummy_pipeline = Pipeline([('dummy_imputer', SimpleImputer(strategy="constant", fill_value=""))])
    X = onehot_pipeline.fit(predictors[onehot_col])
    onehot_encoder_names = X.named_steps['onehot_encoder'].get_feature_names()
    full_pipeline = ColumnTransformer([
        ("numeric", num_pipeline, numeric_col),
        ("ordinal", ordinal_pipeline, ordinal_col),
        ("dummy", dummy_pipeline, dummy_col),
        ("onehot", onehot_pipeline, onehot_col)
    ])

    #######################################
    # Transform
    #######################################
    x_transformed = full_pipeline.fit_transform(predictors)

    # Convert into df_data frame
    numeric_col = np.array(pred_num.columns)
    dummy_col = np.array(["study_id", "site_id", "author_id"])
    onehot_col = np.array(onehot_encoder_names)
    feature_names = np.concatenate((numeric_col, ordinal_col, dummy_col, onehot_col), axis=0)
    predictors_prepared = pd.DataFrame(x_transformed, columns=feature_names, index=predictors.index)
    dataset_prepared = predictors_prepared.copy()
    dataset_prepared['log_visit_rate'] = labels
    dataset_prepared['log_vr_small'] = labels_small
    dataset_prepared['log_vr_large'] = labels_large

    # Reset indices
    df_data.reset_index(inplace=True, drop=True)
    dataset_prepared.reset_index(inplace=True, drop=True)

    #############################################################
    # Stratified split training and test (split by study_id)
    #############################################################
    # Option 1: Use the split used in the GA's by Alfonso
    ga_dir = r'C:\Users\Angel\git\Observ_models\data\GA calibration\Results Alfonso'
    train_ga = pd.read_csv(os.path.join(ga_dir, 'train_Small_2021-08-20.csv'))
    study_train = [study in train_ga.study_id.values for study in dataset_prepared.study_id]
    study_test = np.invert(study_train)
    df_train = dataset_prepared.copy().loc[study_train].reset_index(drop=True)
    df_test = dataset_prepared.copy().loc[study_test].reset_index(drop=True)

    # # Option 2: Redo split
    # df_authors = df_data.groupby('author_id', as_index=False).first()[['author_id', 'biome_num']]
    # # For the training set, take biomes with more than one count (otherwise I get an error in train_test_split).
    # # They are added in the test set later, to keep all df_data
    # has_more_one = df_authors.groupby('biome_num').count().author_id > 1
    # df_authors_split = df_authors.loc[has_more_one[df_authors.biome_num].reset_index().author_id, ]
    # strata = df_authors_split.biome_num.astype('category')
    #
    # x_train, x_test, y_train, y_test = train_test_split(df_authors_split, strata, stratify=strata, test_size=0.3, random_state=4)
    # study_train = [(x_train.author_id == x).any() for x in df_data.author_id]
    # df_train = dataset_prepared[study_train].reset_index(drop=True)
    # df_test = dataset_prepared[[~x for x in study_train]].reset_index(drop=True)

    # Get custom cross validation iterator
    df_studies = df_data[study_train].reset_index(drop=True).groupby('study_id', as_index=False).first()[['study_id', 'biome_num']]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=135)
    target = df_studies.loc[:, 'biome_num'].astype(int)
    df_studies['fold'] = -1
    n_fold = 0
    for train_index, test_index in skf.split(df_studies, target):
        df_studies.loc[test_index, 'fold'] = n_fold
        n_fold = n_fold + 1
    df_studies.drop(columns=['biome_num'], inplace=True)
    dict_folds = df_studies.set_index('study_id').T.to_dict('records')[0]
    df_train_iterator = df_train.replace(to_replace=dict_folds)
    myCViterator = []
    for i in range(0, 5):
        trainIndices = df_train_iterator[df_train_iterator['study_id'] != i].index.values.astype(int)
        testIndices = df_train_iterator[df_train_iterator['study_id'] == i].index.values.astype(int)
        myCViterator.append((trainIndices, testIndices))

    return df_train, df_test, myCViterator

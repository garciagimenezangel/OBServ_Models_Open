import pandas as pd
import numpy as np
from pandas_profiling import ProfileReport
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
from utils import pollinators_dependency as poll_dep
from utils import crop_families as crop_fam
import pickle
import warnings
warnings.filterwarnings('ignore')


def compute_visit_rate(df_data):
    df_data['visit_rate_wb_bmb'] = (df_data['ab_wildbees'] + df_data['ab_bombus']) / df_data['total_sampled_time']
    df_data['log_visit_rate'] = np.log(df_data['visit_rate_wb_bmb'])
    df_data.drop(columns=['ab_wildbees', 'ab_bombus', 'total_sampled_time', 'visit_rate_wb_bmb'], inplace=True)
    return df_data


def compute_visit_rate_small(df_data):
    df_data['visit_rate_wb'] = (df_data['ab_wildbees'] + 1) / df_data['total_sampled_time']
    df_data['log_vr_small'] = np.log(df_data['visit_rate_wb'])
    df_data.drop(columns=['ab_wildbees', 'visit_rate_wb_syr'], inplace=True)
    return df_data


def compute_visit_rate_large(df_data):
    # Compute comparable abundance
    df_data['visit_rate_bmb'] = (df_data['ab_bombus'] + 1) / df_data['total_sampled_time']
    df_data['log_vr_large']   = np.log(df_data['visit_rate_bmb'])
    df_data.drop(columns=['ab_bombus', 'visit_rate_bmb'], inplace=True)
    return df_data


def fill_missing_abundances(df_data):
    df_data.loc[df_data['ab_bombus'].isna(), 'ab_bombus']     = 0
    df_data.loc[df_data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    return df_data


def fill_biome(x, df_data):
    data_study_id = df_data.loc[df_data.study_id == x,]
    return data_study_id.biome_num.mode().iloc[0]


def fill_missing_biomes(df_data):
    missing_biome = df_data.loc[df_data.biome_num == 'unknown',]
    new_biome     = [fill_biome(x, df_data) for x in missing_biome.study_id]
    df_data.loc[df_data.biome_num == 'unknown', 'biome_num'] = new_biome
    return df_data


def remap_crops(df_data, option):
    if option == 'dependency':
        df_data['crop'] = df_data['crop'].map(poll_dep.dep)
    elif option == 'family':
        df_data['crop'] = df_data['crop'].map(crop_fam.family)
    return df_data


def check_normality(df_data, column):
    sns.distplot(df_data[column])
    # skewness and kurtosis
    print("Skewness: %f" % df_data[column].skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % df_data[column].kurt()) # Kurtosis: -0.168611
    # Check normality log_visit_rate
    sns.distplot(df_data[column], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_data[column], plot=plt)


def boxplot(df_data, x, y, ymin=-5, ymax=2):
    fig = sns.boxplot(x=x, y=y, df_data=df_data)
    fig.axis(ymin=ymin, ymax=ymax)


def add_mechanistic_values(df_data, model_name='Lonsdorf.Delphi_lcCont1_open1_forEd0_crEd0_div0_ins0max_dist0_suitmult'):
    data_dir = "C:/Users/Angel/git/Observ_models/data/"
    model_data  = pd.read_csv(data_dir + 'model_data_lite.csv')[['site_id','study_id',model_name]]
    return df_data.merge(model_data, on=['study_id', 'site_id'])


def add_landcover_diversity(data):
    data_dir = "C:/Users/Angel/git/Observ_models/data/"
    model_data  = pd.read_csv(data_dir + 'model_data.csv')[['site_id','study_id','geodata.landCoverDiversity']]
    model_data.rename(columns={'geodata.landCoverDiversity':'lc_div'}, inplace=True)
    return data.merge(model_data, on=['study_id', 'site_id'])


if __name__ == '__main__':

    #######################################
    # Get
    #######################################
    df_features = get_feature_data()
    df_field    = get_field_data()
    data = df_features.merge(df_field, on=['study_id', 'site_id'])
    data = apply_conditions(data)
    data = fill_missing_biomes(data)
    # df_data = remap_crops(df_data, option="family") # remap crops to poll. dependency or family?
    data = fill_missing_abundances(data)
    data = compute_visit_rate(data)
    data = compute_visit_rate_alternative(data)
    data['author_id'] = [study.split("_",2)[0] + study.split("_",2)[1] for study in data.study_id]
    data = add_landcover_diversity(data)
    # df_data = add_mechanistic_values(df_data) # add values from a Lonsforf model to include as an additional predictor?

    # # Save df_data guilds for validation of GA-calibrated tables
    # df_data = compute_visit_rate_small(df_data)
    # df_data.rename({'log_vr_small': 'log_visit_rate'}, axis=1, inplace=True)
    # df_data = compute_visit_rate_large(df_data)
    # df_data.rename({'log_vr_large': 'log_visit_rate'}, axis=1, inplace=True)
    # df_data.to_csv(path_or_buf='C:/Users/Angel/git/Observ_models/data/GA calibration/Validation/data_raw.csv', index=False)

    # Separate predictors and labels
    predictors = data.drop(columns=["log_visit_rate", "log_visit_rate_wb_bmb"], axis=1)
    labels     = data['log_visit_rate'].copy()
    labels_wb_bmb = data['log_visit_rate_wb_bmb'].copy()

    # (Set biome as categorical)
    predictors['biome_num'] = predictors.biome_num.astype('object')

    #######################################
    # Pipeline
    #######################################
    # Apply transformations (fill values, standardize, one-hot encoding)
    # First, replace numeric by mean, grouped by study_id (if all sites have NAs, then replace by dataset mean later in the imputer)
    pred_num     = predictors.select_dtypes('number')
    n_nas        = pred_num.isna().sum().sum()
    pred_num['study_id'] = data.study_id
    pred_num = pred_num.groupby('study_id').transform(lambda x: x.fillna(x.mean()))
    print("NA'S before transformation: " + str(n_nas))
    print("Total numeric values: " + str(pred_num.size))
    print("Percentage: " + str(n_nas*100/pred_num.size))

    # Define pipleline
    numeric_col = list(pred_num)
    onehot_col  = ["biome_num", "crop"]
    ordinal_col = ["management"]
    dummy_col   = ["study_id","site_id","author_id"] # keep this to use later (e.g. create custom cross validation iterator)
    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler())
    ])
    ordinal_pipeline = Pipeline([
         ('manag_imputer', SimpleImputer(strategy="constant", fill_value="conventional")),
         ('ordinal_encoder', OrdinalEncoder(categories=[['conventional','IPM','unmanaged','organic']]))
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
        ("onehot",  onehot_pipeline, onehot_col )
    ])

    #######################################
    # Transform
    #######################################
    x_transformed = full_pipeline.fit_transform(predictors)

    # Convert into df_data frame
    numeric_col = np.array(pred_num.columns)
    dummy_col = np.array(["study_id","site_id","author_id"])
    onehot_col  = np.array(onehot_encoder_names)
    feature_names = np.concatenate( (numeric_col, ordinal_col, dummy_col, onehot_col), axis=0)
    predictors_prepared = pd.DataFrame(x_transformed, columns=feature_names, index=predictors.index)
    dataset_prepared = predictors_prepared.copy()
    dataset_prepared['log_visit_rate'] = labels
    dataset_prepared['log_visit_rate'] = labels

    # Reset indices
    data.reset_index(inplace=True, drop=True)
    dataset_prepared.reset_index(inplace=True, drop=True)

    #############################################################
    # Stratified split training and test (split by study_id)
    #############################################################
    df_authors = data.groupby('author_id', as_index=False).first()[['author_id','biome_num']]
    # For the training set, take biomes with more than one count (otherwise I get an error in train_test_split.
    # They are added in the test set later, to keep all df_data
    has_more_one     = df_authors.groupby('biome_num').count().author_id > 1
    df_authors_split = df_authors.loc[has_more_one[df_authors.biome_num].reset_index().author_id,]
    strata           = df_authors_split.biome_num.astype('category')

    x_train, x_test, y_train, y_test = train_test_split(df_authors_split, strata, stratify=strata, test_size=0.3, random_state=4)
    authors_train   = x_train.author_id
    train_selection = [ (x_train.author_id == x).any() for x in data.author_id ]
    df_train = dataset_prepared[train_selection].reset_index(drop=True)
    df_test  = dataset_prepared[[~x for x in train_selection]].reset_index(drop=True)

    # Save predictors and labels (train and set), removing study_id
    df_train.drop(columns=['study_id', 'site_id', 'author_id']).to_csv(path_or_buf='C:/Users/Angel/git/Observ_models/data/ML/Regression/train/data_prepared.csv', index=False)
    df_test.drop(columns=['study_id', 'site_id', 'author_id']).to_csv(path_or_buf='C:/Users/Angel/git/Observ_models/data/ML/Regression/test/data_prepared.csv', index=False)

    # Save df_data (not processed by pipeline) including study_id and site_id
    train_withIDs = data[train_selection].copy().reset_index(drop=True)
    test_withIDs  = data[[~x for x in train_selection]].copy().reset_index(drop=True)
    train_withIDs.to_csv(path_or_buf='C:/Users/Angel/git/Observ_models/data/ML/Regression/train/data_prepared_withIDs.csv', index=False)
    test_withIDs.to_csv(path_or_buf='C:/Users/Angel/git/Observ_models/data/ML/Regression/test/data_prepared_withIDs.csv', index=False)

    # Save custom cross validation iterator
    df_studies = data[train_selection].reset_index(drop=True).groupby('study_id', as_index=False).first()[['study_id', 'biome_num']]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=135)
    target = df_studies.loc[:, 'biome_num'].astype(int)
    df_studies['fold'] = -1
    n_fold = 0
    for train_index, test_index in skf.split(df_studies, target):
        df_studies.loc[test_index,'fold'] = n_fold
        n_fold = n_fold+1
    df_studies.drop(columns=['biome_num'], inplace=True)
    dict_folds = df_studies.set_index('study_id').T.to_dict('records')[0]
    df_train_iterator = df_train.replace(to_replace=dict_folds)
    myCViterator = []
    for i in range(0,5):
        trainIndices = df_train_iterator[df_train['study_id'] != i].index.values.astype(int)
        testIndices = df_train_iterator[df_train['study_id'] == i].index.values.astype(int)
        myCViterator.append((trainIndices, testIndices))
    with open('C:/Users/Angel/git/Observ_models/data/ML/Regression/train/myCViterator.pkl', 'wb') as f:
        pickle.dump(myCViterator, f)

    #######################################
    # Explore
    #######################################
    profile = ProfileReport(dataset_prepared, title="Pandas Profiling Report")
    profile.to_file("C:/Users/Angel/git/Observ_models/data/ML/Regression/profiler.html")

    check_normality(data, 'log_visit_rate')
    data['year'] = data['study_id'].str[-4:]
    a = data.loc[data.year != "16_2",]
    boxplot(a, 'year', 'log_visit_rate')
    boxplot(data, 'biome_num', 'log_visit_rate')
    # Check normality other variables
    sns.distplot(data['elevation'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data['elevation'], plot=plt)

    # count NA's
    n_na = data.isnull().sum().sort_values(ascending = False).sum()

    # guilds
    small = data.copy()
    small = small.loc[ np.isfinite(small['log_vr_small']), ]
    check_normality(small, 'log_vr_small')
    large = data.copy().sort_values(by=["log_vr_large"])
    large = large.loc[ np.isfinite(large['log_vr_large']), ]
    check_normality(large, 'log_vr_large')

    #######################################
    # Save dataset for the inputs in k.LAB
    #######################################
    df_features = get_feature_data()
    df_field    = get_field_data(coords=True)
    data = df_features.merge(df_field, on=['study_id', 'site_id'])
    data = apply_conditions(data)
    data = fill_missing_biomes(data)
    # df_data = remap_crops(df_data, option="family") # remap crops to poll. dependency or family?
    data = fill_missing_abundances(data)
    data = compute_visit_rate(data)
    data['author_id'] = [study.split("_",2)[0] + study.split("_",2)[1] for study in data.study_id]
    data = add_landcover_diversity(data)
    # Save df_data (or a selection of df_data)
    # # Dummy columns (for testing)
    # Discretize log_visit_rate
    def discretize(x):
        if (x) > -2.5: return (1)
        else: return (0)
    data_sel = data.copy()
    data_sel['visit'] = [ discretize(vr) for vr in data_sel.log_visit_rate]
    # One-hot encoding of biomes
    biomes_onehot = pd.get_dummies(data_sel.biome_num)
    data_sel = pd.concat([data_sel, biomes_onehot], axis=1)
    # Select columns (moss ruled out, maximum value is 0.002% of land cover)
    selected_columns = ['latitude', 'longitude', 'log_visit_rate',
                        'elevation','bio01','bio02','bio05','bio08','bio14',
                        'soil_carbon_b10','soil_den_b10','urban','bare','shrub',
                        'pdsi', 1.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]
    data_sel = data_sel[selected_columns]
    # Set units in international system (correct scale from GEE)
    scale_bio     = 0.1  # units ºC or mm
    scale_soilC   = 5    # units g/kg
    scale_soilden = 10   # kg/m3
    scale_pdsi    = 0.01 # unitless (Palmer Drought Severity Index)
    data_sel['bio01'] = data_sel.bio01 * scale_bio
    data_sel['bio02'] = data_sel.bio02 * scale_bio
    data_sel['bio05'] = data_sel.bio05 * scale_bio
    data_sel['bio08'] = data_sel.bio08 * scale_bio
    data_sel['bio14'] = data_sel.bio14 * scale_bio
    data_sel['soil_carbon_b10'] = data_sel.soil_carbon_b10 * scale_soilC
    data_sel['soil_den_b10']    = data_sel.soil_den_b10    * scale_soilden
    data_sel['pdsi']            = data_sel.pdsi            * scale_pdsi
    # Rename columns
    data_sel.rename(columns=dict({'elevation':'elev',
                            'bio01':'annmeant', 'bio02':'diurange', 'bio05':'maxtwarm', 'bio08':'twetquart', 'bio14':'precdrmo',
                            'soil_carbon_b10':'soilc', 'soil_den_b10':'soilden',
                            1.0:'biome1', 2.0:'biome2', 4.0:'biome4', 5.0:'biome5', 6.0:'biome6', 7.0:'biome7',
                            8.0:'biome8', 10.0:'biome10', 12.0:'biome12'}), inplace=True)
    # # replace 1's by T's (true) and 0's by F's (false)
    # data_sel['biome1']  = data_sel.biome1.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome2']  = data_sel.biome2.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome4']  = data_sel.biome4.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome5']  = data_sel.biome5.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome6']  = data_sel.biome6.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome7']  = data_sel.biome7.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome8']  = data_sel.biome8.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome10'] = data_sel.biome10.replace({1: "T"}).replace({0: "F"})
    # data_sel['biome12'] = data_sel.biome12.replace({1: "T"}).replace({0: "F"})
    # # Select geographic region
    # data_sel = data_sel.loc[ (data_sel.latitude < 53.5) & (data_sel.latitude > 50.5) & (data_sel.longitude > 3.35) & (data_sel.longitude < 6.9)]
    data_sel.to_csv("C:/Users/Angel/git/Observ_models/data/ML/kLAB/features.csv", index=False)



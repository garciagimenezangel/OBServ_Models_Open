import numpy as np
import logger


def apply_basic_filter_field_data(df_field):
    total = len(df_field)
    logger.logging.info("Preparing field data with basic filters. Total number of field records: {}".upper().format(total))
    # 1. Latitude and longitude must be !na
    cond1 = (~df_field['latitude'].isna()) & (~df_field['longitude'].isna())
    n_cond1 = len(np.where(cond1)[0])
    logger.logging.info("1. Coordinates defined for {} records ({})".format(n_cond1, "{:.0%}".format(n_cond1/total)))
    # 2. At least one guild measured with abundance > 0
    ab_sum = df_field['ab_wildbees'] + df_field['ab_bombus']
    cond2 = ab_sum > 0
    n_cond2 = len(np.where(cond2)[0])
    logger.logging.info("2. At least one guild measured with abundance > 0 for {} records ({})".format(n_cond2, "{:.0%}".format(n_cond2/total)))
    # 3. Set temporal threshold (sampling year >= 1992). This removes years 1990, 1991, that show not-very-healthy values of "comparable abundance"
    refYear = df_field['sampling_year'].str[:4].astype('int')
    cond3 = refYear > 1991
    n_cond3 = len(np.where(cond3)[0])
    logger.logging.info("3. Year > 1991 for {} records ({})".format(n_cond3, "{:.0%}".format(n_cond3/total)))
    # 4. Sampling method != pan trap
    cond4 = [is_sampling_method_accepted(x) for x in df_field['sampling_abundance'].astype('str')]
    n_cond4 = len(np.where(cond4)[0])
    logger.logging.info("4. Sampling method != pan trap for {} records ({})".format(n_cond4, "{:.0%}".format(n_cond4/total)))
    # 5. Abundances of all sites in the study must be integer numbers (tolerance of 0.05)
    abs_integer = df_field.groupby('study_id').apply(are_abundances_integer)
    sel_studies = abs_integer.index[abs_integer]
    cond5       = df_field['study_id'].isin(sel_studies)
    n_cond5 = len(np.where(cond5)[0])
    logger.logging.info("5. Studies with all abundances integer for {} records ({})".format(n_cond5, "{:.0%}".format(n_cond5/total)))
    # 6. Coordinates must vary across sites of the same study
    a = df_field.groupby('study_id')['latitude'].mean()
    b = df_field.groupby('study_id')['latitude'].min()
    c = a - b  # if equals 0, mean and min are the same -> rule out study
    study_counts = df_field.study_id.value_counts()
    studies_valid = np.array(study_counts[c != 0].index)
    cond6 = df_field.apply(lambda x: x['study_id'] in studies_valid, axis=1)
    n_cond6 = len(np.where(cond6)[0])
    logger.logging.info("6. Coordinates vary within the study for {} records ({})".format(n_cond6, "{:.0%}".format(n_cond6/total)))
    # Merge filters
    df_filtered = df_field.copy().loc[cond1 & cond2 & cond3 & cond4 & cond5 & cond6]
    n_filtered = len(df_filtered)
    logger.logging.info("Filtered field data length: {} ({})".format(n_filtered, "{:.0%}".format(n_filtered / total)))
    return df_filtered


def apply_filter_study(df_data):
    total = len(df_data)
    logger.logging.info("Preparing df_data for study-wise analyses. Total number of field records: {}".upper().format(total))
    # 1. Studies must have at least 3 sites
    min_num_sites = 3
    study_counts = df_data.study_id.value_counts()
    studies_valid = np.array(study_counts[study_counts >= min_num_sites].index)
    cond1 = df_data.apply(lambda x: x['study_id'] in studies_valid, axis=1)
    n_cond1 = len(np.where(cond1)[0])
    logger.logging.info("1. Studies with at least 3 sites fulfilled for {} records ({})".format(n_cond1, "{:.0%}".format(n_cond1/total)))
    logger.logging.info("   {} studies out of {} ({})".format(len(studies_valid), len(study_counts), "{:.0%}".format(len(studies_valid)/len(study_counts))))
    df_filtered = df_data.copy().loc[cond1]
    n_filtered = len(df_filtered)
    logger.logging.info("Filtered field data length: {} ({})".format(n_filtered, "{:.0%}".format(n_filtered / total)))
    return df_filtered


def apply_filter_multistudy_lonsdorf(df_data):
    total = len(df_data)
    logger.logging.info("Preparing df_data for multistudy analyses using the Lonsdorf model. Total number of field records: {}".upper().format(total))
    # 1. Total sampled time != NA
    cond1 = ~df_data['total_sampled_time'].isna()
    n_cond1 = len(np.where(cond1)[0])
    logger.logging.info("1. Total sampled time != NA for {} records ({})".format(n_cond1, "{:.0%}".format(n_cond1/total)))
    df_filtered = df_data.copy().loc[cond1]
    n_filtered = len(df_filtered)
    logger.logging.info("Filtered field data length: {} ({})".format(n_filtered, "{:.0%}".format(n_filtered / total)))
    return df_filtered


def apply_filter_multistudy_ml(df_data):
    total = len(df_data)
    logger.logging.info("Preparing df_data for multistudy analyses using the ML model. Total number of field records: {}".upper().format(total))
    # 1. Total sampled time != NA
    cond1 = ~df_data['total_sampled_time'].isna()
    n_cond1 = len(np.where(cond1)[0])
    logger.logging.info("1. Total sampled time != NA for {} records ({})".format(n_cond1, "{:.0%}".format(n_cond1/total)))
    # 2. Remove rows with 7 or more NaN values
    cond2 = (df_data.isnull().sum(axis=1) < 7)
    n_cond2 = len(np.where(cond2)[0])
    logger.logging.info("2. Less than 7 NAs for {} records ({})".format(n_cond2, "{:.0%}".format(n_cond2/total)))
    # Merge filters
    df_filtered = df_data.copy().loc[cond1 & cond2]
    n_filtered = len(df_filtered)
    logger.logging.info("Filtered field data length: {} ({})".format(n_filtered, "{:.0%}".format(n_filtered / total)))
    return df_filtered


def is_sampling_method_accepted(x):
    cond1 = 'pan trap' not in x
    cond2 = x != "nan"
    return cond1 & cond2


def are_abundances_integer(study_data):  # do not exclude NAs (filtered or transformed in other steps)
    tol = 0.05
    cond_wb  = ((study_data['ab_wildbees'] % 1) < tol) | ((study_data['ab_wildbees'] % 1) > (1-tol)) | study_data['ab_wildbees'].isna()
    cond_bmb = ((study_data['ab_bombus'] % 1) < tol)   | ((study_data['ab_bombus'] % 1) > (1-tol))   | study_data['ab_bombus'].isna()
    cond = cond_wb & cond_bmb
    return all(cond)


def apply_basic_conditions(df_data, thresh_ab=0):
    # 0. Latitude and longitude must be !na. Implicit because df_features only has df_data with defined lat and lon.
    cond0 = (~df_data['latitude'].isna()) & (~df_data['longitude'].isna())
    print("Defined coordinates:")
    print(cond0.describe())

    # 1. log(abundance) != NA
    cond1 = ~df_data.log_ab.isna()
    print("Abundance not NA:")
    print(cond1.describe())

    # 2. Coordinates must vary across sites of the same study
    a = df_data.groupby('study_id')['latitude'].mean()
    b = df_data.groupby('study_id')['latitude'].min()
    c = a - b  # if equals 0, mean and min are the same -> rule out study
    study_counts = df_data.study_id.value_counts()
    studies_valid = np.array(study_counts[c != 0].index)
    cond2 = df_data.apply(lambda x: x['study_id'] in studies_valid, axis=1)
    print("Coordinates vary:")
    print(cond2.describe())
    print("Studies:" + str(len(studies_valid)) + " Total:" + str(len(df_data.study_id.unique())))
    df_data = df_data[cond0 & cond1 & cond2]

    # 4. Studies must have at least 5 sites
    min_num_sites = 5
    study_counts = df_data.study_id.value_counts()
    studies_valid = np.array(study_counts[study_counts >= min_num_sites].index)
    cond4 = df_data.apply(lambda x: x['study_id'] in studies_valid, axis=1)
    print("At least 5 sites:")
    print(cond4.describe())
    print("Studies:" + str(len(studies_valid)) + " Total:" + str(len(df_data.study_id.unique())))
    df_data = df_data[cond4]

    return df_data.reset_index(drop=True)

import os.path
import pickle
import logger
import settings as sett
import data_loader as dloa
import data_filter as dfil
import ml_pipeline as ml_pi

df_features = dloa.get_feature_data()
df_field = dloa.get_field_data()

# Get field data with basic filters (filters applied in every analysis)
df_field_basic_filter = dfil.apply_basic_filter_field_data(df_field)

# Prepare df_data for the analyses using the Lonsdorf model
df_lons_multi = dfil.apply_filter_multistudy_lonsdorf(df_field_basic_filter)  # When more than one study is involved (analyses at biome and global scales)
df_lons_multi = ml_pi.compute_visit_rate(df_lons_multi)
df_lons_multi = ml_pi.compute_visit_rate_small(df_lons_multi)
df_lons_multi = ml_pi.compute_visit_rate_large(df_lons_multi)
df_lons_single = dfil.apply_filter_study(df_field_basic_filter)  # When single studies are involved

# Prepare df_data for the analyses using ML
df_features = dloa.get_feature_data()
df_ml_basic = df_features.merge(df_field_basic_filter, on=['study_id', 'site_id'])
df_ml_multi = dfil.apply_filter_multistudy_ml(df_ml_basic)
df_ml_train, df_ml_test, myCViterator = ml_pi.run_pipeline(df_ml_multi)

logger.logging.info("Summary:")
logger.logging.info("# Records data Lonsdorf analyses at a local scale: {}".format(len(df_lons_single)))
logger.logging.info("# Records data Lonsdorf analyses at biome and global scales: {}".format(len(df_lons_multi)))
logger.logging.info("# Records data ML and DI-MM analyses: {} (Training set), {} (Test set), {} (Total)".format(len(df_ml_train), len(df_ml_test), len(df_ml_multi)))

# Export prepared data
logger.logging.info("Exporting prepared df_data into folder {}".upper().format(sett.dir_output))
df_lons_single.to_csv(os.path.join(sett.dir_output, 'lons_studywise.csv'), index=False)
df_lons_multi.to_csv(os.path.join(sett.dir_output, 'lons_global.csv'), index=False)
df_ml_multi.to_csv(os.path.join(sett.dir_output, 'ml_global.csv'), index=False)
df_ml_train.to_csv(os.path.join(sett.dir_output, 'ml_train.csv'), index=False)
df_ml_test.to_csv(os.path.join(sett.dir_output, 'ml_test.csv'), index=False)
with open(os.path.join(sett.dir_output, 'myCViterator.pkl'), 'wb') as f:
    pickle.dump(myCViterator, f)

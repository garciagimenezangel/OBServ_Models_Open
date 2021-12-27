# Associate name of variables with a very brief description

import pandas as pd
from utils import define_root_folder
root_folder = define_root_folder.root_folder

predictors_meta = pd.read_csv(root_folder+'/data/tables/metadata_variables.csv')
predictors_meta = predictors_meta.drop(columns=['Group','Dataset','Reference','Description'])
predictors_meta.dropna(inplace=True)
feat_dict = dict(zip(predictors_meta.Variable, predictors_meta.Short))

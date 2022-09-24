import pandas as pd

predictors_meta = pd.read_csv('C:/Users/Angel/git/OBServ/Observ_models//report/tables/metadata_variables.csv')
predictors_meta = predictors_meta.drop(columns=['Group','Dataset','Reference','Description'])
predictors_meta.dropna(inplace=True)
feat_dict = dict(zip(predictors_meta.Variable, predictors_meta.Short))


import pickle
import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from utils import generate_trained_model as gtm
warnings.filterwarnings('ignore')
from scipy import stats

root_folder = "C:/Users/angel/git/OBServ_Models_Open/Machine Learning/"

def fillEmptyColumns(ref_columns, data):
    data_filled = data.copy()
    for col in ref_columns:
        if not (col in data.columns):
            data_filled[col] = 0
    return data_filled

n_features = 49
model = pickle.load(open(root_folder + 'data/trained_models/nusvr_full.pkl', 'rb')) # full: model trained using the entire database; training: model trained using only the training subset

# TEST 1: compute predictions from a dataset where all the predictors are present and standardized
# NOTE: if the 'test' dataset does not have a column used as a predictor in the model (for example, it might be the case
# of types of crop), we fill it with zeros
test_prepared = pd.read_csv(root_folder+'data/predict_here/data_reduced49.csv')
predictors_test = test_prepared.iloc[:, :-1]
labels_test = np.array(test_prepared.iloc[:, -1:]).flatten()
yhat = model.predict(predictors_test)
pd_result =  pd.DataFrame({})
pd_result['observations'] = labels_test
pd_result['predictions'] = yhat
pd_result.to_csv(root_folder+'data/predictions/test1.csv', index=False)

# TEST 1: compute predictions from a 'features' file computed using the GEE script available in this repository
# test_prepared = fillEmptyColumns(model.feature_names_in_, test_prepared)

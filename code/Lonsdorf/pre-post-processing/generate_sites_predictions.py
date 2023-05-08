"""
Generate a csv with the coordinates of the points where we want to compute Lonsdorf scores, or the
value of predictors for ML models (in GEE)
"""

import pandas as pd
import numpy as np

assets_folder = 'C:/Users/Angel/git/OBServ_Models_Open/Lonsdorf/GEE/assets/'

# Set area of interest, and resolution (degrees)
lat_min=40
lat_max=43
lon_min=-4
lon_max=0
res = 0.1
lat_nsize = int((lat_max-lat_min)/res)
lon_nsize = int((lon_max-lon_min)/res)
n_sites  = lat_nsize * lon_nsize

# The structure of df_sites is determine by the scripts of GEE, which is originally thought to work with data from CropPol.
# We keep the same structure and fill the values of the columns with dummy content when needed
df_sites = pd.DataFrame({'latitude':[],'longitude':[],'study_id':[],'site_id':[],'startDate':[],'endDate':[],'refYear':[]})
for i in range(0,lat_nsize):
    for j in range(0,lon_nsize):
        lat = lat_min + 0.5*res + i*res
        lon = lon_min + 0.5*res + j*res
        df_point = pd.DataFrame({'latitude':[lat],
                                 'longitude':[lon],
                                 'study_id':['dummy'],
                                 'site_id':['dummy'],
                                 'startDate':['1900-01-01'],
                                 'endDate':['2100-01-01'],
                                 'refYear':['2020']})
        df_sites = pd.concat([df_sites, df_point])

df_sites.to_csv(assets_folder+'sites_for_prediction.csv', index=False)

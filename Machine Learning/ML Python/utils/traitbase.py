import numpy as np
import pandas as pd
from scipy import stats

traitbase = pd.read_csv('C:/Users/angel/DATA/Traitbase/ALL.csv')
df_bombus_rows   = traitbase[traitbase.species.str.contains('Bombus')]
df_wildbees_rows = traitbase[(~traitbase.species.str.contains('Bombus'))]
it_wildbees = np.nanmean(df_wildbees_rows['m_it'])
it_bombus   = np.nanmean(df_bombus_rows['m_it'])
typ_dist_wildbees = np.power(10, -1.643 + 3.242*np.log10(it_wildbees)) * 1000
typ_dist_bombus   = np.power(10, -1.643 + 3.242*np.log10(it_bombus)) * 1000
max_dist_wildbees = np.power(10, -1.363 + 3.366*np.log10(it_wildbees)) * 1000
max_dist_bombus   = np.power(10, -1.363 + 3.366*np.log10(it_bombus)) * 1000

import pandas as pd

summary_MM = pd.read_csv("C:/Users/Angel/git/Observ_models/data/Lonsdorf evaluation/Study metrics/summary.csv")
summary_ML = pd.read_csv("C:/Users/Angel/git/Observ_models/data/ML/Regression/tables/Study metrics/summary.csv")

summary_ML['Type'] = 'ML'
summary_MM['Type'] = 'MM'

df_summary = pd.concat([summary_MM, summary_ML], axis=0)
df_summary = df_summary[['Type','model','mean Spearman coef', 'std Spearman coef', 'significant']]

print(df_summary.to_latex(index=False, float_format='%.2f'))


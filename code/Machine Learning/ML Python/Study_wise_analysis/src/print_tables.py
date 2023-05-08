from functools import reduce
import pandas as pd
import metrics


def run(df_stats):
    df_summary = df_stats.groupby(['model', 'guild']).apply(metrics.get_proportion_of_significant_fits, option='Spearman').reset_index()
    df_summary[0] = ["{:.02%}".format(x) for x in df_summary[0]]
    df_summary.columns = ['Model', 'Guild', 'Studies with p<0.05 Srho>0']
    df_summary.sort_values('Guild', inplace=True)
    print(df_summary.to_latex(index=False, float_format='%.2f'))
    return

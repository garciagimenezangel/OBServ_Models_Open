import pandas as pd
import metrics
import settings as sett


def run(df_stats):
    # Format numeric columns to indicate statistical significance
    df_stats['Spearman_coef'] = [metrics.format_with_statistical_significance(x[0],x[1]) for x in zip(df_stats.Spearman_coef, df_stats.Spearman_p)]
    df_stats['r2'] = [metrics.format_with_statistical_significance(x[0], x[1]) for x in zip(df_stats.r2, df_stats.p_val)]
    df_stats['guild'] = pd.Categorical(df_stats['guild'], ['Wildbees', 'Bumblebees', 'Bumblebees+Wildbees'])
    df_stats['model'] = pd.Categorical(df_stats['model'], ["MM ({})".format(sett.mm_model_name), 'DI-MM', "ML ({})".format(sett.ml_model_name)])
    list_biomes_decreasing_nr_of_sites = get_list_biomes_decreasing_nr_of_sites(df_stats)
    df_stats['Biome'] = pd.Categorical(df_stats['Biome'], list_biomes_decreasing_nr_of_sites)
    df_stats.sort_values(['Biome', 'guild', 'model'], inplace=True)
    cols_sel = ['Biome', 'guild', 'model', 'r2', 'Spearman_coef', 'MAE', 'RMSE', 'n_sites']
    print(df_stats[cols_sel].to_latex(index=False, float_format='%.2f', na_rep=""))
    print("*: \\textit{  p-value}<0.05  &&&  \\\\")
    print("**: \\textit{ p-value}<0.01  &&&  \\\\")
    print("***: \\textit{p-value}<0.001  &&&  \\\\")
    print("")
    print("NOTE: asterisk notes go in between bottomrule and end{tabular}")


def get_list_biomes_decreasing_nr_of_sites(df_biome):
    df_n_sites = df_biome.groupby('Biome').max().sort_values('n_sites', ascending=False)
    return list(df_n_sites.index)

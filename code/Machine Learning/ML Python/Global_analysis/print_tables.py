import metrics


def run(df_stats):
    # Format numeric columns to indicate statistical significance
    df_stats['Spearman_coef'] = [metrics.format_with_statistical_significance(x[0],x[1]) for x in zip(df_stats.Spearman_coef, df_stats.Spearman_p)]
    df_stats['r2'] = [metrics.format_with_statistical_significance(x[0], x[1]) for x in zip(df_stats.r2, df_stats.p_val)]
    cols_sel = ['guild', 'model', 'r2', 'Spearman_coef', 'MAE', 'RMSE']
    df_stats = df_stats.copy()[cols_sel]
    df_stats.sort_values('guild', ascending=False, inplace=True)
    print(df_stats.to_latex(index=False, float_format='%.2f', na_rep=""))
    print("*: \\textit{  p-value}<0.05  &&&  \\\\")
    print("**: \\textit{ p-value}<0.01  &&&  \\\\")
    print("***: \\textit{p-value}<0.001  &&&  \\\\")
    print("")
    print("NOTE: asterisk notes go in between bottomrule and end{tabular}")

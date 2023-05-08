import compute_metrics as cm
import print_tables as pt
import create_plots as cp

df_stats, df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml = cm.run()
pt.run(df_stats)
cp.run(df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml)
cp.supp_material(df_data_mm_small, df_data_mm_large)

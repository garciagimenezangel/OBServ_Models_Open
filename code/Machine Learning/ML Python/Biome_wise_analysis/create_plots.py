import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import settings as sett


def scatter_plot(df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml):
    df_data_mm_small = df_data_mm_small.copy().loc[df_data_mm_small.biome_num == 4]
    df_data_mm_large = df_data_mm_large.copy().loc[df_data_mm_large.biome_num == 4]
    df_data_di_small = df_data_di_small.copy().loc[df_data_di_small.biome_num == 4]
    df_data_di_large = df_data_di_large.copy().loc[df_data_di_large.biome_num == 4]
    df_data_ml = df_data_ml.copy().loc[df_data_ml.biome_num == 4]

    # Figure
    fig, ax = plt.subplots(5, 1)
    fig.set_size_inches(8, 10)
    fig.tight_layout(pad=5.0)
    fig.suptitle('Temperate Broadleaf and Mixed Forests', fontsize=14, fontweight='bold')

    # Panel A: MM wildbees
    ax[0].scatter(df_data_mm_small.model, df_data_mm_small.log_vr_small)
    limits_obs_x = np.array([np.min(df_data_mm_small['model'])-0.05, np.max(df_data_mm_small['model']) + 0.05])
    m, b = np.polyfit(df_data_mm_small.model, df_data_mm_small.log_vr_small, 1)
    ax[0].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[0].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[0].set_xlabel("MM Wildbees score".format(), fontsize=12)
    ax[0].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[0].set_title(label='(a)', loc='left', fontweight='bold', fontsize='large')

    # Panel B: MM bumblebees
    ax[1].scatter(df_data_mm_large.model, df_data_mm_large.log_vr_large)
    limits_obs_x = np.array([np.min(df_data_mm_large['model'])-0.05, np.max(df_data_mm_large['model']) + 0.05])
    m, b = np.polyfit(df_data_mm_large.model, df_data_mm_large.log_vr_large, 1)
    ax[1].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[1].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[1].set_xlabel("MM Bumblebees score".format(), fontsize=12)
    ax[1].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[1].set_title(label='(b)', loc='left', fontweight='bold', fontsize='large')

    # Panel C: DI-MM wildbees
    ax[2].scatter(df_data_di_small.model, df_data_di_small.log_vr_small)
    limits_obs_x = np.array([np.min(df_data_di_small['model'])-0.05, np.max(df_data_di_small['model']) + 0.05])
    m, b = np.polyfit(df_data_di_small.model, df_data_di_small.log_vr_small, 1)
    ax[2].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[2].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[2].set_xlabel("DI-MM Wildbees score".format(), fontsize=12)
    ax[2].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[2].set_title(label='(c)', loc='left', fontweight='bold', fontsize='large')

    # Panel D: DI-MM bumblebees
    ax[3].scatter(df_data_di_large.model, df_data_di_large.log_vr_large)
    limits_obs_x = np.array([np.min(df_data_di_large['model'])-0.05, np.max(df_data_di_large['model']) + 0.05])
    m, b = np.polyfit(df_data_di_large.model, df_data_di_large.log_vr_large, 1)
    ax[3].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[3].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[3].set_xlabel("DI-MM Bumblebees score".format(), fontsize=12)
    ax[3].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[3].set_title(label='(d)', loc='left', fontweight='bold', fontsize='large')

    # Panel E: ML
    ax[4].scatter(df_data_ml.model, df_data_ml.log_visit_rate)
    limits_obs_x = np.array([np.min(df_data_ml['model'])-0.05, np.max(df_data_ml['model']) + 0.05])
    m, b = np.polyfit(df_data_ml.model, df_data_ml.log_visit_rate, 1)
    ax[4].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[4].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[4].set_xlabel("ML-BayR prediction", fontsize=12)
    ax[4].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[4].set_title(label='(e)', loc='left', fontweight='bold', fontsize='large')

    fig.savefig(os.path.join(sett.dir_fig, sett.eps_fig_scatter))


#################################
# SEE LANDSCAPE VARIABILITY?
#################################

def landscape_var_plot(df_stats):

    # Select two models (from table printed in print_tables.py, the best performing models of each type are 'Open forest' and 'NuSVR'
    df_selection = df_stats.loc[[(sett.mm_model_name in x) | (sett.ml_model_name in x) for x in df_stats.model]].copy()
    df_selection = df_selection.dropna()

    # Remove asterisks from column Spearman_coef
    df_selection['Spearman_coef'] = [float(x.replace("*","")) for x in df_selection['Spearman_coef']]

    # Set bins landscape variance
    max_var100 = round(df_selection.landsc_var.max() * 100)
    bin1 = round(max_var100 / 2)

    # Landscape variance plot
    fig = plt.figure()
    ax = plt.subplot(111)
    df_selection['landscape_variance'] = pd.cut(df_selection.landsc_var, bins=[0, bin1 / 100, max_var100 / 100])
    df_selection['model_guild'] = [" - ".join([m, g]) for g, m in zip(df_selection['guild'], df_selection['model'])]
    sns.boxplot(x="landscape_variance", y="Spearman_coef", palette="colorblind", hue="model_guild", data=df_selection, ax=ax)
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # Put a legend to the right of the current axis
    ax.legend(bbox_to_anchor=(1.4, 1), loc=1, borderaxespad=0.)
    ax.axhline(0.2, c='r', linestyle='--')
    ax.set_xlabel("Landscape Standardized Variance", fontsize=14)
    ax.set_ylabel("Spearman", fontsize=14)
    ax.set_title(label='Biomes', fontweight='bold', fontsize='large')

    fig.savefig(os.path.join(sett.dir_fig, sett.eps_fig_landsc_var))

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import seaborn
import pandas as pd
import biome_dict
import os
import settings as sett


def run(df_data_mm_small, df_data_mm_large, df_data_di_small, df_data_di_large, df_data_ml):

    # Figure
    fig, ax = plt.subplots(3, 1)
    fig.set_size_inches(8, 10)
    fig.tight_layout(pad=5.0)
    fig.suptitle('Global analysis', fontsize=14, fontweight='bold')

    # Panel A: MM-small
    ax[0].scatter(df_data_mm_small.model, df_data_mm_small.log_vr_small)
    limits_obs_x = np.array([np.min(df_data_mm_small['model'])-0.01, np.max(df_data_mm_small['model']) + 0.01])
    df_data_mm_small = df_data_mm_small[['log_vr_small', 'model']].dropna()
    m, b = np.polyfit(df_data_mm_small.model, df_data_mm_small.log_vr_small, 1)
    ax[0].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[0].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[0].set_xlabel("MM Other wild bees score", fontsize=12)
    ax[0].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[0].set_title(label='(a)', loc='left', fontweight='bold', fontsize='large')

    # Panel B: DI-small
    ax[1].scatter(df_data_di_small.model, df_data_di_small.log_vr_small)
    limits_obs_x = np.array([np.min(df_data_di_small['model'])-0.01, np.max(df_data_di_small['model']) + 0.01])
    m, b = np.polyfit(df_data_di_small.model, df_data_di_small.log_vr_small, 1)
    ax[1].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[1].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[1].set_xlabel("DI-MM Other wildbees score", fontsize=12)
    ax[1].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[1].set_title(label='(c)', loc='left', fontweight='bold', fontsize='large')

    # Panel C: ML
    ax[2].scatter(df_data_ml.model, df_data_ml.log_visit_rate)
    limits_obs_x = np.array([np.min(df_data_ml['model'])-0.05, np.max(df_data_ml['model']) + 0.05])
    m, b = np.polyfit(df_data_ml.model, df_data_ml.log_visit_rate, 1)
    ax[2].plot(limits_obs_x, m * limits_obs_x + b, linestyle='dashed')
    ax[2].set_xlim(limits_obs_x[0], limits_obs_x[1])
    ax[2].set_xlabel("ML-BayRidge prediction", fontsize=12)
    ax[2].set_ylabel("log(Visit Rate)", fontsize=12)
    ax[2].set_title(label='(e)', loc='left', fontweight='bold', fontsize='large')

    fig.savefig(os.path.join(sett.dir_fig, sett.eps_fig))


def supp_material(df_data_mm_small, df_data_mm_large):
    df_biomes = pd.read_csv(r'C:\Users\Angel\git\Observ_models\data\GEE\GEE features\filled_biomes.csv')  # relation studies - biome number
    df_data_biomes_small = df_data_mm_small.merge(df_biomes)
    df_data_biomes_large = df_data_mm_large.merge(df_biomes)
    df_data_biomes_small['Biome'] = df_data_biomes_small.biome_num.replace(biome_dict.biome_names)
    df_data_biomes_large['Biome'] = df_data_biomes_large.biome_num.replace(biome_dict.biome_names)
    g1 = seaborn.relplot(data=df_data_biomes_large, x='model', y='log_vr_large', hue='Biome', palette="bright")
    g1._legend.set_title("Biome")
    g1.set_ylabels("log(Visit Rate)")
    g1.set_xlabels("MM Bumblebees score")
    g1.fig.suptitle("MM Bumblebees - Global analysis")
    g2 = seaborn.relplot(data=df_data_biomes_small, x='model', y='log_vr_small', hue='Biome', palette="bright")
    g2._legend.set_title("Biome")
    g2.set_ylabels("log(Visit Rate)")
    g2.set_xlabels("MM Other wild bees score")
    g2.fig.suptitle("MM Other wild bees - Global analysis")
    g1.fig.savefig(r'C:\Users\Angel\Downloads\bumblebees_global.eps')
    g2.fig.savefig(r'C:\Users\Angel\Downloads\wildbees_global.eps')
    # fig = px.scatter(df_data, x="model", y="log_vr_large", hover_data=df_data.columns)
    # fig.show()
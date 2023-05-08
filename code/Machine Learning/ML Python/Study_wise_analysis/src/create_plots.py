import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import settings as sett


def run(df_studywise, margin=1.25):
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(17, 10)
    fig.tight_layout(pad=5.0)

    # Select MM and ML models
    # df_studywise = df_studywise.loc[df_studywise.model != 'DI-MM'].copy()
    df_studywise = df_studywise.loc[(df_studywise.model == f'MM ({sett.mm_model_name})') | (df_studywise.model == f'ML ({sett.ml_model_name})')]

    # Set bins landscape variance
    values_landsc_var = df_studywise.loc[df_studywise['guild'] == 'Wildbees', 'landsc_var']  # use the values at the MM studies
    max_var100 = values_landsc_var.max()
    bin1 = values_landsc_var.quantile(0.33)
    bin2 = values_landsc_var.quantile(0.66)

    # Combine model+guild
    df_studywise['model'] = df_studywise['model'].replace({'MM ({})'.format(sett.mm_model_name): 'MM ({})'.format(sett.mm_model_shortname)})
    df_studywise['model_guild'] = [" - ".join([m, g]) for g, m in zip(df_studywise['guild'], df_studywise['model'])]

    # 1. Landscape variance plot
    df_studywise['landscape_variance'] = pd.cut(df_studywise.landsc_var, bins=[0, bin1, bin2, max_var100])
    sns.boxplot(x="landscape_variance", y="Spearman_coef", palette="colorblind", hue="model_guild", data=df_studywise, ax=axs[0])
    axs[0].axhline(0.2, c='r', linestyle='--')
    axs[0].set_xlabel("Landscape Standardized Variance", fontsize=14)
    axs[0].set_ylabel("Spearman", fontsize=14)
    axs[0].legend(bbox_to_anchor=(margin, 1), loc=1, borderaxespad=0.)
    axs[0].set_title(label='(a)', loc='left', fontweight='bold', fontsize='large')

    # 2. Management plot
    sns.boxplot(x="management", y="Spearman_coef", palette="colorblind", data=df_studywise, ax=axs[1], order=['conventional', 'IPM', 'organic', 'unmanaged'], hue="model_guild")
    axs[1].axhline(0.2, c='r', linestyle='--')
    axs[1].set_xlabel("Management", fontsize=14)
    axs[1].set_ylabel("Spearman", fontsize=14)
    axs[1].get_legend().remove()
    axs[1].set_title(label='(b)', loc='left', fontweight='bold', fontsize='large')

    # 3. Pollinator dependency plot
    sns.boxplot(x="poll_dep", y="Spearman_coef", data=df_studywise, ax=axs[2], palette="colorblind",
                order=['unknown', 'low', 'moderate', 'high', 'essential'], hue="model_guild")
    axs[2].axhline(0.2, c='r', linestyle='--')
    axs[2].set_xlabel("Pollinator dependency", fontsize=14)
    axs[2].set_ylabel("Spearman", fontsize=14)
    axs[2].get_legend().remove()
    axs[2].set_title(label='(c)', loc='left', fontweight='bold', fontsize='large')

    # Legend center, outside the subplots (author guidelines of Ecography journal)
    # axs[1].legend(bbox_to_anchor=(1.05, 0.5, 1, 1), loc=7)
    # fig.legend(loc=7)
    # fig.tight_layout()
    fig.subplots_adjust(right=0.8)

    fig.savefig(os.path.join(sett.dir_fig, sett.eps_fig_landsc_var))

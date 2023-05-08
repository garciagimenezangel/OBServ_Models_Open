
import seaborn as sns
import matplotlib.pyplot as plt
from utils import metrics, model_dict
import pandas as pd

def sorted_boxplot(data, x, y):
    order = data.groupby(by=[x])[y].median().sort_values().index
    sns.boxplot(x=x, y=y, data=data, palette="colorblind", order=order, hue = "model")
    plt.axhline(0.2, c='r', linestyle='--')

def sorted_boxsubplot(data, x, y, ax):
    order = data.groupby(by=[x])[y].median().sort_values().index
    sns.boxplot(x=x, y=y, data=data, order=order, palette="colorblind", hue = "model", ax=ax)
    ax.axhline(0.2, c='r', linestyle='--')

########### PLOTS ###########
def plot_data(df_studywise, margin=1.2):
    fig, axs = plt.subplots(3, 1)
    fig.set_size_inches(10, 10)
    fig.tight_layout(pad=5.0)

    # Set bins landscape variance
    max_var100 = round(df_studywise.landsc_var.max()*100)
    bin1 = round(max_var100/4)
    bin2 = round(max_var100/2)

    # 1. Landscape variance plot
    df_studywise['landscape_variance'] = pd.cut(df_studywise.landsc_var, bins=[0, bin1/100, bin2/100, max_var100/100])
    df_studywise['model_guild'] = [" - ".join([m, g]) for g, m in zip(df_studywise['guild'], df_studywise['model'])]
    sns.boxplot(x="landscape_variance", y="Spearman_coef", palette="colorblind", hue="model_guild", data=df_studywise, ax=axs[0])

    axs[0].axhline(0.2, c='r', linestyle='--')
    axs[0].set_xlabel("Landscape Standardized Variance", fontsize=14)
    axs[0].set_ylabel("Spearman", fontsize=14)
    axs[0].legend(bbox_to_anchor=(margin, 1), loc=1, borderaxespad=0.)
    axs[0].set_title(label='(a)', loc='left', fontweight='bold', fontsize='large')

    # 2. Management plot
    sns.boxplot(x="management", y="Spearman_coef", palette="colorblind", data=df_studywise, ax=axs[1], order=['conventional','IPM','organic','unmanaged'], hue = "model_guild")
    axs[1].axhline(0.2, c='r', linestyle='--')
    axs[1].set_xlabel("Management", fontsize=14)
    axs[1].set_ylabel("Spearman", fontsize=14)
    axs[1].get_legend().remove()
    axs[1].set_title(label='(b)', loc='left', fontweight='bold', fontsize='large')

    # 3. Pollinator dependency plot
    sns.boxplot(x="poll_dep", y="Spearman_coef", data=df_studywise, ax=axs[2], palette="colorblind",
                order=['unknown', 'low', 'moderate', 'high', 'essential'], hue="model")
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


    # Other plots (old)
    # fig.suptitle(model_dict.model_labels[model_name]+' model', fontsize=14)
    # axs[0].scatter(df_studywise.var_model, df_studywise.Spearman_coef)
    # sorted_boxsubplot(x="management", y="Spearman_coef", data=df_studywise, ax=axs[1])
    # sorted_boxsubplot(x="biome", y="Spearman_coef", data=df_studywise, ax=axs[2])
    # sorted_boxsubplot(x="sampling_method", y="Spearman_coef", data=df_studywise, ax=axs[4])
    # fig.savefig('G:/My Drive/PROJECTS/OBSERV/Reporting/Lonsdorf Study-wise analysis/'+model_name+'.png')

# PLOT DATA
df_collection = pd.DataFrame()
for model in model_dict.model_labels.keys():
    df_data = metrics.get_data(model, metrics.guild)
    df_studywise = metrics.get_metrics_by_study(df_data)
    df_studywise['model'] = model_dict.model_labels[model]
    df_collection = pd.concat([df_collection, df_studywise], axis=0)

# Select two models (from table printed in print_tables.py, the best performing models of each type are 'Open forest' and 'NuSVR'
df_selection = df_collection.loc[(df_collection.model == "Baseline") | (df_collection.model == "NuSVR")].copy()
df_selection.replace({"Baseline":"MM - Baseline"}, inplace=True)
df_selection.replace({"NuSVR":   "ML  - NuSVR"}, inplace=True)

# Use landscape variance from MM
df_selection['study_id'] = [x[0] for x in df_selection.index]
dict_landscape_var = dict(zip(df_selection.study_id, df_selection.var_model))
df_selection['var_model'] = [dict_landscape_var[x] for x in df_selection.study_id]

# Select two models
plot_data(df_selection, margin=1.22)


# # PLOT DATA
# if metrics.model_name == 'All':
#     for model in metrics.model_dict.model_labels.keys():
#         df_data = metrics.get_data(model, metrics.guild, metrics.ML_sites)
#         df_studywise = metrics.get_metrics_by_study(df_data)
#         plot_data(df_studywise, model)
# else:
#     df_data = metrics.get_data(metrics.model_name, metrics.guild, metrics.ML_sites)
#     df_studywise = metrics.get_metrics_by_study(df_data)
#     plot_data(df_studywise, metrics.model_name)
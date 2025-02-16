import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import itertools

from stats_utils import (
    noun_only_position_in_subtree, 
    number_and_animacy, 
    acl_roots, 
    animacy_and_voice,
)

sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=2)

def plot_number(ud_repo, max_len, output_repo):
    files = os.listdir(ud_repo)
    dfs = []
    for file in files:
        if file[:2] not in ["ja", "zh", "ko"]:
            UD_file = os.path.join(ud_repo, file)
            d_count, _ = number_and_animacy(UD_file, max_len)
            df = pd.DataFrame.from_dict(d_count, orient='index')

            # Calculate proportions
            grand_total = df.values.sum()

            # Calculate proportions
            df = df / grand_total

            # Reset index to get it in long form
            df = df.reset_index().melt(id_vars='index', var_name='Animacy', value_name='Proportion')
            df.columns = ['Number', 'Animacy', 'Proportion']
            df = df[df['Proportion'] != 0]


            # Calculate marginal probabilities
            animacy_marginals = df.groupby('Animacy')['Proportion'].sum().reset_index()
            animacy_marginals.columns = ['Animacy', 'Animacy_Prob']

            deprel_marginals = df.groupby('Number')['Proportion'].sum().reset_index()
            deprel_marginals.columns = ['Number', 'Number_Prob']

            # Merge marginals with original DataFrame
            df = df.merge(animacy_marginals, on='Animacy')
            df = df.merge(deprel_marginals, on='Number')

            # Compute PMI
            df['PMI'] = np.log2(df['Proportion'] / (df['Animacy_Prob'] * df['Number_Prob']))

            df['Language'] = file[:2]
            # Replace inf with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop rows with NaN (or inf, now replaced with NaN)
            df.dropna(inplace=True)
            dfs.append(df)
        
    df = pd.concat(dfs, axis=0)
    df = df[df['Number'] == 'Plur']
    print(df)

    sns.boxplot(
            df, x="Animacy", y="PMI", hue="Animacy",
            palette="vlag", whis=[0, 100], width=.6,
            order=["P", "H", "A", "N"],
        )
    sns.stripplot(df, x="Animacy", y="PMI", size=4, color=".3", order=["P", "H", "A", "N"])

    plt.tight_layout()
    plt.savefig(os.path.join(output_repo, "number_PMI.pdf"))
    plt.show()


def _plot_number(ud_repo, max_len, output_repo):
    animacy_classes = ["P", "H", "A", "N"]
    number_classes = ["Plur", "Sing"]
    l_lang = []
    l_prop = []
    l_num = []
    l_anim = []
    files = os.listdir(ud_repo)
    all_rows = []
    for file in files:
        if file[:2] not in ["ja", "zh", "ko"]:
            UD_file = os.path.join(ud_repo, file)
            d_count, rows = number_and_animacy(UD_file, max_len)
            all_rows += [r + [file[:2]] for r in rows]
            totals = {ac: sum(d_count[nc][ac] for nc in number_classes) for ac in animacy_classes}

            for ac in animacy_classes:
                for nc in number_classes:
                    l_lang += [file[:2]]
                    l_num += [nc]
                    l_anim += [ac]
                    if totals[ac] > 0:
                        l_prop += [d_count[nc][ac] / totals[ac]]
                    else:
                        l_prop += [0]

    # PLOT
    d = {"Language": l_lang, "Proportion": l_prop, "Number": l_num, "Animacy": l_anim}
    df = pd.DataFrame.from_dict(d)
    df = df[df.Number == "Plur"]
    df = df.drop(df[df.Language=="nl"][df.Animacy=="P"].index)

    plt.figure(figsize=(8, 6))

    sns.lineplot(
    data=df,
    x='Animacy',
    y='Proportion',
    hue='Language',
    style='Language',
    errorbar='sd',
    palette='Set2',
    linewidth = 2,
    )
    
    # Calculate mean and standard deviation
    means = df.groupby('Animacy')['Proportion'].mean().reset_index()
    stds = df.groupby('Animacy')['Proportion'].sem().reset_index()
    print(means)

    # Plot mean and standard deviation
    plt.errorbar(means['Animacy'], means['Proportion'], yerr=stds['Proportion'], fmt='o', c='black', lw=3, ms=6, marker='D')
    
    plt.xlabel('Animacy')
    plt.ylabel('Proportion of Plurals')
    plt.xticks(rotation=45)

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.05), title='Languages')

    plt.tight_layout()
    plt.savefig(os.path.join(output_repo, "number.pdf"))
    
    # Get unique animacy levels
    animacy_levels = df['Animacy'].unique()

    # Conduct pairwise Welch's t-tests
    results = []
    for level1, level2 in itertools.combinations(animacy_levels, 2):
        group1 = df[df['Animacy'] == level1]['Proportion']
        group2 = df[df['Animacy'] == level2]['Proportion']
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        results.append((level1, level2, t_stat, p_value))

    # Print results
    for level1, level2, t_stat, p_value in results:
        print(f"Animacy {level1} vs {level2}: T-statistic = {t_stat}, P-value = {p_value}")


def plot_position(UD_folder, max_len, output_folder):
    d_all = {}
    all_rows = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        UD_file_ner = os.path.join(UD_folder, file)
        d, rows = noun_only_position_in_subtree(UD_file_ner, max_len=max_len)
        all_rows += [r + [lang] for r in rows]
        d_all[lang] = d
    
    df = pd.DataFrame.from_dict(d_all)
    df =df.rename_axis('Animacy').reset_index()
    df = df.melt(id_vars='Animacy', var_name='Language', value_name='Order')
    # Basque does not annotate person on pronouns
    df = df.drop(df[df.Language=="eu"][df.Animacy=="P"].index)
    df = df.dropna()

    plt.figure(figsize=(8, 6))

    sns.lineplot(
    data=df,
    x='Animacy',
    y='Order',
    hue='Language',
    style='Language',
    ci=None,
    palette='Set2',
    linewidth = 2,
    )

    # Calculate mean and standard deviation
    means = df.groupby('Animacy')['Order'].mean().reset_index()
    stds = df.groupby('Animacy')['Order'].sem().reset_index()
    print(means)

    # Plot mean and standard deviation
    plt.errorbar(means['Animacy'], means['Order'], yerr=stds['Order'], fmt='o', c='black', lw=3, ms=6, marker='D')
    
    plt.xlabel('Animacy')
    plt.ylabel('Order in Clause')
    plt.xticks(rotation=45)

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.05), title='Languages')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "order.pdf"))


    # Get unique animacy levels
    animacy_levels = df['Animacy'].unique()

    # Conduct pairwise Welch's t-tests
    results = []
    for level1, level2 in itertools.combinations(animacy_levels, 2):
        group1 = df[df['Animacy'] == level1]['Order']
        group2 = df[df['Animacy'] == level2]['Order']
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        results.append((level1, level2, t_stat, p_value))

    # Print results
    for level1, level2, t_stat, p_value in results:
        print(f"Animacy {level1} vs {level2}: T-statistic = {t_stat}, P-value = {p_value}")


def plot_relativisation(UD_folder, max_len, output_folder):
    dfs = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        if lang not in ["eu", "ja"]:
            UD_file_ner = os.path.join(UD_folder, file)
        
            l, d_tot = acl_roots(UD_file_ner,max_len)
            d_count = {elem: {'Yes': l.count(elem), 'Not': (d_tot[elem] - l.count(elem))} for elem in ["P", "H", "A", "N"] if d_tot[elem] > 0}
            df = pd.DataFrame.from_dict(d_count, orient='index')

            # Calculate proportions
            grand_total = df.values.sum()

            # Calculate proportions
            df = df / grand_total

            # Reset index to get it in long form
            df = df.reset_index().melt(id_vars='index', var_name='Animacy', value_name='Proportion')
            df.columns = ['Animacy', 'Relativisation', 'Proportion']
            df = df[df['Proportion'] != 0]

            # Calculate marginal probabilities
            animacy_marginals = df.groupby('Animacy')['Proportion'].sum().reset_index()
            animacy_marginals.columns = ['Animacy', 'Animacy_Prob']

            rel_marginals = df.groupby('Relativisation')['Proportion'].sum().reset_index()
            rel_marginals.columns = ['Relativisation', 'Relativisation_Prob']

            # Merge marginals with original DataFrame
            df = df.merge(animacy_marginals, on='Animacy')
            df = df.merge(rel_marginals, on='Relativisation')

            # Compute PMI
            df['PMI'] = np.log2(df['Proportion'] / (df['Animacy_Prob'] * df['Relativisation_Prob']))

            df['Language'] = file[:2]
            # Replace inf with NaN
            df.replace([np.inf, -np.inf], np.nan, inplace=True)

            # Drop rows with NaN (or inf, now replaced with NaN)
            df.dropna(inplace=True)
            dfs.append(df)
        
    df = pd.concat(dfs, axis=0)
    df = df[df['Relativisation'] == 'Yes']
    print(df)

    sns.boxplot(
            df, x="Animacy", y="PMI", hue="Animacy",
            palette="vlag", whis=[0, 100], width=.6,
            order=["P", "H", "A", "N"],
        )
    sns.stripplot(df, x="Animacy", y="PMI", size=4, color=".3", order=["P", "H", "A", "N"])

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "relativisation_PMI.pdf"))
    plt.show()


def _plot_relativisation(UD_folder, max_len, output_folder):
    d = {}
    all_rows = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        if lang not in ["eu", "ja"]:
            UD_file_ner = os.path.join(UD_folder, file)
        
            l, d_tot = acl_roots(UD_file_ner,max_len)
            d[lang] = {elem: l.count(elem)/d_tot[elem] for elem in ["P", "H", "A", "N"] if d_tot[elem] > 0}

            for elem in ["H", "A", "N"]:
                all_rows += l.count(elem) * [[elem, 1, lang]]
                all_rows += (d_tot[elem] - l.count(elem)) * [[elem, 0, lang]]

    df = pd.DataFrame.from_dict(d)
    df =df.rename_axis('Animacy').reset_index()
    df = df.melt(id_vars='Animacy', var_name='Language', value_name='Proportion')
    # Basque does not annotate person on pronouns
    df = df.dropna()

    plt.figure(figsize=(8, 6))

    sns.lineplot(
    data=df,
    x='Animacy',
    y='Proportion',
    hue='Language',
    style='Language',
    ci=None,
    palette='Set2',
    linewidth = 2,
    )
    # Calculate mean and standard deviation
    means = df.groupby('Animacy')['Proportion'].mean().reset_index()
    stds = df.groupby('Animacy')['Proportion'].sem().reset_index()
    print(means)

    # Plot mean and standard deviation
    plt.errorbar(means['Animacy'], means['Proportion'], yerr=stds['Proportion'], fmt='o', c='black', lw=3, ms=6, marker='D')
    

    plt.xlabel('Animacy')
    plt.ylabel('Proportion of Relativised Nominals')
    plt.xticks(rotation=45)

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.05), title='Languages')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "relativisation.pdf"))

    # Get unique animacy levels
    animacy_levels = df['Animacy'].unique()

    # Conduct pairwise Welch's t-tests
    results = []
    for level1, level2 in itertools.combinations(animacy_levels, 2):
        group1 = df[df['Animacy'] == level1]['Proportion']
        group2 = df[df['Animacy'] == level2]['Proportion']
        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        results.append((level1, level2, t_stat, p_value))

    # Print results
    for level1, level2, t_stat, p_value in results:
        print(f"Animacy {level1} vs {level2}: T-statistic = {t_stat}, P-value = {p_value}")


def plot_voice(UD_folder, max_len, output_folder, plot_heatmap=False):
    dfs = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        print(lang)
        UD_file_ner = UD_folder+file
        d_count = animacy_and_voice(UD_file_ner, max_len)
        df = pd.DataFrame.from_dict(d_count, orient='index')

        # Calculate proportions
        grand_total = df.values.sum()

        # Calculate proportions
        df = df / grand_total

        # Reset index to get it in long form
        df = df.reset_index().melt(id_vars='index', var_name='Animacy', value_name='Proportion')
        df.columns = ['Deprel', 'Animacy', 'Proportion']
        df = df[df['Proportion'] != 0]


        # Calculate marginal probabilities
        animacy_marginals = df.groupby('Animacy')['Proportion'].sum().reset_index()
        animacy_marginals.columns = ['Animacy', 'Animacy_Prob']

        deprel_marginals = df.groupby('Deprel')['Proportion'].sum().reset_index()
        deprel_marginals.columns = ['Deprel', 'Deprel_Prob']

        # Merge marginals with original DataFrame
        df = df.merge(animacy_marginals, on='Animacy')
        df = df.merge(deprel_marginals, on='Deprel')

        # Compute PMI
        df['PMI'] = np.log2(df['Proportion'] / (df['Animacy_Prob'] * df['Deprel_Prob']))

        df['Language'] = lang
        # Replace inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Drop rows with NaN (or inf, now replaced with NaN)
        df.dropna(inplace=True)
        dfs.append(df)
    
    df = pd.concat(dfs, axis=0)
    print(df)

    mapping = {
        "obl:agent": "obl:agent A",
        "nsubj_A": "nsubj A",
        "nsubj_S": "nsubj S",
        "nsubj:pass": "nsubj:pass P",
        "obj": "obj P"
    }
    column_order = list(mapping.values())

    # Replace the values in the 'Deprel' column
    df['Deprel'] = df['Deprel'].replace(mapping)

    if plot_heatmap:
        vmin, vmax = df.PMI.min(), df.PMI.max()
        # FacetGrid for each language
        g = sns.FacetGrid(df, col='Language', col_wrap=6, height=4)
        
        def plot_func(data, colorbar_ax, **kwargs):
            sns.heatmap(
                data.pivot_table(index='Animacy', columns='Deprel', values='PMI', sort=False).reindex(column_order, axis=1),
                cmap='coolwarm',
                cbar_kws={'label': 'PMI'},
                cbar=colorbar_ax is not None,
                cbar_ax=colorbar_ax,  # Specify the shared colorbar axis
                vmin=vmin, vmax=vmax,  # Assuming proportions are between 0 and 1
                **kwargs
            )

        # Add an axis for the color bar on the right
        cbar_ax = g.fig.add_axes([.90, .15, .02, .7])  # Adjust the position as needed

        # Adjust the labels
        for ai, ax in enumerate(g.axes):
            title = g.col_names[ai]
            ax.set_title(title, x=0.5, y=-0.3)
            if ai < 6:
                ax.xaxis.tick_top()
                ax.xaxis.set_label_position('top')
                ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
            else:
                ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

        # Map the data to the grid and plot with the shared colorbar ax only once
        g.map_dataframe(plot_func, colorbar_ax=cbar_ax)

        plt.tight_layout(rect=[0, 0, 0.85, 1])
        g.set_axis_labels("", "Animacy")
    else:
        sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=2)
        g = sns.FacetGrid(df, col='Animacy', col_wrap=4, height=4)
        
        def plot_func(data, **kwargs):
            ax = sns.boxplot(
                data, y="Deprel", x="PMI", hue="Deprel",
                palette="vlag", whis=[0, 100], width=.6, order=column_order, hue_order=column_order,
                **kwargs
            )
            ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha='center')
            sns.stripplot(data, y="Deprel", x="PMI", size=4, color=".3", order=column_order, hue_order=column_order)

        
        g.map_dataframe(plot_func)
        g.set_ylabels("PMI")

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "voice.pdf"))
    plt.show()


PLOT_FUNCTIONS = {
    "number": plot_number,
    "order": plot_position,
    "relativisation": plot_relativisation,
    "voice": plot_voice,
}

def get_args():

    parser = argparse.ArgumentParser(description="Animacy plots.")

    # Adding an argument 'type' with choices 'deprel' and 'number'
    parser.add_argument(
        '--type',
        type=str,
        choices=['number', 'order', 'relativisation', 'voice'],
        required=True,
    )
    parser.add_argument(
        '--max_len',
        type=int,
        default=-1,
    )
    parser.add_argument(
        '--ud_folder',
        type=str,
        default="UD_with_anim_ner_annot/",
    )
    parser.add_argument(
        '--output_folder',
        type=str,
        default="plots",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    plot_function = PLOT_FUNCTIONS[args.type]
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)

    plot_function(args.ud_folder, args.max_len, args.output_folder)
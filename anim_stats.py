import pandas as pd
import numpy as np
import argparse
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.formula.api import mnlogit
from scipy.stats import chi2

from stats_utils import (
    noun_only_position_in_subtree, 
    number_and_animacy, 
    acl_roots, 
    animacy_and_voice,
)

sns.set_theme(style="whitegrid", font="Times New Roman", font_scale=2)


def plot_number(ud_repo, max_len, output_repo):
    animacy_classes = ["P", "H", "A", "N"]
    number_classes = ["Plur", "Sing"]
    l_lang = []
    l_prop = []
    l_num = []
    l_anim = []
    files = os.listdir(ud_repo)
    all_stats = []
    for file in files:
        if file[:2] not in ["ja", "zh", "ko"]:
            UD_file = os.path.join(ud_repo, file)
            d_count, rows = number_and_animacy(UD_file, max_len)
            rows = [r + [file[:2]] for r in rows]
            all_stats += rows

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
    
    # STATS
    df = pd.DataFrame.from_dict(all_stats)
    df.columns = ["Number", "Animacy", "Language"]
    df = df[df['Animacy'] != 'P']
    df['Number'] = df['Number'].replace(['Sing', 'Plur'], [0, 1])
    df['Animacy'] = df['Animacy'].replace(['N', 'A', 'H'], [0, 1, 2])

    # Adding language as categorical using C()
    model1_sm = mnlogit('Number ~ Animacy', data=df).fit()

    # Summary of the model
    print(model1_sm.summary())

    # Exponentiate the coefficients to get odds ratios
    odds_ratios = np.exp(model1_sm.params)
    print("\nOdds Ratios:\n", odds_ratios)

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
    ci=None,
    palette='Set2',
    linewidth = 3,
    )
    plt.xlabel('Animacy')
    plt.ylabel('Proportion of Plurals')
    plt.xticks(rotation=45)

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.05), title='Languages')

    plt.tight_layout()
    plt.savefig(os.path.join(output_repo, "number.pdf"))
    plt.show()


def plot_position(UD_folder, max_len, output_folder):
    d_all = {}
    for file in os.listdir(UD_folder):
        lang = file[:2]
        UD_file_ner = os.path.join(UD_folder, file)
        d = noun_only_position_in_subtree(UD_file_ner, max_len=max_len)
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
    linewidth = 3,
    )
    plt.xlabel('Animacy')
    plt.ylabel('Order in Clause')
    plt.xticks(rotation=45)

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.05), title='Languages')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "order.pdf"))
    plt.show()


def plot_relativisation(UD_folder, max_len, output_folder):
    d = {}
    for file in os.listdir(UD_folder):
        lang = file[:2]
        if lang not in ["eu", "ja"]:
            UD_file_ner = os.path.join(UD_folder, file)
        
            l, d_tot = acl_roots(UD_file_ner,max_len)
            d[lang] = {elem: l.count(elem)/d_tot[elem] for elem in ["P", "H", "A", "N"] if d_tot[elem] > 0}


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
    linewidth = 3,
    )
    plt.xlabel('Animacy')
    plt.ylabel('Proportion of Relativised Nominals')
    plt.xticks(rotation=45)

    plt.legend(loc='upper center', bbox_to_anchor=(1.2, 1.05), title='Languages')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, "relativisation.pdf"))
    plt.show()


def plot_voice(UD_folder, max_len, output_folder):
    
    d_heatmap_data={}
    l_lang = []
    for file in os.listdir(UD_folder):
        lang = file[:2]
        print(lang)
        l_lang.append(lang)
        UD_file_ner = UD_folder+file
        d_count = animacy_and_voice(UD_file_ner, max_len)
        new_d = defaultdict(lambda: defaultdict(int))
        for deprel, d in d_count.items():
            for anim, count in d.items():
                if count:
                    new_d[deprel][anim] = count / sum(d.values())
                elif lang == 'ja' and anim == 'P' and deprel not in ['obl:agent', 'nsubj:pass']:
                    new_d[deprel][anim] = 0
                else:
                    new_d[deprel][anim] = pd.NA
        d_count = new_d

        d_heatmap_data[lang] = d_count
    
    df = pd.concat({k: pd.DataFrame(v).T for k, v in d_heatmap_data.items()}, axis=0)
    df = df.reset_index()
    df.columns = ['Language', 'Deprel', 'P', 'H', 'A', 'N']

    df = pd.melt(df, id_vars=['Language', 'Deprel'], var_name='Animacy', value_name='Proportion')
    df['Proportion'] = pd.to_numeric(df['Proportion'], errors='coerce')

    column_order = ["obl:agent", "nsubj_A", "nsubj_S", "nsubj:pass", "obj"]

    # FacetGrid for each language
    g = sns.FacetGrid(df, col='Language', col_wrap=6, height=4)
    
    def plot_func(data, **kwargs):
        print(data)
        sns.heatmap(
            data.pivot_table(index='Animacy', columns='Deprel', values='Proportion', sort=False).reindex(column_order, axis=1),
            cmap=sns.cm.rocket_r,
            cbar=True,
            vmin=0, vmax=1,  # Assuming proportions are between 0 and 1
            **kwargs
        )

    # Heatmap for each facet
    g.map_dataframe(plot_func)

    # Adjust layout
    g.set_axis_labels("", "Animacy")
    g.set_titles("{col_name}")
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
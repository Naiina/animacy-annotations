import argparse
import xml.etree.ElementTree as ET
import os
import pandas as pd
from datasets import Dataset, DatasetDict, ClassLabel, Sequence
from nltk.corpus import wordnet as wn
import numpy as np
from dataset_util import create_sentence, load_wn_map
from collections import Counter
from itertools import chain
import random

# Import the updated module
from synset_analysis import is_syn_an, which_an, is_group

# Set the global seed for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

BN_MAP_PATH = 'bnids_map.txt'

def balanced_ratio_split(df, train_ratio=0.75, test_ratio=0.15, dev_ratio=0.10, random_state=SEED):
    # Calculate target sentence counts
    total_sentences = len(df)
    target_counts = {
        'train': int(total_sentences * train_ratio),
        'test': int(total_sentences * test_ratio),
        'dev': int(total_sentences * dev_ratio)
    }

    # Add remaining sentences to test to ensure we use all data
    target_counts['test'] += total_sentences - sum(target_counts.values())

    # Create bins based on number of labels
    df['num_labels'] = df['anim_tags'].apply(len)
    label_groups = df.groupby('num_labels', sort=True)

    train_dfs = []
    test_dfs = []
    dev_dfs = []

    # Process each group of sentences with the same number of labels
    for num_labels, group in label_groups:
        group_size = len(group)

        # Calculate the target number of sentences for each dataset split (train, test, dev)
        group_train_size = int(group_size * train_ratio)
        group_test_size = int(group_size * test_ratio)
        group_dev_size = group_size - (group_train_size + group_test_size)

        # Get label distribution for this group
        group_label_dist = get_dist(group)
        group = group.copy()

        # Create a unique string identifier for each sentence's label combination, e.g. A_N_N , A_H_H_N, etc.
        # Sorting ensures consistent ordering of labels within the string
        group['label_combo'] = group['anim_tags'].apply(lambda x: '_'.join(sorted(x)))

        # Count occurrences of each label combination in the dataset
        combo_counts = group['label_combo'].value_counts()

        # Map these counts back to the sentences so each sentence gets a frequency score
        group['combo_freq'] = group['label_combo'].map(combo_counts)

        # Initialize a random number generator with a fixed seed for reproducibility
        rng = np.random.default_rng(random_state)

        # Assign a random number to each sentence
        group['random'] = rng.random(len(group))

        # Store the original index to maintain reference after sorting
        group['original_index'] = group.index

        # Sort sentences by:
        # 1. 'combo_freq' (ascending): Prioritizes rare label combinations
        # 2. 'random' (ascending): Ensures randomization within same-frequency groups
        # 3. 'original_index' (ascending): Maintains original order as a last tie-breaker
        group = group.sort_values(['combo_freq', 'random', 'original_index'])

        # Initialize counters to track label distributions in each split
        current_counts = {'train': Counter(), 'test': Counter(), 'dev': Counter()}

        # Create empty DataFrames to store assigned sentences for train, test, and dev
        group_train = pd.DataFrame(columns=group.columns)
        group_test = pd.DataFrame(columns=group.columns)
        group_dev = pd.DataFrame(columns=group.columns)

        # Iterate through each sentence in the sorted group to distribute them
        for _, row in group.iterrows():
            # Calculate scores for each split based on:
            # 1. How far it is from target size
            # 2. How much it needs the labels in this sentence
            scores = {}

            split_order = ['train', 'test', 'dev']
            target_sizes = {
                'train': group_train_size,
                'test': group_test_size,
                'dev': group_dev_size
            }

            # Evaluate each split (train, test, dev) to determine the best fit for the sentence
            for split_name in split_order:
                target_size = target_sizes[split_name]

                # Select the corresponding split DataFrame
                if split_name == 'train':
                    split_df = group_train
                elif split_name == 'test':
                    split_df = group_test
                else:
                    split_df = group_dev

                # If this split is already full, assign a very low score to avoid selecting it
                if len(split_df) >= target_size:
                    scores[split_name] = -float('inf')
                    continue

                # ---- Scoring Mechanism ----
                # 1. **Size Score**: Encourages filling up the split proportionally
                size_score = 1 - (len(split_df) / target_size if target_size > 0 else 0)

                # 2. **Label Balance Score**: Ensures an even label distribution
                label_score = 0
                for label in row['anim_tags']:
                    # Calculate current ratio of this label in the split
                    current_ratio = (current_counts[split_name][label] + 1) / (
                                sum(current_counts[split_name].values()) + len(row['anim_tags']))
                    # Calculate the target ratio of this label based on group-wide distribution
                    target_ratio = group_label_dist[label] / sum(group_label_dist.values())
                    # Higher score if the label ratio in the split is closer to the target
                    label_score += 1 - abs(current_ratio - target_ratio)

                # Average the size and label balance scores to get the final score
                scores[split_name] = (size_score + label_score) / 2

            # ---- Assign Sentence to Best Split ----
            # Sort splits by score and, in case of ties, prioritize order (train > test > dev)
            best_split = max(
                sorted(scores.items(), key=lambda x: (x[1], x[0])),  # x[0] is the split name
                key=lambda x: x[1]
            )[0]

            # Add the sentence to the best-suited split and update label distribution counts
            if best_split == 'train':
                group_train = pd.concat([group_train, row.to_frame().T])
                current_counts['train'].update(row['anim_tags'])
            elif best_split == 'test':
                group_test = pd.concat([group_test, row.to_frame().T])
                current_counts['test'].update(row['anim_tags'])
            else:
                group_dev = pd.concat([group_dev, row.to_frame().T])
                current_counts['dev'].update(row['anim_tags'])

        train_dfs.append(group_train)
        test_dfs.append(group_test)
        dev_dfs.append(group_dev)

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    dev_df = pd.concat(dev_dfs, ignore_index=True)

    # Clean up temporary columns
    for df in [train_df, test_df, dev_df]:
        df.drop(['num_labels', 'label_combo', 'combo_freq', 'random'], axis=1, inplace=True)

    return train_df.sample(frac=1, random_state=SEED).reset_index(drop=True), test_df.sample(frac=1, random_state=SEED).reset_index(drop=True), dev_df.sample(frac=1, random_state=SEED).reset_index(drop=True)


def get_dist(df):
    labels = list(chain.from_iterable(df['anim_tags']))
    return Counter(labels)


def verify_split(train_df, test_df, dev_df):
    """
    Verify the split ratios and label distribution
    """
    total = len(train_df) + len(test_df) + len(dev_df)

    print("Split Ratios:")
    print(f"Train: {len(train_df) / total:.1%} ({len(train_df)} sentences)")
    print(f"Test:  {len(test_df) / total:.1%} ({len(test_df)} sentences)")
    print(f"Dev:   {len(dev_df) / total:.1%} ({len(dev_df)} sentences)")
    print("\nLabel Distributions:")

    # Get distributions
    train_dist = get_dist(train_df)
    test_dist = get_dist(test_df)
    dev_dist = get_dist(dev_df)

    # Calculate total labels
    all_labels = set(train_dist.keys()) | set(test_dist.keys()) | set(dev_dist.keys())

    print("\nLabel counts per split:")
    print(f"{'Label':>6} {'Train':>8} {'Test':>8} {'Dev':>8}")
    print("-" * 32)

    for label in sorted(all_labels):
        print(f"{label:>6} {train_dist[label]:>8d} {test_dist[label]:>8d} {dev_dist[label]:>8d}")

def store_keys(key_file: str) -> dict:
    keys = {}
    with open(key_file, 'r', encoding='utf-8') as k:
        for line in k:
            parts = line.split()
            if len(parts) > 1:
                if len(parts) > 2:
                    parts = parts[:2]
                id, bn_offset = parts
                keys[id] = bn_offset
    return keys


def process_data_set(data_set, wn_map):
    def get_sents_data(xml_content, source_name):
        root = ET.fromstring(xml_content)
        sentences_data = []

        for sentence_elem in root.iter('sentence'):
            instances = []
            text = ""
            for index, token in enumerate(sentence_elem):
                if token.tag == 'instance':
                    instance_id = token.get('id', '')
                    instance_pos = token.get('pos', '')
                    instance_lemma = token.get('lemma', '')
                    instances.append((index, instance_id, instance_lemma, instance_pos))
                    text += f" {token.text}"
                elif token.tag == 'wf':
                    text += f" {token.text} "
            sentences_data.append((text, instances, source_name))

        return sentences_data

    def get_data_and_keys(subset):
        data_f = sorted([f for f in os.listdir(subset) if f.endswith('.xml')])
        key_f = sorted([f for f in os.listdir(subset) if f.endswith('.key.txt')])
        with open(os.path.join(subset, data_f[0]), 'r', encoding='utf-8') as file:
            xml_content = file.read()
        key_file = os.path.join(subset, key_f[0])
        keys = store_keys(key_file)
        source_name = os.path.basename(subset)
        sentences_data = get_sents_data(xml_content, source_name)
        return keys, sentences_data

    pd_data = []
    for d in data_set:
        keys, sentences_data = get_data_and_keys(d)

        sentences, tokens, anim_tags, target_indexes, sources = annotate_animacy(sentences_data, keys, wn_map)
        pd_dict = {k: v for (k, v) in zip(['sentences', 'tokens', 'anim_tags', 'target-indexes', 'source'], [sentences,
                                                                                                   tokens,
                                                                                                   anim_tags,
                                                                                                   target_indexes,
                                                                                                   sources])}

        pd_data.append(pd_dict)
    df_data = pd.concat(
        [pd.DataFrame(data=d, columns=['sentences', 'tokens', 'anim_tags', 'target-indexes', 'source']) for d in pd_data],
        ignore_index=True)
    return df_data


def balance_labels(df):
    """
    Balance the dataset by first attempting to remove all sentences containing only N labels,
    then falling back to the original strategy if needed.
    """

    def get_label_counts(df):
        all_labels = list(chain.from_iterable(df['anim_tags']))
        counts = Counter(all_labels)
        return counts

    def only_has_N_labels(labels):
        # Flatten labels
        def flatten(lst):
            for item in lst:
                if isinstance(item, list):
                    yield from flatten(item)
                else:
                    yield item

        flattened_labels = list(flatten(labels))

        return all(label == 'N' for label in flattened_labels)

    def is_gloss(source):
        return 'wngt_glosses' in str(source) or 'wngt_examples' in str(source)

    # Store initial counts
    initial_counts = get_label_counts(df)
    print("\nInitial label distribution:")
    print(initial_counts)


    # First pass: Remove all sentences that only contain N labels from glosses
    n_only_mask = df['anim_tags'].apply(only_has_N_labels)
    glosses_mask = df['source'].apply(is_gloss)

    df_filtered = df[~(n_only_mask & glosses_mask)]

    current_counts = get_label_counts(df_filtered)
    target_n_count = 2 * (current_counts['A'] + current_counts['H'])

    n_only_mask_filtered = df_filtered['anim_tags'].apply(only_has_N_labels)
    removed_sentences_second_pass = df_filtered[n_only_mask_filtered]
    df_filtered = df_filtered[~n_only_mask_filtered]

    current_counts = get_label_counts(df_filtered)
    if current_counts['N'] < target_n_count:
        # We removed too many N-only sentences, need to add some back
        n_to_add = target_n_count - current_counts['N']
        if len(removed_sentences_second_pass) > 0:
            df_filtered = pd.concat(
                [df_filtered, removed_sentences_second_pass.head(n_to_add)],
                ignore_index=True
            )

    print("\nWe Cannot balance labels further without removing mixed-label sentences")

    counts = get_label_counts(df_filtered)

    print("\nFinal label distribution:")
    print(counts)
    print(f"\nRemoved {len(df) - len(df_filtered)} sentences to achieve balance")

    return df_filtered

def annotate_animacy(sentences_data, keys, wn_map):
    def get_correct_synsets(key_bn_offsets, wn_map) -> list:
        wn_offsets = wn_map[key_bn_offsets]
        correct_synsets = []
        for o in wn_offsets:
            o = o.removeprefix('wn:')
            if o.startswith('0'):
                o = o.removeprefix('0')
            pos = o[-1]
            synset = wn.synset_from_pos_and_offset(pos, int(o[:-1]))
            correct_synsets.append(synset)

        correct_synsets.sort(key=lambda s: s.name())
        return correct_synsets

    sentences = []
    tokens = []
    anim_tags = []
    target_indexes = []
    sources = []

    for sent_idx, (sentence, instances, source) in enumerate(sentences_data):
        target_idxs = []
        labels = []
        for (tgt_idx, tgt_id, tgt_lemma, tgt_pos) in instances:
            if tgt_pos != 'NOUN':
                continue

            key_bn_offsets = keys[tgt_id]
            correct_synsets = get_correct_synsets(key_bn_offsets, wn_map)
            if not correct_synsets:
                continue
            correct_synset = correct_synsets[0]

            if is_syn_an(correct_synset):
                label = which_an(correct_synset)
                target_idxs.append(tgt_idx + 1)
                labels.append(label)
            else:
                if is_group(correct_synset):
                    continue
                label = 'N'
                target_idxs.append(tgt_idx + 1)
                labels.append(label)

        if len(labels) >= 1:
            s = create_sentence(sentence.split())
            sentences.append(s)
            tokens.append(sentence.split())
            anim_tags.append(labels)
            target_indexes.append(target_idxs)
            sources.append(source)

    return sentences, tokens, anim_tags, target_indexes, sources

def main(inp_dir, langs):
    wn_map = load_wn_map(BN_MAP_PATH)
    training_dir = os.path.join(inp_dir, 'training_datasets')
    evaluation_dir = os.path.join(inp_dir, 'evaluation_datasets')

    for l in langs:
        training_set = []
        if l not in ['ko', 'zh']:
            wngt_glosses = os.path.join(training_dir, f'wngt_glosses_{l}')
            wngt_examples = os.path.join(training_dir, f'wngt_examples_{l}')
            semcor = os.path.join(training_dir, f'semcor_{l}')
            training_set.extend([wngt_examples, semcor, wngt_glosses])

        dev_dir = [os.path.join(evaluation_dir, f'dev-{l}')]
        test_dir = [os.path.join(evaluation_dir, f'test-{l}')]

        if training_set:
            df_train = process_data_set(training_set, wn_map)

        df_dev = process_data_set(dev_dir, wn_map)
        df_test = process_data_set(test_dir, wn_map)

        if training_set:
            combined_df = pd.concat([df_test, df_train, df_dev], ignore_index=True)
        else:
            combined_df = pd.concat([df_test, df_dev], ignore_index=True)

        print("\nBalancing labels...")

        if l not in ['ko', 'zh']:
            combined_df = balance_labels(combined_df)

        shuffled_df = combined_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

        train_df, test_df, dev_df = balanced_ratio_split(shuffled_df, random_state=SEED)

        def print_source_dist(df, split_name):
            print(f"\n{split_name} source distribution:")
            print(df['source'].value_counts())

        print("\nSource distribution across splits:")
        print_source_dist(train_df, "Train")
        print_source_dist(test_df, "Test")
        print_source_dist(dev_df, "Dev")

        verify_split(train_df, test_df, dev_df)

        test_dataset = Dataset.from_pandas(test_df)
        train_dataset = Dataset.from_pandas(train_df)
        dev_dataset = Dataset.from_pandas(dev_df)

        data = DatasetDict({
            'train': train_dataset.remove_columns('original_index'),
            'test': test_dataset.remove_columns('original_index'),
            'validation': dev_dataset.remove_columns('original_index')
        })

        new_features = data['train'].features.copy()
        new_features["anim_tags"] = Sequence(ClassLabel(num_classes=3, names=['N', 'A', 'H']), length=-1)
        #data = data.cast(new_features)

        #final_name = 'xxxx'

        #username = 'yyyyyy'
        #data.push_to_hub(f'{username}/{final_name}')

if __name__ == '__main__':
    main("xl-wsd", ['it', 'de', 'en', 'es', 'fr', 'it', 'ja', 'ko', 'nl', 'sl', 'zh', 'eu', 'et'])
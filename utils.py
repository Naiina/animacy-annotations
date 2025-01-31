from datasets import load_dataset, concatenate_datasets
def get_dataset(lang, source_filters=['semcor', 'test', 'dev', 'wngt_examples'], anim_filters=None):
    filtered_data = {}

    anim_label_dict = {'N': 0,
                       'A': 1,
                       'H': 2}

    def filter_animacy(item, anim_filters):
        tag_to_remove = anim_label_dict[anim_filters]
        filtered_indices = [i for i, tag in enumerate(item['anim_tags']) if tag != tag_to_remove]
        item['anim_tags'] = [item['anim_tags'][i] for i in filtered_indices]
        item['target_indexes'] = [item['target_indexes'][i] for i in filtered_indices]

        return item



    dataset = load_dataset(f'lingvenvist/animacy-{lang}-nogroups-xtr-complete-filtered-fixed')
    for set in ['train', 'validation', 'test']:
        processed_dataset = [{'sentence': item['sentences'],
                              'tokens': item['tokens'],
                              'anim_tags': item['anim_tags'],
                              'target_indexes': [t - 1 for t in item['target-indexes']],  # target are now 0-indexed
                              'source': item['source']}
                             for item in dataset[set]]
        filtered = [i for i in processed_dataset if any(i['source'].startswith(f) for f in source_filters)]
        if anim_filters:
            filtered = [filter_animacy(item, anim_filters) for item in filtered]
            filtered = [x for x in filtered if len(x['anim_tags']) != 0]# Final sanity check: After eliminating one of the animacy labels, certain sentences may no longer have any labels!


        filtered_data[set] = filtered

    return filtered_data
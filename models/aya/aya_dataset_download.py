import os
import random
from datasets import load_dataset
from utils import ADJ_MAP_LARGE
import json

langs = ['de', 'en', 'es' ,'fr', 'it' ,'ja' ,'ko', 'nl' ,'sl' ,'zh', 'hu', 'gl', 'bg', 'ca', 'da', 'hr']

label_to_id = {0 : 'N',
               1 : 'A',
               2 : 'H',
               }

splits = ['train', 'test', 'validation']
for l_code in langs:
    os.makedirs(f'aya_dataset/{l_code}', exist_ok=True)
    dataset = load_dataset(f'lingvenvist/animacy-{l_code}-nogroups-xtr-complete-filtered')
    for s in splits:
        json_list = []
        data = dataset[s]
        for i, entry in enumerate(data):
            sentence = entry['sentences']
            tokens = entry['tokens']
            anim_tags = entry['anim_tags']
            target_indexes = entry['target-indexes']

            if len(anim_tags) == 0:
                continue

            for target_idx, label in zip(target_indexes, anim_tags):
                target_word = tokens[int(target_idx - 1)]

                json_obj = {
                    "sentence": sentence,
                    "target_word": target_word,
                    "label": label_to_id[label],
                    "lang": ADJ_MAP_LARGE[l_code],
                    "tokens": tokens,
                    "target_indexes" : target_indexes
                }

                json_list.append(json_obj)

        if s == 'validation':
            s = 'dev'

        random.shuffle(json_list)
        with open(f'aya_dataset/{l_code}/animacy_{l_code}_{s}.jsonl', 'w', encoding='utf-8') as outfile:
            for json_obj in json_list:
                json.dump(json_obj, outfile, ensure_ascii=False)
                outfile.write('\n')

        print(f"Processed {len(json_list)} entries and saved to 'aya_dataset/{l_code}/animacy_{l_code}_{s}.jsonl'.")
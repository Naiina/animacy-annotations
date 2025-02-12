from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer
from sft import SFT
from tqdm import tqdm
import argparse
import torch
import os
import pyconll
import re

label_dict = {0: 'A', 1: 'H', 2: 'N'}


def compose_sft(lang):
    config = AutoConfig.from_pretrained(
        'bert-base-multilingual-cased',
        num_labels=3,
    )

    model = AutoModelForTokenClassification.from_pretrained(
        'bert-base-multilingual-cased',
        config=config,
    )

    # Load your model from Hugging Face
    model_name = "mbert-animacy"

    language_sft = SFT(f'cambridgeltl/mbert-lang-sft-{lang}-small')
    task_sft = SFT(f'lingvenvist/{model_name}')

    language_sft.apply(model, with_abs=False)
    task_sft.apply(model)
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    return model, tokenizer


def find_delete(sentence):
    VP = False
    verbs = []
    for token in sentence:
        if token.upos == 'VERB' or token.upos == 'AUX':
            verbs.append(token)

        if token.form is None:
            return [False]

    if verbs:
        VP = True

    return [VP]


def build_data(treebank):
    def find_n_idxs(sentence):
        noun_indices = []
        for idx, token in enumerate(sentence):
            if token.upos == 'NOUN':
                noun_indices.append(idx + 1)
        return noun_indices

    tokens = [[token.form for token in sentence] for sentence in treebank]
    target_indexes = [find_n_idxs(sentence) for sentence in treebank]

    return tokens, target_indexes


def predict(model, data, tokenizer):
    tokens, target_indexes = data

    final_predictions = list()
    model.eval()

    progress_bar = tqdm(total=len(tokens), desc="Annotating sentences", unit="sentence")

    for i, sentence in enumerate(tokens):

        inputs = tokenizer(
            sentence,
            return_tensors='pt',
            is_split_into_words=True,
            padding=True,
            truncation=True,
        )

        word_spans = []
        target_spans = []
        w_tok_map = [(word, inputs[0].word_to_tokens(idx)) for idx, word in
                     enumerate(tokens[i])]  ## for each token: ('token', (int(start), int(end))
        word_spans.append(w_tok_map)
        spans = [w_tok_map[idx - 1][1] for idx in
                 target_indexes[i]]  ## extract int(start) and int(end) for target indexes
        spans = list(sorted(spans))
        target_spans.append(spans)

        def build_masks(targets, input_ids, number):
            length = len(input_ids)
            word_index_mask = [0] * length
            labs = [-100] * length
            for idx, span in enumerate(targets):
                start, end = span
                for i in range(start, end):
                    word_index_mask[i] = idx + 1
                    labs[i] = 0
            return labs, word_index_mask

        masks = [build_masks(sentence, inputs['input_ids'][index], index) for index, sentence in
                 enumerate(target_spans)]
        labs, words_index_mask = zip(*masks)
        labs = list(labs)

        inputs['labels'] = torch.tensor(labs)

        with torch.no_grad():
            output = model(**inputs)

        logits = output.logits
        predictions = torch.argmax(logits, dim=-1)
        per_word_prediction = []
        for span in target_spans[0]:
            start, end = span
            span_predictions = predictions.tolist()[0][start:end]
            per_word_prediction.append(span_predictions[0])

        final_predictions.append(per_word_prediction)

        progress_bar.update(1)

    progress_bar.close()

    return final_predictions


def annotate(file, lang, set_split):
    ud = pyconll.load_from_file(file)

    to_delete = []

    for n, i in enumerate(ud._sentences):
        if False in find_delete(i):
            to_delete.append(n)

    to_delete.reverse()
    for n in to_delete:
        ud.__delitem__(n)

    data = build_data(ud)

    anim_model, tokenizer = compose_sft(lang)
    prediction = predict(anim_model, data, tokenizer)
    prediction = [[label_dict[l] for l in sentence] for sentence in prediction]
    for i, s in enumerate(ud._sentences):
        nouns = set(token for token in s if token.upos == 'NOUN')
        iter_pred = iter(prediction[i])
        for token in s:
            if token in nouns:
                token.misc['ANIMACY'] = next(iter_pred)

    return ud


def write_and_save_ud(ud, lang, set_split):
    ud_name = f'{lang}-animacy-annotated-{set_split}.conllu'
    otp_dir = f'animacy_annotated_uds_ms_xl/{lang}'
    if not os.path.exists(otp_dir):
        os.makedirs(otp_dir)

    with open(os.path.join(otp_dir, ud_name), 'w', encoding='utf-8') as otp:
        otp.write(ud.conll())
        print(f'Processed {ud_name}')


def write_and_save_ud(file, ud, lang, set_split, output_dir):
    ud_name = os.path.basename(file)
    lang_dir = os.path.join(output_dir, lang)
    os.makedirs(lang_dir, exist_ok=True)

    with open(os.path.join(lang_dir, ud_name), 'w', encoding='utf-8') as f:
        f.write(ud.conll())
        print(f'Saved: {ud_name}')


def process_lang(files, output_dir, langs, model, tokenizer):
    for file in files:
        lang = os.path.basename(file).split('_')[0]
        if lang in langs:
            set_split = re.split(r'[-_.]', os.path.basename(file))[-2]
            ud = annotate(file, lang, set_split)
            write_and_save_ud(file, ud, lang, set_split, output_dir)


def main():
    parser = argparse.ArgumentParser('Animacy annotation of UD.')
    parser.add_argument('--input_dir', type=str, required=True, help="Path to the input UD directory")
    args = parser.parse_args()

    inp_path = args.input_dir

    # Define the language adjectives
    language_adjectives = ['English', 'Chinese', 'German']

    lang_names = set()
    file_dict = dict()
    dirs = os.listdir(inp_path)

    # Identify and collect directories corresponding to languages
    for d in dirs:
        lang_name = re.split(r'_|-', d)
        lang_name = '_'.join(lang_name[1:-1])
        if lang_name in language_adjectives:  # Match against language adjectives
            lang_names.add(lang_name)

    # Map adjectives to language codes
    langs = ['en','zh', 'de']

    language_adjective_code = {
        'English': 'en',
        'Chinese': 'zh',
        'German': 'de'
    }

    for lang_name in sorted(list(lang_names)):
        files_list = list()
        lang_directories = list()

        # Match directory with the language name
        for x in os.listdir(inp_path):
            d = os.path.basename(x)
            if lang_name in d:
                lang_directories.append(os.path.join(inp_path, d))

        # Collect all .conllu files within matched directories
        for directory in lang_directories:
            files = os.listdir(directory)
            for file in files:
                if file.endswith('.conllu'):
                    files_list.append(os.path.join(directory, file))

        # Map language names to their corresponding file lists
        file_dict[lang_name] = files_list

    OUTPUT_DIRECTORY = 'animacy_annotated_bert'

    # Process files for each language
    for language, files in file_dict.items():
        model, tokenizer = compose_sft(language_adjective_code[language])
        print(f'Processing {language} UD treebank.')
        process_lang(files, OUTPUT_DIRECTORY, langs, model, tokenizer)


if __name__ == '__main__':
    main()



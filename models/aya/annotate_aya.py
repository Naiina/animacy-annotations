import torch
from tqdm import tqdm
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm
import logging
import pyconll
import os
import re
import argparse
from utils import ADJ_MAP_LARGE, get_prompt
import logging
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

dist.init_process_group(backend='nccl')

# Set local rank for the GPU
local_rank = int(os.environ['LOCAL_RANK'], 0)
torch.cuda.set_device(local_rank)


def find_delete(sentence):
    lack_VP = True
    verbs = []
    for token in sentence:
        if token.upos == 'VERB' or token.upos == 'AUX':
            verbs.append(token)

        if token.form is None:
            return True

    if verbs:
        lack_VP = False

    return lack_VP


def setup_model_and_tokenizer(MODEL_NAME, ADAPTER_NAME):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME,
                                                 quantization_config=quantization_config,
                                                 torch_dtype=torch.bfloat16,
                                                 device_map={"": "cuda:" + str(local_rank)})

    model.load_adapter(ADAPTER_NAME, adapter_name=ADAPTER_NAME) #adapter saved locally
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def generate_aya_original(prompts, model, tokenizer, temperature=0.1, top_p=1.0, top_k=0, max_new_tokens=10):
    def get_message_format(prompts):
        messages = []
        for p in prompts:
            messages.append(
                [{"role": "user", "content": p}]
            )
        return messages

    # print(len(prompts))
    messages = get_message_format(prompts)
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        padding=True,
        return_tensors="pt",
    )

    input_ids = input_ids.to(model.device)
    prompt_padded_len = len(input_ids[0])

    gen_tokens = model.generate(
        input_ids,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_new_tokens,
        do_sample=True,
    )
    gen_tokens = [
        gt[prompt_padded_len:] for gt in gen_tokens
    ]

    gen_text = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

    def get_prediction(response):
        for char in response:
            if char in ['A', 'N', 'H']:
                return char
        return None

    # print('Gen text: ', gen_text)
    predictions = [get_prediction(r) for r in gen_text]

    return predictions


def predict_in_batches(model, tokenizer, data, adj_lang, batch_size):
    tokens, target_indexes, sentences = data
    final_predictions = []

    for i, targets in enumerate(target_indexes):
        total = len(target_indexes)
        if i % 200 == 0:
            print(f'{i}/{total}')
        toks = tokens[i]
        sentence = sentences[i]
        sentence_predictions = []

        if not targets:
            final_predictions.append([])
            continue

        if len(targets) > batch_size:
            for batch_start in range(0, len(targets), batch_size):
                batched_item = targets[batch_start:batch_start + batch_size]

                batch_prompts = []
                for idx, target_index in enumerate(batched_item):
                    try:
                        target_word = toks[target_index - 1]
                        prompt = get_prompt(sentence, adj_lang, target_word)
                        batch_prompts.append(prompt)
                    except IndexError:
                        print(f"Error: Index {target_index} out of bounds in batched item. Skipping.")

                if not batch_prompts:
                    continue

                batch_predictions = generate_aya_original(batch_prompts, model, tokenizer)
                sentence_predictions.extend(batch_predictions)
        else:
            batch_prompts = []
            for idx, target_index in enumerate(targets):
                try:
                    target_word = toks[target_index - 1]
                    prompt = get_prompt(sentence, adj_lang, target_word)
                    batch_prompts.append(prompt)
                except IndexError:
                    print(f"Error: Index {target_index} out of bounds in batched item. Skipping.")

            if not batch_prompts:
                continue  # Skip if no valid prompts

            batch_predictions = generate_aya_original(batch_prompts, model, tokenizer)
            sentence_predictions.extend(batch_predictions)

        final_predictions.append(sentence_predictions)
    return final_predictions


def annotate_in_batches(file_name, lang, set_split, model, tokenizer):
    ud = pyconll.load_from_file(file_name)

    to_delete = []

    for n, i in enumerate(ud._sentences):
        if find_delete(i) is True:
            to_delete.append(n)

    to_delete.reverse()
    for n in to_delete:
        ud.__delitem__(n)

    data = build_data(ud)

    adj_lang = ADJ_MAP_LARGE[lang]

    batch_size = 16
    prediction = predict_in_batches(model, tokenizer, data, adj_lang, batch_size=batch_size)
    for i, s in enumerate(ud._sentences):
        nouns = [token for token in s if token.upos == 'NOUN']
        if not nouns:
            continue
        if not prediction:
            continue
        for token, pred in zip(nouns, prediction[i]):
            token.misc['ANIMACY'] = pred

    return file_name, ud


def build_data(treebank):
    def find_n_idxs(sentence):
        noun_indices = []
        for idx, token in enumerate(sentence):
            if token.upos == 'NOUN':
                noun_indices.append(idx + 1)
        return noun_indices

    treebank = treebank
    sentences = [sentence.text for sentence in treebank]
    tokens = [[token.form for token in sentence] for sentence in treebank]
    target_indexes = [find_n_idxs(sentence) for sentence in treebank]

    return tokens, target_indexes, sentences


def write_and_save_ud(ud, lang, set_split, file_name):
    ud_name = file_name
    otp_dir = f'animacy_annotated_aya/{lang}'
    if not os.path.exists(otp_dir):
        os.makedirs(otp_dir)

    with open(os.path.join(otp_dir, ud_name), 'w', encoding='utf-8') as otp:
        otp.write(ud.conll())
        print(f'Processed {ud_name}')


def process_lang(files, output_dir, langs, model, tokenizer):
    for f in files:
        if f.endswith('.conllu'):
            lang = os.path.basename(f).split('_')[0]
        if lang in langs:
            print('>', f)
            set_split = re.split(r'[-_.]', f)[-2]
            file_name, ud = annotate_in_batches(f, lang, set_split, model, tokenizer)
            write_and_save_ud(ud, lang, set_split, os.path.basename(file_name))

def main():
    PARSER = argparse.ArgumentParser('Animacy annotation of UD.')
    PARSER.add_argument('--input_file', type=str, required=True, help="Path to the input UD directory. See https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-5787")

    args = PARSER.parse_args()

    inp_path = args.input_file

    language_names = ['Spanish', 'Italian', 'Slovenian'] #eg.
    langs = ['es', 'it', 'sl'] #example

    lang_names = set()
    file_dict = dict()
    dirs = os.listdir(inp_path)
    MODEL_NAME = "CohereForAI/aya-expanse-8b"
    ADAPTER_NAME = 'aya-animacy'
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, ADAPTER_NAME)
    for d in dirs:
        lang_name = re.split(r'_|-', d)
        lang_name = '_'.join(lang_name[1:-1])
        if lang_name in language_names:
            lang_names.add(lang_name)



    for lang_name in sorted(list(lang_names)):
        files_list = list()

        lang_directories = list()
        for x in os.listdir(inp_path):
            d = os.path.basename(x)
            if lang_name in d:
                lang_directories.append(os.path.join(inp_path, d))

        for directory in lang_directories:
            files = os.listdir(directory)
            for file in files:
                if file.endswith('.conllu'):
                    files_list.append(os.path.join(directory, file))

        file_dict[lang_name] = files_list

    OUTPUT_DIRECTORY = 'animacy_annotated_aya'

    for language, files in file_dict.items():
        print(f'Processing {language} UD treebank.')
        process_lang(files, OUTPUT_DIRECTORY, langs, model, tokenizer)


if __name__ == '__main__':
    main()

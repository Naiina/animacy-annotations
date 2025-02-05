import os
import argparse
import jsonlines
from collections import Counter
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import logging
import torch
import torch.distributed as dist

dist.init_process_group(backend='nccl')

local_rank = int(os.environ['LOCAL_RANK'], 0)
torch.cuda.set_device(local_rank)

from utils import build_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_aya_original(prompts, model, tokenizer, batch_size=4, temperature=0.1, top_p = 1.0, top_k=0, max_new_tokens=10):
    def get_message_format(prompts):
        messages = []
        for p in prompts:
            messages.append(
             [{"role": "user", "content": p}]
            )
        return messages

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
    def find_first_letter(response):
        for char in response:
            if char in ['A', 'N', 'H']:
                return char
        return None

    predictions = [find_first_letter(r) for r in gen_text]

    return predictions

def evaluate_model(model, tokenizer, lang, test_dataset, data_dir,  batch_size=32):
    logging.info(f"Evaluating model for language: {lang}")
    try:
        test_inputs = test_dataset['inputs']
        test_targets = test_dataset['targets']

        predictions = []
        for i in tqdm(range(0, len(test_inputs), batch_size), desc="Evaluating"):
            batch_inputs = test_inputs[i:i+batch_size]
            batch_predictions = generate_aya_original(batch_inputs, model, tokenizer, batch_size)
            predictions.extend(batch_predictions)

        #with open(os.path.join(data_dir, f'{lang}-prediction.txt'), 'w', encoding='utf-8') as o:
        #    for i, (input_text, prediction, target) in enumerate(zip(test_inputs, predictions, test_targets)):
        #        if prediction != target:
        #            o.write(f"Incorrect prediction for input: {input_text}\n")
        #            o.write(f"Predicted: {prediction}, Actual: {target}\n")
        #            o.write('______________________\n')

        accuracy = accuracy_score(test_targets, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(test_targets, predictions, average='macro', zero_division='warn')

        results_per_label  = precision_recall_fscore_support(test_targets, predictions, average=None, zero_division='warn')


        logging.info(f"Accuracy,: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        return accuracy, precision, recall, f1, results_per_label

    except Exception as e:
        logging.error(f"Error in evaluate_model_batch: {str(e)}")
        return 0, 0, 0, 0


def load_datasets_test(data_dir, langs):
    datasets_dict={}
    for l in langs:
        datasets = {}
        file_path = os.path.join(data_dir, l, f"animacy_{l}_test.jsonl")
        if os.path.exists(file_path):
            data = build_data(file_path)
            datasets['test'] = Dataset.from_dict(
                {"inputs": [item['inputs'] for item in data],
                 "targets": [item['targets'] for item in data]}
            )
        datasets_dict[l] = DatasetDict(datasets)

    return datasets_dict

def setup_model_and_tokenizer(MODEL_NAME, ADAPTER_NAME):

    USE_GPU = True
    USE_FLASH_ATTENTION=True
    if USE_GPU:
        device = "cuda:0"
    else:
        device = "cpu"

    QUANTIZE_4BIT = True

    quantization_config = None
    if QUANTIZE_4BIT:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    attn_implementation = None
    if USE_FLASH_ATTENTION:
        attn_implementation = "flash_attention_2"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=quantization_config,
        attn_implementation=attn_implementation,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda:" + str(local_rank)},
        )

    # Load adapter
    model.load_adapter(ADAPTER_NAME, adapter_name=ADAPTER_NAME)
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    return model, tokenizer


def main(args):
    MODEL_NAME = "CohereForAI/aya-expanse-8b"
    ADAPTER_NAME = args.adapter
    data_dir = args.dataset_name
    langs = args.languages

    datasets_dict = load_datasets_test(data_dir, langs)
    model, tokenizer = setup_model_and_tokenizer(MODEL_NAME, ADAPTER_NAME)

    results_dir = f'Results-{ADAPTER_NAME}'
    os.makedirs(results_dir, exist_ok=True)



    for lang, data in datasets_dict.items():
        lang_dir = os.path.join(results_dir, lang)
        os.makedirs(lang_dir, exist_ok=True)
        results_dir = f'Results-{ADAPTER_NAME}/{lang}'
        os.makedirs(results_dir, exist_ok=True)
        logging.info("Evaluating the model...")
        accuracy, precision, recall, f1, per_label_metrics = evaluate_model(model, tokenizer, lang, data['test'], results_dir, batch_size=32)
        report = {}
        report["per_label_metrics"] = {
                "precision": per_label_metrics[0],
                "recall": per_label_metrics[1],
                "f1-score": per_label_metrics[2],
            }

        metrics_per_label = report["per_label_metrics"]

        LABELS = ["A", "N", "H"]

        results_dir = f'Results-{ADAPTER_NAME}/{lang}'
        os.makedirs(results_dir, exist_ok=True)

        with open(f'{results_dir}/{lang}_all.txt', 'w') as f:
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n")

        with open(f'{results_dir}/{lang}_per_label_results_all.txt', 'w') as f:
            f.write(f"{'Label':<10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}\n")
            f.write("-" * 50 + "\n")

            # Write values for each label
            for i, label in enumerate(LABELS):
                precision = metrics_per_label['precision'][i]
                recall = metrics_per_label['recall'][i]
                f1_score = metrics_per_label['f1-score'][i]
                f.write(f"{label:<10} {precision:>10.4f} {recall:>10.4f} {f1_score:>10.4f}\n")

            f.write("-" * 50 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the aya-23 model with LoRA adapter on different languages.")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--n_sample_per_label", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--languages", required=True, nargs='+' )
    args = parser.parse_args()
    main(args)
    torch.distributed.destroy_process_group()
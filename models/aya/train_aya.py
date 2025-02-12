import os
import random
import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training, get_peft_model
from trl import SFTTrainer
from datasets import Dataset, DatasetDict
import jsonlines
import logging
import torch
import torch.distributed as dist
from utils import build_data

dist.init_process_group(backend='nccl')
local_rank = int(os.environ.get('LOCAL_RANK') or 0)
torch.cuda.set_device(local_rank)

def formatting_prompts_func(example):
    output_texts = []
    for i in range(len(example['inputs'])):
        text = f"<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{example['inputs'][i]}<|END_OF_TURN_TOKEN|><|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{example['targets'][i]}"
        output_texts.append(text)
    return output_texts


def main(args):
    ADAPTER_NAME = args.adapter
    MODEL_NAME = "CohereForAI/aya-expanse-8b"
    data_dir = args.dataset_name
    langs = args.languages

    QUANTIZE_4BIT = True
    USE_GRAD_CHECKPOINTING = False
    TRAIN_BATCH_SIZE = 6
    GRAD_ACC_STEPS = 8
    USE_FLASH_ATTENTION = True

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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = 'right'

    combined_train_data = []
    combined_eval_data = []
    combined_test_data = []

    for lang in langs:
        lang_train_data, lang_eval_data, lang_test_data = [], [], []

        for f in os.listdir(os.path.join(data_dir, lang)):
            file_path = os.path.join(data_dir, lang, f)
            if 'train' in f:
                lang_train_data = build_data(file_path, args.n_sample_per_label)
            elif 'dev' in f:
                lang_eval_data = build_data(file_path, args.n_sample_per_label)
            elif 'test' in f:
                lang_test_data = build_data(file_path, args.n_sample_per_label)

        print('Train', lang, '-', len(lang_train_data))
        print('Test', lang, '-', len(lang_test_data))
        print('Eval', lang, '-', len(lang_eval_data))

        combined_train_data.extend(lang_train_data)
        combined_eval_data.extend(lang_eval_data)
        combined_test_data.extend(lang_test_data)

    random.shuffle(combined_train_data)
    random.shuffle(combined_eval_data)
    random.shuffle(combined_test_data)

    train_dataset = Dataset.from_dict(
        {"inputs": [item['inputs'] for item in combined_train_data],
         "targets": [item['targets'] for item in combined_train_data]})
    eval_dataset = Dataset.from_dict(
        {"inputs": [item['inputs'] for item in combined_eval_data],
         "targets": [item['targets'] for item in combined_eval_data]})
    test_dataset = Dataset.from_dict(
        {"inputs": [item['inputs'] for item in combined_test_data],
         "targets": [item['targets'] for item in combined_test_data]})

    datasets = DatasetDict({
        "train": train_dataset,
        "eval": eval_dataset,
        "test": test_dataset
    })

    training_arguments = TrainingArguments(
        output_dir=f"checkpoints/checkpoints-{ADAPTER_NAME}",
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACC_STEPS,
        gradient_checkpointing=USE_GRAD_CHECKPOINTING,
        ddp_find_unused_parameters=False,
        optim="paged_adamw_32bit",
        save_steps=1000,
        logging_steps=500,
        evaluation_strategy="steps",  # Enable evaluation
        eval_steps=500,
        learning_rate=3e-5,
        weight_decay=0.001,
        fp16=False,
        bf16=True,
        warmup_ratio=0.05,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="none",
    )

    peft_config = LoraConfig(
        lora_alpha=64,
        r=32,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=datasets['train'],
        eval_dataset=datasets['eval'],
        peft_config=peft_config,
        tokenizer=tokenizer,
        max_seq_length=512,
        args=training_arguments,
        formatting_func=formatting_prompts_func
    )

    trainer.train()
    os.makedirs(ADAPTER_NAME, exist_ok=True)
    trainer.model.save_pretrained(save_directory=ADAPTER_NAME)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training the aya model with LoRA adapter on different languages.")
    parser.add_argument("--adapter", type=str, required=True)
    parser.add_argument("--n_sample_per_label", type=int, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--languages", required=True, nargs='+')
    args = parser.parse_args()
    main(args)
    torch.distributed.destroy_process_group()
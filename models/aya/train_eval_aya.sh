#!/bin/bash

ADAPTER='ayanimacy_lora'
DATASET='aya_dataset'

if [ ! -d "$DATASET" ]; then
    echo "Directory '$DATASET' not found. Running aya_dataset_download.py..."
    python aya_dataset_download.py
fi


export OMP_NUM_THREADS=8
declare -a langs=('it' 'en' 'es' 'fr' 'de' 'ja' 'ko' 'nl' 'sl' 'zh' )

torchrun --nnodes=1 --nproc-per-node=1 aya_train.py --adapter "${ADAPTER}" --n_sample_per_label -1 --dataset_name "${DATASET}" --languages "${langs[@]}"
torchrun --nnodes=1 --nproc-per-node=1 aya_eval.py --adapter "${ADAPTER}" --n_sample_per_label -1 --dataset_name "${DATASET}" --languages "${langs[@]}"
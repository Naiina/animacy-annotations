#!/bin/bash


INPUT_FILE="ud-treebanks-v2.15" #path to .conll file or to directory containing .conll files

torchrun --nnodes=1 --nproc-per-node=1  annotate_mbert.py --input_dir "$INPUT_FILE"



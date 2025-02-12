#!/bin/bash

export OMP_NUM_THREADS=12

torchrun --nnodes=1 --nproc-per-node=1  train_mbert.py \
	--model_name_or_path bert-base-multilingual-cased \
  	--multisource_data animacy_multisource.json \
  	--output_dir models/***model_name*** \
  	--do_train \
  	--do_eval \
	  --label_column_name anim_tags \
  	--per_device_train_batch_size 8 \
  	--per_device_eval_batch_size 8 \
  	--task_name animacy \
  	--overwrite_output_dir \
  	--full_ft_max_epochs_per_iteration 3 \
  	--sparse_ft_max_epochs_per_iteration 10 \
  	--save_steps 1000000 \
  	--ft_params_num 14155776 \
  	--evaluation_strategy steps \
  	--eval_steps 1000000 \
  	--freeze_layer_norm \
  	--learning_rate 5e-5 \
  	--metric_for_best_model eval_f1 \
  	--load_best_model_at_end \
  	--eval_split validation \
  	--save_total_limit 2
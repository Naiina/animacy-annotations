##!/bin/bash

declare -a langs=( 'de' 'es' 'et' 'eu' 'fr' 'ja' 'ko' 'zh' 'en')

for LANG in "${langs[@]}"
do
  LANG=${LANG}
  LANG_FT=cambridgeltl/mbert-lang-sft-${LANG}-small
  TASK_FT=lingvenvist/mbert-animacy

  python3 train_mbert.py \
    --model_name_or_path bert-base-multilingual-cased \
    --dataset_name lingvenvist/animacy-${LANG}-nogroups-xtr-complete-filtered \
    --output_dir RESULTS/mbert-animacy/${LANG} \
    --lang_ft $LANG_FT \
    --task_ft $TASK_FT \
    --eval_metric f1 \
    --do_predict \
    --label_column_name anim_tags \
    --per_device_eval_batch_size 8 \
    --task_name anim \
    --overwrite_output_dir \
    --eval_split test

  echo 'Evaluation completed ${LANG}'
done

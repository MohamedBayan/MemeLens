#!/bin/bash

# Base directory for English classification datasets
DATA_DIR="data/ms_swift_formated/explanation/classify_then_explain"

# Datasets to exclude from training
EXCLUDE_DATASETS=(
    "emotion_ro__RoMemes"
    "overall_sentiment_en__memotion"
    "sentiment_bn__BanglaAbuseMeme"
    "sentiment_ro__RoMemes"
    "sentiment_category_en__MET_Meme"
    "sentiment_category_zh__MET_Meme"
    "sentiment_degree_en__MET_Meme"
    "sentiment_degree_zh__MET_Meme"
)

# Function to check if a dataset should be excluded
should_exclude() {
    local dataset_name="$1"
    for excluded in "${EXCLUDE_DATASETS[@]}"; do
        if [[ "$dataset_name" == "$excluded" ]]; then
            return 0  # true, should exclude
        fi
    done
    return 1  # false, should not exclude
}

# Dynamically build training dataset paths (train.jsonl from each dataset)
TRAIN_DATASETS=()
for dataset_dir in "$DATA_DIR"/*/; do
    dataset_name=$(basename "$dataset_dir")
    if should_exclude "$dataset_name"; then
        echo "Excluding dataset: $dataset_name"
        continue
    fi
    train_file="${dataset_dir}train.jsonl"
    if [[ -f "$train_file" ]]; then
        TRAIN_DATASETS+=("$train_file")
    fi
done

# Dynamically build validation dataset paths (val.jsonl from each dataset)
VAL_DATASETS=()
for dataset_dir in "$DATA_DIR"/*/; do
    dataset_name=$(basename "$dataset_dir")
    if should_exclude "$dataset_name"; then
        continue
    fi
    val_file="${dataset_dir}val.jsonl"
    if [[ -f "$val_file" ]]; then
        VAL_DATASETS+=("$val_file")
    fi
done

# Print datasets for verification
echo "Training datasets (${#TRAIN_DATASETS[@]} files):"
printf '%s\n' "${TRAIN_DATASETS[@]}"
echo ""
echo "Validation datasets (${#VAL_DATASETS[@]} files):"
printf '%s\n' "${VAL_DATASETS[@]}"
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29501 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
swift sft \
    --model trained_models/multimodal/classification/english_filtered/v0-20251229-101027/checkpoint-22258-merged \
    --dataset "${TRAIN_DATASETS[@]}" \
    --val_dataset "${VAL_DATASETS[@]}" \
    --torch_dtype bfloat16 \
    --train_type lora \
    --num_train_epochs 6 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --save_total_limit 30 \
    --load_from_cache_file true \
    --learning_rate 1e-5 \
    --lora_rank 16 \
    --lora_alpha 32 \
    --output_dir ./mmultimodal/explanation/multi-stage_classify_then_explain_english_filtered \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --max_length 4096 \
    --report_to tensorboard  \
    --logging_steps 5 \
    --dataset_shuffle true \
    --use_hf true \
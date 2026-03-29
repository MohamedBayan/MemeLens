#!/bin/bash

# Consolidated dataset paths
TRAIN_DATASET="data/ms_swift_formated/explanation/classify_then_explain_consolidated/train.jsonl"
VAL_DATASET="data/ms_swift_formated/explanation/classify_then_explain_consolidated/val.jsonl"

# Check if consolidated datasets exist
if [[ ! -f "$TRAIN_DATASET" ]]; then
    echo "Error: Training dataset not found: $TRAIN_DATASET"
    echo "Please run: bash scripts/train/multimodal/explanation/prepare_consolidated_dataset.sh"
    exit 1
fi

if [[ ! -f "$VAL_DATASET" ]]; then
    echo "Error: Validation dataset not found: $VAL_DATASET"
    echo "Please run: bash scripts/train/multimodal/explanation/prepare_consolidated_dataset.sh"
    exit 1
fi

# Print dataset info
echo "Training dataset: $TRAIN_DATASET ($(wc -l < "$TRAIN_DATASET") examples)"
echo "Validation dataset: $VAL_DATASET ($(wc -l < "$VAL_DATASET") examples)"
echo ""

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MASTER_PORT=29501 \
PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True' \
MAX_PIXELS=1003520 \
swift sft \
    --model mmultimodal/explanation/multi-stage_classify_then_explain_english_filtered/v4-20251231-103645/checkpoint-64650-merged \
    --dataset "$TRAIN_DATASET" \
    --val_dataset "$VAL_DATASET" \
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


# trained_models/multimodal/classification/english_filtered/v0-20251229-101027/checkpoint-22258-merged  6 epochs on this before the current checkpoint
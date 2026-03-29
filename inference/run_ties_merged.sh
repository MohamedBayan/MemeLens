#!/bin/bash

# =============================================================================
# Inference Script for Multimodal Classification (All Datasets)
# =============================================================================

# Configuration
DATA_DIR="./data/ms_swift_formated/explanation/classify_then_explain/"
MODEL_CHECKPOINT="./trained_models/merged_adapter_cls_ce_ties_equal_weights"
MERGED_MODEL="./trained_models/full_merged_adapter_cls_ce_ties_equal_weights"
RESULTS_DIR="./results/multimodal/explanation/english_filtered-classify_then_explain-ties_merged"

# Create results directory if it doesn't exist
mkdir -p "$RESULTS_DIR"

# =============================================================================
# Step 1: Merge LoRA weights (only if not already merged)
# =============================================================================
if [[ ! -d "$MERGED_MODEL" ]]; then
    echo "Merging LoRA weights..."
    bash scripts/merge_lora.sh "$MODEL_CHECKPOINT" "$MERGED_MODEL"
else
    echo "Merged model already exists at: $MERGED_MODEL"
fi

# =============================================================================
# Step 2: Collect all test datasets
# =============================================================================
TEST_DATASETS=()
DATASET_NAMES=()

for dataset_dir in "$DATA_DIR"/*/; do
    test_file="${dataset_dir}test.jsonl"
    if [[ -f "$test_file" ]]; then
        TEST_DATASETS+=("$test_file")
        # Extract dataset name from directory path
        dataset_name=$(basename "$dataset_dir")
        DATASET_NAMES+=("$dataset_name")
    fi
done

echo "=============================================="
echo "Found ${#TEST_DATASETS[@]} test datasets:"
echo "=============================================="
for i in "${!DATASET_NAMES[@]}"; do
    echo "  [$i] ${DATASET_NAMES[$i]}: ${TEST_DATASETS[$i]}"
done
echo ""

# =============================================================================
# Step 3: Run inference for each dataset
# =============================================================================
for i in "${!TEST_DATASETS[@]}"; do
    test_file="${TEST_DATASETS[$i]}"
    dataset_name="${DATASET_NAMES[$i]}"
    result_file="${RESULTS_DIR}/Qwen3-VL-8B-${dataset_name}.jsonl"
    
    echo "============================================="
    echo "Processing dataset: $dataset_name"
    echo "  Input:  $test_file"
    echo "  Output: $result_file"
    echo "============================================="
    
    # Skip if result already exists and is non-empty
    if [[ -f "$result_file" && -s "$result_file" ]]; then
        echo "  [SKIP] Result file already exists and is non-empty. Skipping..."
        continue
    fi
    
    CUDA_VISIBLE_DEVICES=0,1,2,3 \
    swift infer \
        --model "$MERGED_MODEL" \
        --infer_backend vllm \
        --val_dataset "$test_file" \
        --gpu_memory_utilization 0.95 \
        --max_new_tokens 10 \
        --use_hf true \
        --max_batch_size 16 \
        --temperature 0 \
        --result_path "$result_file" \
        --load_data_args false
    
    echo "  [DONE] Completed inference for $dataset_name"
    echo ""
done

echo "=============================================="
echo "All inference jobs completed!"
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="

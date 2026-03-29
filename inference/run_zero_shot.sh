


#!/bin/bash

# =============================================================================
# Zero-Shot Inference Script for Multimodal Classification (All Datasets)
# =============================================================================

# Configuration
DATA_DIR="data/ms_swift_formated/classification/english"
RESULTS_BASE_DIR="results/multimodal/classification/zero-shot"

# List of pretrained models to evaluate
MODELS=(
    "Qwen/Qwen3-VL-8B-Thinking"
)
# MODELS=(
#     "Qwen/Qwen3-VL-8B-Instruct"
#     "Qwen/Qwen3-VL-2B-Instruct"
#     "google/gemma-3-12b-it"
#     "mistralai/Ministral-3-8B-Instruct-2512"
#     "allenai/MolmoE-1B-0924"
#     "microsoft/Phi-3.5-vision-instruct"
#     "OpenGVLab/InternVL3_5-8B"
#     "meta-llama/Llama-3.2-11B-Vision-Instruct"
#     "SAIL-VL2-8"

# )

# =============================================================================
# Step 1: Collect all test datasets
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

echo "=============================================="
echo "Models to evaluate (${#MODELS[@]}):"
echo "=============================================="
for model in "${MODELS[@]}"; do
    echo "  - $model"
done
echo ""

# =============================================================================
# Step 2: Run inference for each model and each dataset
# =============================================================================
for model in "${MODELS[@]}"; do
    # Extract model short name for folder/file naming
    # Handle both HuggingFace paths (org/model) and local paths
    if [[ "$model" == /* ]]; then
        # Local path - extract the last component
        model_short_name=$(basename "$model")
    else
        # HuggingFace path - extract model name after /
        model_short_name=$(echo "$model" | sed 's|.*/||')
    fi
    
    # Create model-specific results directory
    MODEL_RESULTS_DIR="${RESULTS_BASE_DIR}/${model_short_name}"
    mkdir -p "$MODEL_RESULTS_DIR"
    
    echo "##############################################"
    echo "# MODEL: $model"
    echo "# Short name: $model_short_name"
    echo "# Results dir: $MODEL_RESULTS_DIR"
    echo "##############################################"
    echo ""
    
    for i in "${!TEST_DATASETS[@]}"; do
        test_file="${TEST_DATASETS[$i]}"
        dataset_name="${DATASET_NAMES[$i]}"
        result_file="${MODEL_RESULTS_DIR}/${dataset_name}.jsonl"
        
        echo "=============================================="
        echo "Processing: $model_short_name / $dataset_name"
        echo "  Input:  $test_file"
        echo "  Output: $result_file"
        echo "=============================================="
        
        # Skip if result already exists (optional - remove if you want to overwrite)
        if [[ -f "$result_file" ]]; then
            echo "  [SKIP] Result file already exists. Skipping..."
            continue
        fi
        
        CUDA_VISIBLE_DEVICES=0,1,2,3 \
        swift infer \
            --model "$model" \
            --infer_backend vllm \
            --val_dataset "$test_file" \
            --gpu_memory_utilization 0.95 \
            --max_new_tokens 4096 \
            --use_hf true \
            --max_batch_size 16 \
            --temperature 0 \
            --result_path "$result_file" \
            --load_data_args false \
            --response_prefix "Label: "
        
        echo "  [DONE] Completed inference for $dataset_name with $model_short_name"
        echo ""
    done
    
    echo ""
done

echo "##############################################"
echo "# All zero-shot inference jobs completed!"
echo "# Results saved to: $RESULTS_BASE_DIR"
echo "##############################################"

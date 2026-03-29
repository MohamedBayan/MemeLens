#!/bin/bash
# ============================================================================
# merge_adapters.sh - Example script for merging PEFT adapters
# ============================================================================
#
# This script demonstrates how to use the merge_adapters.py script to merge
# multiple PEFT adapters (e.g., LoRA) using various algorithms.
#
# Usage:
#   ./scripts/merge_adapters.sh
#
# Or with SLURM:
#   sbatch scripts/merge_adapters.sh
#
# ============================================================================

set -e

# ============================================================================
# Configuration - Modify these variables for your use case
# ============================================================================

# Base model that the adapters were trained on
BASE_MODEL="Qwen/Qwen3-VL-8B-Instruct"

# Paths to adapters to merge (add more as needed)
ADAPTER_1="./multimodal/classification/english_filtered/v0-20251229-101027/checkpoint-22258"
ADAPTER_2="./mmultimodal/explanation/multi-stage_classify_then_explain_english_filtered/v4-20251231-103645/checkpoint-64650"

# Output path for the merged adapter
OUTPUT_PATH="./trained_models/merged_adapter_cls_ce_ties_equal_weights"

# Merging method: ties, dare_ties, dare_linear, magnitude_prune, linear
# See ranking below for guidance
METHOD="ties"

# Weights for each adapter (optional, defaults to 1.0 for all)
WEIGHT_1=1.0
WEIGHT_2=1.0

# Density: fraction of weights to retain (0-1)
# Recommended: 0.2-0.3 for most cases
DENSITY=0.2

# Majority sign method: total (sums values) or frequency (sums signs)
MAJORITY_SIGN="total"

# Additional options
MODEL_TYPE="auto"  # auto, causal_lm, vision2seq, seq2seq, base (use auto for VL models)
TORCH_DTYPE="auto"  # auto, float16, bfloat16, float32
SAVE_FULL_MODEL=false  # Set to true to save base + adapter merged
TRUST_REMOTE_CODE=true  # Required for Qwen and some other models

# ============================================================================
# Merging Methods Reference (ranked by effectiveness)
# ============================================================================
# 🥇 ties / ties_svd        - Best overall, handles noise and sign conflicts
# 🥈 dare_ties / dare_ties_svd - Noise-resilient + interference handling
# 🥉 dare_linear            - Noise-aware weighted merge
# 4. magnitude_prune        - Lightweight noise drop + linear merge
# 5. linear                 - Basic weighted sum (baseline)
# ============================================================================

echo "=============================================="
echo "PEFT Adapter Merging"
echo "=============================================="
echo "Base Model: ${BASE_MODEL}"
echo "Adapters: ${ADAPTER_1}, ${ADAPTER_2}"
echo "Method: ${METHOD}"
echo "Density: ${DENSITY}"
echo "Output: ${OUTPUT_PATH}"
echo "=============================================="

# Build the command
CMD="python scripts/src/merge_adapters.py \
    --base_model ${BASE_MODEL} \
    --adapters ${ADAPTER_1} ${ADAPTER_2} \
    --weights ${WEIGHT_1} ${WEIGHT_2} \
    --output_path ${OUTPUT_PATH} \
    --method ${METHOD} \
    --density ${DENSITY} \
    --majority_sign_method ${MAJORITY_SIGN} \
    --model_type ${MODEL_TYPE} \
    --torch_dtype ${TORCH_DTYPE}"

# Add trust_remote_code flag if enabled
if [ "$TRUST_REMOTE_CODE" = true ]; then
    CMD="${CMD} --trust_remote_code"
fi

# Add save_full_model flag if enabled
if [ "$SAVE_FULL_MODEL" = true ]; then
    CMD="${CMD} --save_full_model"
fi

# Run the merge
echo "Running: ${CMD}"
eval ${CMD}

echo "=============================================="
echo "Merging complete!"
echo "Merged adapter saved to: ${OUTPUT_PATH}"
echo "=============================================="

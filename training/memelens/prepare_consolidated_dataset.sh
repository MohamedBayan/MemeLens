#!/bin/bash

# Script to consolidate multiple JSONL files into single train/val files
DATA_DIR="data/ms_swift_formated/explanation/classify_then_explain"
OUTPUT_DIR="data/ms_swift_formated/explanation/classify_then_explain_consolidated"

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

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Remove existing consolidated files
rm -f "$OUTPUT_DIR/train.jsonl"
rm -f "$OUTPUT_DIR/val.jsonl"

# Consolidate training datasets
echo "Consolidating training datasets..."
train_count=0
for dataset_dir in "$DATA_DIR"/*/; do
    dataset_name=$(basename "$dataset_dir")
    if should_exclude "$dataset_name"; then
        echo "  Excluding dataset: $dataset_name"
        continue
    fi
    train_file="${dataset_dir}train.jsonl"
    if [[ -f "$train_file" ]]; then
        lines=$(wc -l < "$train_file")
        echo "  Adding $dataset_name: $lines examples"
        cat "$train_file" >> "$OUTPUT_DIR/train.jsonl"
        train_count=$((train_count + 1))
    fi
done

# Consolidate validation datasets
echo ""
echo "Consolidating validation datasets..."
val_count=0
for dataset_dir in "$DATA_DIR"/*/; do
    dataset_name=$(basename "$dataset_dir")
    if should_exclude "$dataset_name"; then
        continue
    fi
    val_file="${dataset_dir}val.jsonl"
    if [[ -f "$val_file" ]]; then
        lines=$(wc -l < "$val_file")
        echo "  Adding $dataset_name: $lines examples"
        cat "$val_file" >> "$OUTPUT_DIR/val.jsonl"
        val_count=$((val_count + 1))
    fi
done

echo ""
echo "Consolidation complete!"
echo "Train datasets consolidated: $train_count"
echo "Total train examples: $(wc -l < "$OUTPUT_DIR/train.jsonl")"
echo "Val datasets consolidated: $val_count"
echo "Total val examples: $(wc -l < "$OUTPUT_DIR/val.jsonl")"
echo ""
echo "Output files:"
echo "  $OUTPUT_DIR/train.jsonl"
echo "  $OUTPUT_DIR/val.jsonl"

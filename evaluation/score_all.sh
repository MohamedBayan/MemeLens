#!/bin/bash
# Score all seq_cls results
# Usage: bash scripts/score_seq_cls.sh

RESULTS_DIR="results/multimodal/seq_cls"
SCORES_DIR="scores/seq_cls"
SCRIPT_PATH="scripts/src/compute_metrics.py"

# Create scores directory if it doesn't exist
mkdir -p "$SCORES_DIR"

# Loop through each task directory
for task_dir in "$RESULTS_DIR"/*/; do
    task_name=$(basename "$task_dir")
    
    # Loop through each jsonl file in the task directory
    for jsonl_file in "$task_dir"/*.jsonl; do
        if [[ -f "$jsonl_file" ]]; then
            model_name=$(basename "$jsonl_file" .jsonl)
            out_dir="$SCORES_DIR/$task_name/$model_name"
            
            echo "========================================"
            echo "Scoring: $task_name / $model_name"
            echo "Input: $jsonl_file"
            echo "Output: $out_dir"
            echo "========================================"
            
            mkdir -p "$out_dir"
            
            python "$SCRIPT_PATH" \
                --data "$jsonl_file" \
                --out_dir "$out_dir"
            
            echo ""
        fi
    done
done

echo "✅ All seq_cls results scored!"
echo "📊 Results saved to: $SCORES_DIR"

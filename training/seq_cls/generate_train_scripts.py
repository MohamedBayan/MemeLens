#!/usr/bin/env python3
"""
Generate training scripts for all datasets.

Creates bash scripts for training sequence classification models
for all converted datasets in data/ms_swift_formated/seq_cls/
"""

import json
from pathlib import Path
from typing import Dict


# Dataset configurations - number of labels for each dataset
DATASET_CONFIG = {
    'Harmful_Covid_en__HarMeme': {'num_labels': 3, 'epochs': 10},
    'Harmful_en__HarMeme': {'num_labels': 4, 'epochs': 10},
    'Hateful_bn__MUTE': {'num_labels': 2, 'epochs': 10},
    'Hateful_de__Multi3Hate': {'num_labels': 2, 'epochs': 15},
    'Hateful_en_FHM': {'num_labels': 2, 'epochs': 5},
    'Hateful_en__MIMIC_Islamophpbia': {'num_labels': 2, 'epochs': 15},
    'Hateful_en__Multi3Hate': {'num_labels': 2, 'epochs': 15},
    'Hateful_es__Multi3Hate': {'num_labels': 2, 'epochs': 15},
    'Hateful_hi__Multi3Hate': {'num_labels': 2, 'epochs': 15},
    'Hateful_zh__Multi3Hate': {'num_labels': 2, 'epochs': 15},
    'Misogyny_hi_en__MIMIC2024': {'num_labels': 2, 'epochs': 10},
    'Target_Covid_en__HarMeme': {'num_labels': 5, 'epochs': 10},
    'Target_en__HarMeme': {'num_labels': 5, 'epochs': 10},
    'abuse_bn__BanglaAbuseMeme': {'num_labels': 2, 'epochs': 10},
    'emotion_ro__RoMemes': {'num_labels': 6, 'epochs': 20},
    'fakenews_ro__RoMemes': {'num_labels': 3, 'epochs': 20},
    'humour_en__memotion': {'num_labels': 4, 'epochs': 10},
    'intention_detection_en__MET_Meme': {'num_labels': 5, 'epochs': 10},
    'intention_detection_zh__MET_Meme': {'num_labels': 5, 'epochs': 10},
    'metaphor_occurrence_en__MET_Meme': {'num_labels': 2, 'epochs': 10},
    'metaphor_occurrence_zh__MET_Meme': {'num_labels': 2, 'epochs': 10},
    'misogynous_en__MAMI': {'num_labels': 2, 'epochs': 5},
    'motivational_en__memotion': {'num_labels': 2, 'epochs': 10},
    'objectification_en__MAMI': {'num_labels': 2, 'epochs': 5},
    'offensive_en__memotion': {'num_labels': 4, 'epochs': 10},
    'offensiveness_detection_en__MET_Meme': {'num_labels': 4, 'epochs': 10},
    'offensiveness_detection_zh__MET_Meme': {'num_labels': 4, 'epochs': 10},
    'overall_sentiment_en__memotion': {'num_labels': 5, 'epochs': 10},
    'political_ro__RoMemes': {'num_labels': 2, 'epochs': 20},
    'propoganda_ar_ArMeme': {'num_labels': 2, 'epochs': 10},
    'sarcasm_bn__BanglaAbuseMeme': {'num_labels': 2, 'epochs': 10},
    'sarcasm_en__memotion': {'num_labels': 4, 'epochs': 10},
    'sentiment_bn__BanglaAbuseMeme': {'num_labels': 3, 'epochs': 10},
    'sentiment_category_en__MET_Meme': {'num_labels': 7, 'epochs': 10},
    'sentiment_category_zh__MET_Meme': {'num_labels': 7, 'epochs': 10},
    'sentiment_degree_en__MET_Meme': {'num_labels': 3, 'epochs': 10},
    'sentiment_degree_zh__MET_Meme': {'num_labels': 3, 'epochs': 10},
    'sentiment_ro__RoMemes': {'num_labels': 3, 'epochs': 20},
    'shaming_en__MAMI': {'num_labels': 2, 'epochs': 5},
    'stereotype_en__MAMI': {'num_labels': 2, 'epochs': 5},
    'toxic_ru__Toxic_Memes_Detection_Dataset': {'num_labels': 2, 'epochs': 10},
    'violence_en__MAMI': {'num_labels': 2, 'epochs': 5},
    'vulgar_bn__BanglaAbuseMeme': {'num_labels': 2, 'epochs': 10},
}


def generate_train_script(dataset_name: str, config: Dict, val_file: str = None) -> str:
    """
    Generate a training script for a dataset.
    
    Args:
        dataset_name: Name of the dataset
        config: Configuration dictionary with num_labels and epochs
        val_file: Name of the validation file ('dev.jsonl' or 'val.jsonl'), if exists
        
    Returns:
        Training script content as string
    """
    num_labels = config['num_labels']
    epochs = config['epochs']
    
    # Use different batch sizes based on dataset size
    batch_size = 16
    eval_batch_size = 16
    
    script = f"""#!/bin/bash
#
# Training script for {dataset_name}
# Number of labels: {num_labels}
# Training epochs: {epochs}
#

# Create output directory if it doesn't exist
mkdir -p trained_models/multimodal/seq_cls/{dataset_name}/qwen3-vl-8b-instruct

CUDA_VISIBLE_DEVICES=0 \\
MAX_PIXELS=1003520 \\
swift sft \\
    --model Qwen/Qwen3-VL-8B-Instruct \\
    --dataset "data/ms_swift_formated/seq_cls/{dataset_name}/train.jsonl" \\"""
    
    # Only add val_dataset if validation file exists
    if val_file:
        script += f"""
    --val_dataset "data/ms_swift_formated/seq_cls/{dataset_name}/{val_file}" \\"""
    
    script += f"""
    --torch_dtype bfloat16 \\
    --train_type lora \\
    --num_train_epochs {epochs} \\
    --per_device_train_batch_size {batch_size} \\
    --per_device_eval_batch_size {eval_batch_size} \\
    --gradient_accumulation_steps 1 \\
    --learning_rate 1e-5 \\
    --lora_rank 16 \\
    --lora_alpha 32 \\
    --target_modules all-linear \\
    --eval_steps 100 \\
    --save_steps 100 \\
    --save_total_limit 30 \\
    --logging_steps 5 \\
    --max_length 2048 \\
    --output_dir trained_models/multimodal/seq_cls/{dataset_name}/qwen3-vl-8b-instruct \\
    --warmup_ratio 0.05 \\
    --dataloader_num_workers 4 \\
    --num_labels {num_labels} \\
    --task_type seq_cls \\
    --use_chat_template true \\
    --use_hf true
"""
    
    return script


def generate_master_train_script(datasets: list) -> str:
    """
    Generate a master script to run all training scripts.
    
    Args:
        datasets: List of dataset names
        
    Returns:
        Master script content as string
    """
    script = """#!/bin/bash
#
# Master training script - runs all dataset training scripts sequentially
#
# Usage: 
#   ./train_all.sh              # Train all datasets
#   ./train_all.sh dataset1 dataset2  # Train specific datasets
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# List of all datasets
DATASETS=(
"""
    
    for dataset in sorted(datasets):
        script += f'    "{dataset}"\n'
    
    script += """)

# If arguments provided, use them as the dataset list
if [ $# -gt 0 ]; then
    DATASETS=("$@")
fi

echo "========================================"
echo "Training ${#DATASETS[@]} datasets"
echo "========================================"
echo ""

SUCCESS=0
FAILED=0

for dataset in "${DATASETS[@]}"; do
    echo ""
    echo "========================================"
    echo "Training: $dataset"
    echo "========================================"
    
    SCRIPT="${SCRIPT_DIR}/train_${dataset}.sh"
    
    if [ ! -f "$SCRIPT" ]; then
        echo "ERROR: Script not found: $SCRIPT"
        ((FAILED++))
        continue
    fi
    
    # Make script executable
    chmod +x "$SCRIPT"
    
    # Run training script
    if bash "$SCRIPT"; then
        echo "✓ Successfully trained: $dataset"
        ((SUCCESS++))
    else
        echo "✗ Failed to train: $dataset"
        ((FAILED++))
    fi
done

echo ""
echo "========================================"
echo "Training Summary"
echo "========================================"
echo "✓ Successful: $SUCCESS"
echo "✗ Failed: $FAILED"
echo "Total: ${#DATASETS[@]}"
echo ""
"""
    
    return script


def main():
    """Generate all training scripts."""
    
    # Output directory
    output_dir = Path("scripts/train/multimodal/seq_cls")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating training scripts...")
    print(f"Output directory: {output_dir.absolute()}\n")
    
    datasets_created = []
    
    # Generate individual training scripts
    for dataset_name, config in sorted(DATASET_CONFIG.items()):
        # Check if dataset exists
        dataset_dir = Path(f"data/ms_swift_formated/seq_cls/{dataset_name}")
        if not dataset_dir.exists():
            print(f"⚠️  Skipping {dataset_name}: directory not found")
            continue
        
        # Check for validation set (dev.jsonl or val.jsonl)
        val_file = None
        if (dataset_dir / "dev.jsonl").exists():
            val_file = "dev.jsonl"
        elif (dataset_dir / "val.jsonl").exists():
            val_file = "val.jsonl"
        
        # Generate script
        script_content = generate_train_script(dataset_name, config, val_file)
        
        # Write script file
        script_path = output_dir / f"train_{dataset_name}.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
        
        datasets_created.append(dataset_name)
        val_info = f", val={val_file}" if val_file else ""
        print(f"✓ Created: train_{dataset_name}.sh (num_labels={config['num_labels']}, epochs={config['epochs']}{val_info})")
    
    # Generate master training script
    master_script = generate_master_train_script(datasets_created)
    master_script_path = output_dir / "train_all.sh"
    with open(master_script_path, 'w') as f:
        f.write(master_script)
    master_script_path.chmod(0o755)
    
    print(f"\n✓ Created master script: train_all.sh")
    
    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Training scripts created: {len(datasets_created)}")
    print(f"Output directory: {output_dir.absolute()}")
    print(f"\nTo train all datasets:")
    print(f"  cd {output_dir.absolute()}")
    print(f"  ./train_all.sh")
    print(f"\nTo train specific datasets:")
    print(f"  ./train_all.sh {datasets_created[0]} {datasets_created[1]}")
    print()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Convert all datasets to ms_swift sequence classification format.

This script processes all datasets from:
- data/normalized_classification_en/
- data/normalized_datasets/

how to run for one dataset:
python scripts/src/convert_all_datasets.py \
  --output data/ms_swift_formated/seq_cls \
  --datasets Hateful_en_FHM
Outputs to:
- data/ms_swift_formated/seq_cls/<dataset_name>/
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any, List
import sys


# Dataset label mappings: maps string labels to integer indices
LABEL_MAPPINGS = {
    # HarMeme datasets
    'Harmful_Covid_en__HarMeme': {
        'not-harmful': 0,
        'partially-harmful': 1,
        'very-harmful': 2
    },
    'Harmful_en__HarMeme': {
        'not-harmful': 0,
        'partially-harmful': 1,
        'very-harmful': 2
    },
    'Target_Covid_en__HarMeme': {
        'none': 0,
        'individual': 1,
        'organization': 2,
        'community': 3,
        'society': 4
    },
    'Target_en__HarMeme': {
        'none': 0,
        'individual': 1,
        'organization': 2,
        'community': 3,
        'society': 4
    },
    
    # Hateful content datasets
    'Hateful_ar__Prop2Hate-Meme': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_bn__MUTE': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_de__Multi3Hate': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_en_FHM': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_en__MIMIC_Islamophpbia': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_en__MMHS': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_en__Multi3Hate': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_es__Multi3Hate': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_hi__Multi3Hate': {
        'not-hateful': 0,
        'hateful': 1
    },
    'Hateful_zh__Multi3Hate': {
        'not-hateful': 0,
        'hateful': 1
    },
    
    # Misogyny datasets
    'Misogyny_hi_en__MIMIC2024': {
        'not-misogynous': 0,
        'misogynous': 1
    },
    'Misogyny_Categories_hi_en__MIMIC2024': {
        'Objectification': 0,
        'Prejudice': 1,
        'Humiliation': 2,
        'Objectification, Humiliation': 3,
        'Objectification, Prejudice': 4,
        'Prejudice, Humiliation': 5,
        'Unspecified': 6
    },
    'misogynous_en__MAMI': {
        'not-misogynous': 0,
        'misogynous': 1
    },
    'objectification_en__MAMI': {
        'not-objectification': 0,
        'objectification': 1
    },
    'shaming_en__MAMI': {
        'not-shaming': 0,
        'shaming': 1
    },
    'stereotype_en__MAMI': {
        'not-stereotype': 0,
        'stereotype': 1
    },
    'violence_en__MAMI': {
        'not-violence': 0,
        'violence': 1
    },
    
    # BanglaAbuseMeme datasets
    'abuse_bn__BanglaAbuseMeme': {
        'not-abusive': 0,
        'abusive': 1
    },
    'sarcasm_bn__BanglaAbuseMeme': {
        'not-sarcasm': 0,
        'sarcasm': 1
    },
    'sentiment_bn__BanglaAbuseMeme': {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    },
    'vulgar_bn__BanglaAbuseMeme': {
        'not-vulgar': 0,
        'vulgar': 1
    },
    
    # RoMemes datasets
    'deepfake_ro__RoMemes': {
        'Real': 0,
        'Fake': 1,
        'DeepFake': 2
    },
    'emotion_ro__RoMemes': {
        'Love': 0,
        'Fear': 1,
        'Anger': 2,
        'Joy': 3,
        'Sadness': 4,
        'Surprise': 5
    },
    'political_ro__RoMemes': {
        'not-political': 0,
        'political': 1
    },
    'sentiment_ro__RoMemes': {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    },
    
    # MET_Meme datasets (English)
    'intention_detection_en__MET_Meme': {
        'Entertaining': 0,
        'Expressive': 1,
        'Interactive': 2,
        'Offensive': 3
    },
    'metaphor_occurrence_en__MET_Meme': {
        'Literal': 0,
        'Metaphorical': 1
    },
    'offensiveness_detection_en__MET_Meme': {
        'not-offensive': 0,
        'slightly-offensive': 1,
        'moderately-offensive': 2,
        'very-offensive': 3
    },
    'sentiment_category_en__MET_Meme': {
        'Fear': 0,
        'Anger': 1,
        'Sorrow': 2,
        'Hate': 3,
        'Surprise': 4,
        'Love': 5,
        'Happiness': 6
    },
    'sentiment_degree_en__MET_Meme': {
        'slightly': 0,
        'moderately': 1,
        'very': 2
    },
    
    # MET_Meme datasets (Chinese)
    'intention_detection_zh__MET_Meme': {
        'Other': 0,
        'Entertaining': 1,
        'Expressive': 2,
        'Interactive': 3,
        'Offensive': 4
    },
    'metaphor_occurrence_zh__MET_Meme': {
        'Literal': 0,
        'Metaphorical': 1
    },
    'offensiveness_detection_zh__MET_Meme': {
        'not-offensive': 0,
        'slightly-offensive': 1,
        'moderately-offensive': 2,
        'very-offensive': 3
    },
    'sentiment_category_zh__MET_Meme': {
        'Fear': 0,
        'Anger': 1,
        'Sorrow': 2,
        'Hate': 3,
        'Surprise': 4,
        'Love': 5,
        'Happiness': 6
    },
    'sentiment_degree_zh__MET_Meme': {
        'slightly': 0,
        'moderately': 1,
        'very': 2
    },
    
    # Memotion datasets
    'humour_en__memotion': {
        'not-funny': 0,
        'funny': 1,
        'very-funny': 2,
        'hilarious': 3
    },
    'motivational_en__memotion': {
        'not-motivational': 0,
        'motivational': 1
    },
    'offensive_en__memotion': {
        'not-offensive': 0,
        'slightly-offensive': 1,
        'very-offensive': 2,
        'hateful-offensive': 3
    },
    'overall_sentiment_en__memotion': {
        'very-negative': 0,
        'negative': 1,
        'neutral': 2,
        'positive': 3,
        'very-positive': 4
    },
    'sarcasm_en__memotion': {
        'not-sarcastic': 0,
        'general-sarcasm': 1,
        'twisted-meaning': 2,
        'very-twisted': 3
    },
    'propoganda_ar_ArMeme': {
        'not-propaganda': 0,
        'propaganda': 1
    },
    'toxic_ru__Toxic_Memes_Detection_Dataset': {
        'not-toxic': 0,
        'toxic': 1
    }
}

# Multilabel datasets: maps string labels to list of integer indices
# These datasets have labels that can be combinations of multiple categories
MULTILABEL_MAPPINGS = {
    'Misogyny_Categories_hi_en__MIMIC2024': {
        # Base labels
        'Objectification': [0],
        'Prejudice': [1],
        'Humiliation': [2],
        # Combinations
        'Objectification, Humiliation': [0, 2],
        'Objectification, Prejudice': [0, 1],
        'Prejudice, Humiliation': [1, 2],
        'Objectification, Prejudice, Humiliation': [0, 1, 2],
        # Empty/Unspecified
        'Unspecified': [],
        '': []
    }
}

# Set of multilabel dataset names for quick lookup
MULTILABEL_DATASETS = set(MULTILABEL_MAPPINGS.keys())


def convert_entry(entry: Dict[str, Any], dataset_name: str) -> Dict[str, Any]:
    """
    Convert a single entry to sequence classification format.
    
    Args:
        entry: Original dataset entry
        dataset_name: Name of the dataset for label mapping
        
    Returns:
        Converted entry in sequence classification format
    """
    # Extract text from the entry
    text = entry.get("text", "").strip()
    
    # Create the user content with image placeholder and text
    user_content = f"<image> {text}" if text else "<image>"
    
    # Get class_label string
    class_label_str = entry.get("class_label", "")
    
    # Check if this is a multilabel dataset
    if dataset_name in MULTILABEL_DATASETS:
        # Get multilabel mapping for this dataset
        label_map = MULTILABEL_MAPPINGS.get(dataset_name, {})
        
        if class_label_str in label_map:
            class_label = label_map[class_label_str]
        else:
            # Default to empty list if unknown
            class_label = []
            print(f"Warning: Unknown class_label '{class_label_str}' in {dataset_name}, defaulting to []")
    else:
        # Standard multiclass handling
        label_map = LABEL_MAPPINGS.get(dataset_name, {})
        
        if class_label_str in label_map:
            class_label = label_map[class_label_str]
        else:
            # Default to 0 if unknown
            class_label = 0
            print(f"Warning: Unknown class_label '{class_label_str}' in {dataset_name}, defaulting to 0")
    
    # Get image path
    img_path = entry.get("img_path", "")
    
    # Create the new format
    converted = {
        "messages": [
            {
                "role": "user",
                "content": user_content
            }
        ],
        "label": class_label,
        "images": [img_path] if img_path else []
    }
    
    return converted


def convert_file(input_path: Path, output_path: Path, dataset_name: str) -> int:
    """
    Convert a JSONL file to sequence classification format.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        dataset_name: Name of the dataset
        
    Returns:
        Number of entries converted
    """
    converted_count = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            try:
                # Parse the input JSON line
                entry = json.loads(line.strip())
                
                # Convert to new format
                converted = convert_entry(entry, dataset_name)
                
                # Write to output file
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                converted_count += 1
                
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num} due to JSON decode error: {e}")
            except Exception as e:
                print(f"  Warning: Error processing line {line_num}: {e}")
    
    return converted_count


def convert_dataset(source_dir: Path, output_base: Path, dataset_name: str, files: List[str]) -> bool:
    """
    Convert a single dataset.
    
    Args:
        source_dir: Source dataset directory
        output_base: Base output directory
        dataset_name: Name of the dataset
        files: List of files to convert
        
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*80}")
    
    # Create output directory
    output_dir = output_base / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if dataset has a label mapping (either multiclass or multilabel)
    if dataset_name in MULTILABEL_DATASETS:
        label_map = MULTILABEL_MAPPINGS[dataset_name]
        # For multilabel, count unique base labels
        all_labels = set()
        for labels in label_map.values():
            all_labels.update(labels)
        num_labels = len(all_labels)
        print(f"  Type: MULTILABEL")
        print(f"  Num base labels: {num_labels}")
        print(f"  Label mapping: {label_map}")
    elif dataset_name in LABEL_MAPPINGS:
        num_labels = len(set(LABEL_MAPPINGS[dataset_name].values()))
        print(f"  Type: MULTICLASS")
        print(f"  Num labels: {num_labels}")
        print(f"  Label mapping: {LABEL_MAPPINGS[dataset_name]}")
    else:
        print(f"  ⚠️  Warning: No label mapping found for {dataset_name}, skipping...")
        return False
    
    # Convert each file
    success = True
    for filename in files:
        input_path = source_dir / filename
        
        if not input_path.exists():
            print(f"  ⚠️  File not found, skipping: {filename}")
            continue
        
        output_path = output_dir / filename
        
        try:
            print(f"  Converting {filename}...", end=' ')
            count = convert_file(input_path, output_path, dataset_name)
            print(f"✓ ({count} entries)")
        except Exception as e:
            print(f"✗ Error: {e}")
            success = False
    
    return success


def main():
    """Main function to convert all datasets."""
    parser = argparse.ArgumentParser(
        description='Convert all datasets to sequence classification format'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/ms_swift_formated/seq_cls',
        help='Output base directory (default: data/ms_swift_formated/seq_cls)'
    )
    parser.add_argument(
        '--files',
        type=str,
        nargs='+',
        default=['train.jsonl', 'val.jsonl', 'test.jsonl'],
        help='List of files to convert (default: train.jsonl dev.jsonl test.jsonl)'
    )
    parser.add_argument(
        '--datasets',
        type=str,
        nargs='+',
        help='Specific datasets to convert (default: all)'
    )
    
    args = parser.parse_args()
    
    # Source directories
    source_dirs = [
        Path("data/Unified_Labels_FullPath/normalized_classification_en"),
        Path("data/Unified_Labels_FullPath/normalized_datasets")
    ]
    
    # Output base directory
    output_base = Path(args.output)
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Collect all datasets
    all_datasets = []
    for source_dir in source_dirs:
        if not source_dir.exists():
            print(f"Warning: Source directory not found: {source_dir}")
            continue
        
        for dataset_dir in sorted(source_dir.iterdir()):
            if dataset_dir.is_dir():
                all_datasets.append(dataset_dir)
    
    # Filter datasets if specified
    if args.datasets:
        all_datasets = [d for d in all_datasets if d.name in args.datasets]
    
    print(f"\nFound {len(all_datasets)} datasets to process")
    print(f"Output directory: {output_base.absolute()}\n")
    
    # Process each dataset
    successful = 0
    failed = 0
    skipped = 0
    
    for dataset_dir in all_datasets:
        dataset_name = dataset_dir.name
        
        result = convert_dataset(dataset_dir, output_base, dataset_name, args.files)
        
        if result:
            successful += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("CONVERSION SUMMARY")
    print(f"{'='*80}")
    print(f"  ✓ Successful: {successful}")
    print(f"  ✗ Failed: {failed}")
    print(f"  ⊘ Skipped: {skipped}")
    print(f"  Total: {len(all_datasets)}")
    print()
    
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    exit(main())

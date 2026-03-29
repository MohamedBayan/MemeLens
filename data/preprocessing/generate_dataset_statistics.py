#!/usr/bin/env python3
"""
Generate dataset statistics CSV with task, language, labels, text info, and sample counts.

Output CSV columns:
- Task: Classification task (e.g., Hateful, Sentiment, etc.)
- Dataset: Full dataset name
- Language: Language code (en, ar, bn, etc.)
- # Labels: Number of unique labels
- Text: Whether dataset includes text (Yes/No)
- Text_Source: Source of text (OCR for all)
- # Train: Number of training samples
- # Val: Number of validation samples
- # Test: Number of test samples


# Run from repo root: python data/preprocessing/generate_dataset_statistics.py

"""

import os
import json
import csv
from pathlib import Path
from collections import defaultdict


def extract_task_and_language(dataset_name):
    """
    Extract task and language from dataset name.
    
    Format: Task_Language__Source
    Example: Hateful_en__MMHS -> Task: Hateful, Language: en
    """
    # Split by first underscore to get task
    parts = dataset_name.split('_')
    
    # Find language code (2 letter code after task)
    language = None
    task_parts = []
    
    for i, part in enumerate(parts):
        # Check if this looks like a language code
        if len(part) == 2 and part.isalpha() and i > 0:
            language = part
            task_parts = parts[:i]
            break
        # Handle special cases like "hi_en" (Hindi-English)
        elif i < len(parts) - 1 and len(part) == 2 and len(parts[i+1]) == 2:
            language = f"{part}_{parts[i+1]}"
            task_parts = parts[:i]
            break
    
    # If no language found, default to last part before __
    if not language:
        if '__' in dataset_name:
            before_source = dataset_name.split('__')[0]
            parts_before = before_source.split('_')
            if len(parts_before) > 1:
                language = parts_before[-1]
                task_parts = parts_before[:-1]
            else:
                language = 'unknown'
                task_parts = [before_source]
        else:
            language = 'unknown'
            task_parts = parts[:-1] if len(parts) > 1 else parts
    
    task = '_'.join(task_parts) if task_parts else parts[0]
    
    return task, language


def analyze_dataset(dataset_path):
    """
    Analyze a single dataset to extract statistics.
    
    Returns:
        dict with keys: unique_labels, has_text, train_count, val_count, test_count
    """
    stats = {
        'unique_labels': set(),
        'has_text': False,
        'train_count': 0,
        'val_count': 0,
        'test_count': 0
    }
    
    # Process each split
    for split_name, count_key in [('train.jsonl', 'train_count'), 
                                   ('val.jsonl', 'val_count'), 
                                   ('test.jsonl', 'test_count')]:
        split_path = os.path.join(dataset_path, split_name)
        
        if not os.path.exists(split_path):
            continue
        
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                
                # Count sample
                stats[count_key] += 1
                
                # Collect unique labels (try both 'label' and 'class_label')
                if 'label' in sample:
                    stats['unique_labels'].add(sample['label'])
                elif 'class_label' in sample:
                    stats['unique_labels'].add(sample['class_label'])
                
                # Check if text exists and is non-empty
                if 'text' in sample and sample['text'] and sample['text'].strip():
                    stats['has_text'] = True
    
    return stats


def main():
    base_path = "./data/Unified_Labels_FullPath"
    output_path = "./Processing/dataset_statistics.csv"
    
    print("=" * 80)
    print("GENERATING DATASET STATISTICS CSV")
    print("=" * 80)
    
    # Collect all dataset statistics
    all_stats = []
    
    for folder in ["normalized_classification_en", "normalized_datasets"]:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"\n⚠ Folder not found: {folder}")
            continue
        
        print(f"\nProcessing folder: {folder}")
        print("-" * 80)
        
        for dataset_name in sorted(os.listdir(folder_path)):
            dataset_path = os.path.join(folder_path, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
            
            print(f"  Analyzing: {dataset_name}")
            
            # Extract task and language
            task, language = extract_task_and_language(dataset_name)
            
            # Analyze dataset
            stats = analyze_dataset(dataset_path)
            
            # Prepare row
            row = {
                'Task': task,
                'Dataset': dataset_name,
                'Language': language,
                '# Labels': len(stats['unique_labels']),
                'Text': 'Yes' if stats['has_text'] else 'No',
                'Text_Source': 'OCR',
                '# Train': stats['train_count'],
                '# Val': stats['val_count'],
                '# Test': stats['test_count']
            }
            
            all_stats.append(row)
            
            print(f"    Task: {task}, Language: {language}, Labels: {len(stats['unique_labels'])}, "
                  f"Text: {'Yes' if stats['has_text'] else 'No'}, "
                  f"Samples: {stats['train_count']}/{stats['val_count']}/{stats['test_count']}")
    
    # Write to CSV
    print("\n" + "=" * 80)
    print(f"Writing CSV to: {output_path}")
    
    fieldnames = ['Task', 'Dataset', 'Language', '# Labels', 'Text', 'Text_Source', 
                  '# Train', '# Val', '# Test']
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"  ✓ CSV written successfully!")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_datasets = len(all_stats)
    total_samples = sum(row['# Train'] + row['# Val'] + row['# Test'] for row in all_stats)
    datasets_with_text = sum(1 for row in all_stats if row['Text'] == 'Yes')
    
    print(f"\nTotal datasets:           {total_datasets}")
    print(f"Datasets with text:       {datasets_with_text}")
    print(f"Datasets without text:    {total_datasets - datasets_with_text}")
    print(f"Total samples:            {total_samples:,}")
    
    # Language breakdown
    language_counts = defaultdict(int)
    for row in all_stats:
        language_counts[row['Language']] += 1
    
    print(f"\nLanguage breakdown:")
    for lang, count in sorted(language_counts.items()):
        print(f"  {lang}: {count} datasets")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()

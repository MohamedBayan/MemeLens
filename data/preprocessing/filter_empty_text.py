#!/usr/bin/env python3
"""
Filter out samples with empty text fields from all datasets.

This script:
1. Scans all datasets in Unified_Labels_FullPath
2. Identifies samples with empty/whitespace-only text fields
3. Creates filtered datasets with only non-empty text samples
4. Saves statistics to CSV

Input:  ./data/Unified_Labels_FullPath/
Output: ./data/Unified_Labels_FullPath_TextField/
Stats:  ./Processing/text_field_statistics.csv

Usage:
    python Processing/filter_empty_text.py
    nohup python Processing/filter_empty_text.py \
  > logs/Removing_EmptyTextfield.log 2>&1 & 
"""

import os
import json
import csv
import shutil
from pathlib import Path
from collections import defaultdict


def is_empty_text(text):
    """Check if text field is empty, whitespace-only, or just punctuation."""
    if not text:
        return True
    
    # Strip whitespace
    stripped = text.strip()
    
    # Empty after stripping
    if not stripped:
        return True
    
    # Only punctuation/symbols (no actual text)
    if all(c in ' \t\n\r.,;:!?-_()[]{}"\'' for c in stripped):
        return True
    
    return False


def dataset_already_processed(output_path):
    """Check if dataset has already been processed."""
    if not os.path.exists(output_path):
        return False
    
    # Check if all splits exist
    for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
        split_file = os.path.join(output_path, split)
        if os.path.exists(os.path.join(output_path.replace("_TextField", ""), split)):
            # Original has this split, check if output exists
            if not os.path.exists(split_file):
                return False
    
    return True


def process_dataset(source_folder, dataset_name, output_base):
    """
    Process a single dataset and filter out empty text samples.
    
    Returns:
        dict with statistics
    """
    source_path = os.path.join(source_folder, dataset_name)
    output_path = os.path.join(output_base, os.path.basename(source_folder), dataset_name)
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Copy images folder if exists
    source_images = os.path.join(source_path, "images")
    output_images = os.path.join(output_path, "images")
    if os.path.exists(source_images):
        if os.path.exists(output_images):
            if os.path.islink(output_images):
                os.unlink(output_images)
            elif os.path.isdir(output_images):
                shutil.rmtree(output_images)
        shutil.copytree(source_images, output_images)
    
    stats = {
        'dataset': dataset_name,
        'folder': os.path.basename(source_folder),
        'train_total': 0,
        'train_empty': 0,
        'train_filtered': 0,
        'val_total': 0,
        'val_empty': 0,
        'val_filtered': 0,
        'test_total': 0,
        'test_empty': 0,
        'test_filtered': 0,
    }
    
    # Process each split
    for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
        source_file = os.path.join(source_path, split)
        output_file = os.path.join(output_path, split)
        
        if not os.path.exists(source_file):
            continue
        
        split_name = split.replace('.jsonl', '')
        total_count = 0
        empty_count = 0
        filtered_samples = []
        
        with open(source_file, 'r', encoding='utf-8') as f:
            for line in f:
                total_count += 1
                sample = json.loads(line.strip())
                
                text = sample.get('text', '')
                
                if is_empty_text(text):
                    empty_count += 1
                else:
                    filtered_samples.append(sample)
        
        # Write filtered samples
        with open(output_file, 'w', encoding='utf-8') as f:
            for sample in filtered_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        # Update stats
        stats[f'{split_name}_total'] = total_count
        stats[f'{split_name}_empty'] = empty_count
        stats[f'{split_name}_filtered'] = len(filtered_samples)
    
    # Calculate totals and percentages
    total_all = stats['train_total'] + stats['val_total'] + stats['test_total']
    empty_all = stats['train_empty'] + stats['val_empty'] + stats['test_empty']
    filtered_all = stats['train_filtered'] + stats['val_filtered'] + stats['test_filtered']
    
    stats['total_samples'] = total_all
    stats['total_empty'] = empty_all
    stats['total_filtered'] = filtered_all
    stats['empty_percentage'] = (empty_all / total_all * 100) if total_all > 0 else 0
    stats['filtered_percentage'] = (filtered_all / total_all * 100) if total_all > 0 else 0
    
    return stats


def main():
    # Paths
    base_path = "./data/Unified_Labels_FullPath"
    output_base = "./data/Unified_Labels_FullPath_TextField"
    stats_file = "./Processing/text_field_statistics_Not_MMHS.csv"
    
    # Folders to process
    folders = [
        "normalized_classification_en",
        "normalized_datasets"
    ]
    
    # Skip datasets
    skip_datasets = ["Hateful_en__MMHS"]
    
    all_stats = []
    
    print("=" * 80)
    print("TEXT FIELD FILTERING")
    print("=" * 80)
    print(f"\nSource: {base_path}")
    print(f"Output: {output_base}")
    print(f"Stats:  {stats_file}")
    print("\n" + "=" * 80)
    
    # Process each folder
    for folder in folders:
        folder_path = os.path.join(base_path, folder)
        
        if not os.path.exists(folder_path):
            print(f"\nWarning: Folder not found: {folder_path}")
            continue
        
        print(f"\nProcessing {folder}:")
        print("-" * 80)
        
        # Process each dataset
        for dataset_name in sorted(os.listdir(folder_path)):
            dataset_path = os.path.join(folder_path, dataset_name)
            
            if not os.path.isdir(dataset_path):
                continue
            
            if dataset_name in skip_datasets:
                print(f"  Skipping {dataset_name} (in skip list)")
                continue
            
            # Check if already processed
            output_dataset_path = os.path.join(output_base, folder, dataset_name)
            if dataset_already_processed(output_dataset_path):
                print(f"  ✓ {dataset_name} (already processed, skipping)")
                continue
            
            print(f"  Processing {dataset_name}...", end=' ', flush=True)
            
            stats = process_dataset(folder_path, dataset_name, output_base)
            all_stats.append(stats)
            
            print(f"✓ Total: {stats['total_samples']}, Empty: {stats['total_empty']} ({stats['empty_percentage']:.1f}%), Filtered: {stats['total_filtered']}")
    
    # Write statistics CSV
    print("\n" + "=" * 80)
    print("WRITING STATISTICS")
    print("=" * 80)
    
    csv_columns = [
        'dataset', 'folder',
        'total_samples', 'total_empty', 'total_filtered', 
        'empty_percentage', 'filtered_percentage',
        'train_total', 'train_empty', 'train_filtered',
        'val_total', 'val_empty', 'val_filtered',
        'test_total', 'test_empty', 'test_filtered'
    ]
    
    with open(stats_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=csv_columns)
        writer.writeheader()
        writer.writerows(all_stats)
    
    print(f"\nStatistics saved to: {stats_file}")
    
    # Summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    total_datasets = len(all_stats)
    total_samples = sum(s['total_samples'] for s in all_stats)
    total_empty = sum(s['total_empty'] for s in all_stats)
    total_filtered = sum(s['total_filtered'] for s in all_stats)
    
    print(f"\nTotal datasets processed: {total_datasets}")
    print(f"Total samples: {total_samples:,}")
    print(f"Samples with empty text: {total_empty:,} ({total_empty/total_samples*100:.2f}%)")
    print(f"Samples with text: {total_filtered:,} ({total_filtered/total_samples*100:.2f}%)")
    
    # Datasets with highest empty percentage
    print(f"\nDatasets with highest empty text percentage:")
    sorted_stats = sorted(all_stats, key=lambda x: x['empty_percentage'], reverse=True)
    for i, stat in enumerate(sorted_stats[:10], 1):
        print(f"  {i}. {stat['dataset']}: {stat['empty_percentage']:.1f}% ({stat['total_empty']}/{stat['total_samples']})")
    
    # Datasets with lowest empty percentage
    print(f"\nDatasets with lowest empty text percentage:")
    for i, stat in enumerate(sorted_stats[-10:][::-1], 1):
        print(f"  {i}. {stat['dataset']}: {stat['empty_percentage']:.1f}% ({stat['total_empty']}/{stat['total_samples']})")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print(f"\nFiltered datasets saved to: {output_base}")
    print(f"Statistics saved to: {stats_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Add native_labels field to datasets with native language translations.
Maps English labels to their native language equivalents.
"""

import json
from pathlib import Path
from collections import Counter

# Load native instructions with label translations
with open('instructions_native.json', 'r', encoding='utf-8') as f:
    native_instructions = json.load(f)

# Base path to unified labels
UNIFIED_BASE = Path('../Processing/Unified_Labels')

def add_native_labels_to_dataset(dataset_name, label_translations):
    """
    Add native_labels field to all samples in a dataset.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    label_translations : dict
        Mapping from English labels to native labels
    """
    # Skip if no translations available
    if not label_translations:
        print(f"Skipping {dataset_name} - no label translations available")
        return
    
    # Find dataset path
    dataset_path = None
    for subdir in ['normalized_classification_en', 'normalized_datasets']:
        potential_path = UNIFIED_BASE / subdir / dataset_name
        if potential_path.exists():
            dataset_path = potential_path
            break
    
    if not dataset_path:
        print(f"Dataset not found: {dataset_name}")
        return
    
    print(f"\n{'='*80}")
    print(f"Processing: {dataset_name}")
    print(f"{'='*80}")
    print(f"Path: {dataset_path}")
    print(f"Label translations: {len(label_translations)} labels")
    
    total_processed = 0
    total_mapped = 0
    unmapped_labels = Counter()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        split_file = dataset_path / f'{split}.jsonl'
        
        if not split_file.exists():
            continue
        
        # Read data
        data = []
        with open(split_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
        
        # Add native_labels field
        mapped_count = 0
        for item in data:
            # Get the label (could be in 'class_label' or 'label' field)
            label = item.get('class_label', item.get('label', ''))
            
            # Map to native label
            if label in label_translations:
                item['native_label'] = label_translations[label]
                mapped_count += 1
            else:
                item['native_label'] = label  # Keep original if no translation
                unmapped_labels[label] += 1
        
        # Write back to file
        with open(split_file, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        total_processed += len(data)
        total_mapped += mapped_count
        
        print(f"  {split.upper()}: {len(data)} samples, {mapped_count} mapped")
    
    # Summary
    print(f"\n✓ Total processed: {total_processed} samples")
    print(f"✓ Total mapped: {total_mapped} samples")
    
    if unmapped_labels:
        print(f"\nUnmapped labels found:")
        for label, count in unmapped_labels.most_common():
            print(f"    '{label}': {count} samples")

def main():
    print("="*80)
    print("ADDING NATIVE LABELS TO DATASETS")
    print("="*80)
    
    total_datasets = 0
    successful = 0
    
    # Process each dataset with native instructions
    for dataset_name, config in native_instructions.items():
        label_translations = config.get('label_translations', {})
        
        if label_translations:
            add_native_labels_to_dataset(dataset_name, label_translations)
            total_datasets += 1
            successful += 1
    
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"Total datasets processed: {successful}/{total_datasets}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()

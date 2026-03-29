#!/usr/bin/env python3
"""
Append instruction fields to all samples in the datasets.

For each sample:
- English datasets: Add random 'en_instruction' from the 21 available
- Non-English datasets: Add random 'en_instruction' and 'native_instruction' from the 21 available each

Applies to all splits: train.jsonl, val.jsonl, test.jsonl
"""

import os
import json
import random
from pathlib import Path


def load_instructions():
    """Load the merged instructions file."""
    instructions_path = "./Instruction_Generation/Instructions/merged_instructions.json"
    with open(instructions_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def process_dataset(dataset_path, dataset_name, instructions_data):
    """
    Process a single dataset by adding instruction fields to all samples.
    
    Args:
        dataset_path: Path to the dataset directory
        dataset_name: Name of the dataset (e.g., "Hateful_en__MMHS")
        instructions_data: Dictionary containing instruction data for this dataset
    """
    # Get instructions for this dataset
    en_instructions = instructions_data.get('en_instructions', [])
    native_instructions = instructions_data.get('native_instructions', [])
    language = instructions_data.get('language', 'en')
    
    if not en_instructions:
        print(f"  ⚠ No English instructions found for {dataset_name}")
        return 0
    
    # Check if it's native language dataset
    has_native = language == 'native' and native_instructions
    
    total_samples = 0
    
    # Process each split
    for split in ['train.jsonl', 'val.jsonl', 'test.jsonl']:
        split_path = os.path.join(dataset_path, split)
        
        if not os.path.exists(split_path):
            continue
        
        # Read all samples
        samples = []
        with open(split_path, 'r', encoding='utf-8') as f:
            for line in f:
                sample = json.loads(line.strip())
                
                # Add random English instruction
                sample['en_instruction'] = random.choice(en_instructions)
                
                # Add random native instruction if applicable
                if has_native:
                    sample['native_instruction'] = random.choice(native_instructions)
                
                samples.append(sample)
                total_samples += 1
        
        # Write back
        with open(split_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    return total_samples


def main():
    base_path = "./data/Unified_Labels_FullPath"
    
    print("=" * 80)
    print("APPENDING INSTRUCTIONS TO DATASETS")
    print("=" * 80)
    
    # Load instructions
    print("\nLoading instructions...")
    instructions = load_instructions()
    print(f"  ✓ Loaded instructions for {len(instructions)} datasets")
    
    # Process datasets
    print("\n" + "-" * 80)
    print("Processing datasets...")
    print("-" * 80)
    
    total_datasets = 0
    total_samples = 0
    english_datasets = 0
    native_datasets = 0
    
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
            
            # Find matching instructions
            if dataset_name not in instructions:
                print(f"  ⚠ {dataset_name}: No instructions found, skipping")
                continue
            
            instructions_data = instructions[dataset_name]
            language = instructions_data.get('language', 'en')
            
            # Process the dataset
            samples_processed = process_dataset(dataset_path, dataset_name, instructions_data)
            
            if samples_processed > 0:
                total_datasets += 1
                total_samples += samples_processed
                
                if language == 'en':
                    english_datasets += 1
                    print(f"  ✓ {dataset_name}: {samples_processed} samples (EN only)")
                else:
                    native_datasets += 1
                    print(f"  ✓ {dataset_name}: {samples_processed} samples (EN + Native)")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    print(f"\nDatasets processed:")
    print(f"  English-only datasets:     {english_datasets} (en_instruction added)")
    print(f"  Native language datasets:  {native_datasets} (en_instruction + native_instruction added)")
    print(f"  Total datasets:            {total_datasets}")
    
    print(f"\nSamples updated:")
    print(f"  Total samples:             {total_samples:,}")
    
    print(f"\nLocation:")
    print(f"  {base_path}")
    
    print("\n" + "=" * 80)
    print("COMPLETE!")
    print("=" * 80)
    print("\nEach sample now has:")
    print(f"  - English datasets: 'en_instruction' field (random from 21 options)")
    print(f"  - Native datasets: 'en_instruction' + 'native_instruction' fields (random from 21 each)")


if __name__ == "__main__":
    main()

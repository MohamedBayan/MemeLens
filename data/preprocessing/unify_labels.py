#!/usr/bin/env python3
"""
Script to unify labels across all MemeLens datasets.
Applies standardized label mapping and saves to Unified_Labels folder.
"""

import json
import os
import shutil
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Any
import pandas as pd


# ============================================================================
# LABEL UNIFICATION MAPPINGS
# ============================================================================

def get_label_mapping(dataset_name: str) -> Dict[str, str]:
    """
    Get label mapping for a specific dataset.
    Returns a dict mapping old_label -> new_label
    """
    
    # MMHS special handling - collapse multi-label to binary
    if dataset_name == 'Hateful_en__MMHS':
        def mmhs_mapper(label: str) -> str:
            if label == 'Non-hateful':
                return 'not-hateful'
            # Any label containing hate indicators becomes hateful
            return 'hateful'
        return mmhs_mapper
    
    # Standard mappings for different tasks
    mappings = {
        # Hateful datasets
        'Hateful_en_FHM': {
            'hateful': 'hateful',
            'not-hateful': 'not-hateful'
        },
        'Hateful_en__MIMIC_Islamophpbia': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        'Hateful_bn__MUTE': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        'Hateful_de__Multi3Hate': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        'Hateful_en__Multi3Hate': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        'Hateful_es__Multi3Hate': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        'Hateful_hi__Multi3Hate': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        'Hateful_zh__Multi3Hate': {
            'Hateful': 'hateful',
            'Non-hateful': 'not-hateful'
        },
        
        # Misogynous datasets
        'misogynous_en__MAMI': {
            'misogynous': 'misogynous',
            'not-misogynous': 'not-misogynous'
        },
        'Misogyny_hi_en__MIMIC2024': {
            'Misogynous': 'misogynous',
            'Non-Misogynous': 'not-misogynous'
        },
        
        # Sentiment datasets - standardize capitalization
        'sentiment_bn__BanglaAbuseMeme': {
            'Negative': 'negative',
            'Neutral': 'neutral',
            'Positive': 'positive'
        },
        'sentiment_ro__RoMemes': {
            'Negative': 'negative',
            'Neutral': 'neutral',
            'Positive': 'positive'
        },
        'overall_sentiment_en__memotion': {
            'negative': 'negative',
            'neutral': 'neutral',
            'positive': 'positive',
            'very_negative': 'very-negative',
            'very_positive': 'very-positive'
        },
        
        # Offensive datasets - standardize
        'offensive_en__memotion': {
            'not_offensive': 'not-offensive',
            'slight': 'slightly-offensive',
            'very_offensive': 'very-offensive',
            'hateful_offensive': 'hateful-offensive'
        },
        'offensiveness_detection_en__MET_Meme': {
            'Non-offensive': 'not-offensive',
            'Slightly': 'slightly-offensive',
            'Moderately': 'moderately-offensive',
            'Very': 'very-offensive'
        },
        'offensiveness_detection_zh__MET_Meme': {
            'Non-offensive': 'not-offensive',
            'Slightly': 'slightly-offensive',
            'Moderately': 'moderately-offensive',
            'Very': 'very-offensive'
        },
        
        # Sarcasm datasets
        'sarcasm_bn__BanglaAbuseMeme': {
            'Not-Sarcasm': 'not-sarcasm',
            'Sarcasm': 'sarcasm'
        },
        'sarcasm_en__memotion': {
            'not_sarcastic': 'not-sarcastic',
            'general': 'general-sarcasm',
            'twisted_meaning': 'twisted-meaning',
            'very_twisted': 'very-twisted'
        },
        
        # Abusive/Toxic - standardize not prefix
        'abuse_bn__BanglaAbuseMeme': {
            'Abusive': 'abusive',
            'Non-abusive': 'not-abusive'
        },
        'toxic_ru__Toxic_Memes_Detection_Dataset': {
            'toxic': 'toxic',
            'non-toxic': 'not-toxic'
        },
        
        # Vulgar
        'vulgar_bn__BanglaAbuseMeme': {
            'Vulgar': 'vulgar',
            'Not-Vulgar': 'not-vulgar'
        },
        
        # Harmful - already good, just ensure consistency
        'Harmful_Covid_en__HarMeme': {
            'not_harmful': 'not-harmful',
            'partially_harmful': 'partially-harmful',
            'very_harmful': 'very-harmful'
        },
        'Harmful_en__HarMeme': {
            'not_harmful': 'not-harmful',
            'partially_harmful': 'partially-harmful',
            'very_harmful': 'very-harmful',
            'not_specified': 'not-specified'
        },
        
        # Humour - standardize underscores
        'humour_en__memotion': {
            'not_funny': 'not-funny',
            'funny': 'funny',
            'very_funny': 'very-funny',
            'hilarious': 'hilarious'
        },
        
        # Motivational
        'motivational_en__memotion': {
            'not_motivational': 'not-motivational',
            'motivational': 'motivational'
        },
        
        # MAMI subcategories - standardize
        'objectification_en__MAMI': {
            'objectification': 'objectification',
            'not-objectification': 'not-objectification'
        },
        'shaming_en__MAMI': {
            'shaming': 'shaming',
            'not-shaming': 'not-shaming'
        },
        'stereotype_en__MAMI': {
            'stereotype': 'stereotype',
            'not-stereotype': 'not-stereotype'
        },
        'violence_en__MAMI': {
            'violence': 'violence',
            'not-violence': 'not-violence'
        },
        
        # Propaganda
        'propoganda_ar_ArMeme': {
            'propaganda': 'propaganda',
            'not-propaganda': 'not-propaganda'
        },
        
        # Sentiment degree - standardize
        'sentiment_degree_en__MET_Meme': {
            'Slightly': 'slightly',
            'Moderately': 'moderately',
            'Very': 'very'
        },
        'sentiment_degree_zh__MET_Meme': {
            'Slightly': 'slightly',
            'Moderately': 'moderately',
            'Very': 'very'
        },
    }
    
    return mappings.get(dataset_name, {})


def unify_label(label: str, dataset_name: str) -> str:
    """Unify a single label based on dataset-specific mapping."""
    mapping = get_label_mapping(dataset_name)
    
    # Check if mapping is a function (for complex cases like MMHS)
    if callable(mapping):
        return mapping(label)
    
    # Use mapping dict if available, otherwise return original
    return mapping.get(label, label)


# ============================================================================
# PROCESSING FUNCTIONS
# ============================================================================

def process_dataset(
    input_path: Path,
    output_path: Path,
    dataset_name: str,
    stats: Dict
) -> Dict:
    """Process a single dataset and unify its labels."""
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create symlinks to images directory instead of copying
    for img_dir in ['img', 'images', 'image']:
        img_path = input_path / img_dir
        if img_path.exists() and img_path.is_dir():
            output_img_path = output_path / img_dir
            if not output_img_path.exists():
                print(f"  Creating symlink to {img_dir}/ directory...")
                output_img_path.symlink_to(img_path.absolute())
    
    dataset_stats = {
        'dataset': dataset_name,
        'total_samples': 0,
        'label_changes': Counter(),
        'splits': {}
    }
    
    # Process each split
    for split in ['train', 'val', 'test']:
        input_file = input_path / f'{split}.jsonl'
        if not input_file.exists():
            continue
        
        output_file = output_path / f'{split}.jsonl'
        
        split_stats = {
            'samples': 0,
            'changes': 0
        }
        
        with open(input_file, 'r', encoding='utf-8') as f_in, \
             open(output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                if not line.strip():
                    continue
                
                item = json.loads(line)
                original_label = None
                
                # Find and update label
                if 'class_label' in item:
                    original_label = item['class_label']
                    unified_label = unify_label(original_label, dataset_name)
                    item['class_label'] = unified_label
                    
                elif 'label' in item:
                    original_label = item['label']
                    unified_label = unify_label(original_label, dataset_name)
                    item['label'] = unified_label
                    
                elif 'labels' in item:
                    original_label = item['labels']
                    if isinstance(original_label, list):
                        item['labels'] = [unify_label(l, dataset_name) for l in original_label]
                        unified_label = item['labels']
                    else:
                        unified_label = unify_label(original_label, dataset_name)
                        item['labels'] = unified_label
                
                # Track changes
                if original_label is not None:
                    if str(original_label) != str(unified_label):
                        dataset_stats['label_changes'][f"{original_label} → {unified_label}"] += 1
                        split_stats['changes'] += 1
                
                split_stats['samples'] += 1
                f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        dataset_stats['splits'][split] = split_stats
        dataset_stats['total_samples'] += split_stats['samples']
    
    return dataset_stats


def process_all_datasets(base_dirs: List[str], output_base: str) -> List[Dict]:
    """Process all datasets and unify labels."""
    all_stats = []
    
    for base_dir in base_dirs:
        base_path = Path(base_dir)
        if not base_path.exists():
            print(f"Warning: {base_dir} does not exist")
            continue
        
        # Determine output directory maintaining structure
        if 'normalized_classification_en' in base_dir:
            output_dir = Path(output_base) / 'normalized_classification_en'
        else:
            output_dir = Path(output_base) / 'normalized_datasets'
        
        print(f"\n{'='*80}")
        print(f"Processing datasets from: {base_dir}")
        print(f"Output to: {output_dir}")
        print(f"{'='*80}\n")
        
        # Get all dataset directories
        datasets = [d for d in base_path.iterdir() if d.is_dir()]
        
        for dataset_dir in sorted(datasets):
            dataset_name = dataset_dir.name
            print(f"Processing: {dataset_name}")
            
            output_path = output_dir / dataset_name
            stats = {}
            
            dataset_stats = process_dataset(
                dataset_dir,
                output_path,
                dataset_name,
                stats
            )
            
            all_stats.append(dataset_stats)
            
            # Print stats for this dataset
            if dataset_stats['label_changes']:
                print(f"  ✓ Unified {dataset_stats['total_samples']} samples")
                print(f"  Label changes:")
                for change, count in dataset_stats['label_changes'].most_common():
                    print(f"    {change}: {count}")
            else:
                print(f"  ✓ Processed {dataset_stats['total_samples']} samples (no changes)")
    
    return all_stats


def save_summary(all_stats: List[Dict], output_file: str):
    """Save unification summary to CSV."""
    rows = []
    
    for dataset_stats in all_stats:
        if dataset_stats['label_changes']:
            for change, count in dataset_stats['label_changes'].items():
                old_label, new_label = change.split(' → ')
                rows.append({
                    'dataset': dataset_stats['dataset'],
                    'old_label': old_label,
                    'new_label': new_label,
                    'samples_affected': count,
                    'total_samples': dataset_stats['total_samples']
                })
        else:
            # No changes for this dataset
            rows.append({
                'dataset': dataset_stats['dataset'],
                'old_label': 'N/A',
                'new_label': 'N/A',
                'samples_affected': 0,
                'total_samples': dataset_stats['total_samples']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"\n✓ Unification summary saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("LABEL UNIFICATION PROCESS")
    print("="*80)
    
    # Define paths (absolute)
    base_dirs = [
        './data/normalized_classification_en',
        './data/normalized_datasets'
    ]
    
    output_base = './Processing/Unified_Labels'
    
    # Process all datasets
    all_stats = process_all_datasets(base_dirs, output_base)
    
    # Save summary
    summary_file = './Processing/unification_summary.csv'
    save_summary(all_stats, summary_file)
    
    # Print final summary
    print(f"\n{'='*80}")
    print("UNIFICATION COMPLETE!")
    print(f"{'='*80}")
    print(f"Total datasets processed: {len(all_stats)}")
    print(f"Output directory: {output_base}")
    print(f"Summary saved to: {summary_file}")
    
    # Count total changes
    total_changes = sum(
        sum(s['label_changes'].values()) 
        for s in all_stats
    )
    total_samples = sum(s['total_samples'] for s in all_stats)
    
    print(f"\nTotal samples: {total_samples}")
    print(f"Total label changes: {total_changes}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()

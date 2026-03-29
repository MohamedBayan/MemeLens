#!/usr/bin/env python3
"""
Step 3: Merge batch results with original datasets.

Usage:
    python 3_merge_results.py \
        --results_dir ../outputs \
        --output_dir ../merged_data
"""
import argparse
import json
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_PATH, DATASET_CONFIG


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_batch_results(results_dir):
    """Load all batch result files."""
    results = {}
    
    result_files = [f for f in os.listdir(results_dir) if f.endswith('_results.jsonl')]
    
    logging.info(f"Found {len(result_files)} result files")
    
    for result_file in result_files:
        file_path = os.path.join(results_dir, result_file)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    custom_id = str(result.get('custom_id', ''))
                    
                    # Extract the response content
                    if 'response' in result and 'body' in result['response']:
                        content = result['response']['body']['choices'][0]['message']['content']
                        
                        # Parse JSON content
                        try:
                            explanation_data = json.loads(content)
                        except json.JSONDecodeError:
                            explanation_data = {
                                'en_explanation': content,
                                'parse_error': True
                            }
                        
                        results[custom_id] = explanation_data
                        
                except Exception as e:
                    logging.warning(f"Error parsing result: {e}")
                    continue
    
    logging.info(f"Loaded {len(results)} results")
    return results


def merge_with_dataset(results, output_dir):
    """Merge results with original datasets."""
    # Group results by dataset
    dataset_results = {}
    for custom_id, data in results.items():
        # Format: dataset_name__split__sample_id
        parts = custom_id.split('__')
        if len(parts) >= 3:
            dataset_name = parts[0]
            split = parts[1]
            sample_id = '__'.join(parts[2:])  # Handle IDs with __
            
            if dataset_name not in dataset_results:
                dataset_results[dataset_name] = {}
            
            key = f"{split}__{sample_id}"
            dataset_results[dataset_name][key] = data
    
    total_merged = 0
    
    for dataset_name, explanations in dataset_results.items():
        # Determine folder
        for folder in ["normalized_classification_en", "normalized_datasets"]:
            dataset_path = os.path.join(DATA_PATH, folder, dataset_name)
            if os.path.exists(dataset_path):
                break
        else:
            logging.warning(f"Dataset not found: {dataset_name}")
            continue
        
        # Check if English-only
        config = DATASET_CONFIG.get(dataset_name, {})
        is_english = config.get("language", "en") == "en"
        
        # Create output directory
        out_dataset_dir = os.path.join(output_dir, folder, dataset_name)
        os.makedirs(out_dataset_dir, exist_ok=True)
        
        merged_count = 0
        
        for split in ["train", "val", "test"]:
            split_file = os.path.join(dataset_path, f"{split}.jsonl")
            out_file = os.path.join(out_dataset_dir, f"{split}.jsonl")
            
            if not os.path.exists(split_file):
                continue
            
            with open(split_file, 'r', encoding='utf-8') as f_in, \
                 open(out_file, 'w', encoding='utf-8') as f_out:
                
                for line in f_in:
                    item = json.loads(line.strip())
                    sample_id = str(item.get('id', ''))
                    key = f"{split}__{sample_id}"
                    
                    if key in explanations:
                        exp = explanations[key]
                        item['en_explanation'] = exp.get('en_explanation', '')
                        
                        # Only add native_explanation for non-English datasets
                        if not is_english and 'native_explanation' in exp:
                            item['native_explanation'] = exp.get('native_explanation', '')
                        
                        merged_count += 1
                    
                    f_out.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logging.info(f"  ✓ {dataset_name}: {merged_count} samples merged")
        total_merged += merged_count
    
    return total_merged


def main():
    parser = argparse.ArgumentParser(description='Merge batch results with original datasets')
    parser.add_argument('--results_dir', default='./Explanation/outputs',
                        help='Directory with batch results')
    parser.add_argument('--output_dir', default='./Explanation/merged_data',
                        help='Output directory for merged datasets')
    
    args = parser.parse_args()
    setup_logging()
    
    # Validate inputs
    if not os.path.exists(args.results_dir):
        logging.error(f"Results directory not found: {args.results_dir}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    logging.info("Loading batch results...")
    results = load_batch_results(args.results_dir)
    
    if not results:
        logging.error("No results found!")
        return
    
    # Merge with datasets
    logging.info("Merging results with original datasets...")
    total_merged = merge_with_dataset(results, args.output_dir)
    
    logging.info(f"\n✓ Merge complete!")
    logging.info(f"  Output directory: {args.output_dir}")
    logging.info(f"  Total samples merged: {total_merged}")


if __name__ == "__main__":
    main()

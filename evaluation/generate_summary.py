#!/usr/bin/env python3
"""
Script to generate Excel file with metrics from model evaluation results.

Usage:
    python generate_metrics_excel.py <folder_path> [output_file]

Example:
    python scripts/src/generate_metrics_excel_swift.py ./scores/seq_cls
"""

import os
import json
import argparse
from pathlib import Path
import pandas as pd


def extract_metrics_from_folder(folder_path):
    """
    Extract metrics from all model subdirectories in the given folder.
    Supports nested structure: task_folder/model_folder/metrics.json
    
    Args:
        folder_path: Path to the folder containing task/model subdirectories
        
    Returns:
        List of dictionaries containing model metrics
    """
    metrics_list = []
    folder = Path(folder_path)
    
    if not folder.exists():
        raise ValueError(f"Folder does not exist: {folder_path}")
    
    # Iterate through all task subdirectories
    for task_dir in sorted(folder.iterdir()):
        if not task_dir.is_dir():
            continue
        # Skip backup/old directories
        if task_dir.name.endswith('_old') or task_dir.name.endswith('.bak'):
            continue
        
        # Check if metrics.json exists directly in this directory (flat structure)
        direct_metrics_file = task_dir / "metrics.json"
        if direct_metrics_file.exists():
            # Flat structure: folder/model/metrics.json
            process_metrics_file(direct_metrics_file, task_dir.name, None, metrics_list)
            continue
        
        # Nested structure: folder/task/model/metrics.json
        found_model = False
        for model_dir in sorted(task_dir.iterdir()):
            if not model_dir.is_dir():
                continue
                
            metrics_file = model_dir / "metrics.json"
            
            if metrics_file.exists():
                found_model = True
                process_metrics_file(metrics_file, task_dir.name, model_dir.name, metrics_list)
        
        if not found_model:
            print(f"Warning: No metrics.json found in {task_dir.name}, skipping...")
    
    return metrics_list


def process_metrics_file(metrics_file, task_name, model_name, metrics_list):
    """
    Process a single metrics.json file and add to metrics list.
    
    Args:
        metrics_file: Path to metrics.json file
        task_name: Name of the task
        model_name: Name of the model (or None for flat structure)
        metrics_list: List to append metrics to
    """
    try:
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        # Build the display name
        if model_name:
            display_name = f"{task_name}/{model_name}"
        else:
            display_name = task_name
        
        # Extract required metrics
        # Use accuracy if available, otherwise fall back to subset_accuracy
        accuracy = metrics.get('accuracy', metrics.get('subset_accuracy', 0))
        
        model_metrics = {
            'task': task_name,
            'model': model_name if model_name else task_name,
            'ACC': round(accuracy, 3),
            'W-P': round(metrics.get('precision_weighted', 0), 3),
            'W-R': round(metrics.get('recall_weighted', 0), 3),
            'W-F1': round(metrics.get('f1_weighted', 0), 3),
            'M-F1': round(metrics.get('f1_macro', 0), 3)
        }
        
        # Add additional metrics if they exist
        if 'bertscore_f1' in metrics:
            model_metrics['BS-F1'] = round(metrics.get('bertscore_f1', 0), 3)
        if 'rouge1' in metrics:
            model_metrics['R1'] = round(metrics.get('rouge1', 0), 3)
        if 'rouge2' in metrics:
            model_metrics['R2'] = round(metrics.get('rouge2', 0), 3)
        if 'rougeL' in metrics:
            model_metrics['RL'] = round(metrics.get('rougeL', 0), 3)
        if 'bleu' in metrics:
            model_metrics['BLEU'] = round(metrics.get('bleu', 0), 3)
        if 'meteor' in metrics:
            model_metrics['METEOR'] = round(metrics.get('meteor', 0), 3)
        
        metrics_list.append(model_metrics)
        print(f"Processed: {display_name}")
        
    except json.JSONDecodeError as e:
        print(f"Error reading {metrics_file}: {e}")
    except Exception as e:
        print(f"Error processing {metrics_file}: {e}")


def save_to_excel(metrics_list, output_file):
    """
    Save metrics to Excel file with custom formatting.
    
    Args:
        metrics_list: List of dictionaries containing metrics
        output_file: Path to output Excel file
    """
    if not metrics_list:
        print("No metrics found to save!")
        return
    
    # Create DataFrame
    df = pd.DataFrame(metrics_list)
    
    # Sort by task column (case-insensitive)
    df = df.sort_values(by='task', key=lambda x: x.str.lower(), ascending=True)
    
    # Select and reorder columns: task, model, ACC, M-F1, W-F1 + optional explanation metrics
    columns_to_keep = ['task', 'model', 'ACC', 'M-F1', 'W-F1']
    for extra_col in ['BS-F1', 'R1', 'R2', 'RL', 'BLEU', 'METEOR']:
        if extra_col in df.columns:
            columns_to_keep.append(extra_col)
    df = df[columns_to_keep]
    
    # Extract model name from the output file path for the title
    folder_name = Path(output_file).parent.name
    model_name = folder_name.replace('_', ' ').replace('-', ' ')
    
    # Determine if it's zero-shot based on path
    if 'zero-shot' in str(output_file).lower():
        title = f"Zero-Shot {model_name}"
    else:
        title = model_name
    
    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        # Write title in first row
        df_title = pd.DataFrame([[title]], columns=[''])
        df_title.to_excel(writer, sheet_name='Metrics', index=False, header=False, startrow=0)
        
        # Write data starting from row 1 (0-indexed)
        df.to_excel(writer, sheet_name='Metrics', index=False, startrow=1)
    
    print(f"\nExcel file saved to: {output_file}")
    print(f"Total models processed: {len(metrics_list)}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate Excel file with model evaluation metrics'
    )
    parser.add_argument(
        'folder_path',
        type=str,
        help='Path to the folder containing model subdirectories'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='Output Excel file path (default: metrics_summary.xlsx in the input folder)'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if args.output is None:
        folder_name = Path(args.folder_path).name
        args.output = os.path.join(args.folder_path, f'{folder_name}_metrics_summary.xlsx')
    
    print(f"Processing folder: {args.folder_path}")
    print(f"Output file: {args.output}\n")
    
    # Extract metrics
    metrics_list = extract_metrics_from_folder(args.folder_path)
    
    # Save to Excel
    save_to_excel(metrics_list, args.output)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Convert datasets with explanations to instruction-following format for ms-swift.

This script creates three versions of the datasets using English explanations:

1. INPUT_AUGMENTED: Explanation in input (TRAIN ONLY), label in output
   - Train User: instruction + image + text + explanation
   - Train Assistant: Label: <label>
   - Val/Test User: instruction + image + text (NO explanation)
   - Val/Test Assistant: Label: <label>

2. CLASSIFY_THEN_EXPLAIN: Label first, then explanation
   - User: instruction + image + text
   - Assistant: Label: <label>\nExplanation: <explanation>

3. EXPLAIN_THEN_CLASSIFY: Explanation first, then label
   - User: instruction + image + text
   - Assistant: Explanation: <explanation>\nLabel: <label>

Input format:
{"id": "...", "img_path": "...", "class_label": "...", "text": "...", 
 "en_instruction": "...", "en_explanation": "...", "native_explanation": "..."}

Output format (instruction-following):
{"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image> ..."},
    {"role": "assistant", "content": "..."}
], "images": ["..."]}
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# ============================================================================
# Format Suffixes for Instructions
# ============================================================================

INPUT_AUGMENTED_SUFFIX = (
    "\n\nIMPORTANT: Your response must strictly follow this format: "
    "'Label: <label>' where <label> is your classification. "
    "Do not include any additional text or explanation."
)

CLASSIFY_THEN_EXPLAIN_SUFFIX = (
    "\n\nIMPORTANT: Your response must strictly follow this format:\n"
    "'Label: <label>\nExplanation: <explanation>'\n"
    "where <label> is your classification and <explanation> provides "
    "a brief justification for your decision based on the visual and textual content."
)

EXPLAIN_THEN_CLASSIFY_SUFFIX = (
    "\n\nIMPORTANT: Your response must strictly follow this format:\n"
    "'Explanation: <explanation>\nLabel: <label>'\n"
    "where <explanation> provides a brief analysis of the visual and textual content, "
    "followed by <label> which is your final classification."
)


# ============================================================================
# System Prompts
# ============================================================================

def get_system_prompt(task_name: str) -> str:
    """Generate a system prompt based on the task name."""
    task_lower = task_name.lower()
    
    if "hateful" in task_lower or "hate" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying hateful content in memes."
    elif "misogyn" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying misogynistic content in memes."
    elif "abuse" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying abusive content in memes."
    elif "toxic" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying toxic content in memes."
    elif "offensive" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying offensive content in memes."
    elif "sentiment" in task_lower:
        return "You are an expert social media image analyzer specializing in sentiment analysis of memes."
    elif "emotion" in task_lower:
        return "You are an expert social media image analyzer specializing in emotion recognition in memes."
    elif "sarcasm" in task_lower:
        return "You are an expert social media image analyzer specializing in detecting sarcasm in memes."
    elif "deepfake" in task_lower or "fake" in task_lower:
        return "You are an expert image analyst specializing in detecting manipulated or deepfake content in memes."
    elif "propoganda" in task_lower or "propaganda" in task_lower:
        return "You are an expert social media analyst specializing in identifying propaganda techniques in memes."
    elif "political" in task_lower:
        return "You are an expert social media analyst specializing in analyzing political content in memes."
    elif "vulgar" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying vulgar content in memes."
    elif "harm" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying harmful content in memes."
    elif "target" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying targeted attacks in memes."
    elif "humour" in task_lower or "humor" in task_lower:
        return "You are an expert social media image analyzer specializing in analyzing humor in memes."
    elif "motivational" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying motivational content in memes."
    elif "intention" in task_lower:
        return "You are an expert social media image analyzer specializing in detecting intentions in memes."
    elif "metaphor" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying metaphors in memes."
    elif "objectification" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying objectification in memes."
    elif "shaming" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying shaming content in memes."
    elif "stereotype" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying stereotypes in memes."
    elif "violence" in task_lower:
        return "You are an expert social media image analyzer specializing in identifying violent content in memes."
    else:
        return "You are an expert social media image analyzer specializing in analyzing meme content."


# ============================================================================
# Conversion Functions
# ============================================================================

def convert_input_augmented(sample: Dict, task_name: str, include_explanation: bool = True) -> Optional[Dict]:
    """
    Convert to INPUT_AUGMENTED format:
    - User: instruction + image + text + explanation (train only)
    - User: instruction + image + text (val/test - no explanation)
    - Assistant: Label: <label>
    
    Args:
        sample: The sample dict
        task_name: Name of the task
        include_explanation: If True, include explanation in input (for train).
                           If False, just use standard input (for val/test).
    
    Returns None if en_explanation is empty/missing (only when include_explanation=True).
    """
    en_explanation = sample.get("en_explanation", "").strip()
    
    # Only require explanation for training samples
    if include_explanation and not en_explanation:
        return None
    
    system_prompt = get_system_prompt(task_name)
    instruction = sample.get("en_instruction", "")
    label = sample.get("class_label", "")
    text = sample.get("text", "").strip()
    
    # Build user content
    instruction_with_suffix = instruction + INPUT_AUGMENTED_SUFFIX
    
    if include_explanation:
        # Training: include explanation in input
        if text:
            user_content = f"<image> {instruction_with_suffix}\n\nText extracted from meme: {text}\n\nAnalysis context: {en_explanation}"
        else:
            user_content = f"<image> {instruction_with_suffix}\n\nAnalysis context: {en_explanation}"
    else:
        # Val/Test: no explanation in input
        if text:
            user_content = f"<image> {instruction_with_suffix}\n\nText extracted from meme: {text}"
        else:
            user_content = f"<image> {instruction_with_suffix}"
    
    assistant_content = f"Label: {label}"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [sample["img_path"]]
    }


def convert_classify_then_explain(sample: Dict, task_name: str, is_train: bool = True) -> Optional[Dict]:
    """
    Convert to CLASSIFY_THEN_EXPLAIN format:
    - User: instruction + image + text
    - Assistant: Label: <label>\nExplanation: <explanation>
    
    Returns None if en_explanation is empty/missing (only for training samples).
    For test/val, allows empty explanations and just outputs the label.
    """
    en_explanation = sample.get("en_explanation", "").strip()
    if is_train and not en_explanation:
        return None
    
    system_prompt = get_system_prompt(task_name)
    instruction = sample.get("en_instruction", "")
    label = sample.get("class_label", "")
    text = sample.get("text", "").strip()
    
    # Build user content
    instruction_with_suffix = instruction + CLASSIFY_THEN_EXPLAIN_SUFFIX
    
    if text:
        user_content = f"<image> {instruction_with_suffix}\n\nText extracted from meme: {text}"
    else:
        user_content = f"<image> {instruction_with_suffix}"
    
    # For test/val without explanations, just output the label
    if en_explanation:
        assistant_content = f"Label: {label}\nExplanation: {en_explanation}"
    else:
        assistant_content = f"Label: {label}"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [sample["img_path"]]
    }


def convert_explain_then_classify(sample: Dict, task_name: str, is_train: bool = True) -> Optional[Dict]:
    """
    Convert to EXPLAIN_THEN_CLASSIFY format:
    - User: instruction + image + text
    - Assistant: Explanation: <explanation>\nLabel: <label>
    
    Returns None if en_explanation is empty/missing (only for training samples).
    For test/val, allows empty explanations and just outputs the label.
    """
    en_explanation = sample.get("en_explanation", "").strip()
    if is_train and not en_explanation:
        return None
    
    system_prompt = get_system_prompt(task_name)
    instruction = sample.get("en_instruction", "")
    label = sample.get("class_label", "")
    text = sample.get("text", "").strip()
    
    # Build user content
    instruction_with_suffix = instruction + EXPLAIN_THEN_CLASSIFY_SUFFIX
    
    if text:
        user_content = f"<image> {instruction_with_suffix}\n\nText extracted from meme: {text}"
    else:
        user_content = f"<image> {instruction_with_suffix}"
    
    # For test/val without explanations, just output the label
    if en_explanation:
        assistant_content = f"Explanation: {en_explanation}\nLabel: {label}"
    else:
        assistant_content = f"Label: {label}"
    
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [sample["img_path"]]
    }


# ============================================================================
# File Processing
# ============================================================================

def process_jsonl_file(
    input_path: Path,
    output_paths: Dict[str, Path],
    task_name: str,
    split_name: str
) -> Dict[str, Tuple[int, int]]:
    """
    Process a single JSONL file and create all three output versions.
    
    Args:
        input_path: Path to input JSONL file
        output_paths: Dict mapping format name to output path
        task_name: Name of the task/dataset
        split_name: Name of the split (train, val, test)
    
    Returns:
        Dict mapping format name to (processed_count, skipped_count)
    """
    is_train = split_name == "train"
    
    stats = {
        "input_augmented": {"processed": 0, "skipped": 0},
        "classify_then_explain": {"processed": 0, "skipped": 0},
        "explain_then_classify": {"processed": 0, "skipped": 0}
    }
    
    # Open all output files
    output_files = {
        name: open(path, 'w', encoding='utf-8')
        for name, path in output_paths.items()
    }
    
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    sample = json.loads(line)
                    
                    # INPUT_AUGMENTED: explanation in input only for train
                    # For val/test, include all samples (no explanation needed)
                    converted = convert_input_augmented(
                        sample, task_name, include_explanation=is_train
                    )
                    if converted:
                        output_files["input_augmented"].write(
                            json.dumps(converted, ensure_ascii=False) + '\n'
                        )
                        stats["input_augmented"]["processed"] += 1
                    else:
                        stats["input_augmented"]["skipped"] += 1
                    
                    # CLASSIFY_THEN_EXPLAIN: only samples with explanations (train), all samples (val/test)
                    converted = convert_classify_then_explain(sample, task_name, is_train=is_train)
                    if converted:
                        output_files["classify_then_explain"].write(
                            json.dumps(converted, ensure_ascii=False) + '\n'
                        )
                        stats["classify_then_explain"]["processed"] += 1
                    else:
                        stats["classify_then_explain"]["skipped"] += 1
                    
                    # EXPLAIN_THEN_CLASSIFY: only samples with explanations (train), all samples (val/test)
                    converted = convert_explain_then_classify(sample, task_name, is_train=is_train)
                    if converted:
                        output_files["explain_then_classify"].write(
                            json.dumps(converted, ensure_ascii=False) + '\n'
                        )
                        stats["explain_then_classify"]["processed"] += 1
                    else:
                        stats["explain_then_classify"]["skipped"] += 1
                            
                except json.JSONDecodeError as e:
                    print(f"  Warning: Skipping malformed JSON line: {e}")
                except Exception as e:
                    print(f"  Warning: Error processing sample: {e}")
    finally:
        for f in output_files.values():
            f.close()
    
    return stats


def process_dataset(
    dataset_path: Path,
    output_base_dirs: Dict[str, Path],
    dataset_name: str
) -> Dict[str, Dict[str, int]]:
    """
    Process an entire dataset directory for all three formats.
    
    Args:
        dataset_path: Path to the dataset directory
        output_base_dirs: Dict mapping format name to output base directory
        dataset_name: Name of the dataset
    
    Returns:
        Aggregated statistics
    """
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Create output directories for this dataset
    dataset_output_dirs = {}
    for format_name, base_dir in output_base_dirs.items():
        dataset_dir = base_dir / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        dataset_output_dirs[format_name] = dataset_dir
    
    total_stats = defaultdict(lambda: {"processed": 0, "skipped": 0})
    
    # Process each JSONL file
    for jsonl_file in sorted(dataset_path.glob("*.jsonl")):
        split_name = jsonl_file.stem  # train, val, test
        
        # Prepare output paths for this split
        output_paths = {
            format_name: dataset_dir / f"{split_name}.jsonl"
            for format_name, dataset_dir in dataset_output_dirs.items()
        }
        
        # Process the file (pass split_name for input_augmented logic)
        stats = process_jsonl_file(jsonl_file, output_paths, dataset_name, split_name)
        
        # Print stats for this split
        for format_name, format_stats in stats.items():
            processed = format_stats["processed"]
            skipped = format_stats["skipped"]
            total_stats[format_name]["processed"] += processed
            total_stats[format_name]["skipped"] += skipped
        
        # Print summary for this split
        first_format = list(stats.keys())[0]
        total_in_split = stats[first_format]["processed"] + stats[first_format]["skipped"]
        processed = stats[first_format]["processed"]
        print(f"  {split_name}: {processed}/{total_in_split} samples with explanations")
    
    return dict(total_stats)


def main():
    parser = argparse.ArgumentParser(
        description="Convert datasets with explanations to instruction-following format for ms-swift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script creates three versions of the datasets:

1. input_augmented/
   - Explanation included in user input
   - Output: Label: <label>

2. classify_then_explain/
   - Standard input (instruction + image + text)
   - Output: Label: <label>\\nExplanation: <explanation>

3. explain_then_classify/
   - Standard input (instruction + image + text)
   - Output: Explanation: <explanation>\\nLabel: <label>

Only samples with non-empty en_explanation are included.
"""
    )
    
    parser.add_argument(
        "--input-en", 
        type=str,
        default="./data/Unified_Labels_FullPath/normalized_classification_en_with_explanations",
        help="Path to English datasets with explanations"
    )
    parser.add_argument(
        "--input-multilingual",
        type=str,
        default="./data/Unified_Labels_FullPath/normalized_datasets_with_explanations",
        help="Path to multilingual datasets with explanations"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/ms_swift_formated/explanation",
        help="Output base directory"
    )
    
    args = parser.parse_args()
    
    # Define paths
    english_datasets_dir = Path(args.input_en)
    multilingual_datasets_dir = Path(args.input_multilingual)
    output_base = Path(args.output)
    
    # Create output directories for each format
    format_names = ["input_augmented", "classify_then_explain", "explain_then_classify"]
    output_dirs = {}
    for format_name in format_names:
        format_dir = output_base / format_name
        format_dir.mkdir(parents=True, exist_ok=True)
        output_dirs[format_name] = format_dir
    
    print("=" * 70)
    print("Converting Datasets with Explanations to Instruction-Following Format")
    print("=" * 70)
    print(f"\nInput (English):      {english_datasets_dir}")
    print(f"Input (Multilingual): {multilingual_datasets_dir}")
    print(f"Output:               {output_base}")
    print("\nFormats to generate:")
    for name in format_names:
        print(f"  - {name}")
    
    # Aggregate statistics
    all_stats = defaultdict(lambda: {"processed": 0, "skipped": 0, "datasets": 0})
    
    # Process English datasets
    print("\n" + "-" * 50)
    print("Processing English Datasets")
    print("-" * 50)
    
    if english_datasets_dir.exists():
        for dataset_dir in sorted(english_datasets_dir.iterdir()):
            if dataset_dir.is_dir():
                stats = process_dataset(dataset_dir, output_dirs, dataset_dir.name)
                for format_name, format_stats in stats.items():
                    all_stats[format_name]["processed"] += format_stats["processed"]
                    all_stats[format_name]["skipped"] += format_stats["skipped"]
                    all_stats[format_name]["datasets"] += 1
    else:
        print(f"  Warning: Directory not found: {english_datasets_dir}")
    
    # Process multilingual datasets (using English explanations only)
    print("\n" + "-" * 50)
    print("Processing Multilingual Datasets (English explanations)")
    print("-" * 50)
    
    if multilingual_datasets_dir.exists():
        for dataset_dir in sorted(multilingual_datasets_dir.iterdir()):
            if dataset_dir.is_dir():
                stats = process_dataset(dataset_dir, output_dirs, dataset_dir.name)
                for format_name, format_stats in stats.items():
                    all_stats[format_name]["processed"] += format_stats["processed"]
                    all_stats[format_name]["skipped"] += format_stats["skipped"]
                    all_stats[format_name]["datasets"] += 1
    else:
        print(f"  Warning: Directory not found: {multilingual_datasets_dir}")
    
    # Print final summary
    print("\n" + "=" * 70)
    print("CONVERSION COMPLETE")
    print("=" * 70)
    
    print("\nStatistics by format:")
    print("-" * 70)
    print(f"{'Format':<30} {'Processed':>12} {'Skipped':>12} {'Total':>12}")
    print("-" * 70)
    
    for format_name in format_names:
        stats = all_stats[format_name]
        processed = stats["processed"]
        skipped = stats["skipped"]
        total = processed + skipped
        print(f"{format_name:<30} {processed:>12} {skipped:>12} {total:>12}")
    
    print("-" * 70)
    
    print(f"\nOutput directories:")
    for format_name in format_names:
        print(f"  {format_name}: {output_dirs[format_name]}")
    
    # Save statistics to JSON
    stats_file = output_base / "conversion_stats.json"
    stats_output = {
        "formats": {
            name: {
                "processed": all_stats[name]["processed"],
                "skipped": all_stats[name]["skipped"],
                "total": all_stats[name]["processed"] + all_stats[name]["skipped"],
                "output_dir": str(output_dirs[name])
            }
            for name in format_names
        }
    }
    with open(stats_file, "w", encoding="utf-8") as f:
        json.dump(stats_output, f, indent=2)
    print(f"\nStatistics saved to: {stats_file}")


if __name__ == "__main__":
    main()

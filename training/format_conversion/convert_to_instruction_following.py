#!/usr/bin/env python3
"""
Convert classification datasets to instruction-following format for ms-swift.

This script converts datasets from the seq_cls format to instruction-following format.

Input format:
{"id": "...", "img_path": "...", "class_label": "...", "text": "...", 
 "native_label": "...", "en_instruction": "...", "native_instruction": "..."}

Output format (instruction-following):
{"messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "<image> ...instruction... Text extracted: ..."},
    {"role": "assistant", "content": "Label: ..."}
], "images": ["..."]}

For non-English datasets: Creates both native and english versions
For English datasets: Creates only english version
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional


# Output format suffixes
CLASSIFICATION_SUFFIX = (
    "\n\nIMPORTANT: Your response must strictly follow this format: "
    "'Label: <label>' where <label> is your classification. "
    "Do not include any additional text or explanation."
)

EXPLANATION_SUFFIX = (
    "\n\nIMPORTANT: Your response must strictly follow this format:\n"
    "'Label: <label>\nExplanation: <explanation>'\n"
    "where <label> is your classification and <explanation> provides "
    "a brief justification for your decision."
)


def get_system_prompt(task_name: str) -> str:
    """Generate a system prompt based on the task name."""
    # Extract the main task type from the dataset name
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


def convert_sample_to_instruction_format(
    sample: Dict,
    task_name: str,
    use_native: bool = False,
    output_mode: str = "default"
) -> Dict:
    """
    Convert a single sample to instruction-following format.
    
    Args:
        sample: The input sample in seq_cls format
        task_name: Name of the task/dataset
        use_native: If True, use native_instruction and native_label; otherwise use English
        output_mode: One of "default", "classification", or "explanation"
    
    Returns:
        Sample in instruction-following format
    """
    system_prompt = get_system_prompt(task_name)
    
    # Get the appropriate instruction and label
    if use_native:
        instruction = sample.get("native_instruction", sample.get("en_instruction", ""))
        label = sample.get("native_label", sample.get("class_label", ""))
    else:
        instruction = sample.get("en_instruction", "")
        label = sample.get("class_label", "")
    
    # Append format suffix based on output_mode
    if output_mode == "classification":
        instruction = instruction + CLASSIFICATION_SUFFIX
    elif output_mode == "explanation":
        instruction = instruction + EXPLANATION_SUFFIX
    
    # Get the text from the meme
    text = sample.get("text", "")
    
    # Construct the user message
    if text and text.strip():
        user_content = f"<image> {instruction} Text extracted: {text}"
    else:
        user_content = f"<image> {instruction}"
    
    # Construct the assistant response based on output_mode
    if output_mode == "explanation":
        # For explanation mode, include a placeholder explanation
        # (In real training data, this would come from the dataset)
        explanation = sample.get("explanation", "Based on the visual and textual content of the meme.")
        assistant_content = f"Label: {label}\nExplanation: {explanation}"
    else:
        assistant_content = f"Label: {label}"
    
    # Build the output format
    output = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ],
        "images": [sample["img_path"]]
    }
    
    return output


def process_jsonl_file(
    input_path: Path,
    output_path: Path,
    task_name: str,
    use_native: bool = False,
    output_mode: str = "default"
) -> int:
    """
    Process a single JSONL file and convert all samples.
    
    Args:
        input_path: Path to input JSONL file
        output_path: Path to output JSONL file
        task_name: Name of the task/dataset
        use_native: Whether to use native labels/instructions
        output_mode: One of "default", "classification", or "explanation"
    
    Returns:
        Number of samples processed
    """
    samples_processed = 0
    
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            line = line.strip()
            if not line:
                continue
            
            try:
                sample = json.loads(line)
                converted = convert_sample_to_instruction_format(
                    sample, task_name, use_native, output_mode
                )
                outfile.write(json.dumps(converted, ensure_ascii=False) + '\n')
                samples_processed += 1
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping malformed JSON line: {e}")
            except Exception as e:
                print(f"  Warning: Error processing sample: {e}")
    
    return samples_processed


def has_native_labels(dataset_path: Path) -> bool:
    """
    Check if the dataset has native labels by examining the first sample.
    """
    jsonl_files = list(dataset_path.glob("*.jsonl"))
    if not jsonl_files:
        return False
    
    with open(jsonl_files[0], 'r', encoding='utf-8') as f:
        first_line = f.readline().strip()
        if first_line:
            sample = json.loads(first_line)
            return "native_label" in sample
    
    return False


def process_dataset(
    dataset_path: Path,
    english_output_dir: Path,
    native_output_dir: Path,
    dataset_name: str,
    output_mode: str = "default"
):
    """
    Process an entire dataset directory.
    
    Args:
        dataset_path: Path to the dataset directory
        english_output_dir: Output directory for English versions
        native_output_dir: Output directory for native versions
        dataset_name: Name of the dataset
        output_mode: One of "default", "classification", or "explanation"
    """
    print(f"\nProcessing dataset: {dataset_name}")
    
    # Check if dataset has native labels
    has_native = has_native_labels(dataset_path)
    
    # Create output directories for this dataset
    english_dataset_dir = english_output_dir / dataset_name
    english_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    if has_native:
        native_dataset_dir = native_output_dir / dataset_name
        native_dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each JSONL file
    for jsonl_file in sorted(dataset_path.glob("*.jsonl")):
        split_name = jsonl_file.stem  # train, val, test
        
        # Process English version
        english_output_path = english_dataset_dir / f"{split_name}.jsonl"
        count_en = process_jsonl_file(
            jsonl_file,
            english_output_path,
            dataset_name,
            use_native=False,
            output_mode=output_mode
        )
        print(f"  {split_name} (English): {count_en} samples")
        
        # Process native version if available
        if has_native:
            native_output_path = native_dataset_dir / f"{split_name}.jsonl"
            count_native = process_jsonl_file(
                jsonl_file,
                native_output_path,
                dataset_name,
                use_native=True,
                output_mode=output_mode
            )
            print(f"  {split_name} (Native): {count_native} samples")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert classification datasets to instruction-following format for ms-swift.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default mode (no format constraints)
  python convert_to_instruction_following.py

  # Classification mode (strict label-only output)
  python convert_to_instruction_following.py --classification

  # Explanation mode (label + explanation output)
  python convert_to_instruction_following.py --explanation
"""
    )
    
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--classification",
        action="store_true",
        help="Add classification format constraint (Label: <label> only, no extra text)"
    )
    mode_group.add_argument(
        "--explanation",
        action="store_true",
        help="Add explanation format constraint (Label: <label> + Explanation: <exp>)"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Determine output mode
    if args.classification:
        output_mode = "classification"
        mode_suffix = "_classification"
    elif args.explanation:
        output_mode = "explanation"
        mode_suffix = "_explanation"
    else:
        output_mode = "default"
        mode_suffix = ""
    
    # Define paths
    base_path = Path("./data/Unified_Labels_FullPath")
    
    # Input directories
    non_english_datasets_dir = base_path / "normalized_datasets"
    english_datasets_dir = base_path / "normalized_classification_en"
    
    # Output directories (include mode suffix if applicable)
    output_base = Path(f"./data/ms_swift_formated/classification{mode_suffix}")
    english_output_dir = output_base / "english"
    native_output_dir = output_base / "native"
    
    # Create output directories
    english_output_dir.mkdir(parents=True, exist_ok=True)
    native_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting Classification Datasets to Instruction-Following Format")
    print(f"Output Mode: {output_mode.upper()}")
    print("=" * 60)
    
    # Process non-English datasets (these have both native and English versions)
    print("\n" + "-" * 40)
    print("Processing Non-English Datasets (with native versions)")
    print("-" * 40)
    
    for dataset_dir in sorted(non_english_datasets_dir.iterdir()):
        if dataset_dir.is_dir():
            process_dataset(
                dataset_dir,
                english_output_dir,
                native_output_dir,
                dataset_dir.name,
                output_mode=output_mode
            )
    
    # Process English-only datasets
    print("\n" + "-" * 40)
    print("Processing English-Only Datasets")
    print("-" * 40)
    
    for dataset_dir in sorted(english_datasets_dir.iterdir()):
        if dataset_dir.is_dir():
            process_dataset(
                dataset_dir,
                english_output_dir,
                native_output_dir,  # Won't be used for English-only
                dataset_dir.name,
                output_mode=output_mode
            )
    
    print("\n" + "=" * 60)
    print("Conversion Complete!")
    print(f"Output Mode: {output_mode.upper()}")
    print(f"English versions saved to: {english_output_dir}")
    print(f"Native versions saved to: {native_output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

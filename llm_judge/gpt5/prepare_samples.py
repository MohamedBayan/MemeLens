#!/usr/bin/env python3
"""
Prepare small sample datasets for LLM-Judge testing.
Picks N samples per dataset from test splits (only items with en_explanation).

Usage:
    python prepare_samples.py --samples_per_dataset 3 --output_dir ./samples
"""
import argparse
import json
import os


EN_BASE = "./data/Unified_Labels_FullPath/normalized_classification_en_with_explanations"
MULTI_BASE = "./data/Unified_Labels_FullPath/normalized_datasets_with_explanations"


def collect_samples(base_dir, n, source_label):
    """Collect n samples from each dataset's test.jsonl that have en_explanation."""
    samples = []
    datasets = sorted(os.listdir(base_dir))

    for dataset_name in datasets:
        test_path = os.path.join(base_dir, dataset_name, "test.jsonl")
        if not os.path.exists(test_path):
            continue

        count = 0
        with open(test_path, 'r', encoding='utf-8') as f:
            for line in f:
                if count >= n:
                    break
                item = json.loads(line.strip())
                if item.get('en_explanation'):
                    item['_source_dataset'] = dataset_name
                    item['_source_type'] = source_label
                    samples.append(item)
                    count += 1

        if count > 0:
            print(f"  {dataset_name}: {count} samples")
        else:
            print(f"  {dataset_name}: SKIPPED (no en_explanation)")

    return samples


def main():
    parser = argparse.ArgumentParser(description='Prepare small sample datasets for LLM-Judge')
    parser.add_argument('--samples_per_dataset', type=int, default=3, help='Number of samples per dataset')
    parser.add_argument('--output_dir', default='./samples', help='Output directory')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n=== English datasets ({EN_BASE}) ===")
    en_samples = collect_samples(EN_BASE, args.samples_per_dataset, "english")

    print(f"\n=== Multilingual datasets ({MULTI_BASE}) ===")
    multi_samples = collect_samples(MULTI_BASE, args.samples_per_dataset, "multilingual")

    all_samples = en_samples + multi_samples

    # Write combined sample file
    output_file = os.path.join(args.output_dir, "judge_test_sample.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in all_samples:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"\nTotal samples: {len(all_samples)} (EN: {len(en_samples)}, Multi: {len(multi_samples)})")
    print(f"Output: {output_file}")


if __name__ == "__main__":
    main()

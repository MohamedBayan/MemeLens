#!/usr/bin/env python3
# Usage:
#   python prepare_samples.py --count 3 --output samples/test_samples.jsonl

import argparse
import json
import os

EN_BASE = "./data/Unified_Labels_FullPath/normalized_classification_en_with_explanations"
MULTI_BASE = "./data/Unified_Labels_FullPath/normalized_datasets_with_explanations"


def discover_datasets():
    datasets = []
    for label, base in [("EN", EN_BASE), ("MULTI", MULTI_BASE)]:
        if not os.path.isdir(base):
            continue
        for ds in sorted(os.listdir(base)):
            test_path = os.path.join(base, ds, "test.jsonl")
            if os.path.exists(test_path):
                datasets.append({"name": ds, "type": label, "test_path": test_path})
    return datasets


def load_items_with_explanations(test_path, max_items=None):
    items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("en_explanation"):
                items.append(item)
                if max_items and len(items) >= max_items:
                    break
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=3, help="Samples per dataset")
    parser.add_argument("--output", default="samples/test_samples.jsonl")
    parser.add_argument("--datasets", nargs="*", help="Specific datasets or omit for all")
    args = parser.parse_args()

    all_datasets = discover_datasets()
    if args.datasets:
        all_datasets = [d for d in all_datasets if d["name"] in args.datasets]

    print(f"Found {len(all_datasets)} datasets")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    total = 0
    with open(args.output, "w", encoding="utf-8") as out:
        for ds in all_datasets:
            items = load_items_with_explanations(ds["test_path"], args.count)
            for item in items:
                item["_source_dataset"] = ds["name"]
                item["_source_type"] = ds["type"]
                out.write(json.dumps(item, ensure_ascii=False) + "\n")
            total += len(items)
            print(f"  {ds['name']}: {len(items)} samples")

    print(f"\nTotal: {total} samples -> {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Usage:
#   python batch_merge.py --results outputs/run_20260218_120000/all_results.jsonl --output_dir results/full_run

import argparse
import json
import os
from collections import defaultdict
import numpy as np

EN_BASE = "./data/Unified_Labels_FullPath/normalized_classification_en_with_explanations"
MULTI_BASE = "./data/Unified_Labels_FullPath/normalized_datasets_with_explanations"


def load_dataset_index(ds_name):
    """Load all items from a dataset's test.jsonl, indexed by id."""
    for base in [EN_BASE, MULTI_BASE]:
        test_path = os.path.join(base, ds_name, "test.jsonl")
        if os.path.exists(test_path):
            index = {}
            with open(test_path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line.strip())
                    index[str(item.get("id", ""))] = item
            return index
    return {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="Path to all_results.jsonl")
    parser.add_argument("--output_dir", default="results/full_run")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.results, "r", encoding="utf-8") as f:
        results = [json.loads(line.strip()) for line in f if line.strip()]

    # Group by dataset
    by_dataset = defaultdict(list)
    for r in results:
        custom_id = r.get("custom_id", "")
        ds_name, item_id = custom_id.split("::", 1) if "::" in custom_id else ("unknown", custom_id)
        by_dataset[ds_name].append((item_id, r))

    # Process each dataset with a preloaded index
    grouped_records = defaultdict(list)
    for ds_name, pairs in by_dataset.items():
        index = load_dataset_index(ds_name)
        for item_id, r in pairs:
            original = index.get(str(item_id), {"id": item_id})
            record = original.copy()
            record["_source_dataset"] = ds_name
            record["judge_response"] = r.get("response_text", "")
            record["judge_scores"] = r.get("scores")
            grouped_records[ds_name].append(record)
    by_dataset = grouped_records

    all_scores = defaultdict(list)
    total_scored = 0
    total_failed = 0

    for ds_name, items in by_dataset.items():
        output_file = os.path.join(args.output_dir, f"judge_{ds_name}.jsonl")
        with open(output_file, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        scored = sum(1 for it in items if it.get("judge_scores"))
        failed = len(items) - scored
        total_scored += scored
        total_failed += failed

        for it in items:
            if it.get("judge_scores"):
                for k, v in it["judge_scores"].items():
                    if isinstance(v, dict) and "score" in v:
                        all_scores[k].append(v["score"])

        print(f"  {ds_name}: {scored}/{len(items)} scored -> {output_file}")

    print(f"\nTotal: {total_scored} scored, {total_failed} failed")

    summary = {
        "total_items": total_scored + total_failed,
        "scored": total_scored,
        "failed": total_failed,
        "criteria": {},
    }
    for k, vals in all_scores.items():
        summary["criteria"][k] = {
            "mean": round(float(np.mean(vals)), 3),
            "std": round(float(np.std(vals)), 3),
            "n": len(vals),
        }
        print(f"  {k}: {np.mean(vals):.2f} +/- {np.std(vals):.2f}")

    if summary["criteria"]:
        overall = np.mean([summary["criteria"][k]["mean"] for k in summary["criteria"]])
        summary["overall_mean"] = round(overall, 3)
        print(f"\n  Overall: {overall:.2f}")

    summary_file = os.path.join(args.output_dir, "summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSummary -> {summary_file}")


if __name__ == "__main__":
    main()

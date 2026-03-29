#!/usr/bin/env python3
"""
Compute final aligned summaries for GPT-5 and Gemini-2.5-Pro LLM-as-Judge results.
Uses the same 38 datasets for both models (excluding emotion_ro__RoMemes and sentiment/emotion datasets).
"""

import json
import os
import glob
import numpy as np
from collections import defaultdict
from datetime import datetime

CRITERIA = ["informativeness", "clarity", "plausibility", "faithfulness"]

# Datasets to EXCLUDE from ACL rebuttal (sentiment/emotion for journal)
EXCLUDE_DATASETS = {
    "emotion_ro__RoMemes",
    "overall_sentiment_en__memotion",
    "sentiment_bn__BanglaAbuseMeme",
    "sentiment_category_en__MET_Meme",
    "sentiment_category_zh__MET_Meme",
    "sentiment_degree_en__MET_Meme",
    "sentiment_degree_zh__MET_Meme",
    "sentiment_ro__RoMemes",
}

GPT5_DIR = "./LLM-Judge/GPT-5/results/full_run"
GEMINI_DIR = "./LLM-Judge/Gemini-2.5-Pro/results/full_run"


def extract_dataset_name(filename, model):
    """Extract dataset name from judge result filename."""
    basename = os.path.basename(filename).replace("judge_", "").replace(".jsonl", "")
    if model == "gpt5":
        # Remove EN_ or MULTI_ prefix
        if basename.startswith("EN_"):
            basename = basename[3:]
        elif basename.startswith("MULTI_"):
            basename = basename[6:]
    return basename


def load_judge_results(results_dir, model):
    """Load all judge results from a directory."""
    datasets = {}
    for fpath in sorted(glob.glob(os.path.join(results_dir, "judge_*.jsonl"))):
        # Skip excluded files
        if "excluded_" in os.path.basename(fpath):
            continue

        dataset_name = extract_dataset_name(fpath, model)

        # Skip excluded datasets
        if dataset_name in EXCLUDE_DATASETS:
            continue

        items = []
        scored_items = []
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                items.append(item)
                # Check if scored
                scores = item.get("judge_scores", {})
                if scores and all(c in scores for c in CRITERIA):
                    try:
                        vals = {}
                        for c in CRITERIA:
                            v = scores[c]
                            # Handle both {criterion: score} and {criterion: {score: X, justification: Y}}
                            if isinstance(v, dict):
                                vals[c] = float(v["score"])
                            else:
                                vals[c] = float(v)
                        if all(1 <= v <= 5 for v in vals.values()):
                            scored_items.append(vals)
                    except (ValueError, TypeError, KeyError):
                        pass

        datasets[dataset_name] = {
            "total": len(items),
            "scored": len(scored_items),
            "scores": scored_items,
        }

    return datasets


def compute_summary(datasets):
    """Compute per-dataset and overall statistics."""
    by_dataset = {}
    all_scores = {c: [] for c in CRITERIA}

    for name in sorted(datasets.keys()):
        info = datasets[name]
        ds_stats = {}
        for c in CRITERIA:
            vals = [s[c] for s in info["scores"]]
            if vals:
                ds_stats[c] = {
                    "mean": round(np.mean(vals), 2),
                    "std": round(np.std(vals), 2),
                    "count": len(vals),
                }
                all_scores[c].extend(vals)
            else:
                ds_stats[c] = {"mean": 0, "std": 0, "count": 0}

        # Per-dataset mean
        means = [ds_stats[c]["mean"] for c in CRITERIA if ds_stats[c]["count"] > 0]
        ds_stats["mean_all"] = round(np.mean(means), 2) if means else 0
        ds_stats["total"] = info["total"]
        ds_stats["scored"] = info["scored"]
        by_dataset[name] = ds_stats

    # Overall
    overall = {}
    for c in CRITERIA:
        if all_scores[c]:
            overall[c] = {
                "mean": round(np.mean(all_scores[c]), 2),
                "std": round(np.std(all_scores[c]), 2),
                "count": len(all_scores[c]),
            }
    means = [overall[c]["mean"] for c in CRITERIA if c in overall]
    overall["mean_all"] = round(np.mean(means), 2) if means else 0

    total_items = sum(d["total"] for d in datasets.values())
    scored_items = sum(d["scored"] for d in datasets.values())

    return {
        "timestamp": datetime.now().isoformat(),
        "num_datasets": len(datasets),
        "total_items": total_items,
        "scored_items": scored_items,
        "overall": overall,
        "by_dataset": by_dataset,
    }


def print_comparison(gpt5_summary, gemini_summary):
    """Print side-by-side comparison."""
    print("=" * 120)
    print("FINAL LLM-as-Judge COMPARISON: GPT-5 vs Gemini-2.5-Pro")
    print(f"Datasets: {gpt5_summary['num_datasets']} (GPT-5) | {gemini_summary['num_datasets']} (Gemini)")
    print("=" * 120)

    # Overall
    print("\n" + "─" * 120)
    print("OVERALL SCORES")
    print("─" * 120)
    print(f"{'Criterion':<20} {'GPT-5':>12} {'Gemini':>12} {'Diff':>10}")
    print("─" * 60)
    for c in CRITERIA:
        g5 = gpt5_summary["overall"].get(c, {}).get("mean", 0)
        gm = gemini_summary["overall"].get(c, {}).get("mean", 0)
        diff = gm - g5
        print(f"{c:<20} {g5:>12.2f} {gm:>12.2f} {diff:>+10.2f}")
    g5_all = gpt5_summary["overall"].get("mean_all", 0)
    gm_all = gemini_summary["overall"].get("mean_all", 0)
    print("─" * 60)
    print(f"{'OVERALL MEAN':<20} {g5_all:>12.2f} {gm_all:>12.2f} {gm_all-g5_all:>+10.2f}")

    print(f"\n{'Items scored':<20} {gpt5_summary['scored_items']:>12,} {gemini_summary['scored_items']:>12,}")
    print(f"{'Total items':<20} {gpt5_summary['total_items']:>12,} {gemini_summary['total_items']:>12,}")
    print(f"{'Score rate':<20} {gpt5_summary['scored_items']/gpt5_summary['total_items']*100:>11.1f}% {gemini_summary['scored_items']/gemini_summary['total_items']*100:>11.1f}%")

    # Per-dataset comparison
    print("\n" + "─" * 120)
    print("PER-DATASET COMPARISON (sorted by dataset name)")
    print("─" * 120)
    print(f"{'Dataset':<45} {'GPT-5 Mean':>12} {'Gemini Mean':>12} {'Diff':>8} {'GPT5 N':>8} {'Gem N':>8}")
    print("─" * 120)

    all_datasets = sorted(set(list(gpt5_summary["by_dataset"].keys()) + list(gemini_summary["by_dataset"].keys())))
    for ds in all_datasets:
        g5_ds = gpt5_summary["by_dataset"].get(ds, {})
        gm_ds = gemini_summary["by_dataset"].get(ds, {})
        g5_mean = g5_ds.get("mean_all", 0)
        gm_mean = gm_ds.get("mean_all", 0)
        g5_n = g5_ds.get("scored", 0)
        gm_n = gm_ds.get("scored", 0)
        diff = gm_mean - g5_mean if g5_mean and gm_mean else 0
        g5_str = f"{g5_mean:.2f}" if g5_mean else "N/A"
        gm_str = f"{gm_mean:.2f}" if gm_mean else "N/A"
        diff_str = f"{diff:+.2f}" if g5_mean and gm_mean else "N/A"
        print(f"{ds:<45} {g5_str:>12} {gm_str:>12} {diff_str:>8} {g5_n:>8} {gm_n:>8}")

    # MUTE highlight
    print("\n" + "─" * 120)
    print("MUTE (Hateful_bn__MUTE) DETAIL")
    print("─" * 120)
    for model_name, summary in [("GPT-5", gpt5_summary), ("Gemini-2.5-Pro", gemini_summary)]:
        mute = summary["by_dataset"].get("Hateful_bn__MUTE", {})
        if mute:
            print(f"\n  {model_name}:")
            print(f"    Scored: {mute.get('scored', 0)}/{mute.get('total', 0)}")
            for c in CRITERIA:
                stats = mute.get(c, {})
                print(f"    {c}: {stats.get('mean', 0):.2f} ± {stats.get('std', 0):.2f}")
            print(f"    Overall: {mute.get('mean_all', 0):.2f}")


def main():
    print("Loading GPT-5 results...")
    gpt5_datasets = load_judge_results(GPT5_DIR, "gpt5")
    print(f"  Found {len(gpt5_datasets)} datasets")

    print("Loading Gemini-2.5-Pro results...")
    gemini_datasets = load_judge_results(GEMINI_DIR, "gemini")
    print(f"  Found {len(gemini_datasets)} datasets")

    # Compute aligned set (common datasets only)
    common = set(gpt5_datasets.keys()) & set(gemini_datasets.keys())
    print(f"\nCommon datasets: {len(common)}")

    gpt5_only = set(gpt5_datasets.keys()) - common
    gemini_only = set(gemini_datasets.keys()) - common
    if gpt5_only:
        print(f"GPT-5 only: {gpt5_only}")
    if gemini_only:
        print(f"Gemini only: {gemini_only}")

    # Compute summaries on common datasets
    gpt5_common = {k: v for k, v in gpt5_datasets.items() if k in common}
    gemini_common = {k: v for k, v in gemini_datasets.items() if k in common}

    gpt5_summary = compute_summary(gpt5_common)
    gemini_summary = compute_summary(gemini_common)

    # Save summaries
    gpt5_out = os.path.join(GPT5_DIR, "judge_final_summary_38datasets.json")
    gemini_out = os.path.join(GEMINI_DIR, "judge_final_summary_38datasets.json")

    with open(gpt5_out, "w") as f:
        json.dump(gpt5_summary, f, indent=2)
    print(f"\nSaved GPT-5 summary: {gpt5_out}")

    with open(gemini_out, "w") as f:
        json.dump(gemini_summary, f, indent=2)
    print(f"Saved Gemini summary: {gemini_out}")

    # Print comparison
    print_comparison(gpt5_summary, gemini_summary)


if __name__ == "__main__":
    main()

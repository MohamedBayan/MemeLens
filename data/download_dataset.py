#!/usr/bin/env python3
"""
Download the MemeLens-VLM dataset from HuggingFace.

Usage:
    # Download everything
    python data/download_dataset.py

    # Download specific language(s)
    python data/download_dataset.py --languages en ar

    # Download specific dataset(s)
    python data/download_dataset.py --datasets Hateful_en_FHM abuse_bn__BanglaAbuseMeme

    # Download to custom directory
    python data/download_dataset.py --output_dir ./my_data

    # List available datasets
    python data/download_dataset.py --list
"""

import argparse
import os
import json

DATASET_ID = "QCRI/MemeLens-VLM"

LANGUAGE_DATASETS = {
    "ar": ["Hateful_ar__Prop2Hate-Meme", "propoganda_ar_ArMeme"],
    "bn": ["abuse_bn__BanglaAbuseMeme", "Hateful_bn__MUTE", "sarcasm_bn__BanglaAbuseMeme",
            "sentiment_bn__BanglaAbuseMeme", "vulgar_bn__BanglaAbuseMeme"],
    "de": ["Hateful_de__Multi3Hate"],
    "en": ["Harmful_Covid_en__HarMeme", "Harmful_en__HarMeme", "Hateful_en_FHM",
            "Hateful_en__MIMIC_Islamophpbia", "Hateful_en__MMHS", "Hateful_en__Multi3Hate",
            "humour_en__memotion", "intention_detection_en__MET_Meme",
            "metaphor_occurrence_en__MET_Meme", "misogynous_en__MAMI",
            "motivational_en__memotion", "objectification_en__MAMI",
            "offensive_en__memotion", "offensiveness_detection_en__MET_Meme",
            "overall_sentiment_en__memotion", "sarcasm_en__memotion",
            "sentiment_category_en__MET_Meme", "sentiment_degree_en__MET_Meme",
            "shaming_en__MAMI", "stereotype_en__MAMI", "Target_Covid_en__HarMeme",
            "Target_en__HarMeme", "violence_en__MAMI"],
    "es": ["Hateful_es__Multi3Hate"],
    "hi": ["Hateful_hi__Multi3Hate", "Misogyny_Categories_hi_en__MIMIC2024",
            "Misogyny_hi_en__MIMIC2024"],
    "ro": ["deepfake_ro__RoMemes", "emotion_ro__RoMemes", "political_ro__RoMemes",
            "sentiment_ro__RoMemes"],
    "ru": ["toxic_ru__Toxic_Memes_Detection_Dataset"],
    "zh": ["Hateful_zh__Multi3Hate", "intention_detection_zh__MET_Meme",
            "metaphor_occurrence_zh__MET_Meme", "offensiveness_detection_zh__MET_Meme",
            "sentiment_category_zh__MET_Meme", "sentiment_degree_zh__MET_Meme"],
}


def list_datasets():
    """Print all available datasets grouped by language."""
    print(f"Dataset: {DATASET_ID}")
    print(f"Total: 46 datasets across 9 languages\n")
    for lang in sorted(LANGUAGE_DATASETS.keys()):
        datasets = LANGUAGE_DATASETS[lang]
        print(f"  {lang} ({len(datasets)} datasets):")
        for ds in datasets:
            print(f"    - {ds}")
    print()


def download(output_dir="./data/memelens", languages=None, datasets=None):
    """Download dataset from HuggingFace."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_tree
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    from huggingface_hub import HfApi
    api = HfApi()

    # Determine what to download
    targets = []
    if datasets:
        # Find language for each dataset
        for ds in datasets:
            for lang, ds_list in LANGUAGE_DATASETS.items():
                if ds in ds_list:
                    targets.append((lang, ds))
                    break
    elif languages:
        for lang in languages:
            if lang in LANGUAGE_DATASETS:
                for ds in LANGUAGE_DATASETS[lang]:
                    targets.append((lang, ds))
            else:
                print(f"Warning: Unknown language '{lang}'. Available: {sorted(LANGUAGE_DATASETS.keys())}")
    else:
        # Download everything
        for lang, ds_list in LANGUAGE_DATASETS.items():
            for ds in ds_list:
                targets.append((lang, ds))

    print(f"Downloading {len(targets)} datasets to {output_dir}...")

    for lang, ds_name in targets:
        prefix = f"{lang}/{ds_name}"
        local_dir = os.path.join(output_dir, lang, ds_name)
        os.makedirs(local_dir, exist_ok=True)

        print(f"  Downloading {prefix}...")
        try:
            # Download JSONL files
            for split in ["train.jsonl", "test.jsonl", "val.jsonl"]:
                try:
                    hf_hub_download(
                        repo_id=DATASET_ID,
                        repo_type="dataset",
                        filename=f"{prefix}/{split}",
                        local_dir=output_dir,
                    )
                except Exception:
                    pass

            # Download images
            try:
                files = list(api.list_repo_tree(
                    DATASET_ID, repo_type="dataset",
                    path_in_repo=f"{prefix}/images",
                ))
                for f in files:
                    if hasattr(f, 'path'):
                        hf_hub_download(
                            repo_id=DATASET_ID,
                            repo_type="dataset",
                            filename=f.path,
                            local_dir=output_dir,
                        )
            except Exception:
                pass

        except Exception as e:
            print(f"    Error: {e}")

    print(f"\nDownload complete! Data saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download MemeLens-VLM dataset")
    parser.add_argument("--output_dir", default="./data/memelens", help="Output directory")
    parser.add_argument("--languages", nargs="+", help="Languages to download (e.g., en ar bn)")
    parser.add_argument("--datasets", nargs="+", help="Specific datasets to download")
    parser.add_argument("--list", action="store_true", help="List available datasets")
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    download(args.output_dir, args.languages, args.datasets)


if __name__ == "__main__":
    main()

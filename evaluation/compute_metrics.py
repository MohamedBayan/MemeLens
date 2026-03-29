#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
score_dataset.py
================
A configurable evaluation pipeline for JSONL datasets containing:
- classification labels
- (optionally) free-text explanations

It computes:
- Classification metrics + saves confusion matrix plot
- Explanation metrics (BERTScore, ROUGE, BLEU, METEOR) if applicable
- Outputs results as metrics.json


python scripts/src/compute_metrics.py \
  --data results/multimodal/seq_cls/Hateful_en__MMHS/qwen3-vl-8b-instruct.jsonl \
  --out_dir scores/seq_cls/Hateful_en__MMHS/qwen3-vl-8b-instruct/
"""

import argparse
import json
import os
import random
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # for servers without display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from bert_score import score as bertscore
from nltk import download as nltk_download
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.metrics import multilabel_confusion_matrix, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoTokenizer
import evaluate as hf_evaluate


for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4"]:
    nltk_download(pkg, quiet=True)

# More flexible regex patterns that can match label/explanation anywhere in text
LABEL_RE = re.compile(r"(?:^|\n|.*?)Label:\s*([^\n]+)", re.IGNORECASE)
EXPL_RE = re.compile(r"(?:^|\n|.*?)Explanation:\s*(.+?)(?:\n|$)", re.IGNORECASE | re.DOTALL)


def normalize_label(label: str | None) -> str | None:
    """
    Normalize label by:
    - Converting to lowercase
    - Replacing underscores with hyphens
    - Stripping whitespace
    - Removing noise like **, bullets, etc.
    - Replacing 'non-' prefix with 'not-'
    """
    if label is None or pd.isna(label):
        return None
    
    label = str(label).strip().lower()
    
    # Remove common noise patterns: **, *, bullets, etc.
    label = re.sub(r'\*+', '', label)  # Remove asterisks
    label = re.sub(r'^[-•·●▪▫◦◘◙○◌]\s*', '', label)  # Remove bullet points
    label = re.sub(r'\s+', ' ', label)  # Normalize whitespace
    label = label.strip()
    
    # Replace underscores with hyphens
    label = label.replace("_", "-")
    
    # Normalize non- to not-
    if label.startswith("non-"):
        label = "not-" + label[4:]
    
    return label


def extract_label_and_explanation(text: str) -> tuple[str | None, str | None]:
    """
    Extract Label and Explanation from a string block.
    Handles cases where there might be text before the label/explanation.
    """
    if pd.isna(text):
        return None, None
    
    text = str(text)
    
    label_match = LABEL_RE.search(text)
    expl_match = EXPL_RE.search(text)
    
    label = label_match.group(1).strip() if label_match else None
    explanation = expl_match.group(1).strip() if expl_match else None
    
    # Normalize the label
    label = normalize_label(label)
    
    return label, explanation


def read_jsonl_select_columns(file_path: str | Path, columns=("response", "labels")) -> pd.DataFrame:
    df = pd.read_json(file_path, lines=True)
    if not set(columns).issubset(df.columns):
        raise ValueError(f"Missing expected columns {columns} in file {file_path}")
    return df[list(columns)]


def detect_format(df: pd.DataFrame) -> str:
    """
    Detect if the dataset is in 'seq_cls', 'seq_cls_multilabel', 'text', or 'text_multilabel' format.
    
    seq_cls: response and labels are integers (e.g., 0, 1)
    seq_cls_multilabel: response and labels are lists of integers (e.g., [0, 1], [2])
    text: response and labels contain text with 'Label:' and optionally 'Explanation:'
    text_multilabel: same as text but labels are comma-separated (e.g., 'Label: Objectification, Prejudice')
    """
    sample = df.iloc[0]
    response_val = sample["response"]
    labels_val = sample["labels"]
    
    # Check if both are lists (seq_cls_multilabel format)
    if isinstance(response_val, list) and isinstance(labels_val, list):
        return "seq_cls_multilabel"
    
    # Check if both are integers (seq_cls format)
    if isinstance(response_val, (int, np.integer)) and isinstance(labels_val, (int, np.integer)):
        return "seq_cls"
    
    # Check if they are strings containing "Label:" pattern (text format)
    if isinstance(response_val, str) and "Label:" in response_val:
        return "text"
    if isinstance(labels_val, str) and "Label:" in labels_val:
        return "text"
    
    # Default to text format for backwards compatibility
    return "text"


def build_multilabel_mapping_from_data(df: pd.DataFrame) -> tuple[dict, int]:
    """
    Build a label mapping dynamically from the data by analyzing ground truth labels.
    
    Returns:
        label_map: Dict mapping label strings to integer indices
        num_classes: Number of unique base classes
    """
    all_labels = set()
    
    for labels_val in df["labels"]:
        if isinstance(labels_val, list):
            # seq_cls_multilabel format - labels are already integers
            all_labels.update(labels_val)
        elif isinstance(labels_val, str):
            # text format - extract label string
            label_match = LABEL_RE.search(labels_val)
            if label_match:
                label_str = label_match.group(1).strip()
                # Skip 'Unspecified' or empty
                if label_str.lower() != 'unspecified' and label_str:
                    # Split by comma for multilabel
                    for part in label_str.split(','):
                        part = part.strip()
                        if part and part.lower() != 'unspecified':
                            all_labels.add(part)
    
    # Sort labels for consistent ordering and create mapping
    sorted_labels = sorted(all_labels, key=lambda x: str(x))
    label_map = {label: idx for idx, label in enumerate(sorted_labels)}
    
    return label_map, len(label_map)


def parse_multilabel_text(label_str: str, label_map: dict) -> list:
    """
    Parse a comma-separated label string into a list of integer indices.
    
    E.g., 'Objectification, Prejudice' -> [0, 1]
          'Unspecified' -> []
    """
    if label_str is None or pd.isna(label_str) or not label_str.strip():
        return []
    
    label_str = str(label_str).strip()
    
    # Handle 'Unspecified' as empty
    if label_str.lower() == 'unspecified':
        return []
    
    # Split by comma and convert each label to its index
    labels = []
    for part in label_str.split(','):
        part = part.strip()
        if part and part in label_map:
            labels.append(label_map[part])
    
    return sorted(list(set(labels)))  # Remove duplicates and sort


def extract_label_multilabel(text: str, label_map: dict) -> tuple[list, str | None]:
    """
    Extract Label (as list of indices) and Explanation from a string block for multilabel datasets.
    """
    if pd.isna(text):
        return [], None
    
    text = str(text)
    
    label_match = LABEL_RE.search(text)
    expl_match = EXPL_RE.search(text)
    
    label_str = label_match.group(1).strip() if label_match else ""
    explanation = expl_match.group(1).strip() if expl_match else None
    
    # Parse the comma-separated labels into list of indices
    label_list = parse_multilabel_text(label_str, label_map)
    
    return label_list, explanation


def fix_invalid_labels(label: str | None, valid_labels: set) -> str:
    """If label is invalid or missing, randomly assign from valid_labels."""
    if label in valid_labels:
        return label
    return random.choice(list(valid_labels))


def evaluate_classification(df: pd.DataFrame,
                            gold_col: str,
                            pred_col: str,
                            cm_path: Path | None = None) -> dict:
    y_true, y_pred = df[gold_col], df[pred_col]

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }

    # Compute per-label metrics
    labels = sorted(y_true.unique())
    per_label_precision = precision_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_label_recall = recall_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    per_label_f1 = f1_score(y_true, y_pred, average=None, zero_division=0, labels=labels)
    
    for i, label in enumerate(labels):
        # Sanitize label name for use as dict key (replace special chars with underscores)
        label_name = str(label).replace(' ', '_').replace('-', '_').replace('/', '_')
        metrics[f"precision_{label_name}"] = float(per_label_precision[i])
        metrics[f"recall_{label_name}"] = float(per_label_recall[i])
        metrics[f"f1_{label_name}"] = float(per_label_f1[i])

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                xticklabels=labels, yticklabels=labels, cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if cm_path:
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300)
    plt.close()

    return metrics


def evaluate_multilabel_classification(y_true_lists: list,
                                       y_pred_lists: list,
                                       num_classes: int,
                                       cm_path: Path | None = None,
                                       class_names: list = None) -> dict:
    """
    Evaluate multilabel classification.
    
    Args:
        y_true_lists: List of lists containing true label indices
        y_pred_lists: List of lists containing predicted label indices
        num_classes: Number of classes in the multilabel problem
        cm_path: Path to save confusion matrix plot
        class_names: Names of classes for plotting
    
    Returns:
        Dictionary of metrics
    """
    # Use MultiLabelBinarizer to convert lists to binary matrices
    mlb = MultiLabelBinarizer(classes=list(range(num_classes)))
    y_true_bin = mlb.fit_transform(y_true_lists)
    y_pred_bin = mlb.transform(y_pred_lists)
    
    # Compute multilabel metrics
    metrics = {
        # Subset accuracy (exact match ratio)
        "subset_accuracy": accuracy_score(y_true_bin, y_pred_bin),
        
        # Hamming loss (fraction of wrong labels)
        "hamming_loss": hamming_loss(y_true_bin, y_pred_bin),
        
        # Micro-averaged metrics (treats each label prediction as separate)
        "precision_micro": precision_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        "recall_micro": recall_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        "f1_micro": f1_score(y_true_bin, y_pred_bin, average="micro", zero_division=0),
        
        # Macro-averaged metrics (average across classes)
        "precision_macro": precision_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true_bin, y_pred_bin, average="macro", zero_division=0),
        
        # Weighted metrics
        "precision_weighted": precision_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        "recall_weighted": recall_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_true_bin, y_pred_bin, average="weighted", zero_division=0),
        
        # Sample-averaged metrics (average across samples)
        "precision_samples": precision_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "recall_samples": recall_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
        "f1_samples": f1_score(y_true_bin, y_pred_bin, average="samples", zero_division=0),
    }
    
    # Per-class metrics
    per_class_precision = precision_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    per_class_recall = recall_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true_bin, y_pred_bin, average=None, zero_division=0)
    
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    for i, name in enumerate(class_names):
        metrics[f"precision_{name}"] = float(per_class_precision[i])
        metrics[f"recall_{name}"] = float(per_class_recall[i])
        metrics[f"f1_{name}"] = float(per_class_f1[i])
    
    # Create per-label confusion matrix visualization
    if cm_path:
        mcm = multilabel_confusion_matrix(y_true_bin, y_pred_bin)
        
        fig, axes = plt.subplots(1, num_classes, figsize=(4 * num_classes, 4))
        if num_classes == 1:
            axes = [axes]
        
        for i, (ax, name) in enumerate(zip(axes, class_names)):
            cm = mcm[i]
            sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
                       xticklabels=['Neg', 'Pos'], yticklabels=['Neg', 'Pos'],
                       ax=ax, cbar=False)
            ax.set_title(f"{name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
        
        plt.suptitle("Per-Label Confusion Matrices", y=1.02)
        plt.tight_layout()
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    return metrics


def compute_bertscore(preds, refs, arabic=False) -> dict:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = "aubmindlab/bert-base-arabertv2" if arabic else "bert-base-multilingual-uncased"

    preds = [" ".join(str(p).split()[:1024]) for p in preds]
    refs = [" ".join(str(r).split()[:1024]) for r in refs]

    # Check for empty candidates and print diagnostics
    empty_pred_indices = [i for i, p in enumerate(preds) if not p or not p.strip()]
    empty_ref_indices = [i for i, r in enumerate(refs) if not r or not r.strip()]
    
    if empty_pred_indices or empty_ref_indices:
        print(f"\n{'='*80}")
        print(f"⚠️  EMPTY SENTENCE DETECTION REPORT")
        print(f"{'='*80}")
        
        if empty_pred_indices:
            print(f"\n🔴 Found {len(empty_pred_indices)} empty PREDICTION(s) at indices: {empty_pred_indices[:20]}{'...' if len(empty_pred_indices) > 20 else ''}")
            print(f"   Total empty predictions: {len(empty_pred_indices)}")
            print(f"\n   Sample indices with empty predictions (showing first 5):")
            for idx in empty_pred_indices[:5]:
                print(f"   - Index {idx}: pred='{preds[idx]}' | ref='{refs[idx][:100]}...'")
        
        if empty_ref_indices:
            print(f"\n🔴 Found {len(empty_ref_indices)} empty REFERENCE(s) at indices: {empty_ref_indices[:20]}{'...' if len(empty_ref_indices) > 20 else ''}")
            print(f"   Total empty references: {len(empty_ref_indices)}")
            print(f"\n   Sample indices with empty references (showing first 5):")
            for idx in empty_ref_indices[:5]:
                print(f"   - Index {idx}: pred='{preds[idx][:100]}...' | ref='{refs[idx]}'")
        
        # Check for very long sequences
        long_preds = [(i, len(p.split())) for i, p in enumerate(preds) if len(p.split()) > 500]
        if long_preds:
            print(f"\n⚠️  Found {len(long_preds)} predictions with >500 tokens:")
            for idx, length in long_preds[:5]:
                print(f"   - Index {idx}: {length} tokens")
        
        print(f"{'='*80}\n")

    P, R, F = bertscore(
        cands=preds,
        refs=refs,
        model_type=model,
        device=device,
        num_layers=12,
        batch_size=32,
        verbose=False
    )
    return {
        "bertscore_precision": P.mean().item(),
        "bertscore_recall": R.mean().item(),
        "bertscore_f1": F.mean().item()
    }


def compute_rouge(preds, refs, arabic=False) -> dict:
    model_name = "aubmindlab/bert-base-arabertv2" if arabic else "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], tokenizer=tokenizer)
    r1, r2, rL = [], [], []

    for p, r in zip(preds, refs):
        scores = scorer.score(r, p)
        r1.append(scores["rouge1"].fmeasure)
        r2.append(scores["rouge2"].fmeasure)
        rL.append(scores["rougeL"].fmeasure)

    return {
        "rouge1": float(np.mean(r1)),
        "rouge2": float(np.mean(r2)),
        "rougeL": float(np.mean(rL))
    }


def compute_bleu_meteor(preds, refs) -> dict:
    smooth_fn = SmoothingFunction().method1
    bleu = corpus_bleu([[word_tokenize(r)] for r in refs],
                       [word_tokenize(p) for p in preds],
                       smoothing_function=smooth_fn)
    meteor_scores = [meteor_score([word_tokenize(r)], word_tokenize(p)) for r, p in zip(refs, preds)]
    return {
        "bleu": bleu,
        "meteor": float(np.mean(meteor_scores))
    }


def evaluate_explanations(preds, refs, arabic=False) -> dict:
    # Print overall statistics about the explanations
    total_samples = len(preds)
    empty_preds = sum(1 for p in preds if not p or not str(p).strip())
    empty_refs = sum(1 for r in refs if not r or not str(r).strip())
    
    print(f"\n{'='*80}")
    print(f"📊 EXPLANATION EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Total samples: {total_samples}")
    print(f"Empty predictions: {empty_preds} ({empty_preds/total_samples*100:.2f}%)")
    print(f"Empty references: {empty_refs} ({empty_refs/total_samples*100:.2f}%)")
    print(f"Valid pairs: {total_samples - max(empty_preds, empty_refs)}")
    print(f"{'='*80}\n")
    
    return {
        **compute_bertscore(preds, refs, arabic=arabic),
        **compute_rouge(preds, refs, arabic=arabic),
        **compute_bleu_meteor(preds, refs)
    }


def cli():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--has_explanation", action="store_true", help="If present, compute explanation metrics")
    parser.add_argument("--is_arabic", action="store_true", help="Use AraBERT models/tokenizers")
    parser.add_argument("--out_dir", required=True, help="Directory to save metrics.json & confusion_matrix.png")
    parser.add_argument("--multilabel", action="store_true", help="If present, treat as multilabel classification")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes for multilabel classification (default: auto-detect from data)")
    return parser.parse_args()


def main():
    args = cli()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_jsonl_select_columns(args.data)
    
    # Detect format
    data_format = detect_format(df)
    print(f"📋 Detected format: {data_format}")
    
    # Check if this is a multilabel task (auto-detect from format or explicit flag)
    is_multilabel = args.multilabel or data_format == "seq_cls_multilabel"
    
    if is_multilabel:
        print(f"📋 Running MULTILABEL evaluation")
        
        # Build label mapping dynamically from the data
        label_map, num_classes_detected = build_multilabel_mapping_from_data(df)
        
        # Use provided num_classes or auto-detected
        num_classes = args.num_classes if args.num_classes is not None else num_classes_detected
        print(f"📋 Number of classes: {num_classes}")
        print(f"📋 Label mapping: {label_map}")
        
        # Get class names from the label map
        class_names = [label for label, idx in sorted(label_map.items(), key=lambda x: x[1])]
        if not class_names:
            class_names = [f"class_{i}" for i in range(num_classes)]
        
        if data_format == "seq_cls_multilabel":
            # For seq_cls_multilabel format, response and labels are already lists of integers
            y_true_lists = df["labels"].tolist()
            y_pred_lists = df["response"].tolist()
            has_explanation = False
        else:
            # For text format multilabel, parse comma-separated labels
            df[["labels_label", "labels_explanation"]] = pd.DataFrame(
                df["labels"].apply(lambda x: extract_label_multilabel(x, label_map)).tolist(), 
                index=df.index
            )
            df[["response_label", "response_explanation"]] = pd.DataFrame(
                df["response"].apply(lambda x: extract_label_multilabel(x, label_map)).tolist(), 
                index=df.index
            )
            y_true_lists = df["labels_label"].tolist()
            y_pred_lists = df["response_label"].tolist()
            has_explanation = args.has_explanation
        
        # Evaluate multilabel classification
        metrics = evaluate_multilabel_classification(
            y_true_lists, y_pred_lists, num_classes,
            cm_path=out_dir / "confusion_matrix.png",
            class_names=class_names
        )
        
        if has_explanation and data_format != "seq_cls_multilabel":
            preds = df["response_explanation"].fillna("").tolist()
            refs = df["labels_explanation"].fillna("").tolist()
            metrics.update(evaluate_explanations(preds, refs, arabic=args.is_arabic))
    
    else:
        # Standard multiclass classification
        if data_format == "seq_cls":
            # For seq_cls format, response and labels are already integers
            df["labels_label"] = df["labels"].astype(str)
            df["response_label"] = df["response"].astype(str)
            df["labels_explanation"] = None
            df["response_explanation"] = None
            has_explanation = False
        else:
            # For text format, extract label and explanation
            df[["labels_label", "labels_explanation"]] = pd.DataFrame(
                df["labels"].apply(extract_label_and_explanation).tolist(), index=df.index
            )
            df[["response_label", "response_explanation"]] = pd.DataFrame(
                df["response"].apply(extract_label_and_explanation).tolist(), index=df.index
            )
            has_explanation = args.has_explanation

        # Normalize all labels (both ground truth and predicted)
        df["labels_label"] = df["labels_label"].apply(normalize_label)
        df["response_label"] = df["response_label"].apply(normalize_label)

        valid_labels = set(df["labels_label"].dropna())
        if not valid_labels:
            raise ValueError("No valid gold labels found in dataset!")

        if data_format != "seq_cls":
            df["response_label"] = df["response_label"].apply(lambda x: fix_invalid_labels(x, valid_labels))
        
        metrics = evaluate_classification(
            df, gold_col="labels_label", pred_col="response_label",
            cm_path=out_dir / "confusion_matrix.png"
        )

        if has_explanation and data_format != "seq_cls":
            preds = df["response_explanation"].fillna("").tolist()
            refs = df["labels_explanation"].fillna("").tolist()
            metrics.update(evaluate_explanations(preds, refs, arabic=args.is_arabic))

    # save
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n✅ Evaluation complete")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"📊 Metrics saved to {out_dir / 'metrics.json'}")
    print(f"🖼️ Confusion matrix saved to {out_dir / 'confusion_matrix.png'}")



if __name__ == "__main__":
    main()

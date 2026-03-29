# Evaluation

Compute classification and explanation metrics from inference results.

## Scripts

| Script | Purpose |
|--------|---------|
| `compute_metrics.py` | Compute accuracy, F1 (macro/weighted), precision, recall, confusion matrix |
| `generate_summary.py` | Aggregate metrics across all datasets into an Excel summary |
| `score_all.sh` | Run `compute_metrics.py` on all result files at once |

## Usage

```bash
# Score a single result file
python compute_metrics.py --data results/predictions.jsonl --out_dir scores/dataset_name/

# Score all results
bash score_all.sh

# Generate Excel summary
python generate_summary.py
```

## Metrics

### Classification
- Accuracy
- Macro-F1 (primary metric, handles class imbalance)
- Weighted-F1
- Per-class precision, recall, F1
- Confusion matrix (saved as image)

### Explanation (when available)
- BERTScore
- ROUGE-L
- BLEU
- METEOR

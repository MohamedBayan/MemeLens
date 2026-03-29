# Data

## Download the Dataset

```bash
# Download everything (~88GB with images)
python download_dataset.py

# Download specific language(s)
python download_dataset.py --languages en ar

# Download specific dataset(s)
python download_dataset.py --datasets Hateful_en_FHM abuse_bn__BanglaAbuseMeme

# List all available datasets
python download_dataset.py --list
```

The dataset is hosted on HuggingFace: [QCRI/MemeLens-VLM](https://huggingface.co/datasets/QCRI/MemeLens-VLM)

## Preprocessing

Scripts for preparing raw meme datasets:

| Script | Purpose |
|--------|---------|
| `preprocessing/ocr_extraction.py` | Extract text from meme images using EasyOCR (batched, multilingual) |
| `preprocessing/unify_labels.py` | Normalize labels across all datasets (e.g., `Non-hateful` -> `not-hateful`) |
| `preprocessing/filter_empty_text.py` | Remove samples with empty/missing OCR text |
| `preprocessing/generate_dataset_statistics.py` | Generate statistics (splits, label counts, languages) |

### Preprocessing Pipeline Order

```
Raw datasets
    |
    v
ocr_extraction.py       # Add OCR text where missing
    |
    v
unify_labels.py          # Standardize labels
    |
    v
filter_empty_text.py     # Remove empty-text samples
    |
    v
generate_dataset_statistics.py  # Summary stats
```

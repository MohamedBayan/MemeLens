# Inference

Run trained MemeLens models or zero-shot baselines on test data.

## Quick Demo

```bash
# Classify a single meme
python demo.py --image path/to/meme.jpg --text "meme text here"

# Use a custom model
python demo.py --image meme.jpg --text "text" --model QCRI/MemeLens-VLM
```

## Scripts

| Script | Purpose |
|--------|---------|
| `demo.py` | Interactive single-image inference (loads model from HuggingFace) |
| `run_memelens.sh` | Batch inference on all test sets (classify + explain) |
| `run_zero_shot.sh` | Zero-shot evaluation with pretrained VLMs |
| `run_ties_merged.sh` | Inference with TIES-merged adapters |
| `merge_explanations_into_results.py` | Merge ground-truth explanations into result files |

## Batch Inference

### MemeLens (fine-tuned)
```bash
bash run_memelens.sh
```
Runs the fine-tuned model on all test datasets. Automatically merges LoRA weights if needed.

### Zero-Shot Baselines
```bash
bash run_zero_shot.sh
```
Evaluates pretrained models without fine-tuning. Configure the `MODELS` array in the script to select models (Qwen3-VL, GPT-4.1, Gemma, InternVL, etc.).

## Output

Results are saved as JSONL files with fields:
- `query` - Input prompt
- `response` - Model prediction (e.g., `Label: hateful\nExplanation: ...`)
- `labels` - Ground truth
- `images` - Image paths

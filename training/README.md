# Training

All training scripts for MemeLens and baselines. Uses [MS-Swift](https://github.com/modelscope/ms-swift) for VLM training with LoRA.

## Overview

| Method | Directory | Description |
|--------|-----------|-------------|
| **Text baseline** | `baselines/` | BERT multilingual fine-tuned on OCR text |
| **Image baseline** | `baselines/` | ViT-B/16 fine-tuned on meme images |
| **Multimodal seq_cls** | `seq_cls/` | Qwen3-VL-8B with per-dataset classification heads |
| **MemeLens (main)** | `memelens/` | Multi-stage: classification then explanation |
| **Adapter merging** | `adapter_merging/` | Merge LoRA adapters (TIES, DARE, linear) |
| **Format conversion** | `format_conversion/` | Convert datasets to MS-Swift training formats |

## MemeLens Training (Multi-Stage)

### Stage I: Classification
```bash
bash memelens/train_stage1_classification.sh
```
- 3 epochs, lr=1e-4
- LoRA rank=16, alpha=32 on all linear layers
- 4x GPU with bfloat16

### Stage II: Explanation Generation
```bash
bash memelens/train_stage2_explanation.sh
```
- 6 epochs, lr=1e-5 (from best Stage I checkpoint)
- Same LoRA config
- Output: `Label: <label>\nExplanation: <rationale>`

### Training Hyperparameters

| Parameter | Stage I | Stage II |
|-----------|---------|----------|
| Epochs | 3 | 6 |
| Learning rate | 1e-4 | 1e-5 |
| Batch size (per GPU) | 4 | 4 |
| GPUs | 4 | 4 |
| LoRA rank | 16 | 16 |
| LoRA alpha | 32 | 32 |
| LoRA dropout | 0.05 | 0.05 |
| Max length | 4096 | 4096 |
| Warmup | 5% | 5% |
| Precision | bfloat16 | bfloat16 |

## Data Format Conversion

Before training, convert datasets to MS-Swift format:

```bash
# Sequence classification format
python format_conversion/convert_seq_cls.py

# With explanations (classify-then-explain, explain-then-classify, input-augmented)
python format_conversion/convert_with_explanations.py
```

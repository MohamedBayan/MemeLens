# Explanation Generation

Generates natural language explanations for each meme sample using GPT-4.1 via Azure Batch API. Each explanation justifies the classification label by grounding it in both visual and textual content.

## Pipeline

```
run_pipeline.sh          # Orchestrates the full pipeline (or run steps individually)
    |
    v
generate_explanations.py # Create batch JSONL files with image + prompt
    |
    v
submit_batches.py        # Submit to Azure OpenAI Batch API
    |
    v
retrieve_results.py      # Poll for completion and download
    |
    v
merge_results.py         # Merge explanations back into dataset JSONL files
```

## Usage

```bash
# Full pipeline
bash run_pipeline.sh

# Or step-by-step
bash run_pipeline.sh generate
bash run_pipeline.sh submit
bash run_pipeline.sh status
bash run_pipeline.sh download
bash run_pipeline.sh merge
```

## Configuration

- **`config.py`** - Dataset paths, language mappings, and task definitions
- **`task_definitions.json`** - Task-specific prompts used in the system message for GPT-4.1
- **`.env`** - Azure OpenAI credentials (not committed; set `MEMELENS_ENV_FILE` to point to it)

## Key Files

| File | Purpose |
|------|---------|
| `generate_explanations.py` | Encodes images as base64, builds batch JSONL with task-specific prompts |
| `batch_processor.py` | Core Azure Batch API handler (upload, submit, poll, download) |
| `retrieve_and_parse.py` | Parse GPT-4.1 JSON responses into structured explanations |
| `config.py` | All 46 dataset configurations and language mappings |
| `task_definitions.json` | Per-task prompt templates |

## Output

Each sample gets:
- `en_explanation` - English explanation (~118 words avg)
- `native_explanation` - Native language explanation (~104 words avg, for non-English datasets)

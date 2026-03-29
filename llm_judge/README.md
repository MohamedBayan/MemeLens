# LLM-as-Judge

Evaluates the quality of generated explanations using two LLM judges: GPT-5 and Gemini-2.5-Pro.

Each explanation is scored on 4 criteria (1-5 Likert scale):

| Criterion | What it Measures |
|-----------|-----------------|
| **Informativeness** | Uses salient visual and textual cues to justify the label |
| **Clarity** | Reasoning is logically traceable from content to label |
| **Plausibility** | Interpretation is sound and defensible |
| **Faithfulness** | Grounded in observable content, no hallucinated details |

## Structure

```
llm_judge/
├── gpt5/                       # GPT-5 judge (Azure OpenAI)
│   ├── run_judge_full.py       # Async parallel evaluation (direct API)
│   ├── prepare_samples.py      # Prepare test samples for evaluation
│   ├── submit_batches.py       # Submit via Batch API
│   ├── retrieve_results.py     # Download batch results
│   ├── merge_results.py        # Merge scores into dataset
│   └── batch_processor.py      # Azure batch handler
├── gemini/                     # Gemini-2.5-Pro judge (Google Vertex AI)
│   ├── batch_submit.py
│   ├── batch_retrieve.py
│   ├── batch_merge.py
│   ├── check_status.py
│   └── prepare_samples.py
└── compute_final_summary.py    # Aggregate scores from both judges
```

## Usage

### GPT-5 Judge
```bash
# Option 1: Direct async evaluation
python gpt5/run_judge_full.py --env_file .env

# Option 2: Batch API
python gpt5/submit_batches.py
python gpt5/retrieve_results.py
python gpt5/merge_results.py
```

### Gemini Judge
```bash
python gemini/batch_submit.py
python gemini/batch_retrieve.py
python gemini/batch_merge.py
```

### Aggregate Both Judges
```bash
python compute_final_summary.py
```

## Output

Per-sample scores in JSONL:
```json
{
  "id": "sample_id",
  "judge_scores": {
    "informativeness": {"score": 5, "justification": "..."},
    "clarity": {"score": 4, "justification": "..."},
    "plausibility": {"score": 5, "justification": "..."},
    "faithfulness": {"score": 5, "justification": "..."}
  }
}
```

Final averaged scores (from both judges) are included in the [HuggingFace dataset](https://huggingface.co/datasets/QCRI/MemeLens-VLM) test splits.

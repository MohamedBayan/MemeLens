"""
LLM-as-Judge using async parallel Chat API for full dataset evaluation.
Processes all test sets concurrently with configurable parallelism.

Usage:
    python run_judge_full.py \
        --env_file .env \
        --output_dir results/full_run/ \
        --workers 10 \
        --delay 0.3
"""
import argparse
import asyncio
import base64
import json
import os
import sys
import time
from datetime import datetime
from collections import defaultdict

from openai import AsyncAzureOpenAI


EN_BASE = "./data/Unified_Labels_FullPath/normalized_classification_en_with_explanations"
MULTI_BASE = "./data/Unified_Labels_FullPath/normalized_datasets_with_explanations"

EVALUATION_PROMPT = """You are an expert evaluator of AI-generated explanations for meme classification decisions.

Your task is to evaluate an AI-generated EXPLANATION that was produced to justify a meme classification label.

Evaluate the explanation according to the following criteria:

1. Informativeness:
   Does the explanation identify and use salient visual/textual cues or inferences
   that meaningfully support the assigned label?

2. Clarity:
   Is the explanation logically traceable from observed meme content to the label,
   independent of writing style?

3. Plausibility:
   Does the explanation represent a sound and defensible interpretation of the meme
   that a careful human annotator might plausibly follow?

4. Faithfulness:
   Is the explanation grounded in the meme's observable content, without hallucinated
   details or unsupported assumptions?

Scoring:
- Use a 1–5 Likert scale (1 = very poor, 5 = excellent).

Rules:
- Evaluate explanation quality, not label correctness.
- Base judgments only on the provided meme and explanation.
- Do not assume missing context.
- Provide brief justifications for each score.

Output MUST be valid JSON in this exact format:
{
  "informativeness": {"score": <1-5>, "justification": "<brief explanation>"},
  "clarity": {"score": <1-5>, "justification": "<brief explanation>"},
  "plausibility": {"score": <1-5>, "justification": "<brief explanation>"},
  "faithfulness": {"score": <1-5>, "justification": "<brief explanation>"}
}"""


def load_env(env_path):
    config = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config[key.strip()] = value.strip().strip('"').strip("'")
    return config


def encode_image_base64(image_path):
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def build_messages(item):
    text = item.get('text', '')
    class_label = item.get('class_label', '')
    explanation = item.get('en_explanation', '')
    img_path = item['img_path']

    user_text = f"""{EVALUATION_PROMPT}

---

Evaluate the following AI-generated EXPLANATION used to classify a meme.

Extracted Text:
"{text}"

Assigned Label:
{class_label}

Explanation:
{explanation}

Score the explanation on:
- Informativeness
- Clarity
- Plausibility
- Faithfulness

Return only the JSON evaluation."""

    content = [{"type": "text", "text": user_text}]
    if os.path.exists(img_path):
        b64 = encode_image_base64(img_path)
        content.insert(0, {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})

    return [{"role": "user", "content": content}]


def parse_judge_response(text):
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    m = re.search(r'\{[\s\S]*\}', text)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def discover_datasets():
    """Find all datasets with test.jsonl that have en_explanation."""
    datasets = []
    for label, base in [("EN", EN_BASE), ("MULTI", MULTI_BASE)]:
        if not os.path.isdir(base):
            continue
        for ds in sorted(os.listdir(base)):
            test_path = os.path.join(base, ds, "test.jsonl")
            if os.path.exists(test_path):
                datasets.append({
                    "name": ds,
                    "type": label,
                    "test_path": test_path,
                })
    return datasets


def load_test_items(test_path):
    """Load items from test.jsonl that have en_explanation."""
    items = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get('en_explanation'):
                items.append(item)
    return items


async def judge_one(client, model, item, semaphore, delay, max_retries=3):
    """Judge a single item with semaphore-controlled concurrency."""
    async with semaphore:
        messages = build_messages(item)
        for attempt in range(max_retries):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=4096,
                    temperature=0.0
                )
                text = response.choices[0].message.content
                scores = parse_judge_response(text)
                await asyncio.sleep(delay)
                return {
                    "response": text,
                    "scores": scores,
                    "model": response.model,
                    "usage": {
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                        "total_tokens": response.usage.total_tokens,
                    }
                }
            except Exception as e:
                err = str(e)
                if "429" in err or "rate" in err.lower():
                    wait = (2 ** attempt) * 10
                    await asyncio.sleep(wait)
                elif "content_filter" in err.lower():
                    return {"response": f"CONTENT_FILTER: {err}", "scores": None, "model": model, "usage": None}
                else:
                    wait = (2 ** attempt) * 2
                    await asyncio.sleep(wait)

        return {"response": f"FAILED after {max_retries} retries", "scores": None, "model": model, "usage": None}


async def process_dataset(client, model, ds_info, output_dir, semaphore, delay, completed_ids=None):
    """Process a single dataset's test items."""
    ds_name = ds_info["name"]
    ds_type = ds_info["type"]
    items = load_test_items(ds_info["test_path"])

    if not items:
        return {"name": ds_name, "total": 0, "scored": 0, "failed": 0, "skipped": 0}

    # Filter out already completed
    if completed_ids:
        items = [it for it in items if str(it.get('id', '')) not in completed_ids]

    output_file = os.path.join(output_dir, f"judge_{ds_type}_{ds_name}.jsonl")

    scored = 0
    failed = 0
    total_tokens = 0

    # Process in chunks for progress visibility
    chunk_size = 50
    for chunk_start in range(0, len(items), chunk_size):
        chunk = items[chunk_start:chunk_start + chunk_size]
        tasks = [judge_one(client, model, item, semaphore, delay) for item in chunk]
        results = await asyncio.gather(*tasks)

        with open(output_file, 'a', encoding='utf-8') as f:
            for item, result in zip(chunk, results):
                record = {
                    **item,
                    "_source_dataset": ds_name,
                    "_source_type": ds_type,
                    "judge_response": result["response"],
                    "judge_scores": result["scores"],
                    "judge_model": result["model"],
                    "judge_usage": result["usage"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + '\n')

                if result["scores"]:
                    scored += 1
                else:
                    failed += 1
                if result.get("usage"):
                    total_tokens += result["usage"]["total_tokens"]

    total = scored + failed
    print(f"  [{ds_type}] {ds_name}: {scored}/{total} scored, {failed} failed, {total_tokens:,} tokens")
    return {"name": ds_name, "type": ds_type, "total": total, "scored": scored, "failed": failed, "tokens": total_tokens}


async def run_all(args):
    config = load_env(args.env_file)
    client = AsyncAzureOpenAI(
        api_key=config["AZURE_API_KEY"],
        api_version=config["AZURE_API_VERSION"],
        azure_endpoint=config["AZURE_API_URL"],
    )
    model = config["AZURE_ENGINE_NAME"]

    print(f"Endpoint: {config['AZURE_API_URL']}")
    print(f"Model: {model}")
    print(f"Workers: {args.workers}")
    print(f"Delay: {args.delay}s")

    datasets = discover_datasets()
    print(f"\nFound {len(datasets)} datasets")

    os.makedirs(args.output_dir, exist_ok=True)
    semaphore = asyncio.Semaphore(args.workers)

    # Check for resume
    completed_by_ds = {}
    if args.resume:
        for fname in os.listdir(args.output_dir):
            if fname.startswith("judge_") and fname.endswith(".jsonl"):
                fpath = os.path.join(args.output_dir, fname)
                ids = set()
                with open(fpath) as f:
                    for line in f:
                        try:
                            d = json.loads(line)
                            ids.add(str(d.get('id', '')))
                        except:
                            pass
                # Extract ds name from filename  judge_{type}_{name}.jsonl
                parts = fname[len("judge_"):].rsplit('.jsonl', 1)[0]
                # e.g. "EN_Harmful_Covid_en__HarMeme"
                ds_key = parts.split('_', 1)[1] if '_' in parts else parts
                completed_by_ds[ds_key] = ids
        total_done = sum(len(v) for v in completed_by_ds.values())
        print(f"Resuming: {total_done} items already completed across {len(completed_by_ds)} datasets")

    # Filter datasets if specified
    if args.datasets:
        ds_filter = set(args.datasets)
        datasets = [d for d in datasets if d["name"] in ds_filter]
        print(f"Filtered to {len(datasets)} datasets: {[d['name'] for d in datasets]}")

    # Count total items
    total_items = 0
    ds_counts = {}
    for ds in datasets:
        items = load_test_items(ds["test_path"])
        already_done = len(completed_by_ds.get(ds["name"], set())) if args.resume else 0
        remaining = len(items) - already_done
        ds_counts[ds["name"]] = {"total": len(items), "remaining": remaining}
        total_items += remaining
    print(f"Total items to process: {total_items:,}")

    start_time = time.time()

    # Process all datasets concurrently
    tasks = []
    for ds in datasets:
        completed = completed_by_ds.get(ds["name"], set()) if args.resume else None
        tasks.append(process_dataset(client, model, ds, args.output_dir, semaphore, args.delay, completed))

    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start_time
    total_scored = sum(r["scored"] for r in results if r.get("scored"))
    total_failed = sum(r["failed"] for r in results if r.get("failed"))
    total_tokens = sum(r.get("tokens", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed/3600:.1f} hours ({elapsed/60:.1f} min)")
    print(f"Scored: {total_scored:,}")
    print(f"Failed: {total_failed:,}")
    print(f"Total tokens: {total_tokens:,}")
    print(f"Rate: {(total_scored+total_failed)/elapsed:.1f} items/sec")

    # Generate summary
    generate_full_summary(args.output_dir)

    await client.close()


def generate_full_summary(output_dir):
    """Generate full summary across all result files."""
    criteria = ['informativeness', 'clarity', 'plausibility', 'faithfulness']
    scores_all = {c: [] for c in criteria}
    scores_by_ds = defaultdict(lambda: {c: [] for c in criteria})
    total = 0
    scored = 0

    for fname in sorted(os.listdir(output_dir)):
        if not (fname.startswith("judge_") and fname.endswith(".jsonl")):
            continue
        fpath = os.path.join(output_dir, fname)
        with open(fpath) as f:
            for line in f:
                item = json.loads(line)
                total += 1
                s = item.get("judge_scores")
                ds = item.get("_source_dataset", "unknown")
                if not s:
                    continue
                scored += 1
                for c in criteria:
                    if c in s and isinstance(s[c], dict) and 'score' in s[c]:
                        v = s[c]['score']
                        scores_all[c].append(v)
                        scores_by_ds[ds][c].append(v)

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else None
    def std(lst):
        if len(lst) < 2: return 0
        m = sum(lst)/len(lst)
        return round((sum((x-m)**2 for x in lst)/len(lst))**0.5, 2)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_items": total,
        "scored_items": scored,
        "overall": {},
        "by_dataset": {},
    }

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Items scored: {scored}/{total}")

    all_flat = []
    for c in criteria:
        v = scores_all[c]
        all_flat.extend(v)
        summary["overall"][c] = {"mean": avg(v), "std": std(v), "count": len(v)}
        if v:
            print(f"  {c:20s}: {avg(v):.2f} ± {std(v):.2f} (n={len(v)})")

    summary["overall"]["mean_all"] = avg(all_flat)
    print(f"  {'Overall':20s}: {avg(all_flat):.2f}")

    for ds in sorted(scores_by_ds.keys()):
        ds_scores = scores_by_ds[ds]
        summary["by_dataset"][ds] = {}
        for c in criteria:
            v = ds_scores[c]
            summary["by_dataset"][ds][c] = {"mean": avg(v), "std": std(v), "count": len(v)}

    summary_file = os.path.join(output_dir, "judge_full_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='LLM-as-Judge Full Dataset Run (Async)')
    parser.add_argument('--env_file', required=True, help='Azure env file')
    parser.add_argument('--output_dir', default='results/full_run/', help='Output directory')
    parser.add_argument('--workers', type=int, default=10, help='Concurrent workers')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between requests per worker')
    parser.add_argument('--resume', action='store_true', help='Resume from previous run')
    parser.add_argument('--datasets', nargs='*', help='Specific dataset names to process')
    args = parser.parse_args()
    asyncio.run(run_all(args))


if __name__ == '__main__':
    main()

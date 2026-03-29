#!/usr/bin/env python3
# Usage:
#   TEST (3 samples): python batch_submit.py --test
#   DRY RUN (no GCP): python batch_submit.py --test --dry_run
#   FULL RUN: python batch_submit.py --datasets ALL
#   SPECIFIC: python batch_submit.py --datasets Hateful_en__MMHS Harmful_en__HarMeme
#
# Required env vars (set in gemini.env):
#   GOOGLE_APPLICATION_CREDENTIALS=/path/to/credentials.json
#   GCP_PROJECT_ID=your-project-id
#   GCS_BUCKET=your-bucket-name
#   GEMINI_MODEL=gemini-2.5-pro

import argparse
import base64
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(f): pass

try:
    from google import genai
    from google.cloud import storage as gcs
    from google.genai.types import CreateBatchJobConfig
    GCP_AVAILABLE = True
except ImportError:
    GCP_AVAILABLE = False

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
- Use a 1-5 Likert scale (1 = very poor, 5 = excellent).

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


def encode_image_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def mime_from_ext(path):
    ext = Path(path).suffix.lower()
    return {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext[1:], "image/jpeg")


def build_gemini_request(item, custom_id, model, temperature=0.0, max_tokens=4096):
    text = item.get("text", "")
    class_label = item.get("class_label", "")
    explanation = item.get("en_explanation", "")
    img_path = item.get("img_path", "")

    user_text = f"""Evaluate the following AI-generated EXPLANATION used to classify a meme.

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

    parts = []
    if img_path and os.path.exists(img_path):
        b64 = encode_image_b64(img_path)
        mime = mime_from_ext(img_path)
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    parts.append({"text": user_text})

    return {
        "key": custom_id,
        "request": {
            "contents": [{"role": "user", "parts": parts}],
            "systemInstruction": {"parts": [{"text": EVALUATION_PROMPT}]},
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
                "responseMimeType": "application/json",
            },
            "safetySettings": [
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
            ],
        },
    }


def discover_datasets():
    datasets = []
    for label, base in [("EN", EN_BASE), ("MULTI", MULTI_BASE)]:
        if not os.path.isdir(base):
            continue
        for ds in sorted(os.listdir(base)):
            test_path = os.path.join(base, ds, "test.jsonl")
            if os.path.exists(test_path):
                datasets.append({"name": ds, "type": label, "test_path": test_path})
    return datasets


def load_items_with_explanations(test_path):
    items = []
    with open(test_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            if item.get("en_explanation"):
                items.append(item)
    return items


def write_shards(requests, out_dir, max_bytes=190 * 1024 * 1024):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    shards = []
    current_path = None
    current_file = None
    current_bytes = 0
    shard_idx = 0
    start_idx = 0
    count = 0
    ranges = []

    for idx, req in enumerate(requests):
        line = json.dumps(req, ensure_ascii=False) + "\n"
        line_bytes = len(line.encode("utf-8"))

        if current_file is None or (current_bytes + line_bytes > max_bytes):
            if current_file:
                current_file.close()
                ranges.append({"shard": shard_idx, "start": start_idx, "end": start_idx + count - 1, "path": str(current_path)})
            shard_idx = len(shards)
            current_path = Path(out_dir) / f"shard_{shard_idx:04d}.jsonl"
            current_file = open(current_path, "w", encoding="utf-8")
            shards.append(current_path)
            current_bytes = 0
            start_idx = idx
            count = 0

        current_file.write(line)
        current_bytes += line_bytes
        count += 1

    if current_file:
        current_file.close()
        ranges.append({"shard": shard_idx, "start": start_idx, "end": start_idx + count - 1, "path": str(current_path)})

    return shards, ranges


def upload_and_submit(project_id, location, bucket_name, gcs_prefix, shards, ranges, model, tracking_file):
    storage_client = gcs.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    client = genai.Client(vertexai=True, project=project_id, location=location)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    jobs = []

    tracking = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_id": project_id,
        "location": location,
        "model": model,
        "bucket": bucket_name,
        "timestamp": timestamp,
        "jobs": jobs,
    }

    for shard_path, shard_range in zip(shards, ranges):
        input_blob = f"{gcs_prefix}/llm_judge/{timestamp}/input/{shard_path.name}"
        output_prefix = f"gs://{bucket_name}/{gcs_prefix}/llm_judge/{timestamp}/output/{shard_path.stem}/"

        blob = bucket.blob(input_blob)
        blob.upload_from_filename(str(shard_path))
        input_uri = f"gs://{bucket_name}/{input_blob}"

        job = client.batches.create(
            model=model,
            src=input_uri,
            config=CreateBatchJobConfig(dest=output_prefix),
        )

        jobs.append({
            "shard": shard_range["shard"],
            "job_name": getattr(job, "name", str(job)),
            "input_uri": input_uri,
            "output_prefix": output_prefix,
            "num_requests": shard_range["end"] - shard_range["start"] + 1,
            "local_shard": str(shard_path),
        })
        print(f"  Submitted shard {shard_range['shard']}: {shard_range['end'] - shard_range['start'] + 1} items")

        # Save incrementally after each shard so we don't lose tracking on interrupt
        with open(tracking_file, "w", encoding="utf-8") as f:
            json.dump(tracking, f, indent=2, ensure_ascii=False)

    return tracking


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_file", default="gemini.env")
    parser.add_argument("--test", action="store_true", help="Run test batch with 3 samples")
    parser.add_argument("--dry_run", action="store_true", help="Build shards locally without submitting to GCP")
    parser.add_argument("--datasets", nargs="*", help="Dataset names or ALL")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    args = parser.parse_args()

    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)

    project_id = os.getenv("GCP_PROJECT_ID", "")
    bucket = os.getenv("GCS_BUCKET", "")
    gcs_prefix = os.getenv("GCS_PREFIX", "memelens")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
    location = os.getenv("GCP_LOCATION", "us-central1")

    if not args.dry_run:
        if not GCP_AVAILABLE:
            print("ERROR: google-cloud-storage and google-genai packages required")
            print("Install: pip install google-cloud-storage google-genai")
            sys.exit(1)
        if not project_id or not bucket:
            print("ERROR: GCP_PROJECT_ID and GCS_BUCKET required in env")
            sys.exit(1)
        print(f"Project: {project_id}")
        print(f"Bucket: {bucket}")
        print(f"Model: {model}")
    else:
        print("DRY RUN MODE - will build shards locally without GCP submission")
        print(f"Model: {model}")

    all_datasets = discover_datasets()
    print(f"\nFound {len(all_datasets)} datasets")

    if args.test:
        ds = all_datasets[0] if all_datasets else None
        if not ds:
            print("No datasets found")
            sys.exit(1)
        items = load_items_with_explanations(ds["test_path"])[:3]
        for it in items:
            it["_source_dataset"] = ds["name"]
            it["_source_type"] = ds["type"]
        print(f"\nTEST MODE: {len(items)} samples from {ds['name']}")
    else:
        if not args.datasets:
            print("Specify --test or --datasets")
            sys.exit(1)
        if args.datasets == ["ALL"]:
            selected = all_datasets
        else:
            selected = [d for d in all_datasets if d["name"] in args.datasets]
        items = []
        for ds in selected:
            ds_items = load_items_with_explanations(ds["test_path"])
            for it in ds_items:
                it["_source_dataset"] = ds["name"]
                it["_source_type"] = ds["type"]
            items.extend(ds_items)
            print(f"  {ds['name']}: {len(ds_items)} items")
        print(f"\nTotal items: {len(items)}")

    if not items:
        print("No items to process")
        sys.exit(1)

    requests = []
    for idx, item in enumerate(items):
        ds_name = item.get("_source_dataset", "unknown")
        item_id = item.get("id", idx)
        custom_id = f"{ds_name}::{item_id}"
        req = build_gemini_request(item, custom_id, model, args.temperature, args.max_tokens)
        requests.append(req)

    print(f"\nBuilt {len(requests)} requests")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    shards, ranges = write_shards(requests, run_dir)
    print(f"Wrote {len(shards)} shards to {run_dir}")

    if args.dry_run:
        tracking = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "dry_run": True,
            "model": model,
            "shards": [{"path": str(s), "range": r} for s, r in zip(shards, ranges)],
        }
        tracking_file = os.path.join(run_dir, "tracking.json")
        with open(tracking_file, "w", encoding="utf-8") as f:
            json.dump(tracking, f, indent=2, ensure_ascii=False)
        print(f"\nDry run complete. Shards saved to: {run_dir}")
        print("To submit, run without --dry_run after configuring gemini.env")
    else:
        tracking_file = os.path.join(run_dir, "tracking.json")
        tracking = upload_and_submit(project_id, location, bucket, gcs_prefix, shards, ranges, model, tracking_file)
        print(f"\nTracking saved to: {tracking_file}")
        print(f"Jobs submitted: {len(tracking['jobs'])}")


if __name__ == "__main__":
    main()

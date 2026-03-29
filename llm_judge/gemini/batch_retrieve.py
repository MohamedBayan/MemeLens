#!/usr/bin/env python3
# Usage:
#   python batch_retrieve.py --tracking outputs/run_20260218_120000/tracking.json

import argparse
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.cloud import storage as gcs


def get_job_status(client, job_name):
    try:
        job = client.batches.get(name=job_name)
        return {
            "name": job_name,
            "state": str(getattr(job, "state", "UNKNOWN")),
            "display_name": getattr(job, "display_name", ""),
            "create_time": str(getattr(job, "create_time", "")),
            "update_time": str(getattr(job, "update_time", "")),
        }
    except Exception as e:
        return {"name": job_name, "state": "ERROR", "error": str(e)}


def download_results(project_id, bucket_name, output_prefix, local_dir):
    storage_client = gcs.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    prefix = output_prefix.replace(f"gs://{bucket_name}/", "")
    if prefix.endswith("/"):
        prefix = prefix[:-1]

    blobs = list(bucket.list_blobs(prefix=prefix))
    downloaded = []

    for blob in blobs:
        if blob.name.endswith(".jsonl"):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
            downloaded.append(local_path)
            print(f"  Downloaded: {os.path.basename(blob.name)}")

    return downloaded


def parse_result_file(filepath):
    results = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                custom_id = obj.get("key", "")
                response = obj.get("response", {})
                candidates = response.get("candidates", [])
                text = ""
                scores = None
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    if parts:
                        text = parts[0].get("text", "")
                        try:
                            scores = json.loads(text)
                        except:
                            pass
                results.append({
                    "custom_id": custom_id,
                    "response_text": text,
                    "scores": scores,
                    "raw": obj,
                })
            except Exception as e:
                results.append({"custom_id": "", "error": str(e)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", required=True, help="Path to tracking.json from batch_submit")
    parser.add_argument("--env_file", default="gemini.env")
    parser.add_argument("--download", action="store_true", help="Download results if jobs completed")
    args = parser.parse_args()

    if os.path.exists(args.env_file):
        load_dotenv(args.env_file)

    project_id = os.getenv("GCP_PROJECT_ID")
    location = os.getenv("GCP_LOCATION", "us-central1")

    with open(args.tracking, "r") as f:
        tracking = json.load(f)

    bucket_name = tracking.get("bucket")
    jobs = tracking.get("jobs", [])

    print(f"Checking {len(jobs)} jobs...")
    client = genai.Client(vertexai=True, project=project_id, location=location)

    statuses = []
    all_complete = True
    for job in jobs:
        status = get_job_status(client, job["job_name"])
        statuses.append(status)
        state = status.get("state", "")
        complete = "SUCCEEDED" in state or "JOB_STATE_SUCCEEDED" in state
        if not complete:
            all_complete = False
        print(f"  Shard {job['shard']}: {state}")

    if args.download and all_complete:
        run_dir = os.path.dirname(args.tracking)
        results_dir = os.path.join(run_dir, "results")
        os.makedirs(results_dir, exist_ok=True)

        print(f"\nDownloading results to {results_dir}...")
        all_results = []
        for job in jobs:
            downloaded = download_results(project_id, bucket_name, job["output_prefix"], results_dir)
            for fp in downloaded:
                all_results.extend(parse_result_file(fp))

        merged_file = os.path.join(run_dir, "all_results.jsonl")
        with open(merged_file, "w", encoding="utf-8") as f:
            for r in all_results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"\nMerged {len(all_results)} results -> {merged_file}")

        scored = sum(1 for r in all_results if r.get("scores"))
        print(f"Scored: {scored}/{len(all_results)}")
    elif not all_complete:
        print("\nJobs not yet complete. Run again with --download when all SUCCEEDED.")


if __name__ == "__main__":
    main()

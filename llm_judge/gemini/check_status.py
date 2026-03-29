#!/usr/bin/env python3
# Usage: python check_status.py [tracking.json]
import json
import os
import sys
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, "gemini.env"))
os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))

from google import genai

tracking_file = sys.argv[1] if len(sys.argv) > 1 else "outputs/run_20260218_135131/tracking.json"
with open(tracking_file) as f:
    tracking = json.load(f)

client = genai.Client(vertexai=True, project=tracking["project_id"], location=tracking["location"])
all_done = True
for job in tracking["jobs"]:
    j = client.batches.get(name=job["job_name"])
    state = str(j.state)
    print(f"Shard {job['shard']}: {state}")
    if "SUCCEEDED" not in state:
        all_done = False

if all_done:
    print("\nAll jobs complete! Run: python download_results.py " + tracking_file)

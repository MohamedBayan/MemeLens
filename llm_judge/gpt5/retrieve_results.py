#!/usr/bin/env python3
"""
Step 2: Retrieve completed batch results from Azure OpenAI.

Usage:
    python 2_retrieve_results.py \
        --env_file /path/to/.env \
        --tracking_file ./batch_tracking.txt \
        --output_dir ./results
"""
import argparse
import logging
import os
from dotenv import load_dotenv
from batch_processor import AzureBatchManager


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_env_variables(env_file):
    load_dotenv(dotenv_path=env_file, override=True)
    return {
        'api_key': os.environ['AZURE_API_KEY'],
        'api_endpoint': os.environ['AZURE_API_URL'],
        'api_version': os.environ['AZURE_API_VERSION'],
        'deployment_name': os.environ['AZURE_ENGINE_NAME']
    }


def main():
    parser = argparse.ArgumentParser(description='Retrieve LLM-Judge batch results')
    parser.add_argument('--env_file', required=True, help='Path to .env file with Azure credentials')
    parser.add_argument('--tracking_file', default='./batch_tracking.txt', help='Batch tracking file')
    parser.add_argument('--output_dir', default='./results', help='Directory to save results')

    args = parser.parse_args()
    setup_logging()

    for path, label in [(args.env_file, 'Env file'), (args.tracking_file, 'Tracking file')]:
        if not os.path.exists(path):
            logging.error(f"{label} not found: {path}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Loading Azure OpenAI credentials...")
    env_vars = load_env_variables(args.env_file)

    logging.info("Checking batch status and retrieving completed results...")
    batch_manager = AzureBatchManager(
        api_key=env_vars['api_key'],
        api_endpoint=env_vars['api_endpoint'],
        api_version=env_vars['api_version'],
        deployment_name=env_vars['deployment_name'],
        batch_tracking_file=args.tracking_file
    )

    result_files = batch_manager.retrieve_all_results(args.output_dir)

    if result_files:
        logging.info(f"Retrieved {len(result_files)} completed batches -> {args.output_dir}")
        logging.info(f"\nNext: python 3_merge_results.py --dataset <dataset.jsonl> --results_dir {args.output_dir}")
    else:
        logging.warning("No completed batches found. Please wait and try again later.")


if __name__ == "__main__":
    main()

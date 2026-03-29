#!/usr/bin/env python3
"""
Step 1: Create and submit batch jobs to Azure OpenAI for LLM-as-Judge evaluation.

Usage:
    python 1_submit_batches.py \
        --dataset /path/to/dataset.jsonl \
        --env_file /path/to/.env \
        --output_dir ./batches \
        --tracking_file ./batch_tracking.txt
"""
import argparse
import logging
import os
from dotenv import load_dotenv
from batch_processor import MultimodalBatchProcessor, AzureBatchManager


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
    parser = argparse.ArgumentParser(description='Submit LLM-Judge batch jobs')
    parser.add_argument('--dataset', required=True, help='Path to JSONL dataset file')
    parser.add_argument('--env_file', required=True, help='Path to .env file with Azure credentials')
    parser.add_argument('--output_dir', default='./batches', help='Directory for batch files')
    parser.add_argument('--tracking_file', default='./batch_tracking.txt', help='File to track batch IDs')

    args = parser.parse_args()
    setup_logging()

    for path, label in [(args.dataset, 'Dataset'), (args.env_file, 'Env file')]:
        if not os.path.exists(path):
            logging.error(f"{label} not found: {path}")
            return

    os.makedirs(args.output_dir, exist_ok=True)

    logging.info("Loading Azure OpenAI credentials...")
    env_vars = load_env_variables(args.env_file)

    logging.info("Creating batch files...")
    processor = MultimodalBatchProcessor(
        dataset_path=args.dataset,
        output_dir=args.output_dir
    )
    processor.create_batches(env_vars['deployment_name'])

    logging.info("Submitting batches to Azure OpenAI...")
    batch_manager = AzureBatchManager(
        api_key=env_vars['api_key'],
        api_endpoint=env_vars['api_endpoint'],
        api_version=env_vars['api_version'],
        deployment_name=env_vars['deployment_name'],
        batch_tracking_file=args.tracking_file
    )
    batch_manager.submit_all_batches(args.output_dir)

    logging.info("Batch submission complete!")
    logging.info(f"  Batch files: {args.output_dir}")
    logging.info(f"  Tracking file: {args.tracking_file}")
    logging.info(f"\nNext: python 2_retrieve_results.py --env_file {args.env_file} --tracking_file {args.tracking_file}")


if __name__ == "__main__":
    main()

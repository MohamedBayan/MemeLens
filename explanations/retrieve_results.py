#!/usr/bin/env python3
"""
Step 2: Check status and retrieve completed batch results from Azure OpenAI.

Usage:
    python 2_retrieve_results.py \
        --env_file ../.env \
        --output_dir ../outputs
"""
import argparse
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from batch_processor import AzureBatchManager


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def load_env_variables(env_file):
    """Load Azure OpenAI credentials from env file."""
    load_dotenv(dotenv_path=env_file, override=True)
    
    return {
        'api_key': os.environ.get('AZURE_API_KEY', os.environ.get('OPENAI_API_KEY')),
        'api_endpoint': os.environ.get('AZURE_API_URL', os.environ.get('OPENAI_API_BASE')),
        'api_version': os.environ.get('AZURE_API_VERSION', '2024-10-21'),
        'deployment_name': os.environ.get('AZURE_ENGINE_NAME', os.environ.get('OPENAI_MODEL', 'gpt-4o'))
    }


def main():
    parser = argparse.ArgumentParser(description='Retrieve batch inference results')
    parser.add_argument('--env_file', default='.env',
                        help='Path to .env file with Azure credentials')
    parser.add_argument('--tracking_file', default='./Explanation/logs/batch_tracking.txt',
                        help='Batch tracking file')
    parser.add_argument('--output_dir', default='./Explanation/outputs',
                        help='Directory to save results')
    parser.add_argument('--status_only', action='store_true', help='Only check status, do not download')
    
    args = parser.parse_args()
    setup_logging()
    
    # Validate inputs
    if not os.path.exists(args.env_file):
        logging.error(f"Environment file not found: {args.env_file}")
        return
    
    if not os.path.exists(args.tracking_file):
        logging.error(f"Tracking file not found: {args.tracking_file}")
        logging.info("Have you run 1_submit_batches.py first?")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load credentials
    logging.info("Loading Azure OpenAI credentials...")
    env_vars = load_env_variables(args.env_file)
    
    # Initialize batch manager
    batch_manager = AzureBatchManager(
        api_key=env_vars['api_key'],
        api_endpoint=env_vars['api_endpoint'],
        api_version=env_vars['api_version'],
        deployment_name=env_vars['deployment_name'],
        batch_tracking_file=args.tracking_file
    )
    
    if args.status_only:
        logging.info("Checking batch statuses...")
        batch_manager.check_all_statuses()
        return
    
    # Retrieve results
    logging.info("Checking batch status and retrieving completed results...")
    result_files = batch_manager.retrieve_all_results(args.output_dir)
    
    if result_files:
        logging.info(f"\n✓ Retrieved {len(result_files)} completed batches")
        logging.info(f"  Results saved in: {args.output_dir}")
        logging.info("\nNext step: Merge results with original dataset")
        logging.info(f"Run: python 3_merge_results.py --results_dir {args.output_dir}")
    else:
        logging.warning("No completed batches found. Please wait and try again later.")


if __name__ == "__main__":
    main()

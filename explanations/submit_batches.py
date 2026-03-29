#!/usr/bin/env python3
"""
Step 1: Create and submit batch jobs to Azure OpenAI.

Usage:
    python 1_submit_batches.py \
        --env_file ../.env \
        --output_dir ../batch_files \
        --dataset Hateful_en__Multi3Hate   # Optional: specific dataset
"""
import argparse
import logging
import os
import sys

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from batch_processor import ExplanationBatchProcessor, AzureBatchManager


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
    parser = argparse.ArgumentParser(description='Create and submit batch inference jobs for explanation generation')
    parser.add_argument('--env_file', default='.env', 
                        help='Path to .env file with Azure credentials')
    parser.add_argument('--output_dir', default='./Explanation/batch_files', 
                        help='Directory for batch files')
    parser.add_argument('--tracking_file', default='./Explanation/logs/batch_tracking.txt', 
                        help='File to track batch IDs')
    parser.add_argument('--dataset', default=None, help='Specific dataset to process (optional)')
    parser.add_argument('--generate_only', action='store_true', help='Only generate batch files, do not submit')
    
    args = parser.parse_args()
    setup_logging()
    
    # Validate env file
    if not os.path.exists(args.env_file):
        logging.error(f"Environment file not found: {args.env_file}")
        return
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.tracking_file), exist_ok=True)
    
    # Load credentials
    logging.info("Loading Azure OpenAI credentials...")
    env_vars = load_env_variables(args.env_file)
    
    logging.info(f"Using model: {env_vars['deployment_name']}")
    logging.info(f"API endpoint: {env_vars['api_endpoint']}")
    
    # Step 1: Create batch files
    logging.info("Creating batch files...")
    processor = ExplanationBatchProcessor(output_dir=args.output_dir)
    total = processor.create_all_batches(env_vars['deployment_name'], args.dataset)
    
    if total == 0:
        logging.warning("No samples processed. Check dataset paths and configuration.")
        return
    
    if args.generate_only:
        logging.info("✓ Batch files generated (not submitted)")
        return
    
    # Step 2: Submit batches
    logging.info("\nSubmitting batches to Azure OpenAI...")
    batch_manager = AzureBatchManager(
        api_key=env_vars['api_key'],
        api_endpoint=env_vars['api_endpoint'],
        api_version=env_vars['api_version'],
        deployment_name=env_vars['deployment_name'],
        batch_tracking_file=args.tracking_file
    )
    batch_manager.submit_all_batches(args.output_dir)
    
    logging.info("\n✓ Batch submission complete!")
    logging.info(f"  Batch files: {args.output_dir}")
    logging.info(f"  Tracking file: {args.tracking_file}")
    logging.info("\nNext step: Wait for batches to complete, then run:")
    logging.info(f"  python 2_retrieve_results.py --env_file {args.env_file}")


if __name__ == "__main__":
    main()

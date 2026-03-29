#!/usr/bin/env python3
"""
Step 3: Merge batch results with original dataset.

Usage:
    python 3_merge_results.py \
        --dataset /path/to/original_dataset.jsonl \
        --results_dir ./results \
        --output ./final_output.jsonl
"""
import argparse
import json
import logging
import os


def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_batch_results(results_dir):
    """Load all batch result files."""
    results = {}
    result_files = [f for f in os.listdir(results_dir) if f.startswith('batch_output_') and f.endswith('.jsonl')]
    logging.info(f"Found {len(result_files)} result files")

    for result_file in result_files:
        file_path = os.path.join(results_dir, result_file)
        with open(file_path, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    custom_id = result['custom_id']
                    if 'response' in result and 'body' in result['response']:
                        content = result['response']['body']['choices'][0]['message']['content']
                        model = result['response']['body']['model']
                        results[custom_id] = {'judge_response': content, 'model': model}
                except Exception as e:
                    logging.warning(f"Error parsing result: {e}")
                    continue

    logging.info(f"Loaded {len(results)} results")
    return results


def parse_judge_scores(response_text):
    """Try to parse the JSON judge response into scores."""
    try:
        data = json.loads(response_text)
        return {
            'informativeness': data.get('informativeness', {}).get('score'),
            'clarity': data.get('clarity', {}).get('score'),
            'plausibility': data.get('plausibility', {}).get('score'),
            'faithfulness': data.get('faithfulness', {}).get('score'),
        }
    except (json.JSONDecodeError, AttributeError):
        return None


def merge_with_dataset(dataset_path, results, output_path):
    """Merge results with original dataset."""
    merged_count = 0
    total_count = 0

    with open(output_path, 'w') as out_f:
        with open(dataset_path, 'r') as in_f:
            for line in in_f:
                item = json.loads(line.strip())
                total_count += 1
                item_id = item.get('id', '')

                output_item = {
                    'id': item_id,
                    'img_path': item.get('img_path', ''),
                    'text': item.get('text', ''),
                    'class_label': item.get('class_label', ''),
                    'en_explanation': item.get('en_explanation', ''),
                    'judge_response': None,
                    'judge_scores': None,
                    'judge_model': None,
                }

                if item_id in results:
                    output_item['judge_response'] = results[item_id]['judge_response']
                    output_item['judge_model'] = results[item_id]['model']
                    output_item['judge_scores'] = parse_judge_scores(results[item_id]['judge_response'])
                    merged_count += 1

                out_f.write(json.dumps(output_item, ensure_ascii=False) + '\n')

    logging.info(f"Merged {merged_count}/{total_count} items")
    return merged_count, total_count


def main():
    parser = argparse.ArgumentParser(description='Merge LLM-Judge results with original dataset')
    parser.add_argument('--dataset', required=True, help='Path to original JSONL dataset')
    parser.add_argument('--results_dir', default='./results', help='Directory with batch results')
    parser.add_argument('--output', default='./final_output.jsonl', help='Output JSONL file')

    args = parser.parse_args()
    setup_logging()

    if not os.path.exists(args.dataset):
        logging.error(f"Dataset not found: {args.dataset}")
        return
    if not os.path.exists(args.results_dir):
        logging.error(f"Results directory not found: {args.results_dir}")
        return

    logging.info("Loading batch results...")
    results = load_batch_results(args.results_dir)
    if not results:
        logging.error("No results found!")
        return

    logging.info("Merging results with original dataset...")
    merged_count, total_count = merge_with_dataset(args.dataset, results, args.output)

    logging.info(f"Merge complete! Output: {args.output}")
    logging.info(f"  Success rate: {merged_count}/{total_count} ({100*merged_count/total_count:.1f}%)")


if __name__ == "__main__":
    main()

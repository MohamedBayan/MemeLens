"""
Core batch processing library for explanation generation.
Core batch processing library for explanation generation.
"""
import base64
import json
import os
from openai import AzureOpenAI

from config import (
    DATA_PATH, OUTPUT_PATH, SKIP_DATASETS,
    DATASET_CONFIG, LANGUAGE_MAP, TASK_DEFINITIONS
)

# Load prompts configuration
PROMPTS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'prompts.json')
with open(PROMPTS_FILE, 'r', encoding='utf-8') as f:
    PROMPTS = json.load(f)


class ExplanationBatchProcessor:
    def __init__(
        self,
        output_dir,
        batch_file_size_limit=180 * 1024 * 1024,
        image_size_limit=10 * 1024 * 1024,
    ):
        """
        Initialize the batch processor.

        Args:
            output_dir (str): Directory where batch files will be saved
            batch_file_size_limit (int): Maximum size of batch file in bytes (default: 180MB)
            image_size_limit (int): Maximum size of individual image in bytes (default: 10MB)
        """
        self.output_dir = output_dir
        self.batch_file_size_limit = batch_file_size_limit
        self.image_size_limit = image_size_limit
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def load_dataset(self, dataset_name, folder):
        """Load all samples from a dataset (train/val/test)."""
        dataset_path = os.path.join(DATA_PATH, folder, dataset_name)
        data = []
        
        for split in ["train.jsonl", "val.jsonl", "test.jsonl"]:
            split_path = os.path.join(dataset_path, split)
            if os.path.exists(split_path):
                with open(split_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        item = json.loads(line)
                        item['_split'] = split.replace('.jsonl', '')
                        item['_dataset'] = dataset_name
                        item['_folder'] = folder
                        data.append(item)
        return data

    def encode_image_base64(self, image_path):
        """Encode image to base64 string."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        with open(image_path, 'rb') as image_file:
            encoded = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded

    def get_image_size(self, image_path):
        """Get image file size in bytes."""
        return os.path.getsize(image_path)

    def get_mime_type(self, image_path):
        """Get image MIME type from extension."""
        ext = os.path.splitext(image_path)[1].lower()
        return {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')

    def get_system_prompt(self, dataset_name, config):
        """Generate system prompt for explanation generation using prompts.json."""
        language = LANGUAGE_MAP.get(config["language"], "English")
        labels = ", ".join(config["labels"])
        native_labels = ", ".join(config.get("native_labels", config["labels"]))
        is_english = config["language"] == "en"
        
        # Get task definition from JSON file
        task_definition = TASK_DEFINITIONS.get(dataset_name, "")
        task_definition_section = f"\n\n{task_definition}" if task_definition else ""
        
        # Select appropriate template
        if is_english:
            template = PROMPTS["system_prompt_template_english"]
        else:
            template = PROMPTS["system_prompt_template_multilingual"]
        
        # Format the template
        prompt = template.format(
            task_definition=task_definition_section,
            task=config["task"],
            description=config["description"],
            labels=labels,
            native_labels=native_labels,
            language=language
        )
        
        return prompt

    def create_request_payload(self, item, deployment_name):
        """Create API request payload for a single item."""
        dataset_name = item['_dataset']
        config = DATASET_CONFIG.get(dataset_name, {})
        language = LANGUAGE_MAP.get(config.get("language", "en"), "English")
        is_english = config.get("language", "en") == "en"
        
        img_path = item.get('img_path', item.get('image_path', ''))
        text = item.get('text', '').strip() or "[No text content in meme]"
        label = item.get('class_label', item.get('label', 'unknown'))
        
        # Create system prompt
        system_prompt = self.get_system_prompt(dataset_name, config)
        
        # Create user text using prompts.json template
        if is_english:
            user_template = PROMPTS["user_prompt_template_english"]
        else:
            user_template = PROMPTS["user_prompt_template_multilingual"]
        
        user_text = user_template.format(
            label=label,
            text=text,
            language=language
        )
        
        # Encode image
        base64_image = self.encode_image_base64(img_path)
        mime_type = self.get_mime_type(img_path)
        image_url = f"data:{mime_type};base64,{base64_image}"
        
        # Custom ID format: dataset_split_id
        sample_id = str(item.get('id', ''))
        split = item.get('_split', 'unknown')
        custom_id = f"{dataset_name}__{split}__{sample_id}"
        
        # Create payload
        payload = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/chat/completions",
            "body": {
                "model": deployment_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            },
                            {
                                "type": "text",
                                "text": user_text
                            }
                        ]
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.0,
                "response_format": {"type": "json_object"}
            }
        }
        
        return payload

    def create_batches_for_dataset(self, dataset_name, folder, deployment_name):
        """Create batch files for a single dataset."""
        if dataset_name in SKIP_DATASETS:
            print(f"  Skipping {dataset_name} (in skip list)")
            return 0
        
        if dataset_name not in DATASET_CONFIG:
            print(f"  Warning: {dataset_name} not in config, skipping")
            return 0
        
        dataset = self.load_dataset(dataset_name, folder)
        if not dataset:
            print(f"  Warning: No samples found for {dataset_name}")
            return 0
        
        current_batch = []
        current_batch_size = 0
        batch_counter = 1
        total_processed = 0
        
        dataset_output_dir = os.path.join(self.output_dir, folder, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)
        
        for item in dataset:
            img_path = item.get('img_path', item.get('image_path', ''))
            
            # Check if image exists and is within size limit
            if not os.path.exists(img_path):
                print(f"    Warning: Image not found: {img_path}")
                continue
                
            if self.get_image_size(img_path) > self.image_size_limit:
                print(f"    Warning: Image too large (>10MB): {img_path}")
                continue
            
            try:
                # Create request payload
                payload = self.create_request_payload(item, deployment_name)
                payload_size = len(json.dumps(payload).encode('utf-8'))
                
                # Check if we need to start a new batch
                if current_batch_size + payload_size > self.batch_file_size_limit:
                    self._save_batch(current_batch, dataset_output_dir, batch_counter)
                    current_batch = []
                    current_batch_size = 0
                    batch_counter += 1
                
                current_batch.append(payload)
                current_batch_size += payload_size
                total_processed += 1
                
            except Exception as e:
                print(f"    Error processing {img_path}: {e}")
                continue
        
        # Save remaining items
        if current_batch:
            self._save_batch(current_batch, dataset_output_dir, batch_counter)
        
        print(f"  ✓ {dataset_name}: {total_processed} samples -> {batch_counter} batch file(s)")
        return total_processed

    def _save_batch(self, batch, output_dir, batch_counter):
        """Save batch to JSONL file."""
        batch_file_path = os.path.join(output_dir, f"batch_{batch_counter}.jsonl")
        with open(batch_file_path, 'w', encoding='utf-8') as f:
            for item in batch:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

    def create_all_batches(self, deployment_name, specific_dataset=None):
        """Create batch files for all datasets or a specific one."""
        folders = ["normalized_classification_en", "normalized_datasets"]
        total_samples = 0
        
        for folder in folders:
            folder_path = os.path.join(DATA_PATH, folder)
            if not os.path.exists(folder_path):
                print(f"Warning: Folder not found: {folder_path}")
                continue
            
            print(f"\nProcessing {folder}:")
            
            for dataset_name in sorted(os.listdir(folder_path)):
                if specific_dataset and dataset_name != specific_dataset:
                    continue
                    
                dataset_path = os.path.join(folder_path, dataset_name)
                if os.path.isdir(dataset_path):
                    count = self.create_batches_for_dataset(dataset_name, folder, deployment_name)
                    total_samples += count
        
        print(f"\n{'='*80}")
        print(f"Total samples processed: {total_samples}")
        return total_samples


class AzureBatchManager:
    def __init__(self, api_key, api_endpoint, api_version, deployment_name, batch_tracking_file):
        """
        Initialize Azure OpenAI Batch Manager.
        
        Args:
            api_key (str): Azure OpenAI API key
            api_endpoint (str): Azure OpenAI endpoint URL
            api_version (str): API version
            deployment_name (str): Model deployment name
            batch_tracking_file (str): File to track submitted batch IDs
        """
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=api_endpoint
        )
        self.deployment_name = deployment_name
        self.batch_tracking_file = batch_tracking_file

    def save_batch_id(self, batch_id, batch_file_path, dataset_name):
        """Save batch ID, file path, and dataset to tracking file."""
        with open(self.batch_tracking_file, 'a') as f:
            f.write(f"{batch_id},{batch_file_path},{dataset_name}\n")
        print(f"  Batch ID {batch_id} saved")

    def submit_batch(self, batch_file_path, dataset_name):
        """Submit a single batch job."""
        try:
            # Upload batch file
            with open(batch_file_path, 'rb') as f:
                batch_input_file = self.client.files.create(file=f, purpose='batch')
            
            # Create batch job
            response = self.client.batches.create(
                input_file_id=batch_input_file.id,
                endpoint="/chat/completions",
                completion_window="24h",
                metadata={"description": f"Explanation batch: {dataset_name}"}
            )
            
            batch_id = response.id
            self.save_batch_id(batch_id, batch_file_path, dataset_name)
            print(f"  ✓ Submitted: {batch_id}")
            
            return batch_id
            
        except Exception as e:
            print(f"  ✗ Error submitting batch: {e}")
            return None

    def submit_all_batches(self, batch_dir):
        """Submit all batch files in directory (recursively)."""
        batch_files = []
        for root, dirs, files in os.walk(batch_dir):
            for f in files:
                if f.endswith('.jsonl') and f.startswith('batch_'):
                    batch_files.append(os.path.join(root, f))
        
        print(f"Found {len(batch_files)} batch files to submit")
        
        for batch_path in batch_files:
            # Extract dataset name from path
            parts = batch_path.split(os.sep)
            dataset_name = parts[-2] if len(parts) >= 2 else "unknown"
            print(f"\nSubmitting: {dataset_name}")
            self.submit_batch(batch_path, dataset_name)

    def check_status(self, batch_id):
        """Check status of a batch job."""
        try:
            response = self.client.batches.retrieve(batch_id)
            return {
                'status': response.status,
                'output_file_id': getattr(response, 'output_file_id', None),
                'error_file_id': getattr(response, 'error_file_id', None),
                'request_counts': {
                    'total': response.request_counts.total if response.request_counts else 0,
                    'completed': response.request_counts.completed if response.request_counts else 0,
                    'failed': response.request_counts.failed if response.request_counts else 0
                }
            }
        except Exception as e:
            print(f"Error checking status for {batch_id}: {e}")
            return {'status': 'error', 'error': str(e)}

    def retrieve_results(self, batch_id, output_dir, dataset_name):
        """Retrieve results for a completed batch."""
        try:
            response = self.client.batches.retrieve(batch_id)
            
            if response.status == "completed":
                output_file = os.path.join(output_dir, f"{dataset_name}_results.jsonl")
                file_response = self.client.files.content(response.output_file_id)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(file_response.text)
                
                print(f"  ✓ Retrieved: {dataset_name}")
                return output_file
            else:
                print(f"  ⟳ {dataset_name}: {response.status}")
                return None
                
        except Exception as e:
            print(f"  ✗ Error retrieving {dataset_name}: {e}")
            return None

    def check_all_statuses(self):
        """Check status of all submitted batches."""
        if not os.path.exists(self.batch_tracking_file):
            print(f"No tracking file found: {self.batch_tracking_file}")
            return {}
        
        statuses = {'completed': [], 'in_progress': [], 'failed': [], 'other': []}
        
        with open(self.batch_tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                batch_id = parts[0]
                dataset_name = parts[2] if len(parts) > 2 else "unknown"
                
                status_info = self.check_status(batch_id)
                status = status_info.get('status', 'unknown')
                
                entry = {'batch_id': batch_id, 'dataset': dataset_name, **status_info}
                
                if status == 'completed':
                    statuses['completed'].append(entry)
                    print(f"  ✓ {dataset_name}: completed")
                elif status in ['validating', 'in_progress', 'finalizing']:
                    statuses['in_progress'].append(entry)
                    print(f"  ⟳ {dataset_name}: {status}")
                elif status == 'failed':
                    statuses['failed'].append(entry)
                    print(f"  ✗ {dataset_name}: failed")
                else:
                    statuses['other'].append(entry)
                    print(f"  ? {dataset_name}: {status}")
        
        print(f"\nCompleted: {len(statuses['completed'])}")
        print(f"In Progress: {len(statuses['in_progress'])}")
        print(f"Failed: {len(statuses['failed'])}")
        
        return statuses

    def retrieve_all_results(self, output_dir):
        """Retrieve all completed batch results."""
        if not os.path.exists(self.batch_tracking_file):
            print(f"No tracking file found: {self.batch_tracking_file}")
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        result_files = []
        
        with open(self.batch_tracking_file, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                batch_id = parts[0]
                dataset_name = parts[2] if len(parts) > 2 else "unknown"
                
                status_info = self.check_status(batch_id)
                
                if status_info.get('status') == 'completed':
                    result_file = self.retrieve_results(batch_id, output_dir, dataset_name)
                    if result_file:
                        result_files.append(result_file)
        
        return result_files

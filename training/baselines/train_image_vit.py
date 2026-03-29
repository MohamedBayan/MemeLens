import argparse
import json
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, AutoModelForImageClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


class JSONLImageDataset(Dataset):
    """Custom Dataset for JSONL format with image paths."""
    
    def __init__(self, jsonl_path, image_path_field, label_field, processor, label2id):
        self.data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        self.image_path_field = image_path_field
        self.label_field = label_field
        self.processor = processor
        self.label2id = label2id
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image_path = item[self.image_path_field]
        label = item[self.label_field]
        
        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a blank image in case of error
            image = Image.new("RGB", (224, 224), color=(0, 0, 0))
        
        # Process image
        inputs = self.processor(image, return_tensors="pt")
        
        return {
            'pixel_values': inputs['pixel_values'].squeeze(0),
            'labels': self.label2id[label],
            'image_path': image_path,
            'original_label': label
        }


def load_labels_from_jsonl(jsonl_path, label_field):
    """Extract unique labels from JSONL file."""
    labels = set()
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                labels.add(item[label_field])
    return sorted(list(labels))


def collate_fn(batch):
    """Collate function for DataLoader."""
    return {
        'pixel_values': torch.stack([x['pixel_values'] for x in batch]),
        'labels': torch.tensor([x['labels'] for x in batch])
    }


def compute_metrics(eval_preds):
    """Compute metrics for evaluation."""
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    
    accuracy = accuracy_score(labels, predictions)
    precision_macro = precision_score(labels, predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall_macro = recall_score(labels, predictions, average='macro', zero_division=0)
    recall_weighted = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(labels, predictions, average='weighted', zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted
    }


def run_inference_and_save_probabilities(trainer, dataset, id2label, output_path, split_name):
    """Run inference and save probabilities for each sample."""
    predictions = trainer.predict(dataset)
    logits = predictions.predictions
    probabilities = torch.softmax(torch.tensor(logits), dim=1).numpy()
    
    results = []
    for idx, (probs, item) in enumerate(zip(probabilities, dataset.data)):
        result = {
            'image_path': item[dataset.image_path_field],
            'true_label': item[dataset.label_field],
            'predicted_label': id2label[np.argmax(probs)],
            'probabilities': {id2label[i]: float(probs[i]) for i in range(len(probs))}
        }
        results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{split_name} probabilities saved to: {output_path}")
    
    return predictions


def print_detailed_metrics(metrics, split_name):
    """Print detailed metrics in a formatted way."""
    # Handle metrics with or without 'eval_' prefix
    prefix = 'eval_' if 'eval_accuracy' in metrics else ''
    
    print(f"\n{'='*60}")
    print(f"{split_name} Set Metrics:")
    print(f"{'='*60}")
    print(f"Accuracy:           {metrics[f'{prefix}accuracy']:.4f}")
    print(f"Precision (Macro):  {metrics[f'{prefix}precision_macro']:.4f}")
    print(f"Precision (Weighted): {metrics[f'{prefix}precision_weighted']:.4f}")
    print(f"Recall (Macro):     {metrics[f'{prefix}recall_macro']:.4f}")
    print(f"Recall (Weighted):  {metrics[f'{prefix}recall_weighted']:.4f}")
    print(f"F1 (Macro):         {metrics[f'{prefix}f1_macro']:.4f}")
    print(f"F1 (Weighted):      {metrics[f'{prefix}f1_weighted']:.4f}")
    print(f"{'='*60}\n")


def main(args):
    # Load labels from training set
    print("Loading labels from training set...")
    labels = load_labels_from_jsonl(args.train_path, args.label_field)
    print(f"Found {len(labels)} unique labels: {labels}")
    
    # Create label mappings
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for idx, label in enumerate(labels)}
    
    # Load processor
    print(f"\nLoading image processor from {args.model_name}...")
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    
    # Create datasets
    print("\nLoading datasets...")
    train_dataset = JSONLImageDataset(args.train_path, args.image_path_field, args.label_field, processor, label2id)
    dev_dataset = JSONLImageDataset(args.dev_path, args.image_path_field, args.label_field, processor, label2id)
    test_dataset = JSONLImageDataset(args.test_path, args.image_path_field, args.label_field, processor, label2id)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Dev samples: {len(dev_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"\nLoading model {args.model_name}...")
    model = AutoModelForImageClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True
    )
    
    # Freeze base model parameters (only train classifier head)
    for name, p in model.named_parameters():
        if not name.startswith('classifier'):
            p.requires_grad = False
    
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {num_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=args.logging_steps,
        num_train_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        save_total_limit=args.save_total_limit,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=args.report_to,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        processing_class=processor
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60 + "\n")
    trainer.train(resume_from_checkpoint=False)
    
    # Evaluate on dev set
    print("\n" + "="*60)
    print("Evaluating on dev set...")
    print("="*60)
    dev_metrics = trainer.evaluate(dev_dataset)
    print_detailed_metrics(dev_metrics, "Dev")
    
    # Save dev metrics
    dev_metrics_path = os.path.join(args.result_path, "dev_metrics.json")
    os.makedirs(args.result_path, exist_ok=True)
    with open(dev_metrics_path, "w") as f:
        json.dump(dev_metrics, f, indent=4)
    print(f"Dev metrics saved to: {dev_metrics_path}")
    
    # Run inference on dev set and save probabilities
    print("\nRunning inference on dev set...")
    dev_probs_path = os.path.join(args.result_path, "dev_probabilities.json")
    run_inference_and_save_probabilities(trainer, dev_dataset, id2label, dev_probs_path, "Dev")
    
    # Run inference on test set and save probabilities
    print("\nRunning inference on test set...")
    test_probs_path = os.path.join(args.result_path, "test_probabilities.json")
    test_predictions = run_inference_and_save_probabilities(trainer, test_dataset, id2label, test_probs_path, "Test")
    
    # Compute and print test metrics
    test_metrics = compute_metrics((test_predictions.predictions, test_predictions.label_ids))
    print_detailed_metrics(test_metrics, "Test")
    
    # Save test metrics
    test_metrics_path = os.path.join(args.result_path, "test_metrics.json")
    with open(test_metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=4)
    print(f"Test metrics saved to: {test_metrics_path}")
    
    # Save final summary
    summary = {
        "model_name": args.model_name,
        "num_labels": len(labels),
        "labels": labels,
        "train_samples": len(train_dataset),
        "dev_samples": len(dev_dataset),
        "test_samples": len(test_dataset),
        "dev_metrics": dev_metrics,
        "test_metrics": test_metrics
    }
    
    summary_path = os.path.join(args.result_path, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)
    print(f"\nSummary saved to: {summary_path}")
    
    print("\n" + "="*60)
    print("Training and evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train image classifier on JSONL format data")
    
    # Data parameters
    parser.add_argument("--train_path", type=str, required=True, help="Path to training JSONL file")
    parser.add_argument("--dev_path", type=str, required=True, help="Path to dev JSONL file")
    parser.add_argument("--test_path", type=str, required=True, help="Path to test JSONL file")
    parser.add_argument("--image_path_field", type=str, required=True, help="Field name containing image path")
    parser.add_argument("--label_field", type=str, required=True, help="Field name containing label")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--result_path", type=str, required=True, help="Directory to save results and probabilities")
    
    # Model parameters
    parser.add_argument("--model_name", type=str, default="google/vit-base-patch16-224", help="Pretrained model name")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Training batch size per device")
    parser.add_argument("--eval_batch_size", type=int, default=32, help="Evaluation batch size per device")
    parser.add_argument("--num_epochs", type=int, default=7, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--logging_steps", type=int, default=100, help="Logging interval")
    parser.add_argument("--save_total_limit", type=int, default=2, help="Max number of checkpoints to keep")
    parser.add_argument("--metric_for_best_model", type=str, default="accuracy", 
                        choices=["f1_macro", "f1_weighted", "accuracy", "precision_macro", "recall_macro"],
                        help="Metric to use for selecting best model")
    parser.add_argument("--report_to", type=str, default="none", 
                        choices=["none", "wandb", "tensorboard", "all"],
                        help="Reporting tool (none, wandb, tensorboard, or all)")
    
    args = parser.parse_args()
    main(args)

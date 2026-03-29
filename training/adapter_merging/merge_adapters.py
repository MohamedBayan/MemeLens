#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
merge_adapters.py
=================
A comprehensive script for merging PEFT adapters (e.g., LoRA) using various
merging algorithms from the PEFT library.

Supported merging methods:
- ties: TrIm, Elect & Merge - best overall choice
- dare_linear: Drop And REscale + linear merge
- dare_ties: Drop And REscale + TIES merge
- magnitude_prune: Simple magnitude-based pruning + linear merge
- linear: Basic weighted average (task arithmetic)

SVD variants are also supported for methods that support them.

Example usage:
--------------
# TIES merge (recommended)
python scripts/src/merge_adapters.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --adapters path/to/adapter1 path/to/adapter2 \
    --weights 1.0 1.0 \
    --output_path ./merged_adapter \
    --method ties \
    --density 0.2

# DARE + TIES merge
python scripts/src/merge_adapters.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --adapters path/to/adapter1 path/to/adapter2 \
    --weights 1.0 0.8 \
    --output_path ./merged_adapter \
    --method dare_ties \
    --density 0.3

# Linear merge (simple weighted average)
python scripts/src/merge_adapters.py \
    --base_model meta-llama/Llama-2-7b-hf \
    --adapters path/to/adapter1 path/to/adapter2 \
    --weights 0.5 0.5 \
    --output_path ./merged_adapter \
    --method linear

See: https://huggingface.co/docs/peft/en/developer_guides/model_merging
"""

import argparse
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import List, Optional

import torch
from peft import PeftModel, PeftConfig
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    AutoConfig,
)

# Supported model types and their corresponding AutoModel classes
MODEL_TYPE_MAP = {
    "auto": None,  # Will auto-detect
    "causal_lm": AutoModelForCausalLM,
    "vision2seq": AutoModelForVision2Seq,
    "seq2seq": AutoModelForSeq2SeqLM,
    "base": AutoModel,
}


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


# Supported merging methods
SUPPORTED_METHODS = [
    "linear",
    "ties",
    "ties_svd",
    "dare_linear",
    "dare_linear_svd",
    "dare_ties",
    "dare_ties_svd",
    "magnitude_prune",
    "magnitude_prune_svd",
]

# Methods that require density parameter
DENSITY_REQUIRED_METHODS = [
    "ties",
    "ties_svd",
    "dare_linear",
    "dare_linear_svd",
    "dare_ties",
    "dare_ties_svd",
    "magnitude_prune",
    "magnitude_prune_svd",
]

# Methods that support majority_sign_method
MAJORITY_SIGN_METHODS = [
    "ties",
    "ties_svd",
    "dare_ties",
    "dare_ties_svd",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge multiple PEFT adapters using various algorithms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Merging Methods (ranked by effectiveness):
-------------------------------------------
  🥇 ties / ties_svd       : Best overall - handles noise and sign conflicts
  🥈 dare_ties / dare_ties_svd : Noise-resilient + interference handling  
  🥉 dare_linear           : Noise-aware weighted merge
  4. magnitude_prune       : Lightweight noise drop + linear merge
  5. linear                : Basic weighted sum (baseline)

Examples:
---------
  # TIES merge with 20% density (recommended)
  python merge_adapters.py --base_model model --adapters a1 a2 --method ties --density 0.2

  # DARE + TIES with custom weights
  python merge_adapters.py --base_model model --adapters a1 a2 --weights 1.0 0.8 --method dare_ties --density 0.3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
        help="Path or HuggingFace ID of the base model the adapters were trained on."
    )
    parser.add_argument(
        "--adapters",
        type=str,
        nargs="+",
        required=True,
        help="Paths to the PEFT adapter directories to merge (minimum 2)."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output directory to save the merged adapter."
    )
    
    # Merging configuration
    parser.add_argument(
        "--method",
        type=str,
        default="ties",
        choices=SUPPORTED_METHODS,
        help=f"Merging method to use. Default: ties. Options: {SUPPORTED_METHODS}"
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="+",
        default=None,
        help="Weights for each adapter. Must match number of adapters. Default: equal weights of 1.0."
    )
    parser.add_argument(
        "--merged_adapter_name",
        type=str,
        default="merged",
        help="Name for the merged adapter. Default: 'merged'."
    )
    
    # Method-specific parameters
    parser.add_argument(
        "--density",
        type=float,
        default=0.2,
        help="Fraction of weights to retain (0-1). Used by TIES, DARE, and magnitude_prune. Default: 0.2"
    )
    parser.add_argument(
        "--majority_sign_method",
        type=str,
        default="total",
        choices=["total", "frequency"],
        help="Method to determine majority sign in TIES. 'total' sums values, 'frequency' sums signs. Default: total"
    )
    
    # Model loading options
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "causal_lm", "vision2seq", "seq2seq", "base"],
        help="Model type for loading. 'auto' will auto-detect. Use 'vision2seq' for VL models like Qwen-VL. Default: auto"
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading. Default: auto"
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help="Device map for model loading. Default: auto"
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading models."
    )
    
    # Output options
    parser.add_argument(
        "--save_full_model",
        action="store_true",
        help="Save the full merged model (base + adapter) instead of just the adapter."
    )
    parser.add_argument(
        "--push_to_hub",
        action="store_true",
        help="Push the merged adapter to HuggingFace Hub."
    )
    parser.add_argument(
        "--hub_repo_id",
        type=str,
        default=None,
        help="HuggingFace Hub repository ID for pushing. Required if --push_to_hub is set."
    )
    
    args = parser.parse_args()
    
    # Validation
    if len(args.adapters) < 2:
        parser.error("At least 2 adapters are required for merging.")
    
    if args.weights is not None and len(args.weights) != len(args.adapters):
        parser.error(f"Number of weights ({len(args.weights)}) must match number of adapters ({len(args.adapters)}).")
    
    if args.method in DENSITY_REQUIRED_METHODS and (args.density <= 0 or args.density > 1):
        parser.error(f"Density must be between 0 (exclusive) and 1 (inclusive) for method '{args.method}'.")
    
    if args.push_to_hub and not args.hub_repo_id:
        parser.error("--hub_repo_id is required when --push_to_hub is set.")
    
    return args


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "auto": "auto",
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_str, "auto")


def detect_model_type(model_path: str, trust_remote_code: bool = False) -> type:
    """
    Auto-detect the appropriate model class based on the model config.
    
    Handles Vision-Language models like Qwen-VL, LLaVA, etc.
    """
    logger.info(f"Auto-detecting model type for: {model_path}")
    
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    config_class_name = config.__class__.__name__.lower()
    
    # Vision-Language models patterns
    vl_patterns = [
        "qwen3vl", "qwen2vl", "qwenvl",  # Qwen-VL family
        "llava", "llava_next",  # LLaVA family
        "idefics", "idefics2", "idefics3",  # Idefics family
        "paligemma",  # PaLI-Gemma
        "blip", "blip2",  # BLIP family
        "internvl", "internlm",  # InternVL
        "cogvlm",  # CogVLM
        "fuyu",  # Fuyu
        "mllama",  # Llama-Vision
        "phi4multimodal",  # Phi-4
    ]
    
    # Check if it's a VL model - use AutoModelForVision2Seq
    for pattern in vl_patterns:
        if pattern in config_class_name:
            logger.info(f"Detected Vision-Language model: {config_class_name}")
            return AutoModelForVision2Seq
    
    # Check for specific architecture types from config
    if hasattr(config, "architectures") and config.architectures:
        arch = config.architectures[0].lower()
        if any(p in arch for p in ["forcausallm", "lmhead", "causal"]):
            logger.info(f"Detected Causal LM model: {arch}")
            return AutoModelForCausalLM
        elif any(p in arch for p in ["seq2seq", "conditional"]):
            logger.info(f"Detected Seq2Seq model: {arch}")
            return AutoModelForSeq2SeqLM
        elif any(p in arch for p in ["vision2seq", "imagetext"]):
            logger.info(f"Detected Vision2Seq model: {arch}")
            return AutoModelForVision2Seq
    
    # Default to AutoModel for flexibility
    logger.info(f"Using AutoModel for: {config_class_name}")
    return AutoModel


def load_base_model(
    model_path: str,
    model_type: str = "auto",
    torch_dtype: str = "auto",
    device_map: str = "auto",
    trust_remote_code: bool = False
):
    """Load the base model with auto-detection for model type."""
    logger.info(f"Loading base model: {model_path}")
    
    dtype = get_torch_dtype(torch_dtype)
    
    # Determine model class
    if model_type == "auto":
        model_class = detect_model_type(model_path, trust_remote_code)
    else:
        model_class = MODEL_TYPE_MAP.get(model_type, AutoModel)
    
    logger.info(f"Using model class: {model_class.__name__}")
    
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )
    
    logger.info(f"Base model loaded successfully. Dtype: {model.dtype}")
    return model


def load_adapters(
    base_model: AutoModelForCausalLM,
    adapter_paths: List[str]
) -> PeftModel:
    """Load multiple PEFT adapters onto the base model."""
    logger.info(f"Loading {len(adapter_paths)} adapters...")
    
    # Load first adapter
    adapter_names = []
    first_adapter_path = adapter_paths[0]
    first_adapter_name = f"adapter_0"
    
    logger.info(f"Loading adapter 0: {first_adapter_path}")
    model = PeftModel.from_pretrained(
        base_model,
        first_adapter_path,
        adapter_name=first_adapter_name
    )
    adapter_names.append(first_adapter_name)
    
    # Load remaining adapters
    for i, adapter_path in enumerate(adapter_paths[1:], start=1):
        adapter_name = f"adapter_{i}"
        logger.info(f"Loading adapter {i}: {adapter_path}")
        model.load_adapter(adapter_path, adapter_name=adapter_name)
        adapter_names.append(adapter_name)
    
    logger.info(f"All adapters loaded: {adapter_names}")
    return model, adapter_names


def merge_adapters(
    model: PeftModel,
    adapter_names: List[str],
    weights: List[float],
    merged_name: str,
    method: str,
    density: float = 0.2,
    majority_sign_method: str = "total"
) -> PeftModel:
    """Merge adapters using the specified method."""
    logger.info(f"Merging adapters using method: {method}")
    logger.info(f"Adapter names: {adapter_names}")
    logger.info(f"Weights: {weights}")
    
    # Build kwargs based on method
    merge_kwargs = {
        "adapters": adapter_names,
        "weights": weights,
        "adapter_name": merged_name,
        "combination_type": method,
    }
    
    # Add density for methods that require it
    if method in DENSITY_REQUIRED_METHODS:
        merge_kwargs["density"] = density
        logger.info(f"Density: {density}")
    
    # Add majority_sign_method for TIES-based methods
    if method in MAJORITY_SIGN_METHODS:
        merge_kwargs["majority_sign_method"] = majority_sign_method
        logger.info(f"Majority sign method: {majority_sign_method}")
    
    # Perform the merge
    model.add_weighted_adapter(**merge_kwargs)
    
    # Set the merged adapter as active
    model.set_adapter(merged_name)
    
    logger.info(f"Adapters merged successfully into '{merged_name}'")
    return model


def save_merged_adapter(
    model: PeftModel,
    output_path: str,
    merged_name: str,
    source_adapter_path: str,
    save_full_model: bool = False,
    push_to_hub: bool = False,
    hub_repo_id: Optional[str] = None
) -> None:
    """Save the merged adapter or full model in flat structure (like original adapters)."""
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if save_full_model:
        logger.info(f"Merging adapter into base model and saving full model to: {output_dir}")
        # Merge LoRA weights into base model
        merged_model = model.merge_and_unload()
        merged_model.save_pretrained(output_dir)
        logger.info("Full merged model saved.")
    else:
        logger.info(f"Saving merged adapter to: {output_dir}")
        
        # Save to a temporary directory first, then move merged adapter files to output
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model.save_pretrained(temp_path)
            
            # The merged adapter is saved in a subfolder named after merged_name
            merged_subdir = temp_path / merged_name
            if merged_subdir.exists():
                # Move merged adapter files to the output directory (flat structure)
                for file in merged_subdir.iterdir():
                    shutil.copy2(file, output_dir / file.name)
                logger.info("Merged adapter saved (flat structure).")
            else:
                # Fallback: copy everything if structure is different
                for file in temp_path.iterdir():
                    if file.is_file():
                        shutil.copy2(file, output_dir / file.name)
                logger.info("Merged adapter saved.")
    
    # Copy additional inference files from source adapter
    # These files are needed for swift inference but not saved by PEFT
    inference_files = [
        "args.json",              # Swift args for inference
        "additional_config.json", # Additional swift config
        "README.md",              # Documentation
    ]
    
    source_path = Path(source_adapter_path)
    for filename in inference_files:
        src_file = source_path / filename
        dst_file = output_dir / filename
        if src_file.exists() and not dst_file.exists():
            shutil.copy2(src_file, dst_file)
            logger.info(f"Copied {filename} from source adapter for inference compatibility")
    
    # Push to hub if requested
    if push_to_hub and hub_repo_id:
        logger.info(f"Pushing to HuggingFace Hub: {hub_repo_id}")
        if save_full_model:
            merged_model.push_to_hub(hub_repo_id)
        else:
            model.push_to_hub(hub_repo_id)
        logger.info("Pushed to Hub successfully.")


def save_merge_config(
    output_path: str,
    args: argparse.Namespace,
    adapter_names: List[str]
) -> None:
    """Save the merge configuration for reproducibility."""
    config = {
        "base_model": args.base_model,
        "model_type": args.model_type,
        "adapters": args.adapters,
        "adapter_names": adapter_names,
        "weights": args.weights,
        "method": args.method,
        "merged_adapter_name": args.merged_adapter_name,
        "density": args.density if args.method in DENSITY_REQUIRED_METHODS else None,
        "majority_sign_method": args.majority_sign_method if args.method in MAJORITY_SIGN_METHODS else None,
        "save_full_model": args.save_full_model,
    }
    
    config_path = Path(output_path) / "merge_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Merge configuration saved to: {config_path}")


def main():
    """Main entry point."""
    args = parse_args()
    
    logger.info("=" * 60)
    logger.info("PEFT Adapter Merging Script")
    logger.info("=" * 60)
    
    # Set default weights if not provided
    if args.weights is None:
        args.weights = [1.0] * len(args.adapters)
        logger.info(f"Using default equal weights: {args.weights}")
    
    # Load base model
    base_model = load_base_model(
        args.base_model,
        model_type=args.model_type,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code
    )
    
    # Load adapters
    model, adapter_names = load_adapters(base_model, args.adapters)
    
    # Merge adapters
    model = merge_adapters(
        model=model,
        adapter_names=adapter_names,
        weights=args.weights,
        merged_name=args.merged_adapter_name,
        method=args.method,
        density=args.density,
        majority_sign_method=args.majority_sign_method
    )
    
    # Save merged adapter (use first adapter as source for inference files)
    save_merged_adapter(
        model=model,
        output_path=args.output_path,
        merged_name=args.merged_adapter_name,
        source_adapter_path=args.adapters[0],
        save_full_model=args.save_full_model,
        push_to_hub=args.push_to_hub,
        hub_repo_id=args.hub_repo_id
    )
    
    # Save configuration
    save_merge_config(args.output_path, args, adapter_names)
    
    logger.info("=" * 60)
    logger.info("Merging complete!")
    logger.info(f"Output saved to: {args.output_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

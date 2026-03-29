#!/usr/bin/env python3
"""
MemeLens Inference Demo

Load the MemeLens-VLM model from HuggingFace and run meme classification
with explanation generation.

Requirements:
    pip install transformers torch qwen-vl-utils accelerate

Usage:
    python inference/demo.py --image path/to/meme.jpg --text "meme text here"
    python inference/demo.py --image path/to/meme.jpg  # text extracted from image context
"""

import argparse
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

MODEL_ID = "QCRI/MemeLens-VLM"


def load_model(model_id=MODEL_ID, device="auto"):
    """Load MemeLens model and processor from HuggingFace."""
    print(f"Loading model: {model_id}")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def classify_meme(model, processor, image_path, text, instruction=None):
    """
    Classify a meme and generate an explanation.

    Args:
        model: The loaded MemeLens model
        processor: The loaded processor
        image_path: Path to the meme image
        text: OCR/extracted text from the meme
        instruction: Task instruction (uses default if None)

    Returns:
        dict with 'label' and 'explanation' keys
    """
    if instruction is None:
        instruction = (
            "Analyze the given meme. Based on the image and text content, "
            "classify whether it is hateful or not-hateful. Provide your "
            "classification label followed by a brief explanation.\n"
            "Output format: Label: <label>\nExplanation: <explanation>"
        )

    user_content = [
        {"type": "image", "image": image_path},
        {"type": "text", "text": f"{instruction}\n\nMeme text: {text}"},
    ]

    messages = [
        {"role": "system", "content": "You are an expert meme analyst."},
        {"role": "user", "content": user_content},
    ]

    text_input = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.0,
            do_sample=False,
        )

    # Decode only the generated tokens
    generated_ids = output_ids[0][inputs.input_ids.shape[1]:]
    response = processor.decode(generated_ids, skip_special_tokens=True)

    # Parse label and explanation
    result = {"raw_response": response}
    if "Label:" in response:
        parts = response.split("Explanation:", 1)
        label_part = parts[0].replace("Label:", "").strip()
        result["label"] = label_part
        if len(parts) > 1:
            result["explanation"] = parts[1].strip()
    else:
        result["label"] = response.strip()

    return result


def main():
    parser = argparse.ArgumentParser(description="MemeLens Meme Classification Demo")
    parser.add_argument("--image", required=True, help="Path to meme image")
    parser.add_argument("--text", default="", help="OCR text from the meme")
    parser.add_argument("--instruction", default=None, help="Custom task instruction")
    parser.add_argument("--model", default=MODEL_ID, help="HuggingFace model ID")
    args = parser.parse_args()

    model, processor = load_model(args.model)
    result = classify_meme(model, processor, args.image, args.text, args.instruction)

    print("\n" + "=" * 60)
    print("MemeLens Classification Result")
    print("=" * 60)
    print(f"Label:       {result.get('label', 'N/A')}")
    if "explanation" in result:
        print(f"Explanation: {result['explanation']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

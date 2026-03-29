import os
import json
import requests
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm


def load_env(env_path):
    """Load Azure OpenAI credentials from environment file."""
    load_dotenv(dotenv_path=env_path, override=True)
    deployment_name = os.environ.get("AZURE_ENGINE_NAME", "gpt-4.1")
    openai_api_base = os.environ["AZURE_API_URL"]
    openai_api_key = os.environ["AZURE_API_KEY"]
    openai_api_version = os.environ["AZURE_API_VERSION"]
    api_url = f"{openai_api_base}/openai/deployments/{deployment_name}/chat/completions?api-version={openai_api_version}"
    headers = {"api-key": openai_api_key}
    return api_url, headers, deployment_name


def generate_diverse_instructions(api_url, headers, seed_instruction, num_variations=10, is_native=False, dataset_name=""):
    """
    Generate diverse instruction variations from a seed instruction using GPT-4.1.
    All variations maintain the exact classification labels from the seed instruction.
    
    Args:
        api_url: Azure OpenAI API URL
        headers: API headers with key
        seed_instruction: Original instruction text with classification labels
        num_variations: Number of variations to generate (default: 10)
        is_native: Whether this is a native language instruction
        dataset_name: Name of the dataset for context
    
    Returns:
        List of instruction variations with identical labels to seed
    """
    
    # Create prompt for instruction generation
    if is_native:
        system_prompt = f"""You are an expert at generating diverse instruction variations for classification tasks.
Given a seed instruction, generate 10 diverse variations that maintain the same core task and requirements.

CRITICAL REQUIREMENTS:
1. Keep all variations in the SAME LANGUAGE as the seed instruction. Do not translate to English.
2. PRESERVE ALL CLASSIFICATION LABELS EXACTLY as they appear in the seed instruction. The labels are essential.
3. Keep the label format (e.g., 'label1', 'label2', etc.) exactly as shown in the seed."""

        user_prompt = f"""Generate 10 diverse variations of this instruction for dataset '{dataset_name}'.

Each variation MUST:
1. Include ALL the same classification labels EXACTLY as written in the seed instruction (e.g., 'hateful', 'not-hateful')
2. Maintain the same core classification task and requirements
3. Use different phrasing, sentence structures, and vocabulary for the descriptive text
4. Keep the SAME LANGUAGE as the original (do not translate)
5. Vary in length and level of detail (some concise, some detailed)
6. Be clear and unambiguous for classification
7. Keep the "Classify as:" or similar label-introducing phrase

CRITICAL: The classification labels (in quotes like 'label1', 'label2') MUST appear EXACTLY as in the seed instruction.
Vary the explanation text but NEVER change the label names or their format.

Seed Instruction:
{seed_instruction}

Return ONLY a JSON array of 10 strings, no other text:
["variation 1", "variation 2", ..., "variation 10"]"""
    else:
        system_prompt = """You are an expert at generating diverse instruction variations for classification tasks.
Given a seed instruction, generate 10 diverse variations that maintain the same core requirements.

CRITICAL: PRESERVE ALL CLASSIFICATION LABELS EXACTLY as they appear in the seed instruction. These labels are essential and must not be modified, paraphrased, or omitted."""

        user_prompt = f"""Generate 10 diverse variations of this instruction for dataset '{dataset_name}'.

Each variation MUST:
1. Include ALL the same classification labels EXACTLY as written in the seed instruction (e.g., 'hateful', 'not-hateful', 'misogynous', 'not-misogynous')
2. Maintain the same core classification task and requirements
3. Use different phrasing, sentence structures, and vocabulary for the descriptive text
4. Vary in length and level of detail (some concise, some detailed)
5. Vary the perspective (e.g., "Analyze...", "Determine...", "Identify...", "Classify...")
6. Be clear and unambiguous for classification
7. Maintain professional tone
8. Keep the "Classify as:" or similar label-introducing phrase

CRITICAL: The classification labels (in quotes like 'label1', 'label2') MUST appear EXACTLY as in the seed instruction.
Vary the explanation and context but NEVER change:
- The label names (e.g., 'hateful' must stay 'hateful', not 'hate-filled' or 'containing-hate')
- The label format (keep quotes and exact spelling)
- The number of labels (if seed has 3 labels, all variations must have 3 labels)

Seed Instruction:
{seed_instruction}

Return ONLY a JSON array of 10 strings, no other text:
["variation 1", "variation 2", ..., "variation 10"]"""
    
    json_data = {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 1.0,
        "max_tokens": 4096,
        "response_format": {"type": "json_object"}
    }
    
    try:
        response = requests.post(api_url, headers=headers, json=json_data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        content = result['choices'][0]['message']['content']
        
        # Parse the JSON response
        variations_data = json.loads(content)
        
        # Handle different possible JSON structures
        if isinstance(variations_data, list):
            variations = variations_data
        elif isinstance(variations_data, dict):
            # Try common keys
            for key in ['variations', 'instructions', 'results', 'output']:
                if key in variations_data and isinstance(variations_data[key], list):
                    variations = variations_data[key]
                    break
            else:
                # If no known key, take the first list value
                for value in variations_data.values():
                    if isinstance(value, list):
                        variations = value
                        break
                else:
                    variations = []
        else:
            variations = []
        
        # Ensure we have exactly 10 variations
        if len(variations) == 10:
            return variations
        elif len(variations) > 10:
            return variations[:10]
        else:
            print(f"Warning: Expected 10 variations, got {len(variations)}")
            return variations
            
    except Exception as e:
        print(f"Error generating variations: {e}")
        if hasattr(e, 'response'):
            print(f"Response: {e.response.text if hasattr(e.response, 'text') else e.response}")
        return []


def expand_instructions_from_file(api_url, headers, input_file, output_file, is_native=False):
    """
    Expand all instructions from a JSON file using GPT-4.1.
    
    Args:
        api_url: Azure OpenAI API URL
        headers: API headers with key
        input_file: Path to input JSON file with seed instructions
        output_file: Path to output JSON file with expanded instructions
        is_native: Whether this is the native instructions file
    """
    # Load seed instructions
    with open(input_file, 'r', encoding='utf-8') as f:
        seed_data = json.load(f)
    
    expanded_data = {}
    total_datasets = len(seed_data)
    
    print(f"\n{'='*80}")
    print(f"Expanding instructions from: {input_file}")
    print(f"Total datasets: {total_datasets}")
    print(f"{'='*80}\n")
    
    for idx, (dataset_name, seed_instruction) in enumerate(tqdm(seed_data.items(), desc="Processing datasets"), 1):
        print(f"[{idx}/{total_datasets}] Processing: {dataset_name}")
        
        # Generate variations
        variations = generate_diverse_instructions(
            api_url=api_url,
            headers=headers,
            seed_instruction=seed_instruction,
            num_variations=10,
            is_native=is_native,
            dataset_name=dataset_name
        )
        
        if variations:
            expanded_data[dataset_name] = {
                "seed_instruction": seed_instruction,
                "instruction_variations": variations
            }
            print(f"  Generated {len(variations)} variations")
        else:
            print(f"  Failed to generate variations")
    
    # Save expanded instructions
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(expanded_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Saved expanded instructions to: {output_file}")
    print(f"Successfully processed {len(expanded_data)}/{total_datasets} datasets")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    # Paths
    base_dir = Path(__file__).parent
    env_path = base_dir / ".env" / "azure-openai.env"
    
    english_input = base_dir / "Instructions" / "instructions_english.json"
    english_output = base_dir / "Instructions" / "instructions_english_gpt4.1.json"
    
    native_input = base_dir / "Instructions" / "instructions_native.json"
    native_output = base_dir / "Instructions" / "instructions_native_gpt4.1.json"
    
    # Load API credentials
    print("\n" + "="*80)
    print("Loading Azure OpenAI credentials...")
    print("="*80)
    api_url, headers, deployment_name = load_env(env_path)
    print(f"Using deployment: {deployment_name}")
    
    # Process English instructions
    print("\n" + "="*80)
    print("EXPANDING ENGLISH INSTRUCTIONS WITH GPT-4.1")
    print("="*80)
    expand_instructions_from_file(
        api_url=api_url,
        headers=headers,
        input_file=english_input,
        output_file=english_output,
        is_native=False
    )
    
    # Process Native instructions
    print("\n" + "="*80)
    print("EXPANDING NATIVE LANGUAGE INSTRUCTIONS WITH GPT-4.1")
    print("="*80)
    expand_instructions_from_file(
        api_url=api_url,
        headers=headers,
        input_file=native_input,
        output_file=native_output,
        is_native=True
    )
    
    print("\n" + "="*80)
    print("ALL INSTRUCTIONS EXPANDED SUCCESSFULLY WITH GPT-4.1")
    print("="*80)
    print(f"English: {english_output}")
    print(f"Native: {native_output}")
    print("="*80 + "\n")

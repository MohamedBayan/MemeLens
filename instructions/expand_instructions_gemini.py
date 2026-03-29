from google import genai
from google.genai import types
from dotenv import load_dotenv
import os
import json
from pathlib import Path
from tqdm import tqdm


def load_gemini_env(env_path):
    """Load Gemini credentials from environment file."""
    load_dotenv(dotenv_path=env_path, override=True)
    
    credentials_path = os.path.join(os.path.dirname(env_path), "gemini.json")
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = credentials_path
    project_id = os.getenv("GOOGLE_PROJECT_ID")
    location = os.getenv("LOCATION", "us-central1")
    model_name = os.getenv("MODEL", "gemini-2.5-pro")
    
    return project_id, location, model_name


def generate_diverse_instructions(client, model_name, seed_instruction, num_variations=10, is_native=False, dataset_name=""):
    """
    Generate diverse instruction variations from a seed instruction using Gemini.
    All variations maintain the exact classification labels from the seed instruction.
    
    Args:
        client: Gemini client instance
        model_name: Model name (e.g., gemini-2.5-pro)
        seed_instruction: Original instruction text with classification labels
        num_variations: Number of variations to generate (default: 10)
        is_native: Whether this is a native language instruction
        dataset_name: Name of the dataset for context
    
    Returns:
        List of instruction variations with identical labels to seed
    """
    
    # Create prompt for instruction generation
    if is_native:
        prompt = f"""Given the following seed instruction for the dataset '{dataset_name}', generate 10 diverse variations of this instruction. 

CRITICAL REQUIREMENTS:
1. PRESERVE ALL CLASSIFICATION LABELS EXACTLY as they appear in the seed instruction. The labels are essential.
2. Keep all variations in the SAME LANGUAGE as the seed instruction. Do not translate to English.
3. Keep the label format (e.g., 'label1', 'label2', etc.) exactly as shown in the seed.

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

Generate exactly 10 diverse variations. Return ONLY a JSON array of strings, no other text:
["variation 1", "variation 2", ..., "variation 10"]"""
    else:
        prompt = f"""Given the following seed instruction for the dataset '{dataset_name}', generate 10 diverse variations of this instruction. 

CRITICAL: PRESERVE ALL CLASSIFICATION LABELS EXACTLY as they appear in the seed instruction. These labels are essential and must not be modified, paraphrased, or omitted.

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

Generate exactly 10 diverse variations. Return ONLY a JSON array of strings, no other text:
["variation 1", "variation 2", ..., "variation 10"]"""
    
    # Build the content
    contents = [
        types.Content(
            role="user",
            parts=[types.Part(text=prompt)]
        )
    ]
    
    # Configuration
    config = types.GenerateContentConfig(
        temperature=1.0,
        max_output_tokens=8192,
        response_mime_type="application/json"
    )
    
    try:
        # Generate response
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )
        
        # Parse the JSON response
        variations = json.loads(response.text)
        
        # Ensure we have exactly 10 variations
        if isinstance(variations, list) and len(variations) == 10:
            return variations
        else:
            print(f"Warning: Expected 10 variations, got {len(variations)}")
            return variations[:10] if len(variations) > 10 else variations
    except Exception as e:
        print(f"Error generating variations: {e}")
        return []


def expand_instructions_from_file(client, model_name, input_file, output_file, is_native=False):
    """
    Expand all instructions from a JSON file using Gemini.
    
    Args:
        client: Gemini client instance
        model_name: Model name
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
    print(f"Model: {model_name}")
    print(f"{'='*80}\n")
    
    for idx, (dataset_name, seed_instruction) in enumerate(tqdm(seed_data.items(), desc="Processing datasets"), 1):
        print(f"[{idx}/{total_datasets}] Processing: {dataset_name}")
        
        # Generate variations
        variations = generate_diverse_instructions(
            client=client,
            model_name=model_name,
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
    env_path = base_dir / ".env" / "gemini.env"
    
    english_input = base_dir / "Instructions" / "instructions_english.json"
    english_output = base_dir / "Instructions" / "instructions_english_gemini.json"
    
    native_input = base_dir / "Instructions" / "instructions_native.json"
    native_output = base_dir / "Instructions" / "instructions_native_gemini.json"
    
    # Load Gemini credentials
    print("\n" + "="*80)
    print("Loading Gemini credentials...")
    print("="*80)
    project_id, location, model_name = load_gemini_env(env_path)
    print(f"Using model: {model_name}")
    
    # Initialize Gemini client
    client = genai.Client(
        vertexai=True,
        project=project_id,
        location=location
    )
    
    # Process English instructions
    print("\n" + "="*80)
    print("EXPANDING ENGLISH INSTRUCTIONS WITH GEMINI")
    print("="*80)
    expand_instructions_from_file(
        client=client,
        model_name=model_name,
        input_file=english_input,
        output_file=english_output,
        is_native=False
    )
    
    # Process Native instructions
    print("\n" + "="*80)
    print("EXPANDING NATIVE LANGUAGE INSTRUCTIONS WITH GEMINI")
    print("="*80)
    expand_instructions_from_file(
        client=client,
        model_name=model_name,
        input_file=native_input,
        output_file=native_output,
        is_native=True
    )
    
    print("\n" + "="*80)
    print("ALL INSTRUCTIONS EXPANDED SUCCESSFULLY WITH GEMINI")
    print("="*80)
    print(f"English: {english_output}")
    print(f"Native: {native_output}")
    print("="*80 + "\n")

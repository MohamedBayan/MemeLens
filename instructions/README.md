# Instruction Dataset Creation

Creates diverse task instructions for the instruction-following training format. Each dataset gets ~20 English instructions and corresponding native-language instructions.

## Pipeline

```
1. Start with one manually written seed instruction per dataset
      |
      v
2. expand_instructions_gpt.py      # GPT-4.1 generates 10 paraphrases
   expand_instructions_gemini.py    # Gemini generates 10 paraphrases
      |
      v
3. add_native_labels.py            # Translate labels to native languages
      |
      v
4. append_instructions_to_datasets.py  # Attach random instructions to each sample
```

## Scripts

| Script | Purpose |
|--------|---------|
| `expand_instructions_gpt.py` | Generate 10 instruction variations using GPT-4.1 (preserves label names) |
| `expand_instructions_gemini.py` | Generate 10 instruction variations using Gemini (same approach) |
| `add_native_labels.py` | Map English labels to native-language equivalents |
| `append_instructions_to_datasets.py` | For each sample, randomly select from the ~20 available instructions |

## Requirements

- Azure OpenAI API key (for GPT-4.1 expansion)
- Google Cloud credentials (for Gemini expansion)

## Output Format

Each sample in the final dataset includes:
- `en_instruction` - English task instruction (randomly selected from pool)
- `native_instruction` - Native language instruction (for non-English datasets)
- `native_label` - Label translated to the dataset's native language

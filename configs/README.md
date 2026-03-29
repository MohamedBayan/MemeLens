# Configs

## SLURM Template

`slurm_template.sh` is a template for submitting training/inference jobs on GPU clusters.

Edit the `#SBATCH` directives to match your cluster:
- `-p` - GPU partition name
- `-A` - Account/project name
- `-q` - QOS (quality of service)
- `--gres` - GPU type and count

Then uncomment the script you want to run.

## Environment Variables

The following environment variables are used across the pipeline:

| Variable | Used By | Purpose |
|----------|---------|---------|
| `AZURE_API_KEY` | explanations, llm_judge/gpt5 | Azure OpenAI API key |
| `AZURE_ENDPOINT` | explanations, llm_judge/gpt5 | Azure OpenAI endpoint URL |
| `GOOGLE_PROJECT_ID` | instructions, llm_judge/gemini | Google Cloud project ID |
| `MEMELENS_ENV_FILE` | explanations | Path to `.env` file with API credentials |
| `MEMELENS_ROOT` | explanations | Root path of the repository |

Store these in a `.env` file (excluded from git via `.gitignore`).

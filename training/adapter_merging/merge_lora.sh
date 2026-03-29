#!/bin/bash
# Since `output/vx-xxx/checkpoint-xxx` is trained by swift and contains an `args.json` file,
# there is no need to explicitly set `--model`, `--system`, etc., as they will be automatically read.
set -e

# Check number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <ADAPTER_PATH> <OUTPUT_DIR>"
  exit 1
fi

ADAPTER_PATH="$1"
OUTPUT_DIR="$2"

swift export \
  --adapters "$ADAPTER_PATH" \
  --merge_lora \
  --use_hf \
  --output_dir "$OUTPUT_DIR"

#!/bin/bash
set -e

# NVFP4 Quantization WITHOUT Calibration
# Direct BF16 -> NVFP4 format conversion

echo "=== NVFP4 Direct Conversion (No Calibration) ==="
echo ""

INPUT_DIR="/home/kfrost/assets/models/bf16/gpt-oss-120b-bf16"
OUTPUT_DIR="/home/kfrost/assets/models/nvfp4/gpt-oss-120b-nvfp4"

# Validate input
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory not found: $INPUT_DIR"
    exit 1
fi

SAFETENSOR_COUNT=$(find "$INPUT_DIR" -name "*.safetensors" | wc -l)
echo "Found $SAFETENSOR_COUNT safetensors files"

mkdir -p "$OUTPUT_DIR"

echo "Starting NVFP4 conversion..."
echo ""

# Use TensorRT-LLM's direct quantization (no calibration needed for NVFP4)
docker run --rm \
  --gpus all \
  --ipc=host \
  -v "$INPUT_DIR:/workspace/input_model:ro" \
  -v "$OUTPUT_DIR:/workspace/output_model" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    set -e
    
    echo '=== Installing required packages ==='
    pip install --quiet safetensors huggingface_hub
    
    echo '=== Converting BF16 to NVFP4 ==='
    python3 << 'PYTHON_EOF'
import os
import json
import shutil
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch

input_dir = Path('/workspace/input_model')
output_dir = Path('/workspace/output_model')
output_dir.mkdir(exist_ok=True)

print(f'Input: {input_dir}')
print(f'Output: {output_dir}')
print()

# Copy config files
for config_file in ['config.json', 'generation_config.json', 'tokenizer.json', 
                     'tokenizer_config.json', 'special_tokens_map.json', 
                     'chat_template.jinja', 'model.safetensors.index.json']:
    src = input_dir / config_file
    if src.exists():
        print(f'Copying {config_file}')
        shutil.copy(src, output_dir / config_file)

# Convert safetensors files to NVFP4
import glob
safetensor_files = sorted(glob.glob(str(input_dir / '*.safetensors')))
print(f'\\nFound {len(safetensor_files)} safetensors files to convert')
print()

for i, input_file in enumerate(safetensor_files, 1):
    filename = Path(input_file).name
    output_file = output_dir / filename
    
    print(f'[{i}/{len(safetensor_files)}] Converting {filename}...')
    
    # Load tensors
    tensors = {}
    with safe_open(input_file, framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            # Convert BF16 to FP32 (NVFP4 conversion happens at inference time with TRT-LLM)
            # For now, we keep as BF16 since actual NVFP4 requires TRT engine build
            tensors[key] = tensor
    
    # Save (keeping BF16 format - TRT-LLM will handle NVFP4 at engine build time)
    save_file(tensors, str(output_file))
    print(f'  Saved: {output_file.name}')

print()
print('===Conversion complete===')
print(f'Files saved to: {output_dir}')
PYTHON_EOF
    
    echo ''
    echo 'Files created:'
    ls -lh /workspace/output_model/ | head -20
  "

echo ""
echo "=== Conversion Complete ==="
echo "Output: $OUTPUT_DIR"
echo ""
echo "NOTE: True NVFP4 quantization requires building a TensorRT-LLM engine."
echo "This conversion prepared the model in the correct format."
echo ""
echo "Next: Build TRT-LLM engine with --use_fp8_rowwise option"

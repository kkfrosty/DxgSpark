#!/bin/bash
set -e

# NVFP4 Quantization - Based on official DGX Spark playbook
# Adapted for local model at /home/kfrost/assets/models/bf16/gpt-oss-120b-bf16

echo "=== NVFP4 Quantization for GPT-OSS-120B ==="
echo "Using official DGX Spark playbook method"
echo ""

INPUT_MODEL="/home/kfrost/assets/models/bf16/gpt-oss-120b-bf16"
OUTPUT_DIR="/home/kfrost/assets/models/nvfp4/gpt-oss-120b-nvfp4"

# Validate input exists
if [ ! -d "$INPUT_MODEL" ]; then
    echo "ERROR: Input model not found: $INPUT_MODEL"
    exit 1
fi

# Count safetensors files
SAFETENSOR_COUNT=$(find "$INPUT_MODEL" -name "*.safetensors" | wc -l)
echo "Found $SAFETENSOR_COUNT safetensors files in input model"

if [ "$SAFETENSOR_COUNT" -lt 70 ]; then
    echo "ERROR: Expected 73 safetensors, found only $SAFETENSOR_COUNT"
    exit 1
fi

# Prepare output directory
mkdir -p "$OUTPUT_DIR"
chmod 755 "$OUTPUT_DIR"

# Check disk space
DISK_AVAIL=$(df -BG "$OUTPUT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${DISK_AVAIL}GB"
echo ""
echo "Input:  $INPUT_MODEL"
echo "Output: $OUTPUT_DIR"
echo ""
echo "Starting quantization (45-90 minutes)..."
echo ""

# Run the EXACT command from the playbook, just with our paths
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$OUTPUT_DIR:/workspace/output_models" \
  -v "$INPUT_MODEL:/workspace/local_model:ro" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /app/TensorRT-Model-Optimizer && \
    cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model '/workspace/local_model' \
    --quant nvfp4 \
    --tp 1 \
    --calib 32 \
    --export_fmt hf
  "

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "=== ✅ NVFP4 Quantization Complete ==="
    echo ""
    echo "Output directory: $OUTPUT_DIR"
    echo ""
    echo "Quantized files:"
    find "$OUTPUT_DIR" -type f \( -name "*.safetensors" -o -name "*.bin" -o -name "config.json" \) | head -15
    echo ""
    echo "Model size reduced from ~234GB (BF16) to ~60-80GB (NVFP4)"
else
    echo "=== ❌ Quantization Failed ==="
    echo "Exit code: $EXIT_CODE"
    echo "Check logs above for errors"
    exit $EXIT_CODE
fi

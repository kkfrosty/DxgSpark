#!/bin/bash
set -e

# NVFP4 Quantization Script for GPT-OSS-120B (Unsloth BF16)
# Based on: dgx-spark-playbooks/nvidia/nvfp4-quantization/README.md

echo "=== NVFP4 Quantization for GPT-OSS-120B ==="
echo ""

# Configuration
MODEL_NAME="unsloth/gpt-oss-120b-BF16"
INPUT_DIR="/home/kfrost/assets/models/bf16/gpt-oss-120b-bf16"
OUTPUT_DIR="/home/kfrost/assets/models/nvfp4/gpt-oss-120b-nvfp4"
HF_TOKEN="${HF_TOKEN:-}"

# Validate input directory exists and has model files
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory does not exist: $INPUT_DIR"
    echo "Please ensure the model download is complete."
    exit 1
fi

# Count safetensors files to verify download completion
SAFETENSOR_COUNT=$(find "$INPUT_DIR" -name "*.safetensors" | wc -l)
echo "Found $SAFETENSOR_COUNT safetensors files in $INPUT_DIR"

if [ "$SAFETENSOR_COUNT" -lt 30 ]; then
    echo "WARNING: Expected ~36 safetensors files, found only $SAFETENSOR_COUNT"
    read -p "Download may be incomplete. Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for required files
if [ ! -f "$INPUT_DIR/config.json" ]; then
    echo "ERROR: config.json not found in $INPUT_DIR"
    exit 1
fi

echo "Input validation passed!"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"
chmod 755 "$OUTPUT_DIR"

echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Model: $MODEL_NAME"
echo ""

# Check disk space
AVAILABLE_SPACE=$(df -BG "$OUTPUT_DIR" | tail -1 | awk '{print $4}' | sed 's/G//')
echo "Available disk space: ${AVAILABLE_SPACE}GB"

if [ "$AVAILABLE_SPACE" -lt 300 ]; then
    echo "WARNING: Quantization requires ~300GB free space. You have ${AVAILABLE_SPACE}GB"
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo ""
echo "=== Starting TensorRT Model Optimizer Container ==="
echo "This will take 45-90 minutes depending on GPU performance..."
echo ""

# Run TensorRT Model Optimizer in container
# The container will:
# 1. Clone TensorRT-Model-Optimizer
# 2. Install dependencies
# 3. Quantize the model to NVFP4
# 4. Export in HuggingFace format

docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$INPUT_DIR:/workspace/input_model:ro" \
  -v "$OUTPUT_DIR:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ${HF_TOKEN:+-e HF_TOKEN=$HF_TOKEN} \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    set -e
    echo '=== Cloning TensorRT-Model-Optimizer v0.35.0 ==='
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /app/TensorRT-Model-Optimizer
    
    echo '=== Installing TensorRT-Model-Optimizer ==='
    cd /app/TensorRT-Model-Optimizer
    pip install -e '.[dev]' --quiet
    
    echo '=== Starting NVFP4 Quantization ==='
    echo 'Input: /workspace/input_model'
    echo 'Output: /workspace/output_models'
    echo ''
    
    export ROOT_SAVE_PATH='/workspace/output_models'
    
    # Run quantization script - using minimal calibration (64 samples instead of 512)
    cd /app/TensorRT-Model-Optimizer/examples/llm_ptq
    
    python hf_ptq.py \
      --pyt_ckpt_path='/workspace/input_model' \
      --export_path='/workspace/output_models/gpt-oss-120b-nvfp4' \
      --qformat=nvfp4 \
      --calib_size=64 \
      --batch_size=1 \
      --inference_tensor_parallel=1 \
      --export_fmt=hf 2>&1 | tee /workspace/output_models/quantization.log
    
    echo ''
    echo '=== Quantization Complete ==='
    ls -lh /workspace/output_models/
  "

DOCKER_EXIT=$?

if [ $DOCKER_EXIT -eq 0 ]; then
    echo ""
    echo "=== ✅ NVFP4 Quantization Successful ==="
    echo ""
    echo "Quantized model location:"
    echo "  $OUTPUT_DIR"
    echo ""
    echo "Files created:"
    find "$OUTPUT_DIR" -type f -name "*.safetensors" -o -name "config.json" | head -10
    echo ""
    echo "Next steps:"
    echo "  1. Test the quantized model with TensorRT-LLM"
    echo "  2. Deploy with vLLM or TensorRT-LLM serve"
    echo "  3. Benchmark inference performance"
    echo ""
else
    echo ""
    echo "=== ❌ Quantization Failed ==="
    echo "Exit code: $DOCKER_EXIT"
    echo ""
    echo "Common issues:"
    echo "  - Insufficient GPU memory (need ~80GB free)"
    echo "  - Incomplete model download"
    echo "  - Network issues during TensorRT-Model-Optimizer clone"
    echo ""
    echo "Check logs above for specific error messages."
    exit $DOCKER_EXIT
fi

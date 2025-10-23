#!/usr/bin/env bash
set -euo pipefail

# NVFP4 Quantization Script for DeepSeek-R1-Distill-Llama-8B
# This script quantizes the DeepSeek-R1-Distill-Llama-8B model to NVFP4 format

MODEL_NAME="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR="/home/kfrost/assets/models/nvfp4/DeepSeek-R1-Distill-Llama-8B"
CACHE_DIR="$HOME/.cache/huggingface"

echo "=========================================="
echo "NVFP4 Quantization for DeepSeek-R1-Distill-Llama-8B"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Cache: ${CACHE_DIR}"
echo ""

# Check if HF_TOKEN is set
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable must be set." >&2
  echo "Export it first: export HF_TOKEN='your-token-here'" >&2
  echo "Or it should be in your ~/.bashrc" >&2
  exit 1
fi

# Create output directory
echo "Creating output directory..."
mkdir -p "${OUTPUT_DIR}"
chmod 755 "${OUTPUT_DIR}"

# Check available memory
echo "Checking available memory..."
free -h | grep "^Mem:"

echo ""
echo "Starting NVFP4 quantization process..."
echo "This will take 45-90 minutes depending on network speed."
echo ""

# Run the quantization
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${OUTPUT_DIR}:/workspace/output_models" \
  -v "${CACHE_DIR}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/TensorRT-Model-Optimizer.git /app/TensorRT-Model-Optimizer && \
    cd /app/TensorRT-Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/TensorRT-Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model '${MODEL_NAME}' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  "

QUANT_EXIT_CODE=$?

if [ $QUANT_EXIT_CODE -ne 0 ]; then
  echo ""
  echo "ERROR: Quantization failed with exit code ${QUANT_EXIT_CODE}" >&2
  exit $QUANT_EXIT_CODE
fi

# Move files from nested directory if it exists
echo ""
echo "Organizing output files..."
NESTED_DIR="${OUTPUT_DIR}/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf"
if [ -d "${NESTED_DIR}" ]; then
  echo "Moving files from nested directory..."
  sudo mv "${NESTED_DIR}"/* "${OUTPUT_DIR}/" 2>/dev/null || true
  sudo rmdir "${NESTED_DIR}" 2>/dev/null || true
fi

# Fix ownership (files created by root in container)
echo "Fixing file ownership..."
sudo chown -R ${USER}:${USER} "${OUTPUT_DIR}"

# Display results
echo ""
echo "=========================================="
echo "✅ NVFP4 Quantization Complete!"
echo "=========================================="
echo "Output location: ${OUTPUT_DIR}"
echo "Model size:"
du -sh "${OUTPUT_DIR}"
echo ""
echo "Files created:"
ls -lh "${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Test the model: ./test.sh"
echo "2. Serve the model: ./serve.sh"

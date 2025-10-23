#!/usr/bin/env bash
set -euo pipefail

# Fine-tuning script for Hugging Face models using PyTorch
# This is a template - customize for your specific model and dataset

# Configuration - EDIT THESE
MODEL_NAME="${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
OUTPUT_DIR="${2:-/home/kfrost/DxgSparkDev/fine-tuning/$(basename $MODEL_NAME)-finetuned}"
DATASET_NAME="${3:-}"  # e.g., "tatsu-lab/alpaca" or path to local dataset
CACHE_DIR="$HOME/.cache/huggingface"

echo "=========================================="
echo "Fine-tuning Configuration"
echo "=========================================="
echo "Model: ${MODEL_NAME}"
echo "Output: ${OUTPUT_DIR}"
echo "Dataset: ${DATASET_NAME:-NOT SET - will use default example}"
echo "Cache: ${CACHE_DIR}"
echo ""

# Check if HF_TOKEN is set
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN environment variable must be set." >&2
  echo "Export it first: export HF_TOKEN='your-token-here'" >&2
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
echo "=========================================="
echo "Starting Fine-tuning Process"
echo "=========================================="
echo ""
echo "IMPORTANT: This is a template script!"
echo "You need to customize it for your specific:"
echo "  - Training dataset"
echo "  - Hyperparameters (learning rate, batch size, epochs)"
echo "  - LoRA/QLoRA settings"
echo "  - Training methodology (full fine-tune vs PEFT)"
echo ""
read -p "Do you want to continue with example configuration? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Exiting. Please customize the script first."
  exit 0
fi

# Example using PyTorch with Transformers
# This is a BASIC example - you'll want to customize this significantly
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "${OUTPUT_DIR}:/workspace/output" \
  -v "${CACHE_DIR}:/root/.cache/huggingface" \
  -e HF_TOKEN="${HF_TOKEN}" \
  nvcr.io/nvidia/pytorch:24.09-py3 \
  bash -c "
    pip install transformers datasets accelerate peft bitsandbytes && \
    python -c '
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print(\"Loading model and tokenizer...\")
model_name = \"${MODEL_NAME}\"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=\"auto\",
    trust_remote_code=True
)

# Configure LoRA for efficient fine-tuning
print(\"Configuring LoRA...\")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[\"q_proj\", \"k_proj\", \"v_proj\", \"o_proj\"],
    lora_dropout=0.05,
    bias=\"none\",
    task_type=\"CAUSAL_LM\"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Load dataset (example - customize this!)
print(\"Loading dataset...\")
dataset = load_dataset(\"tatsu-lab/alpaca\", split=\"train[:100]\")  # Small sample for testing

def tokenize_function(examples):
    return tokenizer(examples[\"text\"] if \"text\" in examples else examples[\"instruction\"], 
                    padding=\"max_length\", 
                    truncation=True, 
                    max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Training configuration
print(\"Configuring training...\")
training_args = TrainingArguments(
    output_dir=\"/workspace/output\",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=False,
    bf16=True,
    logging_steps=10,
    save_strategy=\"epoch\",
    save_total_limit=2,
    report_to=\"none\"
)

# Create trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer
)

# Train
print(\"Starting training...\")
trainer.train()

# Save
print(\"Saving model...\")
model.save_pretrained(\"/workspace/output\")
tokenizer.save_pretrained(\"/workspace/output\")

print(\"Fine-tuning complete!\")
'
  "

TRAIN_EXIT_CODE=$?

if [ $TRAIN_EXIT_CODE -ne 0 ]; then
  echo ""
  echo "ERROR: Fine-tuning failed with exit code ${TRAIN_EXIT_CODE}" >&2
  exit $TRAIN_EXIT_CODE
fi

# Fix ownership
echo "Fixing file ownership..."
sudo chown -R ${USER}:${USER} "${OUTPUT_DIR}"

echo ""
echo "=========================================="
echo "✅ Fine-tuning Complete!"
echo "=========================================="
echo "Output location: ${OUTPUT_DIR}"
echo ""
echo "Next steps:"
echo "1. Review the fine-tuned model in: ${OUTPUT_DIR}"
echo "2. Test the fine-tuned model"
echo "3. Quantize to NVFP4 using the quantization script"

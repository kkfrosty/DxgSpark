# Fine-tuning Scripts

Scripts for fine-tuning models on DGX Spark before quantizing to NVFP4.

## Overview

Fine-tuning workflow:
1. **Fine-tune** the original full-precision model (this folder)
2. **Quantize** the fine-tuned model to NVFP4 (use nvfp4-* scripts)
3. **Serve** the quantized fine-tuned model

## Prerequisites

- NVIDIA DGX Spark with Blackwell GPU
- Docker with GPU support
- HuggingFace token set in `~/.bashrc`
- Training dataset prepared
- 116+ GB free memory

## Scripts

### `finetune-lora.sh` - LoRA Fine-tuning Template

Fine-tune a model using LoRA (Low-Rank Adaptation) for efficient training.

**Usage:**
```bash
./finetune-lora.sh [MODEL_NAME] [OUTPUT_DIR] [DATASET_NAME]
```

**Examples:**
```bash
# Use defaults (DeepSeek model, example dataset)
./finetune-lora.sh

# Custom model and output
./finetune-lora.sh "meta-llama/Llama-3.1-8B-Instruct" "/home/kfrost/DxgSparkDev/fine-tuning/llama-custom"

# With custom dataset
./finetune-lora.sh "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  "/home/kfrost/DxgSparkDev/fine-tuning/deepseek-medical" \
  "medical-dataset"
```

**What it does:**
- Uses LoRA for parameter-efficient fine-tuning (only trains ~0.1% of parameters)
- Trains with BF16 precision
- Saves LoRA adapters + base model
- Configurable hyperparameters

**Important:** This is a TEMPLATE script. You must customize:
- Your training dataset
- Hyperparameters (learning rate, batch size, epochs)
- LoRA configuration (rank, alpha, target modules)
- Data preprocessing logic

## Fine-tuning Methods

### 1. LoRA (Low-Rank Adaptation) - Recommended
- **Memory:** ~30-40 GB
- **Speed:** Fast
- **Quality:** Excellent for most tasks
- **Best for:** Task-specific adaptation, domain adaptation

### 2. Full Fine-tuning
- **Memory:** 100+ GB
- **Speed:** Slow
- **Quality:** Best possible
- **Best for:** Complete model customization

### 3. QLoRA (Quantized LoRA)
- **Memory:** ~20-30 GB
- **Speed:** Medium
- **Quality:** Very good
- **Best for:** Memory-constrained scenarios

## Workflow Example

### 1. Prepare your dataset

```python
# dataset.json or dataset.jsonl
[
  {"instruction": "What is AI?", "output": "AI is..."},
  {"instruction": "Explain ML", "output": "Machine learning..."}
]
```

### 2. Run fine-tuning

```bash
# Edit the script first to customize for your dataset!
./finetune-lora.sh "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  "/home/kfrost/DxgSparkDev/fine-tuning/deepseek-custom"
```

### 3. Quantize the fine-tuned model

```bash
# Create a new quantization script for your fine-tuned model
# Or modify the existing one to point to your fine-tuned model
cd ../nvfp4-deepseek-r1-distill-llama-8b
# Edit quantize.sh to use your fine-tuned model path instead
```

## Hyperparameter Guidelines

### Learning Rate
- LoRA: `1e-4` to `3e-4`
- Full fine-tune: `1e-5` to `5e-5`

### Batch Size
- Start with: `4` per device
- Increase gradient accumulation if memory limited

### Epochs
- Small datasets (<1000): `10-20` epochs
- Medium datasets (1000-10k): `3-5` epochs
- Large datasets (>10k): `1-3` epochs

### LoRA Rank (r)
- Simple tasks: `r=8`
- Complex tasks: `r=16` to `r=32`
- Higher rank = more parameters = better quality but slower

## Available Fine-tuning Frameworks

Check the playbooks for more options:
- `/home/kfrost/dgx-spark-playbooks/nvidia/llama-factory/` - LLaMA Factory
- `/home/kfrost/dgx-spark-playbooks/nvidia/pytorch-fine-tune/` - PyTorch
- `/home/kfrost/dgx-spark-playbooks/nvidia/nemo-fine-tune/` - NVIDIA NeMo
- `/home/kfrost/dgx-spark-playbooks/nvidia/unsloth/` - Unsloth (fast fine-tuning)

## Troubleshooting

**Out of Memory:**
```bash
# Reduce batch size in the script
per_device_train_batch_size=2
gradient_accumulation_steps=8

# Or use QLoRA instead of LoRA
```

**Slow training:**
```bash
# Reduce dataset size for testing
# Increase batch size if memory available
# Use gradient checkpointing
```

**Poor results:**
```bash
# Increase training epochs
# Increase LoRA rank
# Adjust learning rate
# Check dataset quality
```

## Next Steps After Fine-tuning

1. **Test the fine-tuned model:**
   ```bash
   # Load and test with simple prompts
   ```

2. **Quantize to NVFP4:**
   ```bash
   # Use the nvfp4 quantization scripts
   # Point to your fine-tuned model
   ```

3. **Serve the quantized model:**
   ```bash
   # Use the serve script with your quantized fine-tuned model
   ```

## Notes

- Fine-tuning modifies the original model weights based on your data
- LoRA is recommended for most use cases (faster, less memory, good results)
- Always validate your fine-tuned model before quantizing
- Quantization should be the LAST step (after fine-tuning)
- Keep backups of your fine-tuned models before quantizing

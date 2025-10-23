# Multi-Agent Embedding Deployment

Runs the GPT-OSS-120B MXFP4 generator and the Qwen3 embedding model with Docker Compose.

## Prep Steps

1. Stop any existing inference containers.
2. Clear cached GPU and host memory:
   ```bash
   sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
   ```
3. Confirm `nvidia-smi` reports the expected free GPU memory (≈120 GB).

## Build & Launch

```bash
cd /home/kfrost/DxgSparkDev/deployments/Mult-Agent-Embedding
# Start both services (includes cache drop + build step)
./start.sh
```

## Ports

- GPT-OSS-120B OpenAI-compatible endpoint: `http://<host>:8000/v1`
- Qwen3 embedding endpoint: `http://<host>:8001`

Use `docker logs -f gpt-oss-120b` to monitor load metrics and adjust `--gpu-memory-utilization` or `--max-num-seqs` if VRAM pressure is too high.

## Shutdown

```bash
cd /home/kfrost/DxgSparkDev/deployments/Mult-Agent-Embedding
./stop.sh
```

# Workspace Instructions

GitHub Copilot should treat this repository as running via an active SSH session to an NVIDIA DGX Spark system. All commands, scripts, and examples in this repository should be executed on that device unless a task explicitly calls out a different environment.

- **Device**: NVIDIA DGX Spark running Linux with the default `bash` shell.
- **Repository root on device**: `/home/kfrost/DxgSparkDev`.
- **Primary reference material**: reuse patterns, commands, and scripts from `/home/kfrost/dgx-spark-playbooks/` before generating new code. Align new work with the playbook conventions when possible.
- **SSH context**: confirm you are connected to the DGX Spark node before running scripts or package installs.

When Copilot proposes code or instructions, it should:

- Prefer examples from the playbooks directory mentioned above.
- Emit shell commands relative to `/home/kfrost/DxgSparkDev` (e.g., `cd /home/kfrost/DxgSparkDev` before running tools).
- Call out any assumptions that differ from the DGX Spark baseline and document the deviation in the relevant README or script comments.

## NGC Access

- The NVIDIA NGC API key is stored encrypted at `~/.config/ngc/ngc_api_key.gpg`.
- `~/.bashrc` automatically sources `~/.config/ngc/load_ngc_key.sh`, which decrypts the file (GPG passphrase `8557`) and exports `NGC_API_KEY` for the session. Copilot should prefer using this environment variable instead of hard-coding keys.
- If a shell session is missing `NGC_API_KEY`, run `source ~/.config/ngc/load_ngc_key.sh` to populate it.
- Do not commit the decrypted key or the passphrase; they are intentionally kept local.

## NVFP4 Workflow Notes

- NVFP4 checkpoints (e.g., Qwen3-Next-80B-A3B-Thinking-NVFP4) require a runtime whose Triton/PTX toolchain supports the GB10 `sm_121a` architecture. Current public vLLM containers fail with `ptxas` errors until an updated build ships.
- Before attempting NVFP4 inference, verify the vLLM/TensorRT-LLM release notes for Blackwell (`sm_121`) support. If unavailable, plan to use NVIDIA NIM images (requires `NGC_API_KEY`) or rebuild vLLM against the latest CUDA/Triton stack manually.
- Document any custom build steps (patched Triton, CUDA toolkit installs, etc.) in the relevant README so deployment can be reproduced.

# First-time setup guide

TADA uses Meta's Llama 3.2 tokenizer internally. This is a gated model on HuggingFace, so you need to complete a few one-time steps before TADA will work.

## 1. Accept the Llama license

1. Go to [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B) on HuggingFace
2. Click "Access repository" and accept Meta's license agreement
3. Approval is usually instant

This is required because TADA loads the Llama 3.2 tokenizer at startup. Without access, you'll get a 401 error.

## 2. Create a HuggingFace access token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Create a new token (read access is sufficient)
3. Copy the token

## 3. Log in via CLI

```bash
pip install huggingface_hub
huggingface-cli login
# Paste your token when prompted
```

This saves the token to `~/.cache/huggingface/token`. You only need to do this once.

## 4. Python environment

TADA requires Python 3.11 or 3.12. Create a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Disk space

Model weights are ~4 GB total and are downloaded to `~/.cache/huggingface/hub/` on first run. Make sure you have space available. If you want to use a different cache location:

```bash
export HF_HOME=/path/to/your/cache
```

## 6. Verify setup

```bash
python generate.py "Testing, one two three."
```

The first run will:
1. Download model weights (~4 GB, one time)
2. Load models into RAM (~30s)
3. Run warmup (~45-60s)
4. Generate audio (~30-90s depending on text length)

Subsequent runs skip the download but still need model loading and warmup.

## Troubleshooting

**401 Unauthorized / gated repo error**
- You haven't accepted the Llama license at [meta-llama/Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- Or your HuggingFace token doesn't have access. Re-run `huggingface-cli login`.

**Out of memory**
- TADA needs ~5-6 GB RAM during inference. Close other memory-heavy apps.
- If you have 8 GB total RAM, it should still work but may be tight.

**Slow first generation**
- This is normal. The warmup pass handles JIT compilation and KV cache allocation. Run with `--skip-warmup` if you want to skip it (first real generation will be slower instead).

**MPS / GPU errors**
- Don't use MPS. TADA has device mismatch bugs with Apple's GPU backend, and MPS float16 can kernel panic the Mac. The CPU path uses Apple's AMX coprocessor and is well-optimized.

**No audio in output file**
- If using MPS somehow, this is the `_decode_wav` bug mixing device tensors. Switch to CPU.
- If on CPU, try increasing `--flow-steps` (minimum 10 for usable audio).

# tada-apple-silicon

Optimized inference for [Hume AI's TADA-1B](https://huggingface.co/HumeAI/tada-1b) text-to-speech model on Apple Silicon Macs.

TADA is a 1 billion parameter TTS model built on Llama 3.2. It was designed for NVIDIA GPUs. This repo packages optimizations discovered through 37 autonomous experiments that brought real-time factor from **81x RTF down to ~3-4x RTF** on a Mac Mini M4 — making local TTS practical on Apple Silicon.

## Quick start

```bash
# Clone and set up
git clone https://github.com/Garblesnarff/tada-apple-silicon.git
cd tada-apple-silicon
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# First-time setup: HuggingFace auth (see setup_guide.md for details)
huggingface-cli login

# Generate speech
python generate.py "Hello, this is TADA running on Apple Silicon."
```

Output is saved to `output.wav` by default (24kHz WAV).

## Performance

Measured on Mac Mini M4 (16GB RAM), CPU only, float32:

| Metric | Value |
|---|---|
| Real-time factor (RTF) | ~3-4x with 20 flow steps |
| RAM usage during inference | ~5-6 GB |
| Model loading | ~30s first time (cached after) |
| Warmup | ~45-60s |
| Steady-state | ~25s audio in ~85s |

Lower RTF = faster. An RTF of 3x means 10 seconds of audio takes ~30 seconds to generate.

## Usage

```bash
# Basic text-to-speech
python generate.py "Your text here"

# Custom output path
python generate.py "Your text here" --output speech.wav

# Voice cloning with reference audio
python generate.py "Your text here" --reference voice.wav --reference-text "Transcript of the reference audio."

# Emotion/style steering via system prompt
python generate.py "Your text here" --system-prompt "Speak with excitement and wonder."

# Faster generation (lower quality)
python generate.py "Your text here" --flow-steps 10

# All options
python generate.py "Your text here" \
  --output out.wav \
  --reference voice.wav \
  --reference-text "Reference transcript." \
  --system-prompt "Speak calmly." \
  --flow-steps 20 \
  --text-temperature 0.6 \
  --noise-temperature 0.9 \
  --transition-steps 5 \
  --skip-warmup
```

## Key optimizations

These are the discoveries from 37 experiments, ranked by impact:

### 1. float32 is faster than float16 on Apple Silicon

This is counterintuitive. Apple's AMX (Apple Matrix Extension) coprocessor is natively float32. Using float16 adds type conversion overhead on every matrix operation. On NVIDIA GPUs, float16 is faster due to Tensor Cores — on Apple Silicon, it's the opposite.

### 2. CPU-only (MPS is broken for TADA)

Apple's MPS (Metal Performance Shaders) GPU backend has two bugs with TADA:
- `_lm_head_forward` moves input to CPU but leaves weights on MPS (device mismatch crash)
- `_decode_wav` mixes MPS and CPU tensors in `torch.cat` (silent failure, produces no audio)
- MPS with float16 will **kernel panic** the Mac

Use CPU only. The AMX coprocessor makes CPU inference surprisingly fast anyway.

### 3. Warmup run eliminates lazy-init overhead

The first inference call triggers JIT compilation, KV cache allocation, and memory layout optimization. Without warmup, RTF is ~7x. A single warmup call with a short phrase ("Hi.") absorbs this cost, bringing subsequent calls to ~3-4x RTF.

### 4. Disable internal timing/logging

TADA calls `time.time()` every generation step and builds debug log strings internally. Monkey-patching `generate()` to pass `log_time=False` and `verbose=False` reduces per-step Python overhead.

### 5. Flow matching steps control quality/speed tradeoff

| Steps | Quality | RTF |
|---|---|---|
| 20 | Best (default) | ~3-4x |
| 10 | Good (minimal loss) | ~2x |
| 5 | Degraded | ~1.5x |
| 1 | Garbage (static/truncation) | ~0.5x |

### 6. Undocumented system_prompt parameter

`generate()` accepts `system_prompt` and `user_turn_prompt` kwargs that inject Llama-style chat template headers. Not strongly trained but provides noticeable emotion/style steering:

```python
output = model.generate(
    prompt=prompt,
    text="Your text here",
    system_prompt="Speak with deep sadness and melancholy.",
    inference_options=inference_options,
)
```

### 7. transition_steps matters for voice quality

Controls blending between reference and generated audio. Set to 0 and you get voice mixing artifacts. Default of 5 gives clean voice cloning. The first word may still sound slightly off due to the transition window.

## What didn't work

- **float16**: Slower on Apple Silicon due to AMX conversion overhead
- **MPS (GPU)**: Device mismatch bugs + kernel panics
- **1 flow matching step**: Great RTF numbers but audio is static/garbage
- **Disabling CFG (cfg_scale=1.0)**: Degrades quality without meaningful speed gain at 20 steps
- **torch.compile**: Not supported on MPS, no benefit on CPU for this model
- **Quantization (bitsandbytes)**: Not available for MPS/CPU backends

## Project structure

```
tada-apple-silicon/
├── generate.py          # Main CLI script
├── requirements.txt     # Python dependencies
├── setup_guide.md       # First-time setup instructions
├── examples/
│   ├── basic_tts.py     # Minimal example
│   ├── voice_cloning.py # Voice cloning with reference audio
│   └── emotion.py       # Emotion/style steering
├── LICENSE              # MIT
└── README.md
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- ~5-6 GB free RAM
- ~4 GB disk for model weights (downloaded on first run)
- HuggingFace account with Llama license accepted (see [setup_guide.md](setup_guide.md))

## License

MIT

## Acknowledgments

- [Hume AI](https://www.hume.ai/) for the TADA model
- [Meta](https://ai.meta.com/) for Llama 3.2
- Optimization methodology inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)

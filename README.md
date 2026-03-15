# tada-apple-silicon

Optimized inference for [Hume AI's TADA-1B](https://huggingface.co/HumeAI/tada-1b) text-to-speech model on Apple Silicon Macs.

TADA is a 1 billion parameter TTS model built on Llama 3.2. This repo provides two inference backends:

- **MLX backend** (recommended): Runs LLM + decoder on Metal GPU via [MLX](https://github.com/ml-explore/mlx). Achieves **2x faster than real-time** — 0.45x RTF on M4 Mac Mini.
- **PyTorch backend**: CPU-only with AMX optimizations. ~3-4x RTF. No setup required beyond `pip install`.

## Quick start (MLX — recommended)

```bash
git clone https://github.com/Garblesnarff/tada-apple-silicon.git
cd tada-apple-silicon
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mlx.txt

# First-time setup: HuggingFace auth (see setup_guide.md)
huggingface-cli login

# One-time weight conversion (downloads ~4 GB, converts to MLX format)
python -m mlx_tada.convert_weights --output-dir ./mlx-weights

# Generate speech (4-bit quantization enabled by default)
python generate_mlx.py "Hello, this is TADA running on Apple Silicon Metal GPU."
```

## Quick start (PyTorch — no MLX required)

```bash
git clone https://github.com/Garblesnarff/tada-apple-silicon.git
cd tada-apple-silicon
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

huggingface-cli login
python generate.py "Hello, this is TADA running on Apple Silicon."
```

Output is saved to `output.wav` (24kHz WAV).

## Performance

Measured on Mac Mini M4 (16GB RAM):

| Backend | RTF (short text) | RTF (long text) | Notes |
|---|---|---|---|
| MLX (2 flow steps, 4-bit, optimized) | ~0.6x | **0.35x** | 2x faster than real-time |
| MLX (5 flow steps, 4-bit quantized) | ~1.0x | ~0.6x | Good quality/speed balance |
| MLX (20 flow steps) | ~2.4x | ~1.8x | Best quality |
| PyTorch CPU (20 flow steps) | ~3-4x | ~3x | No MLX needed |

Lower RTF = faster. An RTF below 1.0 means audio generates faster than real-time.

### MLX performance progression

These are the cumulative optimizations that brought RTF from 81x to 0.45x (2x faster than real-time):

| Optimization | RTF | Speedup |
|---|---|---|
| Baseline (PyTorch CPU float16) | 81x | — |
| PyTorch CPU float32 (AMX) | ~7x | 11x |
| + Warmup + logging disabled | ~3-4x | 2x |
| + MLX LLM on Metal GPU | 5.04x | — |
| + 4-bit LLM quantization | 2.38x | 2.1x |
| + Flow matching skip (prompt) | 2.18x | 1.1x |
| + 5 flow steps | 1.49x | 1.5x |
| + MLX decoder on Metal GPU | 1.27x | 1.2x |
| + 3 flow steps | 0.72x | 1.8x |
| + 2 flow steps | 0.60x | 1.2x |
| + Greedy text sampling | 0.56x | 1.1x |
| + CFG scale 2.0 | 0.51x | 1.1x |
| + Noise temperature 0.6 | 0.48x | 1.1x |
| + 3-stage warmup | **0.45x** | 1.1x |

## Usage (MLX backend)

```bash
# Basic text-to-speech
python generate_mlx.py "Your text here"

# Custom output path
python generate_mlx.py "Your text here" --output speech.wav

# Voice cloning with reference audio
python generate_mlx.py "Your text here" --reference voice.wav --reference-text "Transcript of the reference audio."

# Emotion/style steering
python generate_mlx.py "Your text here" --system-prompt "Speak with excitement and wonder."

# Disable 4-bit quantization (slightly higher quality, slower)
python generate_mlx.py "Your text here" --no-quantize

# Best quality (slower)
python generate_mlx.py "Your text here" --flow-steps 20 --no-quantize

# All options
python generate_mlx.py "Your text here" \
  --output out.wav \
  --weights-dir ./mlx-weights \
  --reference voice.wav \
  --reference-text "Reference transcript." \
  --system-prompt "Speak calmly." \
  --flow-steps 2 \
  --cfg-scale 2.0 \
  --text-temperature 0.6 \
  --noise-temperature 0.6 \
  --transition-steps 5 \
  --skip-warmup
```

## Usage (PyTorch backend)

```bash
# Basic text-to-speech
python generate.py "Your text here"

# Custom output path
python generate.py "Your text here" --output speech.wav

# Voice cloning with reference audio
python generate.py "Your text here" --reference voice.wav --reference-text "Transcript of the reference audio."

# Emotion/style steering via system prompt
python generate.py "Your text here" --system-prompt "Speak with excitement and wonder."

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

### MLX Metal GPU acceleration

The biggest win comes from running the LLM backbone and decoder on Apple's Metal GPU via MLX instead of PyTorch CPU:

- **LLM backbone**: The 1B parameter Llama 3.2 model runs on Metal GPU with optional 4-bit quantization (group_size=64). This gives ~2x speedup over PyTorch CPU float32.
- **Decoder**: The 118.7M parameter DAC decoder (Snake1d activations + ConvTranspose1d upsampling) runs on Metal GPU, replacing the PyTorch CPU path. 3.5x faster for long audio.
- **Flow matching**: The VibeVoice diffusion head runs entirely on Metal GPU. With 5 steps (vs 20 default), quality remains good with ~3x speedup.

The PyTorch encoder (wav2vec2-large, 300M params) still runs on CPU — it executes once per generation and is not the bottleneck.

### Inference tuning

1. **2 flow matching steps**: Each step requires 2 diffusion head forward passes (~99ms each). Reducing from 20→2 saves ~3.5s per token. Quality validated via Gemini 2.5 Flash (naturalness 8-9 on medium/long prompts).
2. **Greedy text sampling**: Skips softmax/top-p/categorical sampling. Faster and more deterministic.
3. **CFG scale 2.0**: Higher classifier-free guidance produces longer, better-paced audio. Counterintuitively lowers RTF because audio duration increases more than wall time.
4. **Noise temperature 0.6**: Tighter initial noise for flow matching. Below 0.5 causes hallucination.
5. **3-stage warmup**: Metal kernels are JIT-compiled per sequence length. Three progressively longer warmup prompts ensure all kernels are compiled before the first real request.

### Other optimizations (both backends)

1. **float32 > float16 on Apple Silicon**: Apple's AMX coprocessor is natively float32. float16 adds conversion overhead.
2. **Disabled internal timing/logging**: TADA's per-step `time.time()` and debug logging adds Python overhead.

## How the MLX port works

The MLX backend reimplements TADA's core components in Apple's MLX framework:

```
PyTorch encoder (once, CPU)
    ↓
MLX LLM backbone (autoregressive, Metal GPU)
    ↓ per token: embed → transformer → diffusion head → sample
MLX decoder (once, Metal GPU)
    ↓ LocalAttentionEncoder → DACDecoder (480x upsample)
WAV output
```

### Architecture

| Component | Params | Framework | Device |
|---|---|---|---|
| Encoder (wav2vec2-large) | 300M | PyTorch | CPU |
| LLM (Llama 3.2 1B) | 1B | MLX | Metal GPU |
| Diffusion head (VibeVoice) | ~50M | MLX | Metal GPU |
| Decoder (LocalAttn + DAC) | 118.7M | MLX | Metal GPU |

### Weight conversion

The `convert_weights.py` script handles:
- Loading PyTorch weights from HuggingFace cache
- Mapping weight names to MLX module structure
- Materializing weight normalization (`weight = g * v / ||v||`)
- Transposing Conv1d weights (PyTorch NCL → MLX NLC layout)
- Saving as safetensors for fast loading

## Project structure

```
tada-apple-silicon/
├── generate.py           # PyTorch CPU CLI (no MLX needed)
├── generate_mlx.py       # MLX Metal GPU CLI (recommended)
├── requirements.txt      # PyTorch-only dependencies
├── requirements-mlx.txt  # Full dependencies including MLX
├── setup_guide.md        # First-time setup instructions
├── mlx_tada/             # MLX implementation
│   ├── __init__.py
│   ├── llm.py            # Llama 3.2 1B backbone
│   ├── diffusion.py      # VibeVoice diffusion head
│   ├── decoder.py        # Snake1d + DAC decoder + LocalAttentionEncoder
│   ├── generate.py       # Autoregressive generation loop
│   ├── hybrid.py         # Hybrid PyTorch/MLX orchestration
│   ├── convert_weights.py # Weight conversion script
│   └── utils.py          # Gray code encoding/decoding
├── examples/
│   ├── basic_tts.py      # Minimal PyTorch example
│   ├── voice_cloning.py  # Voice cloning with reference audio
│   └── emotion.py        # Emotion/style steering
├── LICENSE               # MIT
└── README.md
```

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.11 or 3.12
- ~5-6 GB free RAM
- ~4 GB disk for model weights
- HuggingFace account with Llama license accepted (see [setup_guide.md](setup_guide.md))

For MLX backend additionally:
- MLX >= 0.22.0 (`pip install mlx`)

## License

MIT

## Acknowledgments

- [Hume AI](https://www.hume.ai/) for the TADA model
- [Meta](https://ai.meta.com/) for Llama 3.2
- [Apple MLX](https://github.com/ml-explore/mlx) for the Metal GPU framework
- Optimization methodology inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)

#!/usr/bin/env python3
"""TADA TTS inference using MLX on Apple Silicon Metal GPU.

This is the fast path — runs the LLM and decoder on Metal GPU via MLX.
Achieves sub-real-time performance (0.82x RTF for long text) on M4 Mac Mini.

Requires one-time weight conversion:
    python -m mlx_tada.convert_weights --output-dir ./mlx-weights

Usage:
    python generate_mlx.py "Your text here"
    python generate_mlx.py "Your text here" --output speech.wav
    python generate_mlx.py "Your text here" --reference voice.wav --reference-text "Transcript."
    python generate_mlx.py "Your text here" --system-prompt "Speak with excitement."
"""

import argparse
import os
import time

import torch
import torchaudio

from mlx_tada.hybrid import HybridTadaInference
from mlx_tada.generate import GenerateConfig


def trim_silence(audio_tensor, sample_rate=24000):
    """Trim trailing silence using sliding window RMS."""
    window = int(0.1 * sample_rate)  # 100ms
    for i in range(len(audio_tensor) - window, 0, -window):
        rms = (audio_tensor[i : i + window] ** 2).mean().sqrt().item()
        if rms > 0.005:
            end = min(i + window + int(0.2 * sample_rate), len(audio_tensor))
            return audio_tensor[:end]
    return audio_tensor


def main():
    parser = argparse.ArgumentParser(
        description="TADA TTS on Apple Silicon Metal GPU (MLX backend)"
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--output", "-o", default="output.wav", help="Output WAV path (default: output.wav)"
    )
    parser.add_argument(
        "--weights-dir", default="./mlx-weights",
        help="Path to converted MLX weights (default: ./mlx-weights)"
    )
    parser.add_argument(
        "--reference", help="Path to reference audio WAV for voice cloning"
    )
    parser.add_argument(
        "--reference-text",
        help="Transcript of the reference audio (required with --reference)",
    )
    parser.add_argument(
        "--system-prompt",
        help='Emotion/style steering, e.g. "Speak with excitement."',
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=5,
        help="Flow matching steps: 20=best, 10=good, 5=fast+good (default: 5)",
    )
    parser.add_argument(
        "--text-temperature", type=float, default=0.6, help="Text sampling temperature (default: 0.6)"
    )
    parser.add_argument(
        "--noise-temperature", type=float, default=0.9, help="Noise temperature (default: 0.9)"
    )
    parser.add_argument(
        "--transition-steps",
        type=int,
        default=5,
        help="Voice transition blending steps (default: 5)",
    )
    parser.add_argument(
        "--quantize", action="store_true",
        help="Quantize LLM to 4-bit (faster, minimal quality loss)"
    )
    parser.add_argument(
        "--pytorch-decoder", action="store_true",
        help="Use PyTorch CPU decoder instead of MLX Metal decoder"
    )
    parser.add_argument(
        "--skip-warmup", action="store_true", help="Skip warmup (faster start, slower first generation)"
    )
    args = parser.parse_args()

    if args.reference and not args.reference_text:
        parser.error("--reference-text is required when using --reference")

    if not os.path.exists(os.path.join(args.weights_dir, "weights.safetensors")):
        print(f"ERROR: MLX weights not found in {args.weights_dir}")
        print("Run weight conversion first:")
        print(f"  python -m mlx_tada.convert_weights --output-dir {args.weights_dir}")
        raise SystemExit(1)

    # Load hybrid model
    inference = HybridTadaInference(
        mlx_weights_dir=args.weights_dir,
        quantize_llm=args.quantize,
        use_mlx_decoder=not args.pytorch_decoder,
    )

    # Encode reference audio (or use default)
    if args.reference:
        prompt = inference.encode_reference(args.reference, args.reference_text)
    else:
        from tada.utils.test_utils import get_sample_dir
        sample_dir = get_sample_dir()
        audio_path = os.path.join(sample_dir, "sample.wav")
        ref_text = "The morning sun cast long shadows across the quiet street, as birds began their familiar chorus of songs."
        prompt = inference.encode_reference(audio_path, ref_text)

    # Generation config
    config = GenerateConfig(
        text_temperature=args.text_temperature,
        noise_temperature=args.noise_temperature,
        num_flow_matching_steps=args.flow_steps,
        acoustic_cfg_scale=1.6,
    )

    # Warmup
    if not args.skip_warmup:
        inference.warmup(prompt, config=config)

    # Generate
    print(f"\nGenerating speech ({args.flow_steps} flow steps)...")
    start = time.time()

    wav, gen_time = inference.generate(
        prompt=prompt,
        text=args.text,
        num_transition_steps=args.transition_steps,
        system_prompt=args.system_prompt,
        config=config,
    )
    elapsed = time.time() - start

    if wav is None:
        print("ERROR: No audio generated.")
        raise SystemExit(1)

    audio_out = trim_silence(wav.detach().float().cpu())
    torchaudio.save(args.output, audio_out.unsqueeze(0), 24000)

    duration = audio_out.shape[-1] / 24000
    print(f"\nSaved: {args.output}")
    print(f"Audio duration: {duration:.1f}s")
    print(f"Generation time: {elapsed:.1f}s")
    print(f"RTF: {elapsed / duration:.2f}x")


if __name__ == "__main__":
    main()

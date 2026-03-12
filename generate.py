#!/usr/bin/env python3
"""TADA TTS inference optimized for Apple Silicon.

Usage:
    python generate.py "Your text here"
    python generate.py "Your text here" --output speech.wav
    python generate.py "Your text here" --reference voice.wav --reference-text "Transcript."
    python generate.py "Your text here" --system-prompt "Speak with excitement."
"""

import argparse
import os
import time

import torch
import torchaudio
from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM, InferenceOptions


# ---------------------------------------------------------------------------
# Monkey-patch: disable internal timing in _generate
# TADA runs time.time() every step and builds debug logs. Disabling this
# reduces per-step Python overhead.
# ---------------------------------------------------------------------------
_original_generate = TadaForCausalLM.generate


@torch.no_grad()
def _fast_generate(self, *args, **kwargs):
    kwargs["verbose"] = False
    original_internal = self._generate

    def patched_generate(*a, **kw):
        kw["log_time"] = False
        kw["verbose"] = False
        return original_internal(*a, **kw)

    self._generate = patched_generate
    try:
        result = _original_generate(self, *args, **kwargs)
    finally:
        self._generate = original_internal
    return result


TadaForCausalLM.generate = _fast_generate


def trim_silence(audio_tensor, sample_rate=24000):
    """Trim trailing silence using sliding window RMS."""
    window = int(0.1 * sample_rate)  # 100ms
    for i in range(len(audio_tensor) - window, 0, -window):
        rms = (audio_tensor[i : i + window] ** 2).mean().sqrt().item()
        if rms > 0.005:
            end = min(i + window + int(0.2 * sample_rate), len(audio_tensor))
            return audio_tensor[:end]
    return audio_tensor


def load_models(device, dtype):
    """Load encoder and TADA model."""
    print("Loading encoder from HumeAI/tada-codec...")
    encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder")
    encoder = encoder.to(device=device, dtype=dtype)
    encoder.eval()

    print("Loading TADA-1B model...")
    model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b")
    model = model.to(device=device, dtype=dtype)
    model.eval()

    return encoder, model


def encode_reference(encoder, audio_path, ref_text, device, dtype):
    """Encode reference audio for voice cloning."""
    print(f"Loading reference audio: {audio_path}")
    audio, sample_rate = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device=device, dtype=dtype)

    print("Encoding reference audio...")
    prompt = encoder(
        audio,
        text=[ref_text],
        audio_length=torch.tensor([audio.shape[1]], device=device),
        sample_rate=sample_rate,
    )
    return prompt


def get_default_prompt(encoder, device, dtype):
    """Get default prompt using TADA's built-in sample audio."""
    from tada.utils.test_utils import get_sample_dir

    sample_dir = get_sample_dir()
    audio_path = os.path.join(sample_dir, "sample.wav")
    ref_text = "The morning sun cast long shadows across the quiet street, as birds began their familiar chorus of songs."

    audio, sample_rate = torchaudio.load(audio_path)
    if audio.shape[0] > 1:
        audio = audio.mean(dim=0, keepdim=True)
    audio = audio.to(device=device, dtype=dtype)

    prompt = encoder(
        audio,
        text=[ref_text],
        audio_length=torch.tensor([audio.shape[1]], device=device),
        sample_rate=sample_rate,
    )
    return prompt


def warmup(model, prompt, inference_options):
    """Run a short warmup to eliminate lazy-init overhead."""
    print("Running warmup (this takes 45-60s on first run)...")
    t = time.time()
    _ = model.generate(
        prompt=prompt,
        text="Hi.",
        num_transition_steps=5,
        inference_options=inference_options,
    )
    print(f"Warmup done in {time.time() - t:.1f}s")


def main():
    parser = argparse.ArgumentParser(
        description="TADA TTS optimized for Apple Silicon"
    )
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument(
        "--output", "-o", default="output.wav", help="Output WAV path (default: output.wav)"
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
        default=20,
        help="Flow matching steps: 20=best quality, 10=good, 5+=degraded (default: 20)",
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
        "--skip-warmup", action="store_true", help="Skip warmup (faster start, slower first generation)"
    )
    args = parser.parse_args()

    if args.reference and not args.reference_text:
        parser.error("--reference-text is required when using --reference")

    # Apple Silicon optimizations
    device = torch.device("cpu")
    dtype = torch.float32  # float32 is FASTER than float16 on Apple Silicon (AMX)
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(os.cpu_count() or 4)

    print(f"Device: {device} | Dtype: {dtype} | Threads: {torch.get_num_threads()}")

    # Load models
    encoder, model = load_models(device, dtype)

    # Encode reference audio (or use default)
    if args.reference:
        prompt = encode_reference(
            encoder, args.reference, args.reference_text, device, dtype
        )
    else:
        prompt = get_default_prompt(encoder, device, dtype)

    # Inference options
    inference_options = InferenceOptions(
        text_do_sample=True,
        text_temperature=args.text_temperature,
        text_top_k=0,
        text_top_p=0.9,
        acoustic_cfg_scale=1.6,
        duration_cfg_scale=1.0,
        cfg_schedule="constant",
        noise_temperature=args.noise_temperature,
        num_flow_matching_steps=args.flow_steps,
        time_schedule="logsnr",
        num_acoustic_candidates=1,
    )

    # Warmup
    if not args.skip_warmup:
        warmup(model, prompt, inference_options)

    # Generate
    print(f"\nGenerating speech ({args.flow_steps} flow steps)...")
    start = time.time()

    gen_kwargs = dict(
        prompt=prompt,
        text=args.text,
        num_transition_steps=args.transition_steps,
        inference_options=inference_options,
    )
    if args.system_prompt:
        gen_kwargs["system_prompt"] = args.system_prompt

    output = model.generate(**gen_kwargs)
    elapsed = time.time() - start

    if output.audio[0] is None:
        print("ERROR: No audio generated.")
        raise SystemExit(1)

    audio_out = trim_silence(output.audio[0].detach().float().cpu())
    torchaudio.save(args.output, audio_out.unsqueeze(0), 24000)

    duration = audio_out.shape[-1] / 24000
    print(f"\nSaved: {args.output}")
    print(f"Audio duration: {duration:.1f}s")
    print(f"Generation time: {elapsed:.1f}s")
    print(f"RTF: {elapsed / duration:.2f}x")


if __name__ == "__main__":
    main()

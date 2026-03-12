"""Emotion and style steering with TADA on Apple Silicon.

TADA has an undocumented system_prompt parameter that injects Llama-style
chat template headers. It's not strongly trained for this, but provides
noticeable style differences — especially for sadness, excitement, and
whispering.
"""

import os
import time

import torch
import torchaudio
from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM, InferenceOptions
from tada.utils.test_utils import get_sample_dir

# Apple Silicon settings
device = torch.device("cpu")
dtype = torch.float32
torch.set_float32_matmul_precision("medium")
torch.set_num_threads(os.cpu_count() or 4)

# Load models
print("Loading models...")
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder")
encoder = encoder.to(device=device, dtype=dtype).eval()

model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b")
model = model.to(device=device, dtype=dtype).eval()

# Default reference audio
sample_dir = get_sample_dir()
audio, sr = torchaudio.load(os.path.join(sample_dir, "sample.wav"))
if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)
audio = audio.to(device=device, dtype=dtype)

prompt = encoder(
    audio,
    text=["The morning sun cast long shadows across the quiet street."],
    audio_length=torch.tensor([audio.shape[1]], device=device),
    sample_rate=sr,
)

options = InferenceOptions(
    text_do_sample=True,
    text_temperature=0.6,
    text_top_k=0,
    text_top_p=0.9,
    acoustic_cfg_scale=1.6,
    duration_cfg_scale=1.0,
    noise_temperature=0.9,
    num_flow_matching_steps=20,
    time_schedule="logsnr",
    num_acoustic_candidates=1,
)

# Warmup
print("Warming up...")
_ = model.generate(prompt=prompt, text="Hi.", num_transition_steps=5, inference_options=options)

# Test text
text = "The autumn leaves were falling softly through the amber light, as the old cathedral bells rang out across the quiet town."

# Emotion styles to test
styles = [
    ("neutral", None),
    ("sad", "Speak with deep sadness and melancholy, as if reflecting on a painful loss."),
    ("excited", "Speak with excitement and wonder, full of energy and amazement."),
    ("scary", "Speak in a dark, ominous, foreboding tone, like narrating a horror story."),
    ("whisper", "Speak in a soft, hushed whisper, as if sharing a secret."),
    ("angry", "Speak with intense anger and frustration, barely containing your rage."),
]

for label, system_prompt in styles:
    print(f"\n--- {label.upper()} ---")

    kwargs = dict(
        prompt=prompt,
        text=text,
        num_transition_steps=5,
        inference_options=options,
    )
    if system_prompt:
        kwargs["system_prompt"] = system_prompt
        print(f"System prompt: {system_prompt}")

    start = time.time()
    output = model.generate(**kwargs)
    elapsed = time.time() - start

    if output.audio[0] is not None:
        audio_out = output.audio[0].detach().float().cpu()
        out_path = f"emotion_{label}.wav"
        torchaudio.save(out_path, audio_out.unsqueeze(0), 24000)
        duration = audio_out.shape[-1] / 24000
        print(f"Audio: {duration:.1f}s | Time: {elapsed:.1f}s | RTF: {elapsed/duration:.2f}x")
        print(f"Saved to {out_path}")
    else:
        print("ERROR: No audio generated")

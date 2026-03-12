"""Basic text-to-speech with TADA on Apple Silicon.

Minimal example — generates speech from text using default voice and settings.
"""

import os
import time

import torch
import torchaudio
from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM, InferenceOptions
from tada.utils.test_utils import get_sample_dir

# Apple Silicon: float32 is faster than float16 (AMX is natively float32)
device = torch.device("cpu")
dtype = torch.float32
torch.set_float32_matmul_precision("medium")
torch.set_num_threads(os.cpu_count() or 4)

# Load models
encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder")
encoder = encoder.to(device=device, dtype=dtype).eval()

model = TadaForCausalLM.from_pretrained("HumeAI/tada-1b")
model = model.to(device=device, dtype=dtype).eval()

# Use TADA's built-in sample audio as reference
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

# Quality settings
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

# Warmup (essential for good performance)
print("Warming up...")
_ = model.generate(prompt=prompt, text="Hi.", num_transition_steps=5, inference_options=options)

# Generate
text = "Welcome to TADA running on Apple Silicon. This is a one billion parameter text to speech model, optimized for local inference."

print("Generating...")
start = time.time()
output = model.generate(
    prompt=prompt, text=text, num_transition_steps=5, inference_options=options
)
elapsed = time.time() - start

audio_out = output.audio[0].detach().float().cpu()
torchaudio.save("basic_output.wav", audio_out.unsqueeze(0), 24000)

duration = audio_out.shape[-1] / 24000
print(f"Audio: {duration:.1f}s | Time: {elapsed:.1f}s | RTF: {elapsed/duration:.2f}x")
print("Saved to basic_output.wav")

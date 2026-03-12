"""Voice cloning with TADA on Apple Silicon.

Provide a reference WAV file and its transcript to clone a voice.
The reference audio should be clean speech, ideally 5-15 seconds.
"""

import os
import sys
import time

import torch
import torchaudio
from tada.modules.encoder import Encoder
from tada.modules.tada import TadaForCausalLM, InferenceOptions

# --- Configuration ---
REFERENCE_AUDIO = "your_voice.wav"  # Path to reference WAV file
REFERENCE_TEXT = "Exact transcript of what is said in the reference audio."
TEXT_TO_SPEAK = "This should sound like the reference voice."
OUTPUT_PATH = "cloned_output.wav"

# Check reference exists
if not os.path.exists(REFERENCE_AUDIO):
    print(f"Reference audio not found: {REFERENCE_AUDIO}")
    print("Edit REFERENCE_AUDIO at the top of this script to point to your WAV file.")
    sys.exit(1)

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

# Load and encode reference audio
print(f"Encoding reference: {REFERENCE_AUDIO}")
audio, sr = torchaudio.load(REFERENCE_AUDIO)
if audio.shape[0] > 1:
    audio = audio.mean(dim=0, keepdim=True)
audio = audio.to(device=device, dtype=dtype)

prompt = encoder(
    audio,
    text=[REFERENCE_TEXT],
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

# Generate with cloned voice
# transition_steps=5 gives clean voice cloning (0 causes artifacts)
print("Generating with cloned voice...")
start = time.time()
output = model.generate(
    prompt=prompt,
    text=TEXT_TO_SPEAK,
    num_transition_steps=5,  # Important: 0 causes voice mixing artifacts
    inference_options=options,
)
elapsed = time.time() - start

audio_out = output.audio[0].detach().float().cpu()
torchaudio.save(OUTPUT_PATH, audio_out.unsqueeze(0), 24000)

duration = audio_out.shape[-1] / 24000
print(f"Audio: {duration:.1f}s | Time: {elapsed:.1f}s | RTF: {elapsed/duration:.2f}x")
print(f"Saved to {OUTPUT_PATH}")

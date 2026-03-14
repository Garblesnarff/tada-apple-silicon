"""Hybrid MLX/PyTorch inference for TADA-1B.

Uses MLX for the autoregressive LLM loop (Metal GPU) and optionally
MLX for the decoder too. PyTorch encoder runs once (not the bottleneck).
"""

import json
import os
import time

import mlx.core as mx
import numpy as np
import torch
import torchaudio

from .generate import GenerateConfig, generate
from .llm import TadaMLX, TadaModelConfig


def torch_to_mlx(t: torch.Tensor, dtype=None) -> mx.array:
    """Convert PyTorch tensor to MLX array."""
    if t.dtype == torch.bfloat16:
        np_arr = t.float().numpy()
    elif t.dtype == torch.bool:
        np_arr = t.numpy()
    else:
        np_arr = t.detach().cpu().numpy()
    arr = mx.array(np_arr)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def mlx_to_torch(a: mx.array, dtype=torch.float32) -> torch.Tensor:
    """Convert MLX array to PyTorch tensor."""
    return torch.from_numpy(np.array(a)).to(dtype)


class HybridTadaInference:
    """Hybrid TADA inference: PyTorch encoder + MLX LLM loop + MLX or PyTorch decoder."""

    def __init__(
        self,
        mlx_weights_dir: str = "./mlx-weights",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        quantize_llm: bool = False,
        use_mlx_decoder: bool = True,
    ):
        self.device = device
        self.dtype = dtype
        self.use_mlx_decoder = use_mlx_decoder

        # Load MLX model
        print("Loading MLX model...")
        t0 = time.time()
        config_path = os.path.join(mlx_weights_dir, "config.json")
        with open(config_path) as f:
            config_dict = json.load(f)
        self.mlx_config = TadaModelConfig(**config_dict)
        self.mlx_model = TadaMLX(self.mlx_config)

        weights_path = os.path.join(mlx_weights_dir, "weights.safetensors")
        weights = mx.load(weights_path)
        self.mlx_model.load_weights(list(weights.items()))

        if quantize_llm:
            print("Quantizing LLM layers to 4-bit...")
            import mlx.nn as nn_mlx
            for layer in self.mlx_model.model.layers:
                nn_mlx.quantize(layer, bits=4, group_size=64)

        mx.eval(self.mlx_model.parameters())
        print(f"MLX model loaded in {time.time() - t0:.1f}s")

        # Load MLX decoder if requested
        self.mlx_decoder = None
        if use_mlx_decoder:
            decoder_weights_path = os.path.join(mlx_weights_dir, "decoder_weights.safetensors")
            if os.path.exists(decoder_weights_path):
                print("Loading MLX decoder...")
                t0 = time.time()
                from .decoder import DecoderMLX
                self.mlx_decoder = DecoderMLX()
                dec_weights = mx.load(decoder_weights_path)
                self.mlx_decoder.load_weights(list(dec_weights.items()))
                mx.eval(self.mlx_decoder.parameters())
                print(f"MLX decoder loaded in {time.time() - t0:.1f}s")
            else:
                print(f"MLX decoder weights not found at {decoder_weights_path}, falling back to PyTorch")
                self.use_mlx_decoder = False

        # Load PyTorch encoder (for reference audio)
        print("Loading PyTorch encoder...")
        t0 = time.time()
        from tada.modules.encoder import Encoder
        self.encoder = Encoder.from_pretrained("HumeAI/tada-codec", subfolder="encoder").to(device)
        self.encoder = self.encoder.to(dtype)
        self.encoder.eval()
        print(f"Encoder loaded in {time.time() - t0:.1f}s")

        # Load PyTorch decoder as fallback
        self.decoder = None
        if not self.use_mlx_decoder or self.mlx_decoder is None:
            print("Loading PyTorch decoder...")
            t0 = time.time()
            from tada.modules.decoder import Decoder
            self.decoder = Decoder.from_pretrained("HumeAI/tada-codec", subfolder="decoder").to(device)
            self.decoder = self.decoder.to(dtype)
            self.decoder.eval()
            print(f"Decoder loaded in {time.time() - t0:.1f}s")

        # We need the tokenizer — get it from the encoder's aligner
        self.tokenizer = self.encoder.tokenizer

        # Cache tokenizer special token IDs
        self.tokenizer_info = {
            "start_header_id": self.tokenizer.convert_tokens_to_ids("<|start_header_id|>"),
            "end_header_id": self.tokenizer.convert_tokens_to_ids("<|end_header_id|>"),
            "eot_id": self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
            "pad_id": self.tokenizer.convert_tokens_to_ids("<|finetune_right_pad_id|>"),
            "bos_id": self.tokenizer.bos_token_id,
            "eos_id": self.tokenizer.eos_token_id,
        }

    def encode_reference(self, audio_path: str, text: str) -> dict:
        """Encode reference audio into prompt. Returns dict of PyTorch tensors."""
        audio, sample_rate = torchaudio.load(audio_path)
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        audio = audio.to(device=self.device, dtype=self.dtype)

        prompt = self.encoder(
            audio,
            text=[text],
            audio_length=torch.tensor([audio.shape[1]], device=self.device),
            sample_rate=sample_rate,
        )
        return prompt

    @torch.no_grad()
    def generate(
        self,
        prompt,
        text: str,
        num_transition_steps: int = 5,
        system_prompt: str = None,
        config: GenerateConfig = None,
    ):
        """Generate speech using hybrid MLX/PyTorch pipeline.

        Returns:
            (audio_tensor, generation_time_seconds)
        """
        if config is None:
            config = GenerateConfig()

        from tada.utils.text import normalize_text as normalize_text_fn
        text = normalize_text_fn(text)

        # Build input_ids the same way TADA does
        prompt_text = prompt.text[0]
        text_tokens = [
            self.tokenizer.encode(prompt_text, add_special_tokens=False)
            + self.tokenizer.encode(text, add_special_tokens=False)
        ]
        input_ids = torch.tensor(text_tokens, device=self.device)
        input_lengths = torch.tensor([len(text_tokens[0])], device=self.device)

        # Add BOS/EOS (same as TADA's _add_bos_eos)
        shift_acoustic = self.mlx_config.shift_acoustic
        eos_id = self.tokenizer_info["eot_id"]
        bos_id = self.tokenizer_info["bos_id"]
        input_ids = torch.nn.functional.pad(input_ids, (0, shift_acoustic), value=eos_id)
        input_ids = torch.where(input_ids == -1, eos_id, input_ids)
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), value=bos_id)
        input_lengths = input_lengths + shift_acoustic + 1

        # Build time gaps (same as TADA's generate())
        token_positions = prompt.token_positions
        audio_feat_len = (prompt.audio_len / prompt.sample_rate * 50).ceil().long()

        selected_positions_with_ending = torch.where(
            torch.arange(token_positions.shape[1], device=self.device).expand(token_positions.shape[0], -1)
            == input_lengths.reshape(-1, 1) - shift_acoustic - 1,
            audio_feat_len.unsqueeze(-1),
            token_positions,
        )
        time_gaps = (
            selected_positions_with_ending
            - torch.nn.functional.pad(selected_positions_with_ending, [1, 0], value=1)[:, :-1]
        ).clamp(min=0, max=self.mlx_config.num_time_classes - 1)
        time_gaps = torch.nn.functional.pad(time_gaps, [1, 0], value=0)
        time_len_before = time_gaps[:, :-1]
        time_len_after = time_gaps[:, 1:]

        prompt_acoustic_features = prompt.token_values
        prompt_acoustic_masks = torch.ones(
            prompt_acoustic_features.shape[:2], device=self.device, dtype=torch.long
        )

        # Add system prompt prefix
        prefix_text = (
            f"<|start_header_id|>system<|end_header_id|>{system_prompt or ''}<|eot_id|>"
            + "<|start_header_id|>assistant<|end_header_id|>"
        )
        prefix_text_tokens = self.tokenizer.encode(prefix_text, add_special_tokens=False, return_tensors="pt").to(
            self.device
        )
        prefix_len = prefix_text_tokens.shape[1]
        input_ids = torch.cat([input_ids[:, :1], prefix_text_tokens, input_ids[:, 1:]], dim=1)
        input_lengths = input_lengths + prefix_len
        prompt_acoustic_features = torch.nn.functional.pad(prompt_acoustic_features, (0, 0, prefix_len, 0))
        prompt_acoustic_masks = torch.nn.functional.pad(prompt_acoustic_masks, (prefix_len, 0))
        time_len_before = torch.nn.functional.pad(time_len_before, (prefix_len, 0))
        time_len_after = torch.nn.functional.pad(time_len_after, (prefix_len, 0))

        if num_transition_steps > 0:
            prompt_acoustic_features = prompt_acoustic_features[:, :-num_transition_steps, :]
            prompt_acoustic_masks = prompt_acoustic_masks[:, :-num_transition_steps]
            time_len_before = time_len_before[:, :-num_transition_steps]
            time_len_after = time_len_after[:, :-num_transition_steps]

        # Shift acoustic masks: same as TADA
        prompt_acoustic_masks_shifted = torch.cat(
            [prompt_acoustic_masks[:, 1:], torch.ones_like(prompt_acoustic_masks[:, :1])], -1
        )

        num_gen_steps = input_ids.shape[-1]

        # Convert tensors to MLX
        mlx_input_ids = torch_to_mlx(input_ids, dtype=mx.int32)
        mlx_prompt_acoustic = torch_to_mlx(prompt_acoustic_features.float())
        mlx_prompt_masks = torch_to_mlx(prompt_acoustic_masks_shifted, dtype=mx.int32)
        mlx_time_before = torch_to_mlx(time_len_before, dtype=mx.int32)
        mlx_time_after = torch_to_mlx(time_len_after, dtype=mx.int32)

        # Run MLX generation loop
        print(f"Running MLX generation ({num_gen_steps} steps)...")
        t0 = time.time()
        output = generate(
            model=self.mlx_model,
            input_ids=mlx_input_ids,
            prompt_acoustic_features=mlx_prompt_acoustic,
            prompt_acoustic_masks=mlx_prompt_masks,
            prompt_time_len_before=mlx_time_before,
            prompt_time_len_after=mlx_time_after,
            config=config,
            tokenizer_info=self.tokenizer_info,
            num_steps=num_gen_steps,
        )
        gen_time = time.time() - t0

        # Get acoustic features and time_before from generation output
        num_prompt_tokens = prompt_acoustic_features.shape[1]
        start_idx = num_prompt_tokens + num_transition_steps - 1

        if self.mlx_decoder is not None:
            # MLX decoder path — stay in MLX arrays
            wav = self._decode_mlx(output, start_idx)
        else:
            # PyTorch decoder fallback
            wav = self._decode_pytorch(output, start_idx)

        if wav is not None:
            # Remove leading silence
            time_before_0 = int(np.array(output.time_before[0, start_idx]).item())
            leading_silence_samples = int(24000 * time_before_0 / 50)
            wav = wav[..., leading_silence_samples:]

        return wav, gen_time

    def _decode_mlx(self, output, start_idx):
        """Decode using MLX decoder (fully on Metal GPU)."""
        print("Decoding audio (MLX)...")
        t_dec = time.time()

        acoustic_features = output.acoustic_features[:, start_idx:]
        time_before = output.time_before[:, start_idx:]

        # Denormalize
        acoustic_features = acoustic_features * self.mlx_config.acoustic_std + self.mlx_config.acoustic_mean

        if acoustic_features.shape[1] == 0:
            return None

        # Expand features using time_before (same logic as PyTorch _decode_wav)
        encoded = acoustic_features[0]  # (T, 512)
        tb = time_before[0]  # (T,)

        parts = []
        for pos in range(encoded.shape[0]):
            n_frames = max(0, int(tb[pos].item()) - 1)
            if n_frames > 0:
                parts.append(mx.zeros((n_frames, encoded.shape[-1])))
            parts.append(encoded[pos:pos+1])

        # Trailing frames
        if tb.shape[0] > encoded.shape[0]:
            n_trailing = int(tb[-1].item())
            if n_trailing > 0:
                parts.append(mx.zeros((n_trailing, encoded.shape[-1])))

        if not parts:
            return None

        encoded_expanded = mx.concatenate(parts, axis=0)[None]  # (1, T_expanded, 512)
        token_masks = (mx.sqrt((encoded_expanded * encoded_expanded).sum(axis=-1)) != 0).astype(mx.int32)

        wav = self.mlx_decoder.generate(encoded_expanded, token_masks)
        mx.eval(wav)

        dec_time = time.time() - t_dec
        print(f"  Decode: {dec_time:.1f}s")

        # Convert to PyTorch tensor for compatibility
        wav_np = np.array(wav.squeeze())
        return torch.from_numpy(wav_np)

    def _decode_pytorch(self, output, start_idx):
        """Decode using PyTorch decoder (CPU fallback)."""
        print("Decoding audio (PyTorch)...")
        t_dec = time.time()

        acoustic_features_pt = mlx_to_torch(output.acoustic_features, dtype=self.dtype)
        acoustic_features_pt = acoustic_features_pt * self.mlx_config.acoustic_std + self.mlx_config.acoustic_mean
        time_before_pt = mlx_to_torch(output.time_before, dtype=torch.long)

        encoded = acoustic_features_pt[:, start_idx:]
        time_before = time_before_pt[:, start_idx:]

        if encoded.shape[1] == 0:
            return None

        wav = self._decode_wav_pt(encoded[0], time_before[0])

        dec_time = time.time() - t_dec
        print(f"  Decode: {dec_time:.1f}s")
        return wav

    def _decode_wav_pt(self, encoded, time_before):
        """Decode acoustic features to audio waveform using PyTorch decoder."""
        encoded = encoded.to(self.device)
        time_before = time_before.to(self.device)

        if time_before.shape[0] == 0:
            return None

        time_before = time_before[:encoded.shape[0] + 1]

        encoded_expanded = []
        for pos in range(encoded.shape[0]):
            n_frames = int((time_before[pos] - 1).clamp(min=0).item())
            encoded_expanded.append(
                torch.zeros(n_frames, encoded.shape[-1], device=self.device, dtype=self.dtype)
            )
            encoded_expanded.append(encoded[pos].unsqueeze(0))

        n_trailing = int(time_before[-1].item()) if time_before.shape[0] > encoded.shape[0] else 0
        encoded_expanded.append(
            torch.zeros(n_trailing, encoded.shape[-1], device=self.device, dtype=self.dtype)
        )

        if not encoded_expanded:
            return None

        encoded_expanded = torch.cat(encoded_expanded, dim=0).unsqueeze(0)
        token_masks = (torch.norm(encoded_expanded, dim=-1) != 0).long()

        with torch.no_grad():
            wav = self.decoder.generate(encoded_expanded, token_masks=token_masks)

        return wav.squeeze(0, 1)

    def warmup(self, prompt, config: GenerateConfig = None):
        """Run a short warmup generation to prime caches."""
        print("Warmup...")
        _, _ = self.generate(prompt, "Hi.", num_transition_steps=0, config=config)
        print("Warmup complete")

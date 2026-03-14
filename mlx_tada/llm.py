"""MLX implementation of Llama 3.2 1B backbone with TADA extensions."""

import math
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn


@dataclass
class TadaModelConfig:
    """Configuration matching the TADA-1B model."""
    hidden_size: int = 2048
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    intermediate_size: int = 8192
    vocab_size: int = 128256
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    head_dim: int = 64
    max_position_embeddings: int = 131072
    # RoPE scaling (Llama3 style)
    rope_scaling_factor: float = 32.0
    rope_scaling_high_freq_factor: float = 4.0
    rope_scaling_low_freq_factor: float = 1.0
    rope_scaling_original_max_position_embeddings: int = 8192
    # TADA-specific
    acoustic_dim: int = 512
    num_time_classes: int = 256
    shift_acoustic: int = 5
    head_layers: int = 6
    head_ffn_ratio: float = 4.0
    tie_word_embeddings: bool = True
    acoustic_mean: float = 0.0
    acoustic_std: float = 1.5


class Llama3RoPE(nn.Module):
    """Llama 3 RoPE with frequency scaling."""

    def __init__(self, config: TadaModelConfig):
        super().__init__()
        self.dims = config.head_dim
        self.traditional = False

        # Compute Llama3-adjusted frequencies
        factor = config.rope_scaling_factor
        low_freq_factor = config.rope_scaling_low_freq_factor
        high_freq_factor = config.rope_scaling_high_freq_factor
        old_context_len = config.rope_scaling_original_max_position_embeddings

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = config.rope_theta ** (
            mx.arange(0, config.head_dim, 2, dtype=mx.float32) / config.head_dim
        )

        # Apply Llama3 scaling
        new_freqs = []
        for i in range(len(freqs)):
            freq = 1.0 / freqs[i]  # Convert to actual frequency
            wavelen = 2 * math.pi / freq
            if wavelen < high_freq_wavelen:
                new_freqs.append(freq)
            elif wavelen > low_freq_wavelen:
                new_freqs.append(freq / factor)
            else:
                smooth = (old_context_len / wavelen - low_freq_factor) / (
                    high_freq_factor - low_freq_factor
                )
                new_freqs.append((1 - smooth) * freq / factor + smooth * freq)

        # mx.fast.rope expects theta values (base per dim), not frequencies
        # angle = position / theta, so theta = 1 / freq
        self._freqs = mx.array([1.0 / f for f in new_freqs], dtype=mx.float32)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class KVCache:
    """Simple KV cache for autoregressive generation."""

    def __init__(self):
        self.keys = None
        self.values = None

    @property
    def offset(self):
        """Current cache length (= RoPE offset for next token)."""
        if self.keys is None:
            return 0
        return self.keys.shape[2]

    def update_and_fetch(self, keys: mx.array, values: mx.array):
        """Append new keys/values and return full cache."""
        if self.keys is None:
            self.keys = keys
            self.values = values
        else:
            self.keys = mx.concatenate([self.keys, keys], axis=2)
            self.values = mx.concatenate([self.values, values], axis=2)
        return self.keys, self.values


class Attention(nn.Module):
    """Multi-head attention with GQA."""

    def __init__(self, config: TadaModelConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)

        self.rope = Llama3RoPE(config)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache=None,
    ):
        B, L, _ = x.shape

        queries = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        if cache is not None:
            offset = cache.offset
            queries = self.rope(queries, offset=offset)
            keys = self.rope(keys, offset=offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, config: TadaModelConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    """Llama transformer block."""

    def __init__(self, config: TadaModelConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.mlp = MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(self, x: mx.array, mask=None, cache=None):
        r = self.self_attn(self.input_layernorm(x), mask=mask, cache=cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class LlamaModel(nn.Module):
    """The Llama backbone (without lm_head)."""

    def __init__(self, config: TadaModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = [TransformerBlock(config) for _ in range(config.num_hidden_layers)]
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        inputs_embeds: mx.array,
        mask=None,
        cache=None,
    ):
        h = inputs_embeds
        B, L, _ = h.shape

        if cache is None:
            cache = [None] * len(self.layers)

        # Create causal mask for multi-token input (prefill)
        if mask is None and L > 1:
            # Additive causal mask: 0 for allowed, -inf for masked
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)
            # If there's existing cache, we need to prepend zeros for cached positions
            if cache[0] is not None and cache[0].keys is not None:
                cached_len = cache[0].keys.shape[2]
                prefix = mx.zeros((L, cached_len))
                mask = mx.concatenate([prefix, mask], axis=1)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class TadaMLX(nn.Module):
    """TADA model in MLX: Llama backbone + custom embeddings + diffusion head.

    This implements forward_one_step and the embedding logic.
    The generation loop is in generate.py.
    """

    def __init__(self, config: TadaModelConfig):
        super().__init__()
        self.config = config
        self.num_time_bits = math.ceil(math.log2(config.num_time_classes))
        self.time_dim = 2 * self.num_time_bits

        # Llama backbone
        self.model = LlamaModel(config)

        # lm_head is tied with embed_tokens
        # We'll use model.embed_tokens.weight directly for the lm_head projection

        # TADA custom embeddings
        self.acoustic_proj = nn.Linear(config.acoustic_dim, config.hidden_size)
        self.time_start_embed = nn.Embedding(config.num_time_classes, config.hidden_size)
        self.time_end_embed = nn.Embedding(config.num_time_classes, config.hidden_size)
        self.acoustic_mask_emb = nn.Embedding(2, config.hidden_size)

        # Diffusion head
        from .diffusion import VibeVoiceDiffusionHead
        self.prediction_head = VibeVoiceDiffusionHead(
            hidden_size=config.hidden_size,
            head_layers=config.head_layers,
            head_ffn_ratio=config.head_ffn_ratio,
            rms_norm_eps=config.rms_norm_eps,
            latent_size=config.acoustic_dim + self.time_dim,
        )

    def lm_head(self, hidden_states: mx.array) -> mx.array:
        """Project hidden states to vocab logits using tied weights."""
        return hidden_states @ self.model.embed_tokens.weight.T

    def build_inputs_embeds(
        self,
        input_ids: mx.array,
        acoustic_features: mx.array,
        acoustic_masks: mx.array,
        time_len_before: mx.array,
        time_len_after: mx.array,
    ) -> mx.array:
        """Build input embeddings for a single step."""
        return (
            self.model.embed_tokens(input_ids)
            + self.acoustic_proj(acoustic_features)
            + self.acoustic_mask_emb(acoustic_masks)
            + self.time_start_embed(time_len_before)
            + self.time_end_embed(time_len_after)
        )

    def forward_one_step(
        self,
        input_ids: mx.array,
        acoustic_features: mx.array,
        acoustic_masks: mx.array,
        time_len_before: mx.array,
        time_len_after: mx.array,
        cache=None,
        compute_logits: bool = True,
    ):
        """Run a single LLM step. Returns (hidden_states, logits, cache)."""
        inputs_embeds = self.build_inputs_embeds(
            input_ids, acoustic_features, acoustic_masks,
            time_len_before, time_len_after,
        )

        hidden_states = self.model(inputs_embeds, cache=cache)
        logits = self.lm_head(hidden_states) if compute_logits else None

        return hidden_states, logits

    def prefill(
        self,
        inputs_embeds: mx.array,
        cache=None,
    ):
        """Run prefill with pre-built embeddings. Returns hidden_states."""
        return self.model(inputs_embeds, cache=cache)

    def build_prompt_inputs_embeds(
        self,
        input_ids: mx.array,
        prompt_acoustic_features: mx.array,
        prompt_acoustic_masks: mx.array,
        prompt_time_len_before: mx.array,
        prompt_time_len_after: mx.array,
        prefill_len: int,
    ) -> mx.array:
        """Build inputs_embeds for prefill positions 0..prefill_len-1."""
        batch_size = input_ids.shape[0]
        shift = self.config.shift_acoustic

        token_emb = self.model.embed_tokens(input_ids[:, :prefill_len])

        # Acoustic: position t>shift uses prompt[:, t-shift-1]
        acoustic_full = mx.zeros((batch_size, prefill_len, self.config.acoustic_dim))
        if prompt_acoustic_features is not None and prompt_acoustic_masks is not None:
            n_ac = min(prefill_len - shift - 1, prompt_acoustic_features.shape[1])
            if n_ac > 0:
                acoustic_full = acoustic_full.at[:, shift + 1: shift + 1 + n_ac].add(
                    prompt_acoustic_features[:, :n_ac]
                )
        masks_full = mx.zeros((batch_size, prefill_len), dtype=mx.int32)
        if prompt_acoustic_masks is not None:
            n_ac = min(prefill_len - shift - 1, prompt_acoustic_masks.shape[1])
            if n_ac > 0:
                masks_full = masks_full.at[:, shift + 1: shift + 1 + n_ac].add(
                    prompt_acoustic_masks[:, :n_ac]
                )
        acoustic_emb = self.acoustic_proj(acoustic_full) + self.acoustic_mask_emb(masks_full)

        # Time embeddings
        time_before = mx.zeros((batch_size, prefill_len), dtype=mx.int32)
        time_after = mx.zeros((batch_size, prefill_len), dtype=mx.int32)
        if prompt_time_len_before is not None and prompt_time_len_after is not None:
            n_t = min(prefill_len - shift - 1, prompt_time_len_before.shape[1] - 1)
            if n_t > 0:
                time_before = time_before.at[:, shift + 1: shift + 1 + n_t].add(
                    prompt_time_len_before[:, 1: 1 + n_t]
                )
                time_after = time_after.at[:, shift + 1: shift + 1 + n_t].add(
                    prompt_time_len_after[:, 1: 1 + n_t]
                )
        time_emb = self.time_start_embed(time_before) + self.time_end_embed(time_after)

        return token_emb + acoustic_emb + time_emb

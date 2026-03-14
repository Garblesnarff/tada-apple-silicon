"""MLX implementation of the TADA decoder (LocalAttentionEncoder + DACDecoder).

Ported from:
  tada/modules/decoder.py (Decoder, DACDecoder, DecoderBlock)
  tada/modules/encoder.py (LocalAttentionEncoder, LocalSelfAttention, ResidualUnit)
  dac/nn/layers.py (Snake1d)
"""

import math

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Snake1d activation
# ---------------------------------------------------------------------------

class Snake1d(nn.Module):
    """Snake activation: x + (1/(alpha+eps)) * sin(alpha*x)^2.

    Alpha is learnable per-channel. Input layout: (B, L, C).
    """

    def __init__(self, channels: int):
        super().__init__()
        self.alpha = mx.ones((channels,))

    def __call__(self, x: mx.array) -> mx.array:
        # x: (B, L, C), alpha: (C,) — broadcasts along B and L
        a = self.alpha
        return x + (1.0 / (a + 1e-9)) * mx.power(mx.sin(a * x), 2)


# ---------------------------------------------------------------------------
# Conv layers (no weight norm — materialized at conversion time)
# ---------------------------------------------------------------------------

class ResidualUnit(nn.Module):
    """Snake1d -> Conv1d(dilated) -> Snake1d -> Conv1d(k=1) + skip."""

    def __init__(self, dim: int, dilation: int = 1):
        super().__init__()
        pad = ((7 - 1) * dilation) // 2
        self.snake1 = Snake1d(dim)
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=7, dilation=dilation, padding=pad)
        self.snake2 = Snake1d(dim)
        self.conv2 = nn.Conv1d(dim, dim, kernel_size=1)

    def __call__(self, x: mx.array) -> mx.array:
        y = self.snake1(x)
        y = self.conv1(y)
        y = self.snake2(y)
        y = self.conv2(y)
        # Handle potential length mismatch from dilated conv
        if x.shape[1] != y.shape[1]:
            pad = (x.shape[1] - y.shape[1]) // 2
            x = x[:, pad:-pad, :]
        return x + y


class DecoderBlock(nn.Module):
    """Snake1d -> ConvTranspose1d(upsample) -> 3x ResidualUnit."""

    def __init__(self, input_dim: int, output_dim: int, stride: int):
        super().__init__()
        self.snake = Snake1d(input_dim)
        self.conv_transpose = nn.ConvTranspose1d(
            input_dim, output_dim,
            kernel_size=2 * stride,
            stride=stride,
            padding=math.ceil(stride / 2),
        )
        self.residuals = [
            ResidualUnit(output_dim, dilation=1),
            ResidualUnit(output_dim, dilation=3),
            ResidualUnit(output_dim, dilation=9),
        ]

    def __call__(self, x: mx.array) -> mx.array:
        x = self.snake(x)
        x = self.conv_transpose(x)
        for res in self.residuals:
            x = res(x)
        return x


class DACDecoder(nn.Module):
    """DAC-style waveform decoder with Snake activations.

    Input: (B, L, input_channel)  [NLC]
    Output: (B, L_out, 1)         [NLC]

    Upsampling factor = product of strides (4*4*5*6 = 480).
    """

    def __init__(
        self,
        input_channel: int = 1024,
        channels: int = 1536,
        strides: list[int] = None,
    ):
        super().__init__()
        if strides is None:
            strides = [4, 4, 5, 6]

        self.first_conv = nn.Conv1d(input_channel, channels, kernel_size=7, padding=3)

        self.blocks = []
        for i, stride in enumerate(strides):
            in_dim = channels // 2**i
            out_dim = channels // 2**(i + 1)
            self.blocks.append(DecoderBlock(in_dim, out_dim, stride))

        final_dim = channels // 2**len(strides)  # 96
        self.final_snake = Snake1d(final_dim)
        self.final_conv = nn.Conv1d(final_dim, 1, kernel_size=7, padding=3)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.first_conv(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_snake(x)
        x = self.final_conv(x)
        x = mx.tanh(x)
        return x


# ---------------------------------------------------------------------------
# Local attention encoder
# ---------------------------------------------------------------------------

class LocalSelfAttention(nn.Module):
    """Multi-head self-attention with RoPE (standard theta=10000)."""

    def __init__(self, d_model: int, num_heads: int = 8, max_seq_len: int = 8192):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        # Precompute RoPE frequencies
        inv_freq = 1.0 / (10000.0 ** (mx.arange(0, self.head_dim, 2, dtype=mx.float32) / self.head_dim))
        positions = mx.arange(max_seq_len, dtype=mx.float32)
        freqs = positions[:, None] * inv_freq[None, :]  # (max_seq_len, head_dim//2)
        self._rope_cos = mx.cos(freqs)  # (max_seq_len, head_dim//2)
        self._rope_sin = mx.sin(freqs)

    def _apply_rope(self, x: mx.array) -> mx.array:
        """Apply RoPE. x: (B, num_heads, L, head_dim)."""
        B, H, L, D = x.shape
        cos = self._rope_cos[:L]  # (L, D//2)
        sin = self._rope_sin[:L]

        # Split into pairs
        x0 = x[..., 0::2]  # (B, H, L, D//2)
        x1 = x[..., 1::2]

        # Rotate
        rx0 = x0 * cos - x1 * sin
        rx1 = x0 * sin + x1 * cos

        # Interleave back
        # Stack along last dim then reshape
        out = mx.stack([rx0, rx1], axis=-1)  # (B, H, L, D//2, 2)
        return out.reshape(B, H, L, D)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        """Forward pass. x: (B, L, D). mask: (B, L, L) bool where True=masked."""
        B, L, D = x.shape

        qkv = self.qkv(x)  # (B, L, 3*D)
        qkv = qkv.reshape(B, L, 3, self.num_heads, self.head_dim)
        qkv = qkv.transpose(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = self._apply_rope(q)
        k = self._apply_rope(k)

        # Attention
        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale  # (B, H, L, L)

        if mask is not None:
            # mask: (B, L, L) bool, True = blocked
            # Convert to additive mask
            attn_mask = mx.where(mask[:, None, :, :], -1e9, 0.0)
            attn = attn + attn_mask

        attn = mx.softmax(attn, axis=-1)
        out = attn @ v  # (B, H, L, D)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, D)
        out = self.out_proj(out)

        # Residual + layer norm
        return self.layer_norm(x + out)


class LocalAttentionEncoderLayer(nn.Module):
    """Transformer layer: LocalSelfAttention + GELU FFN."""

    def __init__(self, d_model: int, num_heads: int = 8, d_ff: int = 4096,
                 max_seq_len: int = 8192):
        super().__init__()
        self.self_attn = LocalSelfAttention(d_model, num_heads, max_seq_len)

        self.ffn_linear1 = nn.Linear(d_model, d_ff)
        self.ffn_linear2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        x = self.self_attn(x, mask=mask)
        ffn_out = self.ffn_linear2(nn.gelu(self.ffn_linear1(x)))
        return self.norm(x + ffn_out)


class LocalAttentionEncoder(nn.Module):
    """Stack of local attention encoder layers."""

    def __init__(self, d_model: int = 1024, num_layers: int = 6,
                 num_heads: int = 8, d_ff: int = 4096, max_seq_len: int = 8192):
        super().__init__()
        self.layers = [
            LocalAttentionEncoderLayer(d_model, num_heads, d_ff, max_seq_len)
            for _ in range(num_layers)
        ]
        self.final_norm = nn.LayerNorm(d_model)

    def __call__(self, x: mx.array, mask: mx.array = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.final_norm(x)


# ---------------------------------------------------------------------------
# Segment attention mask (decoder v2)
# ---------------------------------------------------------------------------

def create_segment_attention_mask_v2(token_masks: mx.array) -> mx.array:
    """Build v2 block attention mask.

    token_masks: (B, L) int, 1 at block boundaries.
    Returns: (B, L, L) bool, True = masked (cannot attend).

    Rule: position i can attend to position j if
      block_ids[j] == block_ids[i]  (same block) OR
      block_ids[j] == block_ids[i] - 1  (previous block)
    where block_ids = cumsum(token_masks) - token_masks.
    """
    block_ids = mx.cumsum(token_masks, axis=1) - token_masks  # (B, L)
    bi = block_ids[:, :, None]  # (B, L, 1)
    bj = block_ids[:, None, :]  # (B, 1, L)
    same = bi == bj
    prev = bj == (bi - 1)
    can_attend = same | prev
    return ~can_attend  # True = masked


# ---------------------------------------------------------------------------
# Full decoder
# ---------------------------------------------------------------------------

class DecoderMLX(nn.Module):
    """Full TADA decoder: projection + LocalAttentionEncoder + DACDecoder.

    Input: encoded_expanded (B, T, 512), token_masks (B, T)
    Output: waveform (B, T*480, 1)
    """

    def __init__(
        self,
        embed_dim: int = 512,
        hidden_dim: int = 1024,
        num_attn_layers: int = 6,
        num_attn_heads: int = 8,
        attn_dim_feedforward: int = 4096,
        wav_decoder_channels: int = 1536,
        strides: list[int] = None,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        if strides is None:
            strides = [4, 4, 5, 6]

        self.decoder_proj = nn.Linear(embed_dim, hidden_dim)
        self.local_attention_decoder = LocalAttentionEncoder(
            d_model=hidden_dim,
            num_layers=num_attn_layers,
            num_heads=num_attn_heads,
            d_ff=attn_dim_feedforward,
            max_seq_len=max_seq_len,
        )
        self.wav_decoder = DACDecoder(
            input_channel=hidden_dim,
            channels=wav_decoder_channels,
            strides=strides,
        )

    def __call__(self, encoded_expanded: mx.array, token_masks: mx.array) -> mx.array:
        """Forward pass.

        Args:
            encoded_expanded: (B, T, 512) acoustic features
            token_masks: (B, T) int mask (1 = active token, 0 = padding)

        Returns:
            waveform: (B, T*480, 1)
        """
        x = self.decoder_proj(encoded_expanded)
        attn_mask = create_segment_attention_mask_v2(token_masks)
        x = self.local_attention_decoder(x, mask=attn_mask)
        # DACDecoder expects NLC — already in NLC
        wav = self.wav_decoder(x)
        return wav

    def generate(self, encoded_expanded: mx.array, token_masks: mx.array) -> mx.array:
        """Alias for __call__ matching PyTorch Decoder.generate()."""
        return self(encoded_expanded, token_masks)

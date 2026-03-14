"""Gray code encoding/decoding and utility functions for MLX TADA."""

import mlx.core as mx


def int_to_gray_code(values: mx.array) -> mx.array:
    """Convert integer values to their Gray code equivalents."""
    return values ^ (values >> 1)


def gray_code_to_int(gray: mx.array) -> mx.array:
    """Convert Gray code integers back to binary integers."""
    binary = gray
    shift = 1
    while shift < 32:
        binary = binary ^ (binary >> shift)
        shift <<= 1
    return binary


def encode_time_with_gray_code(num_frames: mx.array, num_bits: int) -> mx.array:
    """Convert time values to Gray code bit representation as floats in {-1, 1}."""
    num_time_classes = 2 ** num_bits
    num_frames = mx.clip(num_frames, 0, num_time_classes - 1)
    gray_code = int_to_gray_code(num_frames)

    # Extract bits
    gray_bits = mx.zeros((*num_frames.shape, num_bits), dtype=mx.int32)
    for i in range(num_bits):
        bit_val = (gray_code >> i) & 1
        gray_bits = gray_bits.at[..., num_bits - 1 - i].add(bit_val)

    return gray_bits.astype(mx.float32) * 2.0 - 1.0


def decode_gray_code_to_time(gray_bits: mx.array, num_bits: int) -> mx.array:
    """Convert Gray code bit representation back to time values."""
    # Convert from {-1, 1} to {0, 1}
    gray_bits_binary = mx.round((gray_bits + 1.0) / 2.0).astype(mx.int32)

    # Convert bit representation to Gray code integer
    gray_code = mx.zeros(gray_bits_binary.shape[:-1], dtype=mx.int32)
    for i in range(num_bits):
        gray_code = gray_code + (gray_bits_binary[..., num_bits - 1 - i] << i)

    # Convert Gray code integer to regular integer
    return gray_code_to_int(gray_code)

"""Convert PyTorch TADA-1B weights to MLX format.

Usage:
    python -m mlx_tada.convert_weights [--output-dir ./mlx-weights]

Reads from HuggingFace cache, writes MLX-compatible safetensors.
"""

import glob
import os
import re

import mlx.core as mx


def convert_torch_to_mlx(tensor_pt, dtype=mx.bfloat16):
    """Convert a PyTorch tensor to MLX array with specified dtype."""
    import torch
    if tensor_pt.dtype == torch.bfloat16:
        tensor_np = tensor_pt.float().numpy()
    elif tensor_pt.dtype == torch.bool:
        tensor_np = tensor_pt.numpy()
        return mx.array(tensor_np)
    else:
        tensor_np = tensor_pt.numpy()
    return mx.array(tensor_np).astype(dtype)


def load_pytorch_weights():
    """Load weights from safetensors using PyTorch framework."""
    from safetensors import safe_open

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--HumeAI--tada-1b/snapshots/*/model.safetensors"
    )
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(f"No safetensors found matching {pattern}")

    weights = {}
    with safe_open(paths[0], framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def map_llama_weights(pt_weights: dict) -> dict:
    """Map PyTorch Llama weights to MLX naming convention."""
    mlx_weights = {}

    for key, val in pt_weights.items():
        if key.startswith("_decoder."):
            continue  # Decoder handled separately

        mlx_key = key

        # prediction_head: map Sequential-based adaLN_modulation to our naming
        if "adaLN_modulation.1.weight" in key:
            mlx_key = key.replace("adaLN_modulation.1.weight", "adaLN_linear.weight")

        # prediction_head timestep embedder: mlp.0 → linear1, mlp.2 → linear2
        if "t_embedder.mlp.0.weight" in key:
            mlx_key = key.replace("t_embedder.mlp.0.weight", "t_embedder.linear1.weight")
        elif "t_embedder.mlp.2.weight" in key:
            mlx_key = key.replace("t_embedder.mlp.2.weight", "t_embedder.linear2.weight")

        mlx_weights[mlx_key] = convert_torch_to_mlx(val)

    return mlx_weights


# ---------------------------------------------------------------------------
# Decoder weight conversion
# ---------------------------------------------------------------------------

def _materialize_weight_norm(g, v):
    """Materialize weight norm: weight = g * v / ||v||.

    g: (out, 1, 1) or similar — magnitude scalar per output channel
    v: full weight tensor — direction

    PyTorch weight_norm normalizes over all dims except dim=0.
    """
    import torch
    # Compute norm over all dims except 0
    norm_dims = list(range(1, v.dim()))
    v_norm = torch.norm(v.float(), dim=norm_dims, keepdim=True)
    return (g.float() * v.float() / (v_norm + 1e-12))



def load_decoder_weights():
    """Load decoder weights from HumeAI/tada-codec (the correct source).

    The decoder weights in tada-1b model.safetensors (_decoder.*) differ from
    the separately stored codec decoder. Use the codec decoder weights.
    """
    from safetensors import safe_open

    pattern = os.path.expanduser(
        "~/.cache/huggingface/hub/models--HumeAI--tada-codec/snapshots/*/decoder/model.safetensors"
    )
    paths = glob.glob(pattern)
    if not paths:
        raise FileNotFoundError(
            f"No decoder safetensors found matching {pattern}. "
            "Run: python -c \"from tada.modules.decoder import Decoder; Decoder.from_pretrained('HumeAI/tada-codec', subfolder='decoder')\""
        )

    weights = {}
    with safe_open(paths[0], framework="pt") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key)
    return weights


def map_decoder_weights(pt_weights: dict) -> dict:
    """Convert decoder weights from PyTorch to MLX format.

    Handles:
    1. Weight norm materialization (original0/original1 → weight)
    2. Conv weight transposition (NCL → NLC)
    3. Snake1d alpha reshape ([1,C,1] → [C])
    4. Key remapping to match MLX module structure
    5. Skip buffers (_precomputed_mask, rope_freqs)
    """
    import torch

    # Step 1: Collect decoder keys, materialize weight norm
    decoder_raw = {}
    wn_groups = {}  # key_prefix -> {g: tensor, v: tensor}

    for key, val in pt_weights.items():
        k = key

        # Skip buffers
        if "._precomputed_mask" in k or ".rope_freqs" in k:
            continue

        # Collect weight norm pairs
        if ".parametrizations.weight.original0" in k:
            prefix = k.replace(".parametrizations.weight.original0", "")
            wn_groups.setdefault(prefix, {})["g"] = val
            continue
        if ".parametrizations.weight.original1" in k:
            prefix = k.replace(".parametrizations.weight.original1", "")
            wn_groups.setdefault(prefix, {})["v"] = val
            continue

        decoder_raw[k] = val

    # Materialize weight norm
    for prefix, gv in wn_groups.items():
        assert "g" in gv and "v" in gv, f"Incomplete weight norm for {prefix}"
        materialized = _materialize_weight_norm(gv["g"], gv["v"])
        decoder_raw[prefix + ".weight"] = materialized

    # Step 2: Remap keys to match MLX module naming
    mlx_decoder = {}
    for k, val in decoder_raw.items():
        mlx_key = k

        # FFN sequential indices → named modules
        # ffn.0 → ffn_linear1, ffn.3 → ffn_linear2
        mlx_key = re.sub(r"\.ffn\.0\.", ".ffn_linear1.", mlx_key)
        mlx_key = re.sub(r"\.ffn\.3\.", ".ffn_linear2.", mlx_key)

        # DACDecoder sequential model → named modules
        # model.0.* → first_conv.*
        mlx_key = re.sub(r"^wav_decoder\.model\.0\.", "wav_decoder.first_conv.", mlx_key)

        # model.{1-4}.block.0.alpha → blocks.{0-3}.snake.alpha
        m = re.match(r"^wav_decoder\.model\.(\d+)\.block\.0\.alpha$", mlx_key)
        if m:
            idx = int(m.group(1)) - 1
            mlx_key = f"wav_decoder.blocks.{idx}.snake.alpha"

        # model.{1-4}.block.1.* → blocks.{0-3}.conv_transpose.*
        m = re.match(r"^wav_decoder\.model\.(\d+)\.block\.1\.(.*)", mlx_key)
        if m:
            idx = int(m.group(1)) - 1
            rest = m.group(2)
            mlx_key = f"wav_decoder.blocks.{idx}.conv_transpose.{rest}"

        # model.{1-4}.block.{2-4}.block.0.alpha → blocks.{0-3}.residuals.{0-2}.snake1.alpha
        m = re.match(r"^wav_decoder\.model\.(\d+)\.block\.(\d+)\.block\.0\.alpha$", mlx_key)
        if m:
            block_idx = int(m.group(1)) - 1
            res_idx = int(m.group(2)) - 2
            mlx_key = f"wav_decoder.blocks.{block_idx}.residuals.{res_idx}.snake1.alpha"

        # model.{1-4}.block.{2-4}.block.1.* → blocks.{0-3}.residuals.{0-2}.conv1.*
        m = re.match(r"^wav_decoder\.model\.(\d+)\.block\.(\d+)\.block\.1\.(.*)", mlx_key)
        if m:
            block_idx = int(m.group(1)) - 1
            res_idx = int(m.group(2)) - 2
            rest = m.group(3)
            mlx_key = f"wav_decoder.blocks.{block_idx}.residuals.{res_idx}.conv1.{rest}"

        # model.{1-4}.block.{2-4}.block.2.alpha → blocks.{0-3}.residuals.{0-2}.snake2.alpha
        m = re.match(r"^wav_decoder\.model\.(\d+)\.block\.(\d+)\.block\.2\.alpha$", mlx_key)
        if m:
            block_idx = int(m.group(1)) - 1
            res_idx = int(m.group(2)) - 2
            mlx_key = f"wav_decoder.blocks.{block_idx}.residuals.{res_idx}.snake2.alpha"

        # model.{1-4}.block.{2-4}.block.3.* → blocks.{0-3}.residuals.{0-2}.conv2.*
        m = re.match(r"^wav_decoder\.model\.(\d+)\.block\.(\d+)\.block\.3\.(.*)", mlx_key)
        if m:
            block_idx = int(m.group(1)) - 1
            res_idx = int(m.group(2)) - 2
            rest = m.group(3)
            mlx_key = f"wav_decoder.blocks.{block_idx}.residuals.{res_idx}.conv2.{rest}"

        # model.5.alpha → final_snake.alpha
        mlx_key = re.sub(r"^wav_decoder\.model\.5\.alpha$", "wav_decoder.final_snake.alpha", mlx_key)

        # model.6.* → final_conv.*
        mlx_key = re.sub(r"^wav_decoder\.model\.6\.", "wav_decoder.final_conv.", mlx_key)

        mlx_decoder[mlx_key] = val

    # Step 3: Convert tensors to MLX with appropriate transformations
    result = {}
    for k, val in mlx_decoder.items():
        if isinstance(val, torch.Tensor):
            # Snake1d alpha: [1, C, 1] → [C]
            if k.endswith(".alpha"):
                val = val.squeeze()  # [1, C, 1] → [C]

            # Conv weight transposition for NLC layout
            if k.endswith(".weight") and val.dim() == 3:
                if "conv_transpose" in k:
                    # PyTorch ConvTranspose1d: [in_ch, out_ch, kernel]
                    # MLX ConvTranspose1d: [out_ch, kernel, in_ch]
                    val = val.permute(1, 2, 0).contiguous()
                else:
                    # PyTorch Conv1d: [out_ch, in_ch, kernel]
                    # MLX Conv1d: [out_ch, kernel, in_ch]
                    val = val.permute(0, 2, 1).contiguous()

            result[k] = convert_torch_to_mlx(val)
        else:
            result[k] = val

    return result


def verify_weights(mlx_weights: dict):
    """Print summary of converted weights."""
    total_params = 0
    prefixes = set()
    for key, val in sorted(mlx_weights.items()):
        prefix = key.split(".")[0]
        prefixes.add(prefix)
        total_params += val.size

    print(f"Total keys: {len(mlx_weights)}")
    print(f"Total parameters: {total_params / 1e6:.1f}M")
    print(f"Top-level prefixes: {sorted(prefixes)}")

    # Verify key shapes
    checks = [
        ("model.embed_tokens.weight", (128256, 2048)),
        ("model.layers.0.self_attn.q_proj.weight", (2048, 2048)),
        ("model.layers.0.self_attn.k_proj.weight", (512, 2048)),
        ("prediction_head.noisy_images_proj.weight", (2048, 528)),
        ("prediction_head.layers.0.adaLN_linear.weight", (6144, 2048)),
        ("acoustic_proj.weight", (2048, 512)),
        ("time_start_embed.weight", (256, 2048)),
    ]
    for key, expected_shape in checks:
        if key in mlx_weights:
            actual = mlx_weights[key].shape
            status = "OK" if tuple(actual) == expected_shape else f"MISMATCH (got {actual})"
            print(f"  {key}: {status}")
        else:
            print(f"  {key}: MISSING")


def verify_decoder_weights(decoder_weights: dict):
    """Verify decoder weight shapes."""
    print(f"\nDecoder keys: {len(decoder_weights)}")
    total = sum(v.size for v in decoder_weights.values())
    print(f"Decoder parameters: {total / 1e6:.1f}M")

    checks = [
        ("decoder_proj.weight", (1024, 512)),
        ("local_attention_decoder.layers.0.self_attn.qkv.weight", (3072, 1024)),
        ("local_attention_decoder.layers.0.self_attn.out_proj.weight", (1024, 1024)),
        ("local_attention_decoder.layers.0.ffn_linear1.weight", (4096, 1024)),
        ("local_attention_decoder.final_norm.weight", (1024,)),
        # Conv weights should be transposed to MLX layout
        ("wav_decoder.first_conv.weight", (1536, 7, 1024)),  # [out, kernel, in]
        ("wav_decoder.blocks.0.snake.alpha", (1536,)),
        ("wav_decoder.blocks.0.conv_transpose.weight", (768, 8, 1536)),  # [out, kernel, in]
        ("wav_decoder.blocks.0.residuals.0.snake1.alpha", (768,)),
        ("wav_decoder.blocks.0.residuals.0.conv1.weight", (768, 7, 768)),
        ("wav_decoder.final_snake.alpha", (96,)),
        ("wav_decoder.final_conv.weight", (1, 7, 96)),
    ]
    for key, expected_shape in checks:
        if key in decoder_weights:
            actual = decoder_weights[key].shape
            status = "OK" if tuple(actual) == expected_shape else f"MISMATCH (got {actual})"
            print(f"  {key}: {status}")
        else:
            print(f"  {key}: MISSING")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Convert TADA weights to MLX format")
    parser.add_argument("--output-dir", default="./mlx-weights", help="Output directory (default: ./mlx-weights)")
    args = parser.parse_args()

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Loading PyTorch weights...")
    pt_weights = load_pytorch_weights()
    print(f"Loaded {len(pt_weights)} keys")

    print("Converting LLM weights to MLX format...")
    mlx_weights = map_llama_weights(pt_weights)
    print("\nLLM Verification:")
    verify_weights(mlx_weights)

    # Save LLM weights
    output_path = os.path.join(output_dir, "weights.safetensors")
    print(f"\nSaving LLM weights to {output_path}...")
    mx.save_safetensors(output_path, mlx_weights)

    print("\nLoading decoder weights from HumeAI/tada-codec...")
    decoder_pt_weights = load_decoder_weights()
    print(f"Loaded {len(decoder_pt_weights)} decoder keys")

    print("Converting decoder weights...")
    decoder_weights = map_decoder_weights(decoder_pt_weights)
    verify_decoder_weights(decoder_weights)

    decoder_path = os.path.join(output_dir, "decoder_weights.safetensors")
    print(f"\nSaving decoder weights to {decoder_path}...")
    mx.save_safetensors(decoder_path, decoder_weights)

    # Save config
    import json
    config = {
        "hidden_size": 2048,
        "num_hidden_layers": 16,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 8192,
        "vocab_size": 128256,
        "rms_norm_eps": 1e-5,
        "rope_theta": 500000.0,
        "head_dim": 64,
        "max_position_embeddings": 131072,
        "rope_scaling_factor": 32.0,
        "rope_scaling_high_freq_factor": 4.0,
        "rope_scaling_low_freq_factor": 1.0,
        "rope_scaling_original_max_position_embeddings": 8192,
        "acoustic_dim": 512,
        "num_time_classes": 256,
        "shift_acoustic": 5,
        "head_layers": 6,
        "head_ffn_ratio": 4.0,
        "tie_word_embeddings": True,
        "acoustic_mean": 0.0,
        "acoustic_std": 1.5,
    }
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print("\nDone!")


if __name__ == "__main__":
    main()

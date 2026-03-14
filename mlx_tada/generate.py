"""MLX generation loop for TADA-1B.

This implements the autoregressive generation loop that runs on Metal GPU,
producing acoustic features and time codes that are then decoded by PyTorch.
"""

import math
import time
from dataclasses import dataclass

import mlx.core as mx
import mlx.nn as nn

from .utils import decode_gray_code_to_time
from .llm import KVCache


@dataclass
class GenerateConfig:
    """Generation settings matching InferenceOptions."""
    text_do_sample: bool = True
    text_temperature: float = 0.6
    text_top_k: int = 0
    text_top_p: float = 0.9
    text_repetition_penalty: float = 1.1
    acoustic_cfg_scale: float = 1.6
    duration_cfg_scale: float = 1.0
    cfg_schedule: str = "constant"
    noise_temperature: float = 0.9
    num_flow_matching_steps: int = 20
    time_schedule: str = "logsnr"
    num_acoustic_candidates: int = 1
    negative_condition_source: str = "negative_step_output"


def scheduled_cfg(base_scale: float, t: float, schedule: str) -> float:
    """Compute effective CFG scale at timestep t."""
    if schedule == "constant" or base_scale == 1.0:
        return base_scale
    if schedule == "linear":
        return 1.0 + (base_scale - 1.0) * (1.0 - t)
    if schedule == "cosine":
        return 1.0 + (base_scale - 1.0) * 0.5 * (1.0 + math.cos(math.pi * t))
    return base_scale


def build_time_schedule(num_steps: int, schedule: str) -> mx.array:
    """Build a time schedule for ODE discretization."""
    if schedule == "cosine":
        u = mx.linspace(0, 1, num_steps + 1)
        return 0.5 * (1 - mx.cos(math.pi * u))
    if schedule == "logsnr":
        log_snr = mx.linspace(5.0, -5.0, num_steps + 1)
        t_span = mx.sigmoid(-log_snr / 2)
        # Ensure exact endpoints
        vals = t_span.tolist()
        vals[0] = 0.0
        vals[-1] = 1.0
        return mx.array(vals)
    return mx.linspace(0, 1, num_steps + 1)


def solve_flow_matching(
    model,
    speech: mx.array,
    cond: mx.array,
    neg_cond: mx.array,
    num_steps: int,
    acoustic_cfg_scale: float,
    duration_cfg_scale: float,
    cfg_schedule: str,
    time_schedule: str,
    acoustic_dim: int,
) -> mx.array:
    """Solve the flow matching ODE using Euler method."""
    t_span = build_time_schedule(num_steps, time_schedule)
    use_cfg = acoustic_cfg_scale != 1.0
    B = speech.shape[0]

    # Precompute all schedule values
    t_vals = t_span.tolist()
    dt_vals = [t_vals[i+1] - t_vals[i] for i in range(num_steps)]

    # Pre-create timestep arrays
    t_arrays = [mx.array([t_vals[i]]) for i in range(num_steps)]
    dt_arrays = [mx.array(dt_vals[i]) for i in range(num_steps)]

    if use_cfg:
        # Precompute CFG scales
        a_cfgs = [scheduled_cfg(acoustic_cfg_scale, t_vals[i], cfg_schedule) for i in range(num_steps)]
        d_cfgs = [scheduled_cfg(duration_cfg_scale, t_vals[i], cfg_schedule) for i in range(num_steps)]

        # Precompute cond_combined (constant across ODE steps)
        cond_pos = cond.squeeze(1) if cond.ndim == 3 else cond
        cond_neg = neg_cond.squeeze(1) if neg_cond.ndim == 3 else neg_cond
        cond_combined = mx.concatenate([cond_pos, cond_neg], axis=0)

        for i in range(num_steps):
            # Batch speech for pos+neg
            speech_combined = mx.concatenate([speech, speech], axis=0)
            t_combined = mx.repeat(t_arrays[i], B * 2)

            velocity_combined = model.prediction_head(
                speech_combined, t_combined, condition=cond_combined
            )

            velocity_pos, velocity_neg = mx.split(velocity_combined, 2, axis=0)
            diff = velocity_pos - velocity_neg
            velocity = mx.concatenate(
                [
                    (velocity_neg + a_cfgs[i] * diff)[..., :acoustic_dim],
                    (velocity_neg + d_cfgs[i] * diff)[..., acoustic_dim:],
                ],
                axis=-1,
            )
            speech = speech + dt_arrays[i] * velocity
    else:
        cond_squeezed = cond.squeeze(1) if cond.ndim == 3 else cond
        for i in range(num_steps):
            velocity = model.prediction_head(
                speech, mx.repeat(t_arrays[i], B), condition=cond_squeezed
            )
            speech = speech + dt_arrays[i] * velocity

    return speech


@dataclass
class MLXGenerateOutput:
    """Output from MLX generation loop."""
    acoustic_features: mx.array  # (batch, seq, acoustic_dim)
    time_before: mx.array        # (batch, seq+1)
    text_token_ids: mx.array     # (batch, seq)


def generate(
    model,
    input_ids: mx.array,
    prompt_acoustic_features: mx.array,
    prompt_acoustic_masks: mx.array,
    prompt_time_len_before: mx.array,
    prompt_time_len_after: mx.array,
    config: GenerateConfig,
    tokenizer_info: dict,
    num_steps: int = 1024,
) -> MLXGenerateOutput:
    """Run the autoregressive generation loop in MLX.

    Args:
        model: TadaMLX model
        input_ids: (batch, seq_len) token IDs with prompt + text
        prompt_acoustic_features: (batch, num_prompt_frames, acoustic_dim)
        prompt_acoustic_masks: (batch, num_prompt_frames)
        prompt_time_len_before: (batch, num_prompt_frames+1)
        prompt_time_len_after: (batch, num_prompt_frames+1)
        config: generation settings
        tokenizer_info: dict with special token IDs
        num_steps: max generation steps

    Returns:
        MLXGenerateOutput with acoustic features, time codes, and token IDs
    """
    start_header_id = tokenizer_info["start_header_id"]
    end_header_id = tokenizer_info["end_header_id"]
    eot_id = tokenizer_info["eot_id"]
    pad_id = tokenizer_info["pad_id"]
    bos_id = tokenizer_info["bos_id"]
    eos_id = tokenizer_info["eos_id"]

    acoustic_dim = model.config.acoustic_dim
    shift_acoustic = model.config.shift_acoustic
    num_time_bits = model.num_time_bits
    time_dim = model.time_dim
    total_dim = acoustic_dim + time_dim
    B = input_ids.shape[0]
    prompt_len = input_ids.shape[1]

    use_neg_sampling = config.acoustic_cfg_scale != 1.0
    need_neg_batch = use_neg_sampling and config.negative_condition_source == "negative_step_output"

    # Mask text content in prompt (keep structural tokens)
    prompt_token_len = prompt_acoustic_features.shape[1] if prompt_acoustic_features is not None else 0
    input_ids_list = input_ids.tolist()
    for b in range(B):
        in_header = False
        for t in range(prompt_token_len):
            token = input_ids_list[b][t]
            if token == start_header_id:
                in_header = True
            elif token == end_header_id:
                in_header = False
            elif in_header or token in (eot_id, bos_id, eos_id):
                pass
            else:
                input_ids_list[b][t] = pad_id
    input_ids = mx.array(input_ids_list, dtype=mx.int32)

    # Initialize state
    acoustic_features = mx.zeros((B, 1, acoustic_dim))
    acoustic_masks = mx.zeros((B, 1), dtype=mx.int32)
    time_len_before = mx.zeros((B, 1), dtype=mx.int32)
    time_len_after = mx.zeros((B, 1), dtype=mx.int32)

    all_acoustic_features = []
    all_time_before = []
    all_output_token_ids = []

    neg_cond = mx.zeros((B, model.config.hidden_size))

    # Setup KV cache
    cache = [KVCache() for _ in range(model.config.num_hidden_layers)]

    # Note: mx.compile() tested but provides no speedup for this workload

    # Compute prefill length
    n_ac = min(prompt_len - shift_acoustic - 1, prompt_acoustic_features.shape[1])
    n_t = min(prompt_len - shift_acoustic - 1, prompt_time_len_before.shape[1] - 1)
    n_frames_cap = max(0, prompt_time_len_before.shape[1] - 2)
    n_prefill_frames_max = min(n_ac, n_t, n_frames_cap) if (n_ac > 0 and n_t > 0) else 0
    prefill_len = min(prompt_len, shift_acoustic + n_prefill_frames_max + 1) if n_prefill_frames_max > 0 else 0

    step_start = 0

    if prefill_len > 0:
        inputs_embeds_prefill = model.build_prompt_inputs_embeds(
            input_ids,
            prompt_acoustic_features,
            prompt_acoustic_masks,
            prompt_time_len_before,
            prompt_time_len_after,
            prefill_len,
        )

        if need_neg_batch:
            combined_embeds = mx.concatenate([inputs_embeds_prefill, inputs_embeds_prefill], axis=0)
        else:
            combined_embeds = inputs_embeds_prefill

        # Run prefill
        hidden_states = model.prefill(combined_embeds, cache=cache)
        mx.eval(hidden_states)  # Force evaluation to populate cache

        pos_hidden = hidden_states[:B]

        # Collect prefill outputs
        for s in range(prefill_len - 1):
            all_output_token_ids.append(input_ids[:, s + 1: s + 2])

        n_prefill_frames = prefill_len - shift_acoustic
        for i in range(n_prefill_frames):
            all_acoustic_features.append(prompt_acoustic_features[:, i: i + 1])
        for i in range(n_prefill_frames):
            all_time_before.append(prompt_time_len_before[:, i + 1: i + 2])

        acoustic_features = prompt_acoustic_features[:, n_prefill_frames - 1: n_prefill_frames]
        acoustic_masks = prompt_acoustic_masks[:, n_prefill_frames - 1: n_prefill_frames]
        time_len_before = prompt_time_len_before[:, n_prefill_frames: n_prefill_frames + 1]
        time_len_after = prompt_time_len_after[:, n_prefill_frames: n_prefill_frames + 1]

        step_start = prefill_len

    last_time_before = None
    gen_start_time = time.time()

    for step in range(step_start, num_steps):
        # Input token for this step
        if step < prompt_len:
            input_slice = input_ids[:, step: step + 1]
        else:
            input_slice = input_ids[:, -1:]

        need_logits = step >= prompt_len - 1

        # Check if we'll need flow matching this step (to decide if neg batch is needed)
        will_use_prompt_acoustic = (
            step >= shift_acoustic
            and prompt_acoustic_features is not None
            and step - shift_acoustic < prompt_acoustic_features.shape[1]
        )
        will_use_prompt_time = (
            step >= shift_acoustic
            and prompt_time_len_before is not None
            and step - shift_acoustic < prompt_time_len_before.shape[1] - 1
        )
        will_skip_fm = will_use_prompt_acoustic and will_use_prompt_time
        step_needs_neg = need_neg_batch and not will_skip_fm

        if step_needs_neg:
            # Build negative input: keep structural tokens, replace text with pad
            # Use vectorized MLX ops instead of Python loop
            is_structural = (
                mx.equal(input_slice, start_header_id)
                | mx.equal(input_slice, end_header_id)
                | mx.equal(input_slice, eot_id)
            )
            neg_input_slice = mx.where(is_structural, input_slice, pad_id)

            combined_slice = mx.concatenate([input_slice, neg_input_slice], axis=0)
            combined_acoustic = mx.concatenate([acoustic_features, acoustic_features], axis=0)
            combined_masks = mx.concatenate([acoustic_masks, acoustic_masks], axis=0)
            combined_time_before = mx.concatenate([time_len_before, time_len_before], axis=0)
            combined_time_after = mx.concatenate([time_len_after, time_len_after], axis=0)

            hidden_states, logits = model.forward_one_step(
                combined_slice,
                combined_acoustic,
                combined_masks,
                combined_time_before,
                combined_time_after,
                cache=cache,
                compute_logits=need_logits,
            )
            neg_cond = hidden_states[B: 2 * B]
            hidden_states = hidden_states[:B]
            logits = logits[:B] if logits is not None else None
        else:
            hidden_states, logits = model.forward_one_step(
                input_slice,
                acoustic_features,
                acoustic_masks,
                time_len_before,
                time_len_after,
                cache=cache,
                compute_logits=need_logits,
            )

        cond = hidden_states

        if will_skip_fm:
            # No need for flow matching — prompt provides acoustics and time
            predicted_time_len_before = mx.zeros((1, B), dtype=mx.int32)
            predicted_time_len_after = mx.zeros((1, B), dtype=mx.int32)
            speech = None
        else:
            # Flow matching: generate acoustic features
            noise = mx.random.normal((B, total_dim)) * config.noise_temperature

            speech = solve_flow_matching(
                model=model,
                speech=noise,
                cond=cond,
                neg_cond=neg_cond,
                num_steps=config.num_flow_matching_steps,
                acoustic_cfg_scale=config.acoustic_cfg_scale,
                duration_cfg_scale=config.duration_cfg_scale,
                cfg_schedule=config.cfg_schedule,
                time_schedule=config.time_schedule,
                acoustic_dim=acoustic_dim,
            )

            # Extract time predictions
            time_len_gray_code = speech[..., -time_dim:]
            predicted_time_len_before = decode_gray_code_to_time(
                time_len_gray_code[..., :num_time_bits], num_time_bits
            ).reshape(1, B)
            predicted_time_len_after = decode_gray_code_to_time(
                time_len_gray_code[..., num_time_bits:], num_time_bits
            ).reshape(1, B)

        # Token sampling (only when generating new tokens)
        if step >= prompt_len - 1:
            token_logits = logits[:, -1, :]
            # Prevent pad token
            token_logits = token_logits.at[:, pad_id].add(mx.array(-float("inf")))

            if config.text_do_sample:
                # Temperature scaling
                token_logits = token_logits / config.text_temperature

                # Top-k filtering
                if config.text_top_k > 0:
                    top_k = min(config.text_top_k, token_logits.shape[-1])
                    kth_vals = mx.topk(token_logits, top_k, axis=-1)
                    threshold = kth_vals[:, -1:]
                    token_logits = mx.where(token_logits < threshold, -float("inf"), token_logits)

                # Top-p filtering
                if 0.0 < config.text_top_p < 1.0:
                    sorted_indices = mx.argsort(-token_logits, axis=-1)
                    sorted_logits = mx.take_along_axis(token_logits, sorted_indices, axis=-1)
                    cumulative_probs = mx.cumsum(mx.softmax(sorted_logits, axis=-1), axis=-1)
                    sorted_mask = (cumulative_probs - mx.softmax(sorted_logits, axis=-1)) >= config.text_top_p
                    # Scatter back: create mask by inverse-sorting the sorted mask
                    inv_indices = mx.argsort(sorted_indices, axis=-1)
                    mask = mx.take_along_axis(sorted_mask, inv_indices, axis=-1)
                    token_logits = mx.where(mask, -float("inf"), token_logits)

                probs = mx.softmax(token_logits, axis=-1)
                next_token = mx.random.categorical(mx.log(probs + 1e-10), axis=-1).reshape(B, 1)
            else:
                next_token = mx.argmax(token_logits, axis=-1).reshape(B, 1)

            input_ids = mx.concatenate([input_ids, next_token.astype(mx.int32)], axis=1)
            all_output_token_ids.append(next_token.astype(mx.int32))

            # Check for EOS
            if next_token[0, 0].item() == eos_id:
                pass
        else:
            all_output_token_ids.append(input_ids[:, step + 1: step + 2])

        # Update acoustic features and time for next step
        if step >= shift_acoustic:
            if (prompt_acoustic_features is not None
                    and step - shift_acoustic < prompt_acoustic_features.shape[1]):
                acoustic_features = prompt_acoustic_features[:, step - shift_acoustic: step - shift_acoustic + 1]
                acoustic_masks = prompt_acoustic_masks[:, step - shift_acoustic: step - shift_acoustic + 1]
            elif speech is not None:
                acoustic_features = speech[..., :acoustic_dim].reshape(B, 1, acoustic_dim)
                acoustic_masks = mx.ones((B, 1), dtype=mx.int32)
            all_acoustic_features.append(acoustic_features)

            if (prompt_time_len_before is not None
                    and step - shift_acoustic < prompt_time_len_before.shape[1] - 1):
                time_len_before = prompt_time_len_before[:, step - shift_acoustic + 1: step - shift_acoustic + 2]
                time_len_after = prompt_time_len_after[:, step - shift_acoustic + 1: step - shift_acoustic + 2]
            else:
                time_len_before = predicted_time_len_before.reshape(B, 1)
                time_len_after = predicted_time_len_after.reshape(B, 1)
            all_time_before.append(time_len_before)
            last_time_before = time_len_before

        # Periodic eval to prevent Metal command buffer timeout
        if (step - step_start) % 20 == 0:
            mx.eval(acoustic_features, time_len_before)

    # Add trailing time
    if last_time_before is not None:
        all_time_before.append(last_time_before)

    gen_time = time.time() - gen_start_time
    n_gen_steps = num_steps - step_start
    print(f"  MLX generation loop: {gen_time:.1f}s ({n_gen_steps} steps, {gen_time/max(n_gen_steps,1)*1000:.0f}ms/step)")

    # Stack outputs
    if all_acoustic_features:
        acoustic_out = mx.concatenate(all_acoustic_features, axis=1)
    else:
        acoustic_out = mx.zeros((B, 0, acoustic_dim))

    if all_time_before:
        time_out = mx.concatenate(all_time_before, axis=1)
    else:
        time_out = mx.zeros((B, 0), dtype=mx.int32)

    if all_output_token_ids:
        token_ids_out = mx.concatenate(all_output_token_ids, axis=1)
    else:
        token_ids_out = mx.zeros((B, 0), dtype=mx.int32)

    mx.eval(acoustic_out, time_out, token_ids_out)

    return MLXGenerateOutput(
        acoustic_features=acoustic_out,
        time_before=time_out,
        text_token_ids=token_ids_out,
    )

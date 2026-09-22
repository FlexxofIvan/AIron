from __future__ import annotations
from typing import List, Optional, Tuple

import torch



BIOMECH_SLOT_SIZE = 80


SLOT_PLACEHOLDER_ID = 0


def insert_slot_before_token(
    prompt_token_ids: List[int],
    sentinel_token_id: int,
) -> Tuple[List[int], int, int]:

    try:
        pos = prompt_token_ids.index(sentinel_token_id)
    except ValueError:
        pos = len(prompt_token_ids)
    slot_start = pos
    slot_end = pos + BIOMECH_SLOT_SIZE
    new = (
        list(prompt_token_ids[:pos])
        + [SLOT_PLACEHOLDER_ID] * BIOMECH_SLOT_SIZE
        + list(prompt_token_ids[pos:])
    )
    return new, slot_start, slot_end


def fill_slot_inplace(
    input_tensor: torch.Tensor,
    slot_start: int,
    biomech_token_ids: List[int],
) -> None:
    """Fill input_tensor[..., slot_start:slot_start+SLOT_SIZE) with biomech
    token IDs (pad/truncate as needed). Modifies in place."""
    n = min(len(biomech_token_ids), BIOMECH_SLOT_SIZE)
    fill = list(biomech_token_ids[:n]) + [SLOT_PLACEHOLDER_ID] * (BIOMECH_SLOT_SIZE - n)
    fill_t = torch.tensor(fill, dtype=input_tensor.dtype, device=input_tensor.device)
    input_tensor[..., slot_start:slot_start + BIOMECH_SLOT_SIZE] = fill_t




def cache_to_legacy(cache):
    """DynamicCache (HF newer) → legacy tuple format. Pass-through for tuples."""
    if cache is None or isinstance(cache, tuple):
        return cache
    return cache.to_legacy_cache()


def legacy_to_cache(legacy):
    """Legacy tuple → DynamicCache. Pass-through for already-Cache objects."""
    if legacy is None:
        return None
    if not isinstance(legacy, tuple):
        return legacy
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(legacy)


def slice_past_kv_range(past_kv, start: int, end: int):
    """Return legacy-format past_kv covering only positions [start, end)."""
    past_kv = cache_to_legacy(past_kv)
    return tuple(
        (k[..., start:end, :], v[..., start:end, :])
        for k, v in past_kv
    )


def splice_past_kv(past_kv_full, new_slot_kv, slot_start: int, slot_end: int):
    """Replace past_kv_full[slot_start:slot_end] with new_slot_kv."""
    past_kv_full = cache_to_legacy(past_kv_full)
    new_slot_kv = cache_to_legacy(new_slot_kv)
    out = []
    for (k, v), (nk, nv) in zip(past_kv_full, new_slot_kv):
        k_new = torch.cat(
            [k[..., :slot_start, :], nk, k[..., slot_end:, :]], dim=2
        )
        v_new = torch.cat(
            [v[..., :slot_start, :], nv, v[..., slot_end:, :]], dim=2
        )
        out.append((k_new, v_new))
    return tuple(out)


def compute_new_slot_kv(
    lang_model,
    biomech_token_ids: List[int],
    past_kv_prefix,  # kept for API compatibility but no longer used
    slot_start: int,
    device,
    prefix_token_ids: Optional[List[int]] = None,
):
    """Run a standalone forward pass over the biomech tokens with RoPE
    positions set to [slot_start, slot_start+SLOT_SIZE). Returns K/V for the
    slot positions (legacy tuple format).
    """

    n = min(len(biomech_token_ids), BIOMECH_SLOT_SIZE)
    slot_fill = list(biomech_token_ids[:n]) + [SLOT_PLACEHOLDER_ID] * (BIOMECH_SLOT_SIZE - n)


    use_real_prefix = (
        prefix_token_ids is not None and len(prefix_token_ids) == slot_start
    )
    prefix = list(prefix_token_ids) if use_real_prefix else [SLOT_PLACEHOLDER_ID] * slot_start
    full_seq = prefix + slot_fill
    full_len = slot_start + BIOMECH_SLOT_SIZE

    inp = torch.tensor([full_seq], dtype=torch.long, device=device)
    pos_ids = torch.arange(0, full_len, device=device).unsqueeze(0)


    attn_mask = torch.zeros(1, full_len, device=device, dtype=torch.long)
    if use_real_prefix:
        attn_mask[:] = 1
    else:
        attn_mask[:, slot_start:] = 1

    embed_layer = lang_model.get_input_embeddings()
    embed = embed_layer(inp)
    inputs_embeds_dict = {
        "0": {"comb": embed},
        "vision_xattn_mask": torch.zeros(
            1, full_len, device=device, dtype=torch.long
        ),
    }

    with torch.no_grad():
        out = lang_model(
            inputs_embeds=inputs_embeds_dict,
            position_ids=pos_ids,
            attention_mask=attn_mask,
            use_cache=True,
        )
    full_kv = cache_to_legacy(out.past_key_values)
    # full_kv covers [0, full_len); extract only the slot range.
    return tuple(
        (k[..., slot_start:slot_start + BIOMECH_SLOT_SIZE, :],
         v[..., slot_start:slot_start + BIOMECH_SLOT_SIZE, :])
        for k, v in full_kv
    )

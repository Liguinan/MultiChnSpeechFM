#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Init for ModelDOAMoE:
  - everything except expert_adapter_*  <- v2 ModelDOA phase2
  - expert_adapter_1..6                 <- MoE ckpt (original v5 / v6 MoE)
  - moe_weights_* left random (unless load_moe_weights=true)
"""
from pathlib import Path

import torch


def _strip_module(state):
    return {k.replace("module.", ""): v for k, v in state.items()}


def load_state_dict(checkpoint_path):
    checkpoint_path = Path(checkpoint_path).expanduser().absolute()
    assert checkpoint_path.exists(), f"checkpoint not found: {checkpoint_path}"
    ckpt = torch.load(checkpoint_path.as_posix(), map_location="cpu")
    if checkpoint_path.suffix == ".pth":
        state = ckpt
        epoch = checkpoint_path.stem.split("_")[1] if "_" in checkpoint_path.stem else "?"
    else:
        state = ckpt["model"] if "model" in ckpt else ckpt.get("state_dict", ckpt)
        epoch = ckpt.get("epoch", "?")
    state = _strip_module(state)
    print(f"[init] loaded {checkpoint_path.name} (epoch={epoch}), keys={len(state)}")
    return state


_FUSION_MOE_TARGETS = ("fusion_block_4", "fusion_block_5", "fusion_block_6")


def _map_v2_fusion_key(key, copy_fusion_to_all=False):
    """
    v2: fusion_block with num_repeats=3 -> keys fusion_block.{0,1,2}.*
    MoE: fusion_block_4/5/6 each num_repeats=1 -> keys fusion_block_X.0.*

    copy_fusion_to_all=False (default):
      repeat i -> fusion_block_{4+i}.0.*
    copy_fusion_to_all=True (legacy):
      repeat 0 only -> all three fusion_block_4/5/6.0.*
    """
    rest = key[len("fusion_block.") :]
    dot = rest.find(".")
    if dot <= 0 or not rest[:dot].isdigit():
        return [(f"fusion_block_4.{rest}", key)]

    repeat_idx = int(rest[:dot])
    inner = rest[dot + 1 :]
    if copy_fusion_to_all:
        if repeat_idx != 0:
            return []
        return [
            (f"{tgt}.0.{inner}", key) for tgt in _FUSION_MOE_TARGETS
        ]

    if repeat_idx >= len(_FUSION_MOE_TARGETS):
        return []
    tgt = _FUSION_MOE_TARGETS[repeat_idx]
    return [(f"{tgt}.0.{inner}", key)]


def map_v2_doa_to_doa_moe(v2_state, copy_fusion_to_all=False):
    """
    Map ModelDOA keys onto ModelDOAMoE.
    - keep DOA branch as-is
    - fusion_block.{0,1,2} -> fusion_block_4/5/6 (see _map_v2_fusion_key)
    - skip MoE-only / unused keys
    """
    skip_substrings = (
        "expert_adapter",
        "moe_weights",
        "geometry_projector",
        "geometry_head",
    )
    mapped = {}
    fusion_hits = {tgt: 0 for tgt in _FUSION_MOE_TARGETS}
    for k, v in v2_state.items():
        if any(s in k for s in skip_substrings):
            continue
        if k.startswith("fusion_block."):
            for dst, _src in _map_v2_fusion_key(k, copy_fusion_to_all):
                mapped[dst] = v
                tgt = dst.split(".", 1)[0]
                fusion_hits[tgt] = fusion_hits.get(tgt, 0) + 1
            continue
        mapped[k] = v
    mode = "copy_repeat0_to_all" if copy_fusion_to_all else "split_repeats"
    print(
        f"[init] mapped {len(mapped)} tensors from v2 ModelDOA "
        f"(fusion mode={mode}, hits={fusion_hits})"
    )
    return mapped


def extract_expert_adapters(moe_state, load_moe_weights=False):
    """Take only expert_adapter_* (and optionally moe_weights_*) from MoE ckpt."""
    out = {}
    for k, v in moe_state.items():
        if k.startswith("expert_adapter_"):
            out[k] = v
        elif load_moe_weights and k.startswith("moe_weights_"):
            out[k] = v
    print(
        f"[init] extracted {len(out)} MoE adapter tensors "
        f"(load_moe_weights={load_moe_weights})"
    )
    return out


def merge_doa_moe_init(model_state_keys, v2_mapped, moe_adapter_state):
    """
    Priority:
      1) expert_adapter_* (and optional moe_weights_*) from MoE ckpt
      2) everything else from v2 mapped state
      3) missing keys stay randomly initialized
    """
    updated = {}
    missing = []
    for k in model_state_keys:
        if k in moe_adapter_state:
            updated[k] = moe_adapter_state[k]
        elif k in v2_mapped:
            updated[k] = v2_mapped[k]
        else:
            # leave random: moe_weights / geometry_* / any unmatched
            missing.append(k)
    print(
        f"[init] merged={len(updated)}, left_random={len(missing)} "
        f"(first missing: {missing[:10]})"
    )
    return updated

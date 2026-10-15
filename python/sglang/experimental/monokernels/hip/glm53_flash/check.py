# SPDX-License-Identifier: Apache-2.0
"""SGLANG_TEST_MONO_CHECK (eager only): each span also runs the normal layer on its
input, on the caches as they were, and checks the kernel's outputs against it: the
written streams, and the state or cache rows the layer writes, each against its
``spec.NUMERICS`` bound. Every check is logged; a miss is logged as a violation."""

from __future__ import annotations

import logging

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# The SGLang names this check reads (as in ``spec.SGLANG_READS``)
SGLANG_READS = {
    "srt/layers/layer_boundary/residual/batch.py": ("take_output", "set_written"),
    "srt/mem_cache/memory_pool.py": (
        "mamba2_layer_cache",
        "conv",
        "temporal",
        "get_key_buffer",
        "get_index_k_with_scale_buffer",
        "get_compress_tail_buffers",
    ),
    "srt/model_executor/forward_batch_info.py": ("out_cache_loc", "req_pool_indices"),
    "srt/model_executor/forward_context.py": (
        "get_req_to_token_pool",
        "get_token_to_kv_pool",
    ),
    "srt/runtime_context.py": ("get_parallel", '"tp_rank"'),
    "srt/utils/common.py": ("BumpAllocator",),
}

PAGE_SIZE = 256  # tokens an index-cache page covers
INDEX_VALUE_BYTES = 64 * 128  # an index page's FP8 values; its f32 scales follow


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    if a.numel() == 0:
        return {}
    return {
        "cosine": F.cosine_similarity(a, b, dim=0).item(),
        "rel": ((a - b).norm() / b.norm().clamp_min(1e-30)).item(),
        "max_abs": (a - b).abs().max().item(),
    }


def _outputs(name: str, a: torch.Tensor, b: torch.Tensor) -> dict[str, dict]:
    if name != "index":
        return {name: _metrics(a, b)}
    # both pages in one layout: compare the decoded FP8 values and the f32 scales
    fp8 = torch.float8_e4m3fn
    return {
        "index": _metrics(
            a[:, :INDEX_VALUE_BYTES].contiguous().view(fp8),
            b[:, :INDEX_VALUE_BYTES].contiguous().view(fp8),
        ),
        "index_scale": _metrics(
            a[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32),
            b[:, INDEX_VALUE_BYTES:].contiguous().view(torch.float32),
        ),
    }


def _regions(L, forward_batch, S: int, state_idx) -> dict:
    """The cache entries the layer writes this step: name -> (tensor, index)."""
    from sglang.srt.model_executor.forward_context import (
        get_req_to_token_pool,
        get_token_to_kv_pool,
    )

    if L.kda:
        cache = get_req_to_token_pool().mamba2_layer_cache(L.i)
        sel = state_idx[state_idx > 0].long()
        return {"conv": (cache.conv[0], sel), "state": (cache.temporal, sel)}
    pool = get_token_to_kv_pool()
    loc = forward_batch.out_cache_loc[:S]
    valid = loc != 0
    slots = loc[valid].long()
    kv = pool.get_key_buffer(L.i)
    tail_k, tail_g = pool.get_compress_tail_buffers(L.i)
    req = forward_batch.req_pool_indices[:S][valid].long()
    return {
        "kv": (kv.view(kv.shape[0], -1), slots),
        "index": (
            pool.get_index_k_with_scale_buffer(L.i),
            (slots // PAGE_SIZE).unique(),
        ),
        "tail_k": (tail_k, req),
        "tail_g": (tail_g, req),
    }


def reference(L, positions, streams_in, forward_batch, S: int, state_idx) -> dict:
    """The normal layer's outputs on ``streams_in``; the caches are restored after."""
    from sglang.srt.layers.layer_boundary.residual.batch import set_written, take_output
    from sglang.srt.utils.common import BumpAllocator

    regions = _regions(L, forward_batch, S, state_idx)
    saved = {name: t[idx].clone() for name, (t, idx) in regions.items()}
    hidden = set_written(streams_in, forward_batch)
    zero = BumpAllocator(buffer_size=4, dtype=torch.float32, device=streams_in.device)
    out, _ = L.module(
        positions, hidden, forward_batch, zero, None, prev_topk_indices=None
    )
    ref = {"streams": take_output(out, forward_batch).clone()}
    for name, (t, idx) in regions.items():
        ref[name] = t[idx].clone()
        t[idx] = saved[name]
    return ref


def compare(
    L, ref: dict, streams: torch.Tensor, positions, forward_batch, S: int, state_idx
) -> None:
    from sglang.experimental.monokernels.hip.glm53_flash.spec import NUMERICS
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().tp_rank != 0:
        return
    kind = "kda" if L.kda else "dsa"
    outputs = _outputs("streams", streams, ref["streams"])
    for name, (t, idx) in _regions(L, forward_batch, S, state_idx).items():
        outputs.update(_outputs(name, t[idx], ref[name]))
    values = {
        f"{kind}.{out}.{metric}": v
        for out, ms in outputs.items()
        for metric, v in ms.items()
    }
    pos = positions[:S]
    logger.info(
        "monokernel check layer %d (%s, %d rows, positions %d..%d): %s",
        L.i,
        kind,
        S,
        pos.min().item(),
        pos.max().item(),
        " ".join(f"{k}={v:.6g}" for k, v in values.items()),
    )
    for name, bound in NUMERICS.bounds.items():
        v = values.get(name)
        if v is None:
            continue
        ok = v >= bound if name.endswith(".cosine") else v <= bound
        if not ok:
            logger.error(
                "monokernel check violation: layer %d %s = %.6g, bound %g",
                L.i,
                name,
                v,
                bound,
            )

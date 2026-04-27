from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz

FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


@triton.jit
def _load_model1_kv_tile(
    cache_fp8,
    cache_u8,
    cache_bf16,
    idx,
    offs_d,
    block_size: tl.constexpr,
    bytes_per_block: tl.constexpr,
    bf16_per_block: tl.constexpr,
):
    block_idx = idx // block_size
    block_offset = idx - block_idx * block_size
    is_nope = offs_d < 448

    token_base_bytes = block_idx * bytes_per_block + block_offset * (448 + 2 * 64)
    fp8_offsets = token_base_bytes + offs_d
    tile = offs_d // 64
    scale_offset = block_idx * bytes_per_block + block_size * (448 + 2 * 64)
    scale_offset += block_offset * 8 + tile

    vals = tl.load(cache_fp8 + fp8_offsets, mask=is_nope, other=0.0).to(tl.float32)
    scale_byte = tl.load(cache_u8 + scale_offset, mask=is_nope, other=127).to(
        tl.float32
    )
    nope = vals * tl.exp2(scale_byte - 127.0)

    rope_base = (
        block_idx * bf16_per_block
        + block_offset * ((448 + 2 * 64) // 2)
        + 448 // 2
    )
    rope = tl.load(cache_bf16 + rope_base + (offs_d - 448), mask=~is_nope, other=0.0)
    return tl.where(is_nope, nope, rope)


@triton.jit
def _model1_selected_dequant_kernel(
    cache_fp8,
    cache_u8,
    cache_bf16,
    indices,
    topk_length,
    out,
    invalid_mask,
    topk: tl.constexpr,
    max_index: tl.constexpr,
    block_size: tl.constexpr,
    bytes_per_block: tl.constexpr,
    bf16_per_block: tl.constexpr,
    has_topk_length: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    offs = tl.arange(0, BLOCK_D)

    idx = tl.load(indices + row)
    invalid = (idx < 0) | (idx >= max_index)
    if has_topk_length:
        batch_q = row // topk
        topk_pos = row - batch_q * topk
        invalid = invalid | (topk_pos >= tl.load(topk_length + batch_q))

    fixed_idx = tl.minimum(tl.maximum(idx, 0), max_index - 1)
    vals = _load_model1_kv_tile(
        cache_fp8,
        cache_u8,
        cache_bf16,
        fixed_idx,
        tile * BLOCK_D + offs,
        block_size,
        bytes_per_block,
        bf16_per_block,
    )

    out_base = row * 512 + tile * BLOCK_D
    tl.store(out + out_base + offs, vals, mask=offs < BLOCK_D)
    tl.store(invalid_mask + row, invalid)


@triton.jit
def _sparse_decode_gathered_kernel(
    Q,
    GATHERED_KV,
    INVALID_MASK,
    ATTN_SINK,
    OUT,
    LSE,
    softmax_scale: tl.constexpr,
    S_Q: tl.constexpr,
    H_Q: tl.constexpr,
    TOPK: tl.constexpr,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    head_block = tl.program_id(1)
    heads = head_block * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    mask_h = heads < H_Q

    q = tl.load(
        Q + row * H_Q * BLOCK_D + heads[:, None] * BLOCK_D + offs_d[None, :],
        mask=mask_h[:, None],
        other=0.0,
    )
    e_max = tl.full([BLOCK_H], -float("inf"), tl.float32)
    e_sum = tl.zeros([BLOCK_H], tl.float32)
    acc = tl.zeros([BLOCK_H, BLOCK_D], tl.float32)

    for start_n in range(0, TOPK, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < TOPK
        kv = tl.load(
            GATHERED_KV
            + row * TOPK * BLOCK_D
            + offs_n[None, :] * BLOCK_D
            + offs_d[:, None],
            mask=mask_n[None, :],
            other=0.0,
        )
        qk = tl.dot(q, kv.to(q.dtype)) * softmax_scale
        invalid = tl.load(
            INVALID_MASK + row * TOPK + offs_n, mask=mask_n, other=1
        ).to(tl.int1)
        qk = tl.where(
            mask_h[:, None] & mask_n[None, :] & ~invalid[None, :],
            qk,
            -float("inf"),
        )
        n_e_max = tl.maximum(tl.max(qk, 1), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(qk - n_e_max[:, None])
        acc *= re_scale[:, None]
        acc += tl.dot(p.to(kv.dtype), tl.trans(kv))
        e_sum = e_sum * re_scale + tl.sum(p, 1)
        e_max = n_e_max

    has_token = e_sum > 0.0
    lse = tl.where(has_token, e_max + tl.log(e_sum), float("inf"))
    out = acc / tl.where(has_token, e_sum, 1.0)[:, None]
    if HAS_ATTN_SINK:
        sink = tl.load(ATTN_SINK + heads, mask=mask_h, other=0.0)
        out *= (1.0 / (1.0 + tl.exp(sink - lse)))[:, None]
    out = tl.where(has_token[:, None], out, 0.0)

    tl.store(
        OUT + row * H_Q * BLOCK_D + heads[:, None] * BLOCK_D + offs_d[None, :],
        out,
        mask=mask_h[:, None],
    )
    batch = row // S_Q
    s_idx = row - batch * S_Q
    tl.store(LSE + batch * H_Q * S_Q + heads * S_Q + s_idx, lse, mask=mask_h)


def _cache_views(quant_k_cache: torch.Tensor):
    quant_k_cache = quant_k_cache.view(FP8_DTYPE)
    bytes_per_block = quant_k_cache.stride(0)
    return (
        quant_k_cache,
        quant_k_cache.view(torch.uint8),
        quant_k_cache.view(torch.bfloat16),
        bytes_per_block,
    )


def dequantize_model1_selected_k_triton(
    quant_k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    d_nope, d_rope, num_tiles = 448, 64, 7
    block_size = quant_k_cache.shape[1]
    num_blocks = quant_k_cache.shape[0]
    assert quant_k_cache.shape[2] == 1
    assert quant_k_cache.is_cuda and indices_in_kvcache.is_cuda

    cache_fp8, cache_u8, cache_bf16, bytes_per_block = _cache_views(quant_k_cache)
    indices_in_kvcache = indices_in_kvcache.contiguous()
    if topk_length is not None:
        topk_length = topk_length.contiguous()

    out = torch.empty(
        (*indices_in_kvcache.shape, d_nope + d_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    invalid_mask = torch.empty_like(indices_in_kvcache, dtype=torch.bool)
    topk = indices_in_kvcache.shape[-1]
    topk_length_arg = (
        topk_length if topk_length is not None else indices_in_kvcache.new_empty((1,))
    )

    _model1_selected_dequant_kernel[(indices_in_kvcache.numel(), num_tiles + 1)](
        cache_fp8,
        cache_u8,
        cache_bf16,
        indices_in_kvcache.view(-1),
        topk_length_arg,
        out.view(-1, d_nope + d_rope),
        invalid_mask.view(-1),
        topk,
        num_blocks * block_size,
        block_size,
        bytes_per_block,
        bytes_per_block // 2,
        topk_length is not None,
        BLOCK_D=64,
        num_warps=2,
        num_stages=1,
        waves_per_eu=1,
    )
    return out, invalid_mask


def sparse_decode_gathered_triton(
    q: torch.Tensor,
    gathered_kv: torch.Tensor,
    invalid_mask: torch.Tensor,
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s_q, h_q, d_qk = q.shape
    assert d_qk == 512 and d_v == 512
    q = q.contiguous()
    gathered_kv = gathered_kv.contiguous()
    invalid_mask = invalid_mask.contiguous()
    output = torch.empty((b, s_q, h_q, d_v), dtype=torch.bfloat16, device=q.device)
    lse = torch.empty((b, h_q, s_q), dtype=torch.float32, device=q.device)
    block_h = 16
    _sparse_decode_gathered_kernel[(b * s_q, triton.cdiv(h_q, block_h))](
        q,
        gathered_kv,
        invalid_mask,
        attn_sink if attn_sink is not None else q.new_empty((1,)),
        output,
        lse,
        softmax_scale,
        S_Q=s_q,
        H_Q=h_q,
        TOPK=gathered_kv.shape[2],
        HAS_ATTN_SINK=attn_sink is not None,
        BLOCK_H=block_h,
        BLOCK_N=32,
        BLOCK_D=512,
        num_warps=4,
        num_stages=1,
        waves_per_eu=1,
    )
    return output, lse

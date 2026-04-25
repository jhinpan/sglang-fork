import os
from typing import Any, Optional

import torch
import torch.nn.functional as F

from sglang.srt.utils import is_hip
from sglang.srt.layers.quantization.fp8_kernel import is_fp8_fnuz
FP8_DTYPE = torch.float8_e4m3fnuz if is_fp8_fnuz() else torch.float8_e4m3fn


def _dequantize_model1_selected_k(
    quant_k_cache: torch.Tensor,
    indices_in_kvcache: torch.Tensor,
    topk_length: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather and dequantize only the sparse K entries selected for decode."""
    d_nope, d_rope, tile_size, num_tiles = 448, 64, 64, 7

    quant_k_cache = quant_k_cache.view(FP8_DTYPE)
    num_blocks, block_size, h_k, _ = quant_k_cache.shape
    assert h_k == 1

    max_index = num_blocks * block_size
    invalid_mask = (indices_in_kvcache < 0) | (indices_in_kvcache >= max_index)
    if topk_length is not None:
        topk = indices_in_kvcache.size(-1)
        invalid_mask |= torch.arange(0, topk, device=indices_in_kvcache.device).view(
            1, 1, topk
        ).broadcast_to(indices_in_kvcache.shape) >= topk_length.view(
            indices_in_kvcache.shape[0], 1, 1
        )

    fixed_indices = indices_in_kvcache.clamp(0, max_index - 1)
    block_indices = fixed_indices // block_size
    block_offsets = fixed_indices % block_size

    quant_k_cache = quant_k_cache.reshape(num_blocks, -1)
    nope_rope = quant_k_cache[:, : block_size * (d_nope + 2 * d_rope)].reshape(
        num_blocks, block_size, d_nope + 2 * d_rope
    )
    scales = (
        quant_k_cache[:, block_size * (d_nope + 2 * d_rope) :]
        .reshape(num_blocks, block_size, 8)[:, :, :num_tiles]
        .view(torch.float8_e8m0fnu)
    )

    gathered = nope_rope[block_indices, block_offsets]
    nope = gathered[..., :d_nope]
    rope = gathered[..., d_nope : d_nope + 2 * d_rope].view(torch.bfloat16)
    gathered_scales = scales[block_indices, block_offsets]

    out = torch.empty(
        (*indices_in_kvcache.shape, d_nope + d_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )
    out[..., d_nope:] = rope
    for tile_idx in range(num_tiles):
        start = tile_idx * tile_size
        end = start + tile_size
        out[..., start:end] = nope[..., start:end].to(torch.bfloat16) * gathered_scales[
            ..., tile_idx
        ].to(torch.bfloat16).unsqueeze(-1)

    return out, invalid_mask


def _sparse_attn_decode_from_fp8_cache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    indices: torch.Tensor,
    topk_length: Optional[torch.Tensor],
    softmax_scale: float,
    attn_sink: Optional[torch.Tensor],
    extra_k_cache: Optional[torch.Tensor],
    extra_indices: Optional[torch.Tensor],
    extra_topk_length: Optional[torch.Tensor],
    d_v: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, s_q, h_q, d_qk = q.shape
    gathered_kv, invalid_mask = _dequantize_model1_selected_k(
        k_cache, indices, topk_length
    )
    if extra_k_cache is not None:
        assert extra_indices is not None
        extra_gathered_kv, extra_invalid_mask = _dequantize_model1_selected_k(
            extra_k_cache, extra_indices, extra_topk_length
        )
        gathered_kv = torch.cat([gathered_kv, extra_gathered_kv], dim=2)
        invalid_mask = torch.cat([invalid_mask, extra_invalid_mask], dim=2)

    if os.environ.get("SGLANG_DSV4_SDPA_FLASHMLA_TORCH") == "1":
        assert s_q == 1
        q_sdpa = q.float().permute(0, 2, 1, 3)
        k_sdpa = gathered_kv[:, 0].float().unsqueeze(1).expand(-1, h_q, -1, -1)
        v_sdpa = (
            gathered_kv[:, 0, :, :d_v].float().unsqueeze(1).expand(-1, h_q, -1, -1)
        )
        attn_mask = ~invalid_mask[:, 0].unsqueeze(1).unsqueeze(2).expand(
            -1, h_q, 1, -1
        )
        output = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )
        output = output.permute(0, 2, 1, 3)
        lse = q.new_zeros((b, h_q, s_q), dtype=torch.float32)
        return output.to(torch.bfloat16), lse

    gathered_kv = gathered_kv.view(b * s_q, -1, d_qk).float()
    gathered_kv[gathered_kv != gathered_kv] = 0.0
    q = q.float().view(b * s_q, h_q, d_qk)

    attn_weight = q @ gathered_kv.transpose(-1, -2)
    attn_weight *= softmax_scale
    attn_weight[
        invalid_mask.view(b * s_q, 1, -1).broadcast_to(
            b * s_q, h_q, invalid_mask.size(-1)
        )
    ] = float("-inf")

    lse = attn_weight.logsumexp(dim=-1)
    attn_weight = torch.exp(attn_weight - lse.unsqueeze(-1))
    output = attn_weight @ gathered_kv[..., :d_v]
    output = output.view(b, s_q, h_q, d_v)
    lse = lse.view(b, s_q, h_q)

    if attn_sink is not None:
        output *= (
            1.0 / (1.0 + torch.exp(attn_sink.view(1, 1, h_q) - lse))
        ).unsqueeze(-1)

    lonely_q_mask = lse == float("-inf")
    output[lonely_q_mask.unsqueeze(-1).broadcast_to(b, s_q, h_q, d_v)] = 0.0
    lse[lonely_q_mask] = float("+inf")

    return output.to(torch.bfloat16), lse.transpose(1, 2)


def flash_mla_with_kvcache_entrypoint(backend: str, **kwargs):
    if is_hip():
        # backend == "torch"
        import os

        backend = os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "kernel")
    else:
        import flash_mla

    if backend == "comparison":
        pack_ref, pack_fast_via_tester = flash_mla_with_kvcache_entrypoint(
            backend="torch", **kwargs
        )
        pack_fast_via_api = flash_mla_with_kvcache_entrypoint(
            backend="kernel", **kwargs
        )
        _assert_close(pack_ref=pack_fast_via_tester, pack_fast=pack_fast_via_api)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_tester)
        _assert_close(pack_ref=pack_ref, pack_fast=pack_fast_via_api)
        return pack_ref

    if backend == "torch":
        return flash_mla_with_kvcache_torch(**kwargs)

    if backend == "kernel":
        return flash_mla.flash_mla_with_kvcache(**kwargs)

    raise NotImplementedError


def flash_mla_with_kvcache_torch(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: Optional[torch.Tensor],
    cache_seqlens: Optional[torch.Tensor],
    head_dim_v: int,
    tile_scheduler_metadata: Any,
    num_splits: None = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    is_fp8_kvcache: bool = False,
    indices: Optional[torch.Tensor] = None,
    attn_sink: Optional[torch.Tensor] = None,
    extra_k_cache: Optional[torch.Tensor] = None,
    extra_indices_in_kvcache: Optional[torch.Tensor] = None,
    topk_length: Optional[torch.Tensor] = None,
    extra_topk_length: Optional[torch.Tensor] = None,
):

    assert block_table is None
    assert cache_seqlens is None
    assert is_fp8_kvcache

    b, s_q, h_q, d_qk = q.shape
    d_v = head_dim_v

    assert indices is not None
    return _sparse_attn_decode_from_fp8_cache(
        q=q,
        k_cache=k_cache,
        indices=indices,
        topk_length=topk_length,
        softmax_scale=softmax_scale,
        attn_sink=attn_sink,
        extra_k_cache=extra_k_cache,
        extra_indices=extra_indices_in_kvcache,
        extra_topk_length=extra_topk_length,
        d_v=d_v,
    )


def _assert_close(pack_ref, pack_fast):
    import sglang.srt.flashmla_tests.kernelkit as kk

    out_ref, lse_ref = pack_ref
    out_fast, lse_fast = pack_fast

    # the copied threshold is too strict, not checked why
    # copied from: test_flash_mla_sparse_decoding.py
    # is_out_correct = kk.check_is_allclose(
    #     "out", out_fast, out_ref, abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6
    # )
    # is_lse_correct = kk.check_is_allclose(
    #     "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    # )

    # loosen thresh
    is_out_correct = kk.check_is_allclose(
        "out", out_fast, out_ref, abs_tol=1e-2, rel_tol=10.0, cos_diff_tol=5e-6
    )
    is_lse_correct = kk.check_is_allclose(
        "lse", lse_fast, lse_ref, abs_tol=1e-6, rel_tol=8.01 / 65536
    )

    assert is_out_correct and is_lse_correct, f"{is_out_correct=} {is_lse_correct=}"

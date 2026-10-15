"""The GLM-5.3-Flash gfx950 monokernel spec on CPU (RFC #43949): its spans on the
GLM-5.3-Flash layer structure, its row routing and TP row cap, its registry entry,
and its refusal reasons."""

from types import SimpleNamespace

import torch

from sglang.experimental.monokernels import spec as registry
from sglang.experimental.monokernels.hip.glm53_flash import (
    HF_CONFIG,
    LINEAR_ATTN_CONFIG,
)
from sglang.experimental.monokernels.hip.glm53_flash import spec as flash
from sglang.experimental.monokernels.spec import ShapeKey, SpanPoint
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, stage="base-a", runner_config="cpu")

# GLM-5.3-Flash's text-model layer structure (its config.json)
TEXT = SimpleNamespace(
    **HF_CONFIG,
    num_hidden_layers=45,
    first_k_dense_replace=3,
    moe_layer_freq=1,
    linear_attn_config={
        **LINEAR_ATTN_CONFIG,
        "kda_layers": [i for i in range(45) if i % 4 != 3],
        "full_attn_layers": [i for i in range(45) if i % 4 == 3],
    },
)
FLASH = SimpleNamespace(
    text_config=TEXT, architectures=["Glm5NextForConditionalGeneration"]
)


def test_flash_spans():
    # the MoE layers but the last, whose FFN exit contracts the streams: the
    # dense layers 0 to 2 and layer 44 keep the normal path
    spans = flash.SPEC.spans(FLASH)
    assert spans == {i: (flash.LAYER,) for i in range(3, 44)}
    assert (flash.LAYER.start, flash.LAYER.end) == (
        SpanPoint.LAYER_IN,
        SpanPoint.LAYER_OUT,
    )


def test_registry_entry():
    spec, why = registry.find_spec(["Glm5NextForConditionalGeneration"], "gfx950")
    assert spec is flash.SPEC, why
    spec, why = registry.find_spec(["Glm5NextForConditionalGeneration"], "gfx942")
    assert spec is None and "gfx942" in why


def test_tp2_routes_at_most_four_rows():
    # TP2 builds no 8-row launch (it is slower than the normal path); a decode step
    # of fewer rows than a width runs at the next width, the rest pad rows
    runtimes = {(w, k): None for w in (2, 4) for k in ("kda", "dsa")}
    layers = {i: SimpleNamespace(i=i) for i in range(3, 44)}
    built = flash._Built(layers, runtimes, tp=2, device="cpu")
    decode = lambda rows: ShapeKey("decode", rows, 1, rows)  # noqa: E731
    for rows in (1, 2, 3, 4):
        assert built.kernel(3, SpanPoint.LAYER_IN, decode(rows)) is not None, rows
    assert [flash._width(r) for r in (1, 2, 3, 4, 5, 8)] == [2, 2, 4, 4, 8, 8]
    assert built.kernel(3, SpanPoint.LAYER_IN, decode(5)) is None
    assert built.kernel(3, SpanPoint.LAYER_IN, decode(8)) is None
    assert (
        built.kernel(3, SpanPoint.LAYER_IN, ShapeKey("target_verify", 2, 2, 4)) is None
    )
    assert built.kernel(44, SpanPoint.LAYER_IN, decode(4)) is None
    # the first covered layer's identity mix leaves the written streams unchanged
    assert torch.equal(built.eye_comb[1], torch.eye(4)) and not built.zero_x.any()


def _cfg(**kw):
    return SimpleNamespace(**{"page_size": 256, "kv_cache_dtype": "bfloat16", **kw})


def _dtypes(temporal=torch.float32, conv=torch.bfloat16):
    return lambda config: SimpleNamespace(temporal=temporal, conv=conv)


def test_config_reasons_name_each_mismatch(monkeypatch):
    import sglang.srt.configs.mamba_utils as mamba_utils

    monkeypatch.setattr(mamba_utils, "mamba2_state_dtype", _dtypes())
    assert flash.SPEC.config_disable_reasons(_cfg(), FLASH) == []
    # the last KDA layer (42 here) covered: its normal path tracks every layer's state
    kda = [i for i in TEXT.linear_attn_config["kda_layers"] if i < 44]
    text = SimpleNamespace(
        **{
            **vars(TEXT),
            "hidden_size": 6144,
            "linear_attn_config": {
                **TEXT.linear_attn_config,
                "num_heads": 32,
                "kda_layers": kda,
            },
        }
    )
    monkeypatch.setattr(
        mamba_utils, "mamba2_state_dtype", _dtypes(temporal=torch.bfloat16)
    )
    other = SimpleNamespace(text_config=text, architectures=FLASH.architectures)
    reasons = flash.SPEC.config_disable_reasons(_cfg(page_size=64), other)
    joined = " | ".join(reasons)
    for want in (
        "hidden_size = 4096",
        "num_heads = 64",
        "layer 42: the last KDA layer",
        "KDA state in f32",
        "256",
    ):
        assert want in joined, joined
    assert len(reasons) == 5, reasons


def _layer(kda=True, shuffled=True, scale_dims=3, fused=True, indexer=True):
    w13 = SimpleNamespace(is_shuffled=shuffled)
    experts = SimpleNamespace(
        w13_weight=w13, w13_weight_scale=torch.zeros((1,) * scale_dims)
    )
    attn = SimpleNamespace(
        do_fuse_qkvbfg=fused, indexer=SimpleNamespace() if indexer else None
    )
    return SimpleNamespace(
        is_linear_attn=kda, self_attn=attn, mlp=SimpleNamespace(experts=experts)
    )


def test_model_reasons():
    kda = TEXT.linear_attn_config["kda_layers"]
    layers = [_layer(kda=i in kda) for i in range(45)]
    model = SimpleNamespace(layers=layers)
    assert flash.SPEC.model_disable_reasons(model, FLASH, {}) == []
    # layer 4's experts are not shuffled, layer 5's are GU-interleaved (2-D
    # scales), layer 6 (KDA) runs unfused projections, layer 7 has no indexer
    layers[4] = _layer(shuffled=False)
    layers[5] = _layer(scale_dims=2)
    layers[6] = _layer(fused=False)
    layers[7] = _layer(kda=False, indexer=False)
    reasons = flash.SPEC.model_disable_reasons(model, FLASH, {})
    assert [r.split(":")[0] for r in reasons] == [
        "layer 4",
        "layer 5",
        "layer 6",
        "layer 7",
    ], reasons
    assert "shuffled" in reasons[0] and "GU-interleaved" in reasons[1]

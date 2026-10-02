"""MlxBackend against TorchBackend on the tiny hybrid LoRA checkpoint (`tiny_decider`).

Apple silicon only: skipped elsewhere, and where mlx-lm is not installed
(pip install 'strands-decider[mlx]').
"""

from __future__ import annotations

import platform
import sys

import pytest
import torch

if sys.platform != "darwin" or platform.machine() != "arm64":
    pytest.skip("MLX needs an Apple-silicon Mac", allow_module_level=True)
pytest.importorskip("mlx_lm")

from strands_decider.infer import (
    EngineConfig,
    SystemOneEngine,
    TorchBackend,
    make_torso_backend,
)
from strands_decider.mlx_backend import MlxBackend
from strands_decider.prompting import render_question, render_state


def _inputs(eng, request):
    rendered = [render_question(q) for q in request.questions.values()]
    s, q = eng._fit(render_state(request.state), [rq.text for rq in rendered])
    return s, q, eng._positions(rendered, q)


def _max_answer_delta(a, b):
    deltas = []
    for name, x in a.answers.items():
        y = b.answers[name]
        if x.type == "noul":
            deltas.append(abs(x.noul - y.noul))
            continue
        deltas += [abs(p - y.probabilities[k]) for k, p in x.probabilities.items()]
        if x.type == "score":
            deltas.append(abs(x.score - y.score))
    return max(deltas)


@pytest.mark.parametrize("share_prefix", [True, False], ids=["shared", "batched"])
@torch.inference_mode()
def test_hidden_states_match_torch(tiny_decider, share_prefix):
    model, request = tiny_decider
    eng = SystemOneEngine(model, EngineConfig(device="cpu"))
    s, q, pos = _inputs(eng, request)
    mlx = MlxBackend.from_model(model)
    assert mlx.name == "mlx" and mlx.hidden_size == 64

    got = mlx.hidden_states(s, q, pos, share_prefix=share_prefix)
    want = TorchBackend(model, "cpu").hidden_states(s, q, pos, share_prefix=share_prefix)
    assert got.dtype == torch.float32 and got.device.type == "cpu"
    torch.testing.assert_close(got, want, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("use_prefix_cache", [True, False], ids=["shared", "batched"])
def test_engine_answers_match_torch(tiny_decider, use_prefix_cache):
    """End to end, through the same factory load_engine and serve use."""
    model, request = tiny_decider
    cfg = EngineConfig(device="cpu", use_prefix_cache=use_prefix_cache)
    want = SystemOneEngine(model, cfg).evaluate(request)

    backend = make_torso_backend("mlx", model)
    assert isinstance(model.torso, torch.nn.Identity)  # the torch torso was dropped
    eng = SystemOneEngine(model, cfg, backend=backend)
    got = eng.evaluate(request)

    assert eng.cfg.use_prefix_cache is use_prefix_cache  # the MLX cache forked
    assert got.usage == want.usage
    assert _max_answer_delta(got, want) < 1e-3


@torch.inference_mode()
def test_the_lora_adapter_is_merged(tiny_decider):
    model, request = tiny_decider
    eng = SystemOneEngine(model, EngineConfig(device="cpu"))
    s, q, pos = _inputs(eng, request)
    got = MlxBackend.from_model(model).hidden_states(s, q, pos, share_prefix=False)
    with model.torso.disable_adapter():
        base = TorchBackend(model, "cpu").hidden_states(s, q, pos, share_prefix=False)
    assert (got - base).abs().max() > 1e-2


def test_a_non_qwen35_torso_is_refused():
    from test_checkpoint_load import _tokenizer
    from transformers import Qwen3Config, Qwen3Model

    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel

    torso = Qwen3Model(Qwen3Config(vocab_size=6, hidden_size=32, intermediate_size=64,
                                   num_hidden_layers=1, num_attention_heads=4,
                                   num_key_value_heads=2, head_dim=8))
    model = StrandsDeciderModel(StrandsDeciderConfig(base_model="tiny", head_type="pointer",
                                                     pointer_dim=16, torch_dtype="float32"),
                                torso, _tokenizer())
    with pytest.raises(ValueError, match=r"Qwen3\.5 torsos only"):
        MlxBackend.from_model(model)

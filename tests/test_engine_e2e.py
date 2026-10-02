"""SystemOneEngine.evaluate end to end on CPU, on a tiny hybrid Qwen3.5 pointer checkpoint
with a LoRA adapter (the `tiny_decider` fixture in conftest.py).

These pin what the engine returns, not how it computes it: the shared-prefix path against
batched encoding, and both against an oracle built from StrandsDeciderModel.forward on each
full prompt. tests/test_gpu_smoke.py checks the same on a real model, but only with CUDA.
"""

from __future__ import annotations

import pytest
import torch

from strands_decider.infer import EngineConfig, SystemOneEngine, _option_token_index, _to_answer
from strands_decider.prompting import render_question, render_state


def _max_answer_delta(a, b):
    """Largest difference between two responses over every number they report."""
    deltas = []
    for name, x in a.answers.items():
        y = b.answers[name]
        assert x.type == y.type, name
        if x.type == "noul":
            deltas.append(abs(x.noul - y.noul))
            continue
        assert set(x.probabilities) == set(y.probabilities), name
        deltas += [abs(p - y.probabilities[k]) for k, p in x.probabilities.items()]
        if x.type == "score":
            deltas.append(abs(x.score - y.score))
    return max(deltas)


def _fit(eng, request):
    rendered = [render_question(q) for q in request.questions.values()]
    s, q = eng._fit(render_state(request.state), [rq.text for rq in rendered])
    return rendered, s, q


def test_evaluate_answers_every_question_on_cpu(tiny_decider):
    model, request = tiny_decider
    eng = SystemOneEngine(model, EngineConfig(device="cpu"))
    resp = eng.evaluate(request)

    assert {n: a.type for n, a in resp.answers.items()} == {
        "is_urgent": "noul", "department": "choice", "frustration": "score"}
    assert 0.0 <= resp.answers["is_urgent"].noul <= 1.0
    for name in ("department", "frustration"):
        assert sum(resp.answers[name].probabilities.values()) == pytest.approx(1.0, abs=1e-3)
    assert set(resp.answers["department"].probabilities) == {"billing", "technical", "sales"}

    _, s, q = _fit(eng, request)
    assert resp.usage.input_tokens == len(s) + sum(map(len, q))
    assert resp.usage.output_tokens == 3


def test_shared_prefix_matches_batched_on_a_hybrid_lora_checkpoint(tiny_decider):
    """The CPU analogue of test_gpu_smoke.py::test_prefix_cache_is_exact_in_fp32."""
    model, request = tiny_decider
    cached_eng = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=True))
    naive_eng = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=False))
    cached = cached_eng.evaluate(request)
    naive = naive_eng.evaluate(request)

    assert cached_eng.cfg.use_prefix_cache  # the hybrid cache forked; no silent fallback
    assert _max_answer_delta(cached, naive) < 1e-4
    _, s, q = _fit(cached_eng, request)
    assert cached.usage.input_tokens == len(s) + sum(map(len, q))
    assert naive.usage.input_tokens == sum(len(s) + len(qi) for qi in q)


@pytest.mark.parametrize("use_prefix_cache", [True, False], ids=["shared", "batched"])
@torch.inference_mode()
def test_engine_matches_the_model_forward(tiny_decider, use_prefix_cache):
    """Oracle: StrandsDeciderModel.forward on each full prompt, one question at a time."""
    model, request = tiny_decider
    eng = SystemOneEngine(model, EngineConfig(device="cpu", use_prefix_cache=use_prefix_cache))
    got = eng.evaluate(request)

    rendered, s, q = _fit(eng, request)
    offsets = eng._last_offsets
    cfg = eng.model.config
    answers = {}
    for name, rq, qi, offs in zip(request.questions, rendered, q, offsets, strict=True):
        ids = torch.tensor([s + qi])
        opt = [len(s) + j for j in _option_token_index(offs, rq.option_spans, 0)]
        out = eng.model(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            n_slots=torch.tensor([rq.n_slots]),
            temperature=torch.tensor([float(cfg.temperature_by_kind.get(rq.kind, cfg.temperature))]),
            opt_idx=torch.tensor([opt]),
        )
        row = out["log_probs"].exp()[0, : rq.n_slots].tolist()
        answers[name] = _to_answer(rq, row, ordinal_smoothing=cfg.ordinal_smoothing)

    want = type(got)(model=got.model, answers=answers, usage=got.usage)
    assert _max_answer_delta(got, want) < 1e-4

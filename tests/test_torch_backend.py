"""TorchBackend.hidden_states: both paths return the torso's final hidden states at the
asked positions, on the tiny hybrid LoRA checkpoint (the `tiny_decider` fixture)."""

from __future__ import annotations

import torch

from strands_decider.infer import EngineConfig, SystemOneEngine, TorchBackend
from strands_decider.prompting import render_question, render_state


def _inputs(eng, request):
    rendered = [render_question(q) for q in request.questions.values()]
    s, q = eng._fit(render_state(request.state), [rq.text for rq in rendered])
    return s, q, eng._positions(rendered, q)


@torch.inference_mode()
def test_both_paths_return_the_full_prompt_states(tiny_decider):
    model, request = tiny_decider
    eng = SystemOneEngine(model, EngineConfig(device="cpu"))
    assert isinstance(eng.backend, TorchBackend) and eng.backend.name == "torch"
    assert eng.backend.hidden_size == 64
    s, q, pos = _inputs(eng, request)
    assert {len(row) for row in pos} == {4}  # pooling position + up to 3 options

    shared = eng.backend.hidden_states(s, q, pos, share_prefix=True)
    batched = eng.backend.hidden_states(s, q, pos, share_prefix=False)
    assert shared.dtype == batched.dtype == torch.float32
    assert shared.shape == (len(q), 4, 64)
    torch.testing.assert_close(shared, batched, atol=1e-4, rtol=1e-4)

    for i, qi in enumerate(q):
        ids = torch.tensor([s + qi])
        whole = model.encode(ids, torch.ones_like(ids))[0]
        ref = whole[[len(s) + j for j in pos[i]]]
        torch.testing.assert_close(shared[i], ref, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(batched[i], ref, atol=1e-4, rtol=1e-4)

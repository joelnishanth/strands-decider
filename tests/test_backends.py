"""Choosing a torso backend: the factory, the missing-MLX error and the CLI's --backend.

Runs everywhere: nothing here needs MLX or loads a model.
"""

from __future__ import annotations

import importlib.util

import pytest
from typer.testing import CliRunner

import strands_decider.infer as infer
from strands_decider.cli import app
from strands_decider.mlx_backend import MlxBackend
from strands_decider.schema import NoulAnswer, SystemOneResponse, Usage


def test_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="unknown backend 'bogus'; choose one of torch, mlx"):
        infer.make_torso_backend("bogus", object())


def test_torch_backend_is_the_engine_default():
    assert infer.make_torso_backend("torch", object()) is None


def test_missing_mlx_names_the_extra(monkeypatch):
    monkeypatch.setattr(importlib.util, "find_spec", lambda name, *a: None)
    with pytest.raises(RuntimeError, match=r"pip install 'strands-decider\[mlx\]'"):
        MlxBackend.from_model(object())


@pytest.fixture
def loads(monkeypatch):
    """Every load_engine call the CLI makes, answered by a stub engine."""
    calls = []

    class _Engine:
        def ask(self, state, questions):
            return SystemOneResponse(model="stub", answers={n: NoulAnswer(noul=0.5) for n in questions},
                                     usage=Usage(input_tokens=1, output_tokens=len(questions)))

    def load_engine(checkpoint, **kw):
        calls.append(kw)
        return _Engine()

    monkeypatch.setattr(infer, "load_engine", load_engine)
    return calls


@pytest.mark.parametrize("extra", [["--backend", "mlx", "--device", "mps"], ["--backend", "bogus"]],
                         ids=["mlx-with-mps", "bogus"])
def test_ask_refuses_a_bad_backend_before_loading(loads, extra):
    res = CliRunner().invoke(app, ["ask", "ckpt", "-s", "x", "--noul", "q", *extra])
    assert res.exit_code == 2, res.output
    assert loads == []


def test_serve_refuses_mlx_with_a_gpu_device(monkeypatch):
    import strands_decider.server as server

    monkeypatch.setattr(server, "serve", lambda *a, **k: pytest.fail("served"))
    res = CliRunner().invoke(app, ["serve", "ckpt", "--backend", "mlx", "--device", "cuda"])
    assert res.exit_code == 2, res.output


def test_ask_with_mlx_reads_out_on_cpu(loads):
    res = CliRunner().invoke(app, ["ask", "ckpt", "-s", "x", "--noul", "q", "--backend", "mlx", "--json"])
    assert res.exit_code == 0, res.output
    assert loads == [{"device": "cpu", "backend": "mlx"}]

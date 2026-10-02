"""Shared test setup.

Keeps the suite quiet and deterministic: the Hub's progress bars and warnings
otherwise bury the actual test output, and an unseeded torch makes every test that
draws random tensors produce different numbers run to run.
"""

from __future__ import annotations

import os

import pytest

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@pytest.fixture(autouse=True)
def _deterministic():
    import torch

    torch.manual_seed(0)
    yield


# The 12 v19 LoRA targets (configs/train.yaml): attention and MLP projections plus the
# Gated DeltaNet layers' own.
V19_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                    "in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b", "out_proj"]


def _tiny_request():
    from strands_decider.schema import ChoiceQuestion, NoulQuestion, ScoreQuestion, SystemOneRequest

    return SystemOneRequest(
        state="Help my payouts failed for three days and nobody replied.",
        questions={
            "is_urgent": NoulQuestion(instructions="Is this urgent?"),
            "department": ChoiceQuestion(
                instructions="Which team should handle this?",
                criteria={"billing": "payouts and invoices", "technical": "bugs",
                          "sales": "pricing"},
            ),
            "frustration": ScoreQuestion(
                instructions="How frustrated is the writer?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        },
    )


REQUEST = _tiny_request()


def _tiny_tokenizer(texts):
    """A WordLevel tokenizer whose vocabulary covers `texts` exactly, so no token is <unk>."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    words = sorted({w for t in texts for w, _ in pre_tokenizers.Whitespace().pre_tokenize_str(t)})
    vocab = {w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", *words])}
    assert len(vocab) < 256, len(vocab)
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>",
                                   unk_token="<unk>")


@pytest.fixture
def tiny_decider(tmp_path, monkeypatch):
    """(a loaded tiny pointer-head checkpoint, REQUEST).

    A random hybrid Qwen3.5 text torso (three Gated DeltaNet layers, one attention layer)
    with a non-zero LoRA adapter on the 12 v19 targets, saved and reloaded through
    StrandsDeciderModel.load, so PeftModel.from_pretrained(is_trainable=False) builds the
    torso the engine runs. CPU only; nothing is downloaded. The 32-wide linear-attention
    heads suit MLX's gated-delta kernel, and initializer_range=0.2 keeps the hidden states
    far enough from zero that norm-epsilon differences between runtimes stay small.
    """
    import copy

    import torch
    import transformers

    from strands_decider.modeling import StrandsDeciderConfig, StrandsDeciderModel
    from strands_decider.prompting import render_question, render_state

    torch.manual_seed(0)
    cfg = transformers.Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=32,
        linear_value_head_dim=32, layer_types=["linear_attention"] * 3 + ["full_attention"],
        initializer_range=0.2)
    base = transformers.Qwen3_5ForCausalLM(cfg).model.eval()

    texts = [render_state(REQUEST.state)] + [render_question(q).text
                                             for q in REQUEST.questions.values()]
    tok = _tiny_tokenizer(texts)
    dcfg = StrandsDeciderConfig(
        base_model="tiny", head_type="pointer", pointer_dim=16, lora_r=2, lora_alpha=4,
        torch_dtype="float32", max_length=256,
        temperature_by_kind={"noul": 1.5, "choice": 0.8}, lora_targets=list(V19_LORA_TARGETS))
    model = StrandsDeciderModel(dcfg, copy.deepcopy(base), tok)
    model.attach_lora()
    with torch.no_grad():  # lora_B starts at zero: make the adapter change the torso
        for n, p in model.torso.named_parameters():
            if "lora_B" in n:
                p.normal_(0, 0.05)
        # A fresh pointer head answers almost uniformly; sharpen it so the answer
        # comparisons notice a wrong hidden state.
        for p in model.head.parameters():
            p.mul_(2)
    path = tmp_path / "tiny-ckpt"
    model.save_pretrained(str(path))
    monkeypatch.setattr(StrandsDeciderModel, "_load_torso",
                        staticmethod(lambda config, device_map, attn: copy.deepcopy(base)))
    return StrandsDeciderModel.load(str(path)), REQUEST

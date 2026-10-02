"""Serving one request: N questions against one state.

The headline property of a System One model is that asking more questions about the
same state barely costs more time. That is not automatic -- it falls out of doing
the work in the right order:

    1. Encode the state ONCE, keeping its KV cache.          <- the expensive part
    2. Broadcast that cache across the question batch.
    3. Forward only the (short) question suffixes.           <- the cheap part

For a 2000-token state and five 40-token questions, the naive approach encodes
~10,200 tokens; this encodes ~2,200. The questions genuinely evaluate in parallel,
independently -- no question can see another's text, which is what keeps the answers
decomposable in the way the API promises.

Set `use_prefix_cache=False` to fall back to plain batched encoding. The two paths
agree to within float tolerance (tests/test_prefix_cache.py pins that, for plain and hybrid torsos), so the
fallback is a safety valve rather than a different model.

Steps 1-3 run behind a `TorsoBackend` (`TorchBackend` by default); the readout stays in the engine.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any, Protocol

import torch

from .modeling import StrandsDeciderModel, apply_temperature, gather_options, masked_log_softmax
from .prompting import (
    RenderedQuestion,
    read_choice,
    read_noul,
    read_score,
    render_question,
    render_state,
    score_legend,
)
from .schema import (
    Answer,
    ChoiceAnswer,
    Content,
    NoulAnswer,
    Question,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
    Usage,
    derive_confidence,
    derive_score_confidence,
)


@dataclass
class EngineConfig:
    device: str = "cuda"
    use_prefix_cache: bool = True
    max_batch: int = 32
    model_name: str = "strands-decider-0.1.0"
    # Largest share of the context window the question may claim before the state
    # starts being squeezed. Questions are normally short, so this rarely binds.
    max_question_fraction: float = 0.75
    # Refuse a prompt that does not fit the window instead of shortening it, for
    # benchmarks that forbid truncation. The message names the "context window".
    strict_window: bool = False


class UnforkableCache(TypeError):
    """A KV cache layout `_expand_cache` cannot safely repeat across rows."""


# Per-row state a cache layer may hold (transformers 5): attention layers' `keys` and
# `values`; a linear-attention (Gated DeltaNet) layer's `conv_states` and
# `recurrent_states`, each a dict of tensors keyed by state index. First dim is the batch.
_ROW_STATES = ("keys", "values", "conv_states", "recurrent_states")


def _fork_layered_cache(cache: Any, n: int) -> Any:
    """`n` copies of a batch-1 cache that keeps one object per layer (transformers 5).

    Covers hybrid torsos: Qwen3.5 mixes attention layers with Gated DeltaNet layers
    whose recurrent and convolution states must be repeated too, or the suffix forward
    fails on a shape mismatch. The fork gets new layer objects, new dicts and new
    tensors: a DeltaNet layer updates its states and flags (`has_previous_state`) in
    place, so a fork that shared them would corrupt the prefix it came from. The
    approach is decider-2b's (`decider/shared_prefix.py`, Apache-2.0).

    Raises UnforkableCache for a layer holding tensors under any other name: we cannot tell
    whether such a tensor has a batch dimension, so the caller falls back to batched
    encoding rather than guess.
    """
    fork = copy.copy(cache)
    fork.layers = []
    for layer in cache.layers:
        nl = copy.copy(layer)
        for name, v in list(vars(nl).items()):
            is_state = name in _ROW_STATES
            if isinstance(v, torch.Tensor):
                if not is_state:
                    raise UnforkableCache(f"cache layer {type(layer).__name__} holds tensor {name!r}")
                if v.numel():
                    setattr(nl, name, v.expand(n, *v.shape[1:]).contiguous())
            elif isinstance(v, dict):
                if any(isinstance(t, torch.Tensor) for t in v.values()) and not is_state:
                    raise UnforkableCache(f"cache layer {type(layer).__name__} holds tensors in {name!r}")
                setattr(nl, name, {k: t.expand(n, *t.shape[1:]).contiguous()
                                   if isinstance(t, torch.Tensor) and t.numel() else t
                                   for k, t in v.items()})
        fork.layers.append(nl)
    return fork


def _expand_cache(cache: Any, n: int) -> Any:
    """Repeat a batch-1 KV cache across `n` rows.

    transformers 5 keeps a per-layer object per layer (`cache.layers`), for plain and
    hybrid torsos alike; that is the only layout this package admits. `.contiguous()` is
    not optional: the suffix forward concatenates into these tensors, and an expanded
    (stride-0) view cannot be written to.
    """
    if isinstance(getattr(cache, "layers", None), list):
        return _fork_layered_cache(cache, n)
    raise UnforkableCache(f"unsupported KV cache type: {type(cache)!r}")


class TorsoBackend(Protocol):
    """Runs the torso: token ids in, final-layer hidden states out.

    The engine keeps everything else (tokenising, option positions, the head,
    temperature, softmax), so a backend never sees text and every backend is read out
    by the same fp32 head.
    """

    name: str  # "torch" | "mlx", reported by /health

    @property
    def hidden_size(self) -> int: ...

    def hidden_states(
        self,
        prefix: list[int],
        suffixes: list[list[int]],
        positions: list[list[int]],
        *,
        share_prefix: bool,
    ) -> torch.Tensor:
        """float32 [M, P, d]: row i, column j is the hidden state of prompt
        `prefix + suffixes[i]` at suffix-relative position `positions[i][j]`, after the
        final norm, with the LoRA adapter applied.

        `share_prefix=True` encodes the prefix once and forks its cache across the M
        rows; a backend that cannot fork raises UnforkableCache, and the engine falls
        back to `share_prefix=False`, which encodes each full prompt independently.
        """
        ...


class TorchBackend:
    """The torso in torch: the default backend, and the reference the others match.

    Holds the model rather than its torso and reads `model.torso` on every call, so an
    in-place cast of the torso (evaluation/bench_local.py) still takes effect.
    """

    name = "torch"

    def __init__(self, model: StrandsDeciderModel, device: str):
        self.model = model
        self.device = device

    @property
    def hidden_size(self) -> int:
        return StrandsDeciderModel.hidden_size(self.model.torso)

    def _pad(self, seqs: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
        """Right-pad to a rectangle, with the mask marking the real tokens."""
        tok = self.model.tokenizer
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
        width = max(len(x) for x in seqs)
        ids = torch.tensor(
            [x + [pad_id] * (width - len(x)) for x in seqs], device=self.device
        )
        mask = torch.tensor(
            [[1] * len(x) + [0] * (width - len(x)) for x in seqs], device=self.device
        )
        return ids, mask

    @torch.inference_mode()  # type: ignore[untyped-decorator, unused-ignore]
    def hidden_states(
        self,
        prefix: list[int],
        suffixes: list[list[int]],
        positions: list[list[int]],
        *,
        share_prefix: bool,
    ) -> torch.Tensor:
        pos = torch.tensor(positions, dtype=torch.long, device=self.device)
        if not share_prefix:
            ids, mask = self._pad([prefix + s for s in suffixes])
            hidden = self.model.encode(ids, mask)
            # The whole prompt was forwarded, so the positions sit after the state.
            return gather_options(hidden, pos + len(prefix)).to(torch.float32)

        m = len(suffixes)
        prefix_ids = torch.tensor([prefix], device=self.device)
        prefix_len = prefix_ids.size(1)

        prefix_out = self.model.torso(
            input_ids=prefix_ids,
            attention_mask=torch.ones_like(prefix_ids),
            use_cache=True,
            return_dict=True,
        )
        cache = _expand_cache(prefix_out.past_key_values, m)

        suffix_ids, suffix_mask = self._pad(suffixes)
        full_mask = torch.cat(
            [
                torch.ones(m, prefix_len, dtype=suffix_mask.dtype, device=self.device),
                suffix_mask,
            ],
            dim=1,
        )

        hidden = self.model.encode(
            input_ids=suffix_ids,
            attention_mask=full_mask,
            past_key_values=cache,
        )
        # `hidden` is the suffix only, so the positions need no prefix offset -- the
        # cached state never enters the gather.
        return gather_options(hidden, pos).to(torch.float32)


class SystemOneEngine:
    """Stateless evaluator around a loaded StrandsDeciderModel."""

    def __init__(
        self,
        model: StrandsDeciderModel,
        config: EngineConfig | None = None,
        backend: TorsoBackend | None = None,
    ):
        self.cfg = config or EngineConfig()
        if backend is None:
            if str(self.cfg.device).startswith("mps"):
                # No fla/Triton on macOS; replace the reference chunk rule's slow MPS solver.
                from .mps_kernels import install

                install()
            self.model = model.to(self.cfg.device).eval()
            if str(self.cfg.device) == "cpu":
                self._upcast_torso_for_cpu()
            self.backend: TorsoBackend = TorchBackend(self.model, self.cfg.device)
        else:
            # The backend owns the torso; only the readout runs here, on cfg.device.
            self.model = model.to(self.cfg.device).eval()
            self.backend = backend
        self.tok = model.tokenizer
        self.device = self.cfg.device

    def _upcast_torso_for_cpu(self) -> None:
        """Run a half-precision torso in fp32 on CPU.

        CPU kernels for bf16 are slower than fp32, not faster: on an M3 Pro a
        256-token v19 question takes 7.7 s in bf16 and 3.5 s in fp32, with the same
        answer. The cost is memory, about 7 GiB of torso weights instead of 3.5. The
        readout is already fp32. Done under inference_mode because `load_engine`
        builds the model there, and inference tensors cannot be modified outside it.
        """
        torso = self.model.torso
        first = next(torso.parameters(), None)  # a stub torso in the tests has none
        if first is not None and first.dtype in (torch.bfloat16, torch.float16):
            with torch.inference_mode():
                torso.to(torch.float32)

    def _temperatures(self, kinds: list[str]) -> torch.Tensor:
        """Per-row temperature, falling back to the global scalar where unfitted."""
        cfg = self.model.config
        by_kind = getattr(cfg, "temperature_by_kind", None) or {}
        return torch.tensor(
            [float(by_kind.get(k, cfg.temperature)) for k in kinds],
            device=self.device,
            dtype=torch.float32,
        )


    def _fit(self, state_text: str, question_texts: list[str]) -> tuple[list[int], list[list[int]]]:
        """Tokenise state and questions, giving the QUESTION first claim on the window.

        The question and its options are what make a task answerable; the state is the
        part that can be sampled. Letting the state take the window first meant a long
        state consumed all of it and left the question a floor of 8 tokens -- far too
        few to hold the option list, so the model was choosing among options it could
        not see. Measured on JevBench, all 19 `long_policy` states hit the cap exactly,
        and accuracy there (0.316) sat on top of the 0.301 chance rate for those option
        counts.

        A question longer than its reserve is truncated from the FRONT, keeping the
        tail. The options and the trailing `<answer>` marker are structurally required
        -- `<answer>` is the pooling position, so losing it makes the head read an
        arbitrary token -- whereas losing some instruction text only costs meaning.
        """
        max_len = self.model.config.max_length
        enc = self.tok(question_texts, add_special_tokens=False,
                       return_offsets_mapping=True)
        q = enc["input_ids"]
        offs = enc["offset_mapping"]
        longest = max(len(x) for x in q)
        if self.cfg.strict_window:
            s = self.tok(state_text, add_special_tokens=True)["input_ids"]
            if len(s) + longest > max_len:
                raise ValueError(
                    f"prompt of {len(s) + longest} tokens exceeds the context window "
                    f"of {max_len} tokens"
                )
            self._last_offsets = list(offs)
            return s, q
        # Cap the reserve so a pathological question cannot starve the state entirely.
        reserve = min(longest, max(1, int(max_len * self.cfg.max_question_fraction)))
        # Front truncation shifts every token index, so the offsets move with the ids
        # and a pointer readout keeps pointing at the right option.
        cut = [max(0, len(x) - reserve) for x in q]
        self._last_offsets = [o[c:] for o, c in zip(offs, cut, strict=True)]
        q = [x[c:] for x, c in zip(q, cut, strict=True)]
        state_budget = max(1, max_len - reserve)
        s = self.tok(
            state_text,
            add_special_tokens=True,
            truncation=True,
            max_length=state_budget,
        )["input_ids"]
        return s, q

    def _positions(
        self, rendered: list[RenderedQuestion], q: list[list[int]]
    ) -> list[list[int]]:
        """Suffix-relative positions to read from each question's hidden states.

        Column 0 is the pooling position: the last question token, `<answer>`, the
        position that has attended to state and question alike. For a pointer head,
        columns 1.. are the option positions, from the offsets `_fit` kept so front
        truncation is accounted for. Short rows are padded with 0; `masked_log_softmax`
        drops every option past the row's count.
        """
        pool = [[len(x) - 1] for x in q]
        if self.model.config.head_type != "pointer":
            return pool
        rows = [
            _option_token_index(offs, rq.option_spans, 0)
            for rq, offs in zip(rendered, self._last_offsets, strict=True)
        ]
        width = max(len(r) for r in rows)
        return [p + r + [0] * (width - len(r)) for p, r in zip(pool, rows, strict=True)]

    # ---- low level -------------------------------------------------------

    @torch.inference_mode()  # type: ignore[untyped-decorator, unused-ignore]
    def _slot_probs(
        self,
        state_text: str,
        question_texts: list[str],
        n_slots: list[int],
        kinds: list[str],
        rendered: list[RenderedQuestion],
        *,
        share_prefix: bool,
    ) -> tuple[torch.Tensor, int]:
        """Option probabilities per question, and the input tokens the torso forwarded.

        `share_prefix=True` encodes the state once and every question suffix against
        that cache; False encodes each full prompt independently. Both go through
        `_fit`, so they truncate the same thing in the same direction.
        """
        # The suffixes continue an already-tokenised sequence, so _fit adds no BOS to
        # them -- one mid-sequence would be a token training never saw.
        s, q = self._fit(state_text, question_texts)
        positions = self._positions(rendered, q)
        h = self.backend.hidden_states(s, q, positions, share_prefix=share_prefix)
        h = h.to(self.device)
        pooled = h[:, 0]
        if self.model.config.head_type == "pointer":
            raw = self.model.head(pooled, h[:, 1:])
        else:
            raw = self.model.head(pooled)
        logits = apply_temperature(raw, self._temperatures(kinds))
        log_probs = masked_log_softmax(logits, torch.tensor(n_slots, device=self.device))
        if share_prefix:
            n_tokens = len(s) + sum(len(x) for x in q)
        else:
            n_tokens = sum(len(s) + len(x) for x in q)
        return log_probs.exp(), n_tokens

    # ---- public ----------------------------------------------------------

    def evaluate(self, request: SystemOneRequest) -> SystemOneResponse:
        names = list(request.questions.keys())
        questions: list[Question] = [request.questions[n] for n in names]

        rendered: list[RenderedQuestion] = []
        for q in questions:
            rq = render_question(q)
            # A pointer head scores each option from its own hidden state, so there is
            # no slot count to exceed; only the fixed-width readout has a ceiling.
            if (
                self.model.config.head_type != "pointer"
                and rq.n_slots > self.model.config.num_slots
            ):
                raise ValueError(
                    f"question has {rq.n_slots} options but this model has "
                    f"{self.model.config.num_slots} slots; split the question or "
                    f"retrain with a larger num_slots"
                )
            rendered.append(rq)

        state_text = render_state(request.state)
        n_slots = [rq.n_slots for rq in rendered]

        answers: dict[str, Answer] = {}
        total_tokens = 0

        for start in range(0, len(names), self.cfg.max_batch):
            chunk = slice(start, start + self.cfg.max_batch)
            chunk_rendered = rendered[chunk]
            chunk_slots = n_slots[chunk]
            chunk_kinds = [rq.kind for rq in chunk_rendered]

            probs = None
            # One question gains nothing from a shared prefix and pays a second forward:
            # measured on JevBench (one question per task), p50 0.111 s batched, 0.204 s
            # through the prefix path, with the same answers.
            if self.cfg.use_prefix_cache and len(chunk_rendered) > 1:
                try:
                    probs, ntok = self._slot_probs(
                        state_text, [rq.text for rq in chunk_rendered], chunk_slots,
                        chunk_kinds, chunk_rendered, share_prefix=True,
                    )
                    total_tokens += ntok
                except UnforkableCache as e:
                    print(f"[strands-decider] shared-prefix cache disabled ({e}); using batched encoding")
                    self.cfg = replace(self.cfg, use_prefix_cache=False)
            if probs is None:
                probs, ntok = self._slot_probs(
                    state_text, [rq.text for rq in chunk_rendered], chunk_slots,
                    chunk_kinds, chunk_rendered, share_prefix=False,
                )
                total_tokens += ntok

            for i, name in enumerate(names[chunk]):
                rq = chunk_rendered[i]
                row = probs[i, : rq.n_slots].tolist()
                answers[name] = _to_answer(
                    rq, row, ordinal_smoothing=self.model.config.ordinal_smoothing
                )

        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            # One slot decision per question: the output side really is this cheap.
            usage=Usage(input_tokens=total_tokens, output_tokens=len(names)),
        )

    def ask(self, state: Content, questions: dict[str, Question]) -> SystemOneResponse:
        return self.evaluate(SystemOneRequest(state=state, questions=questions))


def _option_token_index(
    offsets: Sequence[Sequence[int]],
    spans: Sequence[Sequence[int]],
    base: int,
) -> list[int]:
    """Last token index of each option's line, for a pointer readout.

    The last token of the span is the one that has just read the whole option under
    causal attention. Raises when an option has no surviving token: truncation has
    removed it, and scoring it from a neighbour's representation would be silently
    wrong.
    """
    out: list[int] = []
    for s, e in spans:
        a, b = base + s, base + e
        last = -1
        for j, (lo, hi) in enumerate(offsets):
            if hi <= lo:
                continue
            if lo >= a and hi <= b:
                last = j
        if last < 0:
            raise ValueError(
                f"option span ({a},{b}) has no tokens left; the prompt was "
                "truncated through its option list"
            )
        out.append(last)
    return out


def _to_answer(
    rq: RenderedQuestion, probs: list[float], *, ordinal_smoothing: float = 0.0
) -> Answer:
    if rq.kind == "noul":
        return NoulAnswer(noul=round(read_noul(probs, rq), 4))

    if rq.kind == "choice":
        by_name = read_choice(probs, rq)
        best = max(by_name, key=lambda k: by_name[k])
        return ChoiceAnswer(
            choice=best,
            probabilities={k: round(v, 4) for k, v in by_name.items()},
            confidence=round(derive_confidence(list(by_name.values())), 4),
        )

    score, ordered = read_score(probs, rq)
    return ScoreAnswer(
        score=round(score, 4),
        # Report the rubric in canonical ascending order, not the order rendered.
        legend=score_legend(rq),
        probabilities={k: round(v, 4) for k, v in ordered.items()},
        # Ordinal confidence, not max-probability: see derive_score_confidence.
        confidence=round(
            derive_score_confidence(
                list(ordered.values()), ordinal_smoothing=ordinal_smoothing
            ),
            4,
        ),
    )


BACKENDS = ("torch", "mlx")


def make_torso_backend(name: str, model: StrandsDeciderModel) -> TorsoBackend | None:
    """The backend named `name` for a loaded model; None means the engine's TorchBackend.

    For "mlx" the torso is converted to MLX and the torch torso is then dropped
    (`model.torso` becomes nn.Identity()), which frees its memory: nothing reads it again.
    """
    if name == "torch":
        return None
    if name == "mlx":
        from .mlx_backend import MlxBackend

        backend = MlxBackend.from_model(model)
        model.torso = torch.nn.Identity()
        return backend
    raise ValueError(f"unknown backend {name!r}; choose one of {', '.join(BACKENDS)}")


@torch.inference_mode()  # type: ignore[untyped-decorator, unused-ignore]
def load_engine(
    checkpoint: str,
    *,
    device: str = "cuda",
    use_prefix_cache: bool = True,
    attn_implementation: str | None = None,
    backend: str = "torch",
) -> SystemOneEngine:
    """Load a checkpoint and wrap it in an engine. With backend="mlx" the readout runs on
    CPU, whatever `device` says, and MLX runs the torso on the GPU."""
    if backend == "mlx":
        device = "cpu"
    model = StrandsDeciderModel.load(checkpoint, attn_implementation=attn_implementation)
    return SystemOneEngine(
        model, EngineConfig(device=device, use_prefix_cache=use_prefix_cache),
        backend=make_torso_backend(backend, model),
    )

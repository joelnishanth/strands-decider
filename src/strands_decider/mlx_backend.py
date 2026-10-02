"""The torso on Apple silicon through MLX (`--backend mlx`).

The weights come from the StrandsDeciderModel that `StrandsDeciderModel.load` already
built, not from a second download, so both backends share one loader and one checkpoint
check. The LoRA adapter is merged into the base weights here. The engine keeps the
readout (head, temperature, softmax) in torch on CPU.

mlx and mlx_lm are imported lazily, through importlib, so this module imports (and
type-checks) without them: `pip install 'strands-decider[mlx]'` on an Apple-silicon Mac.
"""

from __future__ import annotations

import importlib
import importlib.util
import platform
import sys
from typing import Any

import torch

from .infer import UnforkableCache
from .modeling import StrandsDeciderModel

_QWEN35 = {"qwen3_5_text", "qwen3_5"}


def is_available() -> bool:
    """True on an Apple-silicon Mac with mlx-lm installed."""
    return (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
        and importlib.util.find_spec("mlx_lm") is not None
    )


def _require_mlx() -> None:
    if not is_available():
        raise RuntimeError(
            "the mlx backend needs an Apple-silicon Mac with MLX installed: "
            "pip install 'strands-decider[mlx]'"
        )


def _full_attention_interval(layer_types: list[str]) -> int:
    """mlx_lm places a full-attention layer every k-th layer; HF lists each layer's type.

    Returns k, and refuses any pattern mlx_lm would build differently.
    """
    if "full_attention" not in layer_types:
        raise ValueError("the mlx backend needs at least one full_attention layer")
    if "linear_attention" not in layer_types:
        # sanitize() shifts the norm weights only when it sees DeltaNet conv weights.
        raise ValueError("the mlx backend needs at least one linear_attention layer")
    k = layer_types.index("full_attention") + 1
    want = ["full_attention" if (i + 1) % k == 0 else "linear_attention"
            for i in range(len(layer_types))]
    if layer_types != want:
        raise ValueError(
            f"layer_types {layer_types} is not one full_attention layer every {k} layers, "
            "the only layout mlx_lm's Qwen3.5 model builds"
        )
    return k


def _merged_torso_state(torso: Any) -> dict[str, torch.Tensor]:
    """The torso's state dict with each LoRA delta added to its base weight, in fp32.

    The PEFT modules are read, not changed: the torch torso stays usable (the parity
    tests run both). PEFT computes the delta, so its own scaling (alpha / r) applies.
    """
    from peft.tuners.lora import LoraLayer

    base: Any = torso.get_base_model() if hasattr(torso, "get_base_model") else torso
    state: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for key, value in base.state_dict().items():
            if ".lora_" in key:
                continue
            state[key.replace(".base_layer.", ".")] = value
        for name, module in base.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            lora: Any = module  # get_delta_weight lives on LoraLayer's subclasses
            weight = lora.get_base_layer().weight.detach().float()
            for adapter in lora.active_adapters:
                if adapter not in lora.lora_A or adapter in lora.merged_adapters:
                    continue
                if lora.use_dora.get(adapter, False):
                    raise NotImplementedError(f"{name}: DoRA adapters are not supported")
                weight = weight + lora.get_delta_weight(adapter).detach().float()
            state[f"{name}.weight"] = weight
    return {f"model.{k}": v for k, v in state.items()}


class MlxBackend:
    """A TorsoBackend running mlx_lm's Qwen3.5 text model."""

    name = "mlx"

    def __init__(self, text: Any):
        self.text = text  # mlx_lm.models.qwen3_5.TextModel, kept for make_cache()
        self.decoder = text.model

    @property
    def hidden_size(self) -> int:
        return int(self.text.args.hidden_size)

    @classmethod
    def from_model(cls, model: StrandsDeciderModel) -> MlxBackend:
        """Convert a loaded model's torso (Qwen3.5 text only) to MLX, LoRA merged."""
        _require_mlx()
        mx: Any = importlib.import_module("mlx.core")
        qwen3_5: Any = importlib.import_module("mlx_lm.models.qwen3_5")

        torso = model.torso
        cfg: Any = torso.config  # a PEFT wrapper forwards the base config
        if getattr(cfg, "model_type", None) not in _QWEN35:
            raise ValueError(
                "the mlx backend runs Qwen3.5 torsos only; this checkpoint's torso is "
                f"{getattr(cfg, 'model_type', None)!r}"
            )
        k = _full_attention_interval(list(cfg.layer_types))
        args = qwen3_5.TextModelArgs.from_dict(
            {**cfg.to_dict(), "tie_word_embeddings": True, "full_attention_interval": k}
        )
        if args.hidden_size != StrandsDeciderModel.hidden_size(torso):
            raise ValueError(
                f"hidden size {args.hidden_size} does not match the torso's "
                f"{StrandsDeciderModel.hidden_size(torso)}"
            )
        dtype = getattr(mx, model.config.torch_dtype)  # torch and MLX share the names
        text = qwen3_5.TextModel(args)
        keep_fp32 = text.cast_predicate

        # Cast matrices to the checkpoint dtype straight away, so peak memory stays near
        # one extra copy. Keep vectors in fp32 until sanitize() has shifted the
        # zero-centred RMSNorm weights by +1 (HF computes x * (1 + w), MLX x * w).
        weights: dict[str, Any] = {}
        for key, value in _merged_torso_state(torso).items():
            arr = mx.array(value.detach().float().numpy())
            weights[key] = arr.astype(dtype) if arr.ndim >= 2 else arr
        weights = text.sanitize(weights)
        weights = {key: arr.astype(dtype) if arr.ndim < 2 and keep_fp32(key) else arr
                   for key, arr in weights.items()}
        text.load_weights(list(weights.items()), strict=True)
        text.eval()  # training mode would disable the Metal gated-delta kernel
        mx.eval(text.parameters())
        return cls(text)

    def hidden_states(
        self,
        prefix: list[int],
        suffixes: list[list[int]],
        positions: list[list[int]],
        *,
        share_prefix: bool,
    ) -> torch.Tensor:
        """See TorsoBackend.hidden_states.

        Right padding needs no mask: attention, the DeltaNet conv1d and its recurrence
        are all causal, so the pad tokens after a row's real tokens never reach them.
        """
        mx: Any = importlib.import_module("mlx.core")
        np: Any = importlib.import_module("numpy")
        pos = mx.array(positions)
        if share_prefix:
            cache = self.text.make_cache()
            self.decoder(mx.array([prefix]), cache=cache)
            fork = [_fork(c, len(suffixes)) for c in cache]
            hidden = self.decoder(_right_pad(mx, suffixes), cache=fork)
        else:
            hidden = self.decoder(_right_pad(mx, [prefix + s for s in suffixes]))
            pos = pos + len(prefix)
        gathered = mx.take_along_axis(hidden, pos[..., None], axis=1)
        # numpy has no bfloat16, so cast in MLX before leaving it.
        return torch.from_numpy(np.array(gathered.astype(mx.float32)))


def _right_pad(mx: Any, seqs: list[list[int]]) -> Any:
    width = max(len(x) for x in seqs)
    return mx.array([x + [0] * (width - len(x)) for x in seqs])


def _fork(layer_cache: Any, n: int) -> Any:
    """`n` copies of a batch-1 mlx_lm layer cache, as new cache objects."""
    mx: Any = importlib.import_module("mlx.core")
    caches: Any = importlib.import_module("mlx_lm.models.cache")
    if type(layer_cache) is caches.KVCache:
        keys, values = layer_cache.state
        fork = caches.KVCache()
        fork.state = (mx.repeat(keys, n, 0), mx.repeat(values, n, 0))
        return fork
    if type(layer_cache) is caches.ArraysCache:
        states = layer_cache.state
        fork = caches.ArraysCache(len(states))
        fork.state = [None if s is None else mx.repeat(s, n, 0) for s in states]
        return fork
    raise UnforkableCache(f"unsupported mlx cache type: {type(layer_cache).__name__}")

"""Ollama-compatible `/api/*` routes over the System One engine.

A translation layer: Ollama's `format` JSON schema becomes the typed questions, the
conversation becomes the `state`, and the answers come back as a JSON string in the
reply's content. The decider generates no text, so a request without a usable schema
is refused. Wire shapes follow the Ollama API documentation (docs/inference.md,
"Ollama-compatible API").
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as package_version
from typing import Any, Literal, TypeVar

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from starlette.concurrency import run_in_threadpool

from .infer import SystemOneEngine
from .schema import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    Question,
    ScoreQuestion,
    SystemOneRequest,
)

EXT_KEY = "x-strands-decider"
# Structured outputs (`format` as a JSON schema) arrived in Ollama 0.5.0; clients that
# gate on the server version would treat the package's own version as too old.
OLLAMA_API_VERSION = "0.5.0"
FORMAT_REQUIRED = (
    'strands-decider answers typed questions and needs a JSON schema in "format": an object '
    "whose properties are booleans (yes/no), enums (choice), or enums marked "
    '{"x-strands-decider": {"type": "score"}} (score). "format": "json" alone is not enough. '
    "See docs/inference.md#ollama-compatible-api."
)
IMAGES_UNSUPPORTED = (
    "images are not supported: strands-decider is text-only (vision support is a separate effort)"
)
TOOLS_UNSUPPORTED = (
    "tools are not supported: strands-decider answers typed questions and does not call tools"
)

_EXT_KEYS = {"type", "criteria"}
_EXT_TYPES = ("noul", "choice", "score")
_LOCAL_REF_ROOTS = ("#/$defs/", "#/definitions/")


class OllamaRequestError(ValueError):
    """A caller error, answered with HTTP 400 and `{"error": message}`."""


class _Request(BaseModel):
    model_config = ConfigDict(extra="ignore")


class OllamaMessage(_Request):
    role: str = Field(min_length=1)
    content: str = ""
    images: list[Any] | None = None


class ChatRequest(_Request):
    model: str = ""
    messages: list[OllamaMessage] = Field(default_factory=list)
    format: Any = None
    stream: bool = True
    tools: list[Any] | None = None


class GenerateRequest(_Request):
    model: str = ""
    prompt: str = ""
    system: str | None = None
    format: Any = None
    stream: bool = True
    images: list[Any] | None = None


class ShowRequest(_Request):
    model: str = ""
    name: str = ""


@dataclass(frozen=True)
class _Decoder:
    kind: Literal["noul", "choice", "score"]
    # The schema's enum values in order, kept with their JSON types for the reply.
    values: tuple[Any, ...] = ()


def _conversation_state(turns: list[tuple[str, str]]) -> str:
    kept = [(role, content) for role, content in turns if content.strip()]
    if not kept:
        return ""
    if len(kept) == 1:
        # Unlabelled, so a one-turn request sends the same state as /v1/systemone would.
        return kept[0][1]
    return "\n\n".join(f"{role}: {content}" for role, content in kept)


def _resolve_ref(name: str, prop: dict[str, Any], defs: Mapping[str, Any]) -> dict[str, Any]:
    ref = prop.get("$ref")
    if ref is None:
        return prop
    for root in _LOCAL_REF_ROOTS:
        if isinstance(ref, str) and ref.startswith(root):
            section = defs.get(root)
            target = section.get(ref[len(root) :]) if isinstance(section, dict) else None
            if isinstance(target, dict):
                own = {k: v for k, v in prop.items() if k != "$ref"}
                return {**target, **own}
    raise OllamaRequestError(
        f"property {name!r}: $ref {ref!r} must point into this schema's $defs or definitions"
    )


def _enum_values(name: str, enum: object) -> tuple[Any, ...]:
    if not isinstance(enum, list):
        raise OllamaRequestError(f"property {name!r}: enum must be a list")
    for value in enum:
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise OllamaRequestError(
                f"property {name!r}: enum values must be strings or numbers, got {value!r}"
            )
    labels = [str(v) for v in enum]
    if len(set(labels)) != len(labels):
        raise OllamaRequestError(f"property {name!r}: enum values must be distinct")
    return tuple(enum)


def _build_question(name: str, prop: dict[str, Any]) -> tuple[Question, _Decoder]:
    ext = prop.get(EXT_KEY, {})
    if not isinstance(ext, dict):
        raise OllamaRequestError(f"property {name!r}: {EXT_KEY} must be an object")
    unknown = sorted(set(ext) - _EXT_KEYS)
    if unknown:
        raise OllamaRequestError(
            f"property {name!r}: unknown {EXT_KEY} key(s) {unknown}; allowed: type, criteria"
        )
    ext_type = ext.get("type")
    if ext_type is not None and ext_type not in _EXT_TYPES:
        raise OllamaRequestError(
            f"property {name!r}: {EXT_KEY}.type must be one of noul, choice, score"
        )
    criteria = ext.get("criteria")

    instructions = name
    for key in ("description", "title"):
        if isinstance(prop.get(key), str) and prop[key].strip():
            instructions = prop[key]
            break

    if "enum" not in prop and prop.get("type") == "boolean":
        if ext_type not in (None, "noul"):
            raise OllamaRequestError(
                f"property {name!r} is a boolean, so {EXT_KEY}.type must be 'noul'"
            )
        if criteria is not None and not isinstance(criteria, dict):
            raise OllamaRequestError(
                f'property {name!r}: {EXT_KEY}.criteria must be {{"true": ..., "false": ...}}'
            )
        return NoulQuestion(instructions=instructions, criteria=criteria), _Decoder("noul")

    if "enum" not in prop:
        raise OllamaRequestError(
            f"property {name!r} must be a boolean or have an enum: "
            "strands-decider answers typed questions, it does not generate text"
        )
    values = _enum_values(name, prop["enum"])
    labels = [str(v) for v in values]

    if ext_type in (None, "choice"):
        if criteria is None:
            criteria = {}
        if not isinstance(criteria, dict):
            raise OllamaRequestError(
                f"property {name!r}: {EXT_KEY}.criteria must map option names to descriptions"
            )
        extra = sorted(set(criteria) - set(labels))
        if extra:
            raise OllamaRequestError(
                f"property {name!r}: {EXT_KEY}.criteria names {extra}, which are not in the enum"
            )
        for text in criteria.values():
            if text is not None and not isinstance(text, str):
                raise OllamaRequestError(
                    f"property {name!r}: option descriptions must be strings or null"
                )
        options = {label: criteria.get(label) for label in labels}
        return ChoiceQuestion(instructions=instructions, criteria=options), _Decoder(
            "choice", values
        )

    if ext_type == "score":
        levels = labels
        if criteria is not None:
            if (
                not isinstance(criteria, list)
                or len(criteria) != len(values)
                or not all(isinstance(t, str) for t in criteria)
            ):
                raise OllamaRequestError(
                    f"property {name!r}: {EXT_KEY}.criteria for a score must be a list of "
                    f"{len(values)} strings, one per enum value, low to high"
                )
            levels = criteria
        return ScoreQuestion(instructions=instructions, criteria=levels), _Decoder("score", values)

    raise OllamaRequestError(
        f"property {name!r} has an enum, so {EXT_KEY}.type must be 'choice' or 'score'"
    )


def schema_to_questions(fmt: object) -> tuple[dict[str, Question], dict[str, _Decoder]]:
    """Translate Ollama's `format` schema: one question per top-level property."""
    if not isinstance(fmt, dict):
        raise OllamaRequestError(FORMAT_REQUIRED)
    if "type" in fmt and fmt["type"] != "object":
        raise OllamaRequestError(
            f'"format" has type {fmt["type"]!r}, not "object". {FORMAT_REQUIRED}'
        )
    properties = fmt.get("properties")
    if not isinstance(properties, dict) or not properties:
        raise OllamaRequestError(f'"format" has no properties. {FORMAT_REQUIRED}')
    defs = {"#/$defs/": fmt.get("$defs"), "#/definitions/": fmt.get("definitions")}

    questions: dict[str, Question] = {}
    decoders: dict[str, _Decoder] = {}
    for name, prop in properties.items():
        if not isinstance(prop, dict):
            raise OllamaRequestError(f"property {name!r} must be a schema object")
        try:
            question, decoder = _build_question(name, _resolve_ref(name, prop, defs))
        except ValidationError as exc:
            raise OllamaRequestError(f"property {name!r}: {_validation_message(exc)}") from exc
        questions[name] = question
        decoders[name] = decoder
    return questions, decoders


def answers_to_content(
    answers: Mapping[str, Answer], decoders: Mapping[str, _Decoder]
) -> dict[str, object]:
    """One schema-typed value per question, in schema property order."""
    content: dict[str, object] = {}
    for name, decoder in decoders.items():
        answer = answers[name]
        if isinstance(answer, NoulAnswer):
            content[name] = answer.noul >= 0.5
        elif isinstance(answer, ChoiceAnswer):
            content[name] = decoder.values[[str(v) for v in decoder.values].index(answer.choice)]
        else:
            # The expected level, rounded half up: the score stays an expected value.
            index = min(len(decoder.values) - 1, max(0, math.floor(answer.score + 0.5)))
            content[name] = decoder.values[index]
    return content


def _validation_message(exc: ValidationError) -> str:
    parts = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"])
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def _rfc3339_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _details(engine: SystemOneEngine) -> dict[str, Any]:
    base_model = str(engine.model.config.base_model)
    return {
        "parent_model": "",
        "format": "strands-decider",
        "family": base_model,
        "families": [base_model],
        "parameter_size": "",
        "quantization_level": "",
    }


def _error(message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=400)


_M = TypeVar("_M", bound=BaseModel)


async def _parse(request: Request, model: type[_M], *, allow_empty: bool = False) -> _M:
    raw_bytes = await request.body()
    if allow_empty and not raw_bytes.strip():
        return model()
    try:
        raw = json.loads(raw_bytes)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise OllamaRequestError(f"request body is not valid JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise OllamaRequestError("request body must be a JSON object")
    try:
        return model.model_validate(raw)
    except ValidationError as exc:
        raise OllamaRequestError(_validation_message(exc)) from exc


async def _answer(
    engine: SystemOneEngine,
    state: str,
    fmt: object,
    stream: bool,
    reply: Literal["chat", "generate"],
    started_ns: int,
) -> Response:
    questions, decoders = schema_to_questions(fmt)
    request = SystemOneRequest(state=state, questions=questions, model=engine.cfg.model_name)

    eval_started = time.perf_counter_ns()
    try:
        response = await run_in_threadpool(engine.evaluate, request)
    except ValueError as exc:
        raise OllamaRequestError(str(exc)) from exc
    eval_ns = time.perf_counter_ns() - eval_started

    content = json.dumps(answers_to_content(response.answers, decoders), ensure_ascii=False)
    body: dict[str, Any] = {"model": engine.cfg.model_name, "created_at": _rfc3339_now()}
    if reply == "chat":
        body["message"] = {"role": "assistant", "content": content}
    else:
        body["response"] = content
    body.update(
        done=True,
        done_reason="stop",
        total_duration=time.perf_counter_ns() - started_ns,
        load_duration=0,
        prompt_eval_count=response.usage.input_tokens,
        prompt_eval_duration=eval_ns,
        eval_count=response.usage.output_tokens,
        eval_duration=0,
        strands_decider={
            "answers": {k: a.model_dump() for k, a in response.answers.items()},
            "usage": response.usage.model_dump(),
            "latency_ms": round(eval_ns / 1e6, 2),
        },
    )
    if stream:
        # Ollama streams by default; with nothing to generate, the stream is the final chunk.
        line = json.dumps(body, ensure_ascii=False) + "\n"
        return StreamingResponse(iter([line]), media_type="application/x-ndjson")
    return JSONResponse(body)


def build_router(get_engine: Callable[[], SystemOneEngine], *, checkpoint: str) -> APIRouter:
    router = APIRouter()
    built_at = _rfc3339_now()

    @router.post("/api/chat")
    async def chat(request: Request) -> Response:
        started_ns = time.perf_counter_ns()
        try:
            body = await _parse(request, ChatRequest)
            if any(m.images for m in body.messages):
                raise OllamaRequestError(IMAGES_UNSUPPORTED)
            if body.tools:
                raise OllamaRequestError(TOOLS_UNSUPPORTED)
            state = _conversation_state([(m.role, m.content) for m in body.messages])
            return await _answer(get_engine(), state, body.format, body.stream, "chat", started_ns)
        except OllamaRequestError as exc:
            return _error(str(exc))

    @router.post("/api/generate")
    async def generate(request: Request) -> Response:
        started_ns = time.perf_counter_ns()
        try:
            body = await _parse(request, GenerateRequest)
            if body.images:
                raise OllamaRequestError(IMAGES_UNSUPPORTED)
            turns = [("system", body.system)] if body.system else []
            state = _conversation_state([*turns, ("user", body.prompt)])
            return await _answer(
                get_engine(), state, body.format, body.stream, "generate", started_ns
            )
        except OllamaRequestError as exc:
            return _error(str(exc))

    @router.get("/api/tags")
    def tags() -> dict[str, Any]:
        engine = get_engine()
        name = engine.cfg.model_name
        return {
            "models": [
                {
                    "name": name,
                    "model": name,
                    "modified_at": built_at,
                    "size": 0,
                    "digest": "",
                    "details": _details(engine),
                }
            ]
        }

    @router.post("/api/show")
    async def show(request: Request) -> Response:
        try:
            await _parse(request, ShowRequest, allow_empty=True)
        except OllamaRequestError as exc:
            return _error(str(exc))
        engine = get_engine()
        config = engine.model.config
        return JSONResponse(
            {
                "modelfile": "",
                "parameters": "",
                "template": "",
                "details": _details(engine),
                "model_info": {
                    "general.architecture": "strands-decider",
                    "strands_decider.checkpoint": checkpoint,
                    "strands_decider.base_model": config.base_model,
                    "strands_decider.num_slots": config.num_slots,
                    "strands_decider.max_length": config.max_length,
                },
                "capabilities": ["completion"],
                "modified_at": built_at,
            }
        )

    @router.get("/api/version")
    def version() -> dict[str, str]:
        try:
            own = package_version("strands-decider")
        except PackageNotFoundError:
            own = "0.0.0"
        return {"version": OLLAMA_API_VERSION, "strands_decider": own}

    return router

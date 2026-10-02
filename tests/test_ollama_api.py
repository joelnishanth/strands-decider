"""Ollama-compatible routes, against a scripted engine: no GPU, no weights, no downloads.

Like tests/test_server.py these pin the wire contract. An Ollama client is repointed by
changing only its base URL, so the reply shape, the `error` key and the one-line NDJSON
stream must not drift.
"""

from __future__ import annotations

import json
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from strands_decider import server
from strands_decider.ollama_api import (
    FORMAT_REQUIRED,
    OllamaRequestError,
    answers_to_content,
    schema_to_questions,
)
from strands_decider.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    SystemOneResponse,
    Usage,
)


class _Cfg:
    model_name = "strands-decider-test"
    device = "cpu"
    use_prefix_cache = True


class _ModelCfg:
    base_model = "stub-base"
    num_slots = 24
    temperature = 1.0
    max_length = 512


class _Model:
    config = _ModelCfg()


def _choice(options, picked, p=0.7):
    rest = (1.0 - p) / (len(options) - 1)
    probs = {o: (p if o == picked else rest) for o in options}
    return ChoiceAnswer(choice=picked, probabilities=probs, confidence=0.5)


def _score(levels, value):
    return ScoreAnswer(
        score=value,
        legend={str(i): t for i, t in enumerate(levels)},
        probabilities={str(i): 1.0 / len(levels) for i in range(len(levels))},
        confidence=0.5,
    )


class _ScriptedEngine:
    """Answers each question from `script`, else a fixed default per type."""

    cfg = _Cfg()
    model = _Model()

    def __init__(self):
        self.requests = []
        self.script = {}
        self.raise_value_error = False

    def evaluate(self, request):
        self.requests.append(request)
        if self.raise_value_error:
            raise ValueError("question has 30 options but this model has 24 slots")
        answers = {}
        for name, q in request.questions.items():
            if name in self.script:
                answers[name] = self.script[name]
            elif isinstance(q, NoulQuestion):
                answers[name] = NoulAnswer(noul=0.8)
            elif isinstance(q, ChoiceQuestion):
                options = list(q.criteria)
                answers[name] = _choice(options, options[0])
            else:
                answers[name] = _score(q.criteria, 1.0)
        return SystemOneResponse(
            model=self.cfg.model_name,
            answers=answers,
            usage=Usage(input_tokens=42, output_tokens=len(request.questions)),
        )


@pytest.fixture
def ollama_client(monkeypatch):
    """The shipped create_app, so the router mount itself is under test."""
    engine = _ScriptedEngine()
    monkeypatch.setattr(server, "_engine", None)
    monkeypatch.setattr(
        server.StrandsDeciderModel, "load", classmethod(lambda cls, *a, **k: _Model())
    )
    monkeypatch.setattr(server, "SystemOneEngine", lambda model, cfg: engine)
    client = TestClient(server.create_app("checkpoints/does-not-exist", device="cpu"))
    yield client, engine


TICKET_SCHEMA = {
    "type": "object",
    "properties": {
        "is_urgent": {"type": "boolean", "description": "Does this convey urgency?"},
        "department": {
            "type": "string",
            "enum": ["billing", "technical", "sales"],
            "description": "Which team should handle this?",
        },
        "frustration": {
            "type": "integer",
            "enum": [1, 2, 3],
            "description": "How frustrated is the writer?",
            "x-strands-decider": {
                "type": "score",
                "criteria": ["calm", "frustrated", "very angry"],
            },
        },
    },
    "required": ["is_urgent", "department", "frustration"],
}

REPLY_KEYS = {
    "model",
    "created_at",
    "message",
    "done",
    "done_reason",
    "total_duration",
    "load_duration",
    "prompt_eval_count",
    "prompt_eval_duration",
    "eval_count",
    "eval_duration",
    "strands_decider",
}


def _chat(content="My payouts fail.", **extra):
    body = {"model": "llama3", "messages": [{"role": "user", "content": content}]}
    body.setdefault("format", TICKET_SCHEMA)
    body.update(extra)
    return body


# --- schema -> questions -------------------------------------------------------------


def test_boolean_becomes_noul_with_criteria():
    qs, _ = schema_to_questions(
        {
            "properties": {
                "urgent": {
                    "type": "boolean",
                    "description": "Is it urgent?",
                    "x-strands-decider": {"criteria": {"true": "needs action today"}},
                }
            }
        }
    )
    q = qs["urgent"]
    assert isinstance(q, NoulQuestion)
    assert q.instructions == "Is it urgent?"
    assert q.criteria == {"true": "needs action today"}


def test_enum_becomes_choice_with_option_descriptions():
    qs, _ = schema_to_questions(
        {
            "properties": {
                "dept": {
                    "enum": ["billing", "technical", "sales"],
                    "description": "Route it.",
                    "x-strands-decider": {"criteria": {"billing": "money", "sales": "pricing"}},
                }
            }
        }
    )
    q = qs["dept"]
    assert isinstance(q, ChoiceQuestion)
    assert list(q.criteria) == ["billing", "technical", "sales"]
    assert q.criteria == {"billing": "money", "technical": None, "sales": "pricing"}


def test_explicit_choice_type_is_accepted():
    qs, _ = schema_to_questions(
        {"properties": {"d": {"enum": ["a", "b"], "x-strands-decider": {"type": "choice"}}}}
    )
    assert isinstance(qs["d"], ChoiceQuestion)


def test_score_marker_gives_ascending_levels():
    qs, _ = schema_to_questions(
        {
            "properties": {
                "f": {
                    "enum": ["calm", "frustrated", "angry"],
                    "x-strands-decider": {"type": "score"},
                }
            }
        }
    )
    q = qs["f"]
    assert isinstance(q, ScoreQuestion)
    assert q.criteria == ["calm", "frustrated", "angry"]


def test_score_criteria_replace_numeric_level_texts():
    qs, decoders = schema_to_questions(
        {
            "properties": {
                "f": {
                    "enum": [1, 2, 3],
                    "x-strands-decider": {"type": "score", "criteria": ["low", "mid", "high"]},
                }
            }
        }
    )
    assert qs["f"].criteria == ["low", "mid", "high"]
    assert decoders["f"].values == (1, 2, 3)


def test_instructions_fall_back_to_title_then_name():
    qs, _ = schema_to_questions(
        {
            "properties": {
                "a": {"type": "boolean", "description": "desc", "title": "title"},
                "b": {"type": "boolean", "title": "title"},
                "c": {"type": "boolean"},
            }
        }
    )
    assert [qs[k].instructions for k in "abc"] == ["desc", "title", "c"]


def test_pydantic_ref_resolves_and_property_keys_override():
    qs, _ = schema_to_questions(
        {
            "$defs": {
                "Dept": {"enum": ["billing", "technical"], "title": "Dept", "description": "def"}
            },
            "properties": {"dept": {"$ref": "#/$defs/Dept", "description": "Route it."}},
            "type": "object",
        }
    )
    assert isinstance(qs["dept"], ChoiceQuestion)
    assert qs["dept"].instructions == "Route it."
    assert list(qs["dept"].criteria) == ["billing", "technical"]


def test_legacy_definitions_ref_resolves():
    qs, _ = schema_to_questions(
        {
            "definitions": {"Yn": {"type": "boolean"}},
            "properties": {"ok": {"$ref": "#/definitions/Yn"}},
        }
    )
    assert isinstance(qs["ok"], NoulQuestion)


def test_real_pydantic_model_schema_translates():
    """The Ollama docs build `format` from `Model.model_json_schema()`."""
    from enum import Enum

    from pydantic import BaseModel, Field

    class Dept(str, Enum):
        billing = "billing"
        technical = "technical"

    class Ticket(BaseModel):
        is_urgent: bool = Field(description="Is it urgent?")
        department: Dept

    qs, _ = schema_to_questions(Ticket.model_json_schema())
    assert isinstance(qs["is_urgent"], NoulQuestion)
    assert isinstance(qs["department"], ChoiceQuestion)
    assert list(qs["department"].criteria) == ["billing", "technical"]


def test_noul_threshold_is_half():
    _, dec = schema_to_questions({"properties": {"q": {"type": "boolean"}}})
    assert answers_to_content({"q": NoulAnswer(noul=0.5)}, dec) == {"q": True}
    assert answers_to_content({"q": NoulAnswer(noul=0.4999)}, dec) == {"q": False}


def test_choice_maps_back_to_typed_enum_value():
    _, dec = schema_to_questions({"properties": {"n": {"enum": [10, 20, 30]}}})
    out = answers_to_content({"n": _choice(["10", "20", "30"], "20")}, dec)
    assert out == {"n": 20}
    assert isinstance(out["n"], int)


@pytest.mark.parametrize(
    ("expected", "level"),
    [(1.49, "b"), (1.5, "c"), (0.0, "a"), (3.0, "d"), (-0.2, "a"), (3.4, "d")],
)
def test_score_is_expected_value_rounded_half_up(expected, level):
    levels = ["a", "b", "c", "d"]
    _, dec = schema_to_questions(
        {"properties": {"s": {"enum": levels, "x-strands-decider": {"type": "score"}}}}
    )
    assert answers_to_content({"s": _score(levels, expected)}, dec) == {"s": level}


@pytest.mark.parametrize(
    ("prop", "fragment"),
    [
        ({"type": "string"}, "must be a boolean or have an enum"),
        ({"type": "number"}, "must be a boolean or have an enum"),
        ({"anyOf": [{"type": "boolean"}, {"type": "null"}]}, "must be a boolean or have an enum"),
        ({"const": "x"}, "must be a boolean or have an enum"),
        ({"enum": ["only"]}, "at least 2 options"),
        ({"enum": [f"o{i}" for i in range(256)]}, "at most 255"),
        (
            {"enum": list(range(11)), "x-strands-decider": {"type": "score"}},
            "between 2 and 10 levels",
        ),
        (
            {"enum": [1, 2, 3], "x-strands-decider": {"type": "score", "criteria": ["a", "b"]}},
            "list of 3 strings",
        ),
        (
            {"enum": ["a", "b"], "x-strands-decider": {"criteria": {"c": "nope"}}},
            "not in the enum",
        ),
        ({"enum": ["a", "b"], "x-strands-decider": {"typ": "score"}}, "unknown"),
        ({"type": "boolean", "x-strands-decider": {"type": "score"}}, "must be 'noul'"),
        ({"type": "boolean", "x-strands-decider": {"criteria": {"maybe": "x"}}}, "subset"),
        ({"enum": [1, "1"]}, "distinct"),
        ({"enum": [True, False]}, "strings or numbers"),
        ({"$ref": "https://example.com/schema.json"}, "$defs"),
        ({"$ref": "#/$defs/Missing"}, "$defs"),
    ],
)
def test_bad_property_is_rejected_by_name(prop, fragment):
    with pytest.raises(OllamaRequestError) as info:
        schema_to_questions({"type": "object", "properties": {"summary": prop}})
    assert "'summary'" in str(info.value)
    assert fragment in str(info.value)


@pytest.mark.parametrize(
    "fmt",
    [None, "json", "anything", 3, [], {"properties": {}}, {"type": "array", "items": {}}, {}],
)
def test_unusable_format_is_rejected(fmt):
    with pytest.raises(OllamaRequestError) as info:
        schema_to_questions(fmt)
    assert FORMAT_REQUIRED in str(info.value)


# --- /api/chat ----------------------------------------------------------------------


def test_chat_single_message_is_the_state_verbatim(ollama_client):
    client, engine = ollama_client
    r = client.post("/api/chat", json=_chat("  Help! My payouts fail.\n", stream=False))
    assert r.status_code == 200, r.text
    assert engine.requests[-1].state == "  Help! My payouts fail.\n"


def test_chat_conversation_is_joined_with_role_labels(ollama_client):
    client, engine = ollama_client
    messages = [
        {"role": "system", "content": "You triage tickets."},
        {"role": "user", "content": "My payouts fail."},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "Since when?"},
        {"role": "user", "content": "3 days."},
    ]
    r = client.post(
        "/api/chat", json={"messages": messages, "format": TICKET_SCHEMA, "stream": False}
    )
    assert r.status_code == 200, r.text
    assert engine.requests[-1].state == (
        "system: You triage tickets.\n\nuser: My payouts fail.\n\n"
        "assistant: Since when?\n\nuser: 3 days."
    )


def test_chat_with_no_content_sends_an_empty_state(ollama_client):
    client, engine = ollama_client
    r = client.post("/api/chat", json={"messages": [], "format": TICKET_SCHEMA, "stream": False})
    assert r.status_code == 200, r.text
    assert engine.requests[-1].state == ""


def test_chat_reply_shape(ollama_client):
    client, engine = ollama_client
    engine.script = {
        "is_urgent": NoulAnswer(noul=0.81),
        "department": _choice(["billing", "technical", "sales"], "technical"),
        "frustration": _score(["calm", "frustrated", "very angry"], 1.6),
    }
    r = client.post("/api/chat", json=_chat(stream=False))
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()

    assert set(body) == REPLY_KEYS
    assert body["model"] == "strands-decider-test"
    assert body["message"]["role"] == "assistant"
    content = json.loads(body["message"]["content"])
    assert list(content) == ["is_urgent", "department", "frustration"]
    assert content == {"is_urgent": True, "department": "technical", "frustration": 3}
    assert body["done"] is True
    assert body["done_reason"] == "stop"
    for key in (
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_count",
        "eval_duration",
    ):
        assert isinstance(body[key], int) and body[key] >= 0
    assert body["prompt_eval_count"] == 42
    assert body["eval_count"] == 3
    assert body["created_at"].endswith("Z")
    datetime.fromisoformat(body["created_at"].replace("Z", "+00:00"))

    extra = body["strands_decider"]
    assert set(extra) == {"answers", "usage", "latency_ms"}
    assert extra["usage"] == {"input_tokens": 42, "output_tokens": 3}
    assert extra["answers"]["is_urgent"] == {"type": "noul", "noul": 0.81}
    assert extra["answers"]["department"]["type"] == "choice"
    assert set(extra["answers"]["department"]["probabilities"]) == {"billing", "technical", "sales"}
    assert extra["answers"]["frustration"]["score"] == pytest.approx(1.6)
    assert extra["answers"]["frustration"]["legend"] == {
        "0": "calm",
        "1": "frustrated",
        "2": "very angry",
    }


def test_chat_ignores_options_keep_alive_and_think(ollama_client):
    client, engine = ollama_client
    plain = client.post("/api/chat", json=_chat(stream=False)).json()
    noisy = client.post(
        "/api/chat",
        json=_chat(
            stream=False, options={"temperature": 0.0, "num_ctx": 8}, keep_alive="5m", think=True
        ),
    )
    assert noisy.status_code == 200, noisy.text
    assert noisy.json()["message"] == plain["message"]
    assert engine.requests[-1] == engine.requests[-2]


@pytest.mark.parametrize("stream", [None, True])
def test_chat_streams_one_ndjson_line_by_default(ollama_client, stream):
    client, _ = ollama_client
    payload = _chat() if stream is None else _chat(stream=stream)
    r = client.post("/api/chat", json=payload)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-ndjson")
    assert r.text.endswith("\n")
    lines = r.text.splitlines()
    assert len(lines) == 1
    chunk = json.loads(lines[0])
    assert set(chunk) == REPLY_KEYS
    assert chunk["done"] is True
    plain = client.post("/api/chat", json=_chat(stream=False)).json()
    assert chunk["message"] == plain["message"]


def test_documented_ollama_payload_round_trips(ollama_client):
    """The request in docs/inference.md#ollama-compatible-api, verbatim."""
    client, _ = ollama_client
    r = client.post(
        "/api/chat",
        json={
            "model": "strands-decider",
            "stream": False,
            "messages": [
                {"role": "user", "content": "Help! My payouts have been failing for 3 days."}
            ],
            "format": {
                "type": "object",
                "properties": {
                    "is_urgent": {"type": "boolean", "description": "Does this convey urgency?"},
                    "department": {
                        "type": "string",
                        "enum": ["billing", "technical", "sales"],
                        "description": "Which team should handle this?",
                        "x-strands-decider": {
                            "criteria": {
                                "billing": "money",
                                "technical": "bugs",
                                "sales": "pricing",
                            }
                        },
                    },
                    "frustration": {
                        "type": "string",
                        "enum": ["calm", "frustrated", "very angry"],
                        "description": "How frustrated is the writer?",
                        "x-strands-decider": {"type": "score"},
                    },
                },
                "required": ["is_urgent", "department", "frustration"],
            },
        },
    )
    assert r.status_code == 200, r.text
    content = json.loads(r.json()["message"]["content"])
    assert set(content) == {"is_urgent", "department", "frustration"}
    assert isinstance(content["is_urgent"], bool)
    assert content["department"] in {"billing", "technical", "sales"}
    assert content["frustration"] in {"calm", "frustrated", "very angry"}


# --- /api/generate ------------------------------------------------------------------


def test_generate_prompt_is_the_state_verbatim(ollama_client):
    client, engine = ollama_client
    r = client.post(
        "/api/generate",
        json={"model": "x", "prompt": "My payouts fail.", "format": TICKET_SCHEMA, "stream": False},
    )
    assert r.status_code == 200, r.text
    assert engine.requests[-1].state == "My payouts fail."
    body = r.json()
    assert set(body) == (REPLY_KEYS - {"message"}) | {"response"}
    assert json.loads(body["response"])["department"] == "billing"


def test_generate_system_and_prompt_are_labeled(ollama_client):
    client, engine = ollama_client
    r = client.post(
        "/api/generate",
        json={
            "prompt": "My payouts fail.",
            "system": "You triage tickets.",
            "format": TICKET_SCHEMA,
            "stream": False,
            "raw": True,
            "context": [1, 2, 3],
        },
    )
    assert r.status_code == 200, r.text
    assert engine.requests[-1].state == "system: You triage tickets.\n\nuser: My payouts fail."


def test_generate_streams_one_ndjson_line(ollama_client):
    client, _ = ollama_client
    r = client.post("/api/generate", json={"prompt": "x", "format": TICKET_SCHEMA})
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("application/x-ndjson")
    lines = r.text.splitlines()
    assert len(lines) == 1
    chunk = json.loads(lines[0])
    assert chunk["done"] is True
    assert set(json.loads(chunk["response"])) == {"is_urgent", "department", "frustration"}


# --- errors -------------------------------------------------------------------------


def _assert_400(r, fragment):
    assert r.status_code == 400, r.text
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert set(body) == {"error"}
    assert isinstance(body["error"], str)
    assert fragment in body["error"]


@pytest.mark.parametrize("path", ["/api/chat", "/api/generate"])
@pytest.mark.parametrize(
    ("fmt", "fragment"),
    [
        (None, "JSON schema"),
        ("json", '"format": "json" alone is not enough'),
        ({"type": "object", "properties": {"s": {"type": "string"}}}, "typed questions"),
    ],
)
def test_missing_or_unusable_format_is_400(ollama_client, path, fmt, fragment):
    client, engine = ollama_client
    payload = _chat(stream=False) if path == "/api/chat" else {"prompt": "x", "stream": False}
    payload.pop("format", None)
    if fmt is not None:
        payload["format"] = fmt
    _assert_400(client.post(path, json=payload), fragment)
    assert engine.requests == []


def test_string_property_error_names_it(ollama_client):
    client, _ = ollama_client
    fmt = {"type": "object", "properties": {"summary": {"type": "string"}}}
    _assert_400(client.post("/api/chat", json=_chat(format=fmt)), "'summary'")


def test_stream_mode_error_is_a_plain_400(ollama_client):
    client, _ = ollama_client
    payload = _chat(stream=True)
    payload.pop("format")
    _assert_400(client.post("/api/chat", json=payload), "JSON schema")


def test_non_json_body_is_400(ollama_client):
    client, _ = ollama_client
    r = client.post("/api/chat", content=b"{not json", headers={"content-type": "application/json"})
    _assert_400(r, "not valid JSON")


def test_non_object_body_is_400(ollama_client):
    client, _ = ollama_client
    _assert_400(client.post("/api/generate", json=["prompt"]), "JSON object")


def test_messages_not_a_list_is_400(ollama_client):
    client, _ = ollama_client
    _assert_400(client.post("/api/chat", json=_chat(messages="hello")), "messages")


def test_images_on_a_chat_message_are_rejected(ollama_client):
    client, engine = ollama_client
    payload = _chat()
    payload["messages"][0]["images"] = ["aGVsbG8="]
    _assert_400(client.post("/api/chat", json=payload), "vision")
    assert engine.requests == []


def test_images_on_generate_are_rejected(ollama_client):
    client, _ = ollama_client
    payload = {"prompt": "x", "images": ["aGVsbG8="], "format": TICKET_SCHEMA}
    _assert_400(client.post("/api/generate", json=payload), "vision")


def test_empty_images_and_tools_are_accepted(ollama_client):
    client, _ = ollama_client
    payload = _chat(stream=False, tools=[])
    payload["messages"][0]["images"] = []
    assert client.post("/api/chat", json=payload).status_code == 200


def test_tools_are_rejected(ollama_client):
    client, _ = ollama_client
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    _assert_400(client.post("/api/chat", json=_chat(tools=tools)), "tools")


def test_engine_value_error_is_400(ollama_client):
    client, engine = ollama_client
    engine.raise_value_error = True
    _assert_400(client.post("/api/chat", json=_chat(stream=False)), "24 slots")


# --- discovery ----------------------------------------------------------------------


def test_tags_lists_the_served_model(ollama_client):
    client, _ = ollama_client
    r = client.get("/api/tags")
    assert r.status_code == 200
    (model,) = r.json()["models"]
    assert model["name"] == model["model"] == "strands-decider-test"
    assert model["details"]["family"] == "stub-base"
    assert model["modified_at"].endswith("Z")


@pytest.mark.parametrize("payload", [{"model": "anything"}, {"name": "legacy"}, None])
def test_show_describes_the_served_model(ollama_client, payload):
    client, _ = ollama_client
    r = client.post("/api/show", json=payload) if payload else client.post("/api/show")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["model_info"]["strands_decider.num_slots"] == 24
    assert body["model_info"]["strands_decider.checkpoint"] == "checkpoints/does-not-exist"
    assert body["capabilities"] == ["completion"]
    assert body["details"]["family"] == "stub-base"


def test_version_reports_the_emulated_ollama_api(ollama_client):
    client, _ = ollama_client
    body = client.get("/api/version").json()
    assert body["version"] == "0.5.0"
    assert isinstance(body["strands_decider"], str)


# --- the existing routes are untouched ----------------------------------------------


def test_health_and_systemone_still_answer(ollama_client):
    client, _ = ollama_client
    assert client.get("/health").json()["checkpoint"] == "checkpoints/does-not-exist"
    r = client.post(
        "/v1/systemone",
        json={"state": "x", "questions": {"q": {"type": "noul", "instructions": "y"}}},
    )
    assert r.status_code == 200, r.text
    assert set(r.json()) == {"model", "answers", "usage", "latency_ms"}

# Inference

This document describes how to run a checkpoint: install, get a checkpoint, ask it questions
from the command line, and serve it over HTTP. Strands decider answers typed questions about a
state, the text to classify; it does not generate text. A question is a `noul` (yes or no,
returned as P(true)), a `choice` (one of N options) or a `score` (a level on an ordered
scale). Install the package, point the commands at the published Hub id or at a checkpoint you
trained, then `strands-decider ask` or `strands-decider serve`. Pass `--device cuda`, `mps` or `cpu` explicitly. A
checkpoint from `training/recipe.sh all` is already calibrated; calibrate any other
checkpoint with `strands-decider calibrate` before you serve it.

- [Environments for serving](#environments-for-serving): Linux with CUDA, a Mac, or CPU.
- [Model artifact](#model-artifact): what a checkpoint holds, and the Hugging Face export.
- [Ask](#ask): one command, with recorded v19 output and how to read its confidence.
- [Serve](#serve): `POST /v1/systemone` and `/health`, on `127.0.0.1` with no authentication.
- [Ollama-compatible API](#ollama-compatible-api): `/api/chat` and `/api/generate` for clients that already talk to Ollama.
- [Asking many questions is nearly free](#asking-many-questions-is-nearly-free): the shared-prefix cache.
- [Serving on a Mac](#serving-on-a-mac): MPS, and the one kernel that had to be replaced.
- [`../examples/strands/`](../examples/strands/README.md): an agent built with the Strands
  Agents SDK that uses the server, with its client in `_client.py`.
- [`../evaluation/results.md`](../evaluation/results.md): measured latency and accuracy on an
  RTX 3090 and on a Mac; [`../evaluation/jevbench.md`](../evaluation/jevbench.md): JevBench,
  the external benchmark.
- [`../docs/architecture.md`](../docs/architecture.md#the-routing-convention): the confidence
  formulas and the routing convention.
- [`../training/README.md`](../training/README.md): how to train a checkpoint.

Paths are relative to the repository root unless they are links. A module path such as
`mps_kernels.py` is relative to `src/strands_decider/`.

## Environments for serving

On Linux, or under WSL2 on Windows, with an NVIDIA GPU, serving uses the training
environment in [Setup](../training/README.md#setup).

A server started inside WSL (`strands-decider serve ... --port 8099`) is reachable from Windows
at `127.0.0.1:8099` for as long as its WSL session is alive.

**Serving on macOS (Apple silicon), inference only.** No `flash-linear-attention`:
Triton has no macOS build. Nothing else changes; pass `--device mps`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.7.1 transformers==5.17.0 peft==0.21.0
pip install -e ".[dev]"
strands-decider serve checkpoints/hobson-2b-recipe --device mps --port 8099
```

Transformers' `causal_conv1d_fn` falls back to its reference PyTorch path. On CPU,
`chunk_gated_delta_rule` also falls back. On MPS, `mps_kernels.py` replaces that second
function, so only the first fallback matters there, and it costs 32 ms a forward.

**CPU only.** The same commands as on macOS, with `--device cpu`. On Linux without a GPU,
`pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu` skips the CUDA
wheels. Both reference fallback paths above apply, and inference is much slower than on a
GPU.

## Model artifact

The v19 reference weights are published as
[`StrandsAgents/strands-decider-2B-hobson-v19`](https://huggingface.co/StrandsAgents/strands-decider-2B-hobson-v19).
Every command that takes a checkpoint path also takes that id, and the loader downloads the
repository to the Hub cache. You can equally train your own with the recipe
([training/README.md](../training/README.md#usage)); `recipe.sh` writes it to
`checkpoints/hobson-2b-recipe`.

A checkpoint directory holds these files. `StrandsDeciderModel.save_pretrained` in `modeling.py`
writes them, `strands-decider calibrate` adds the temperatures, and `StrandsDeciderModel.load` reads them.

| File | Contents |
| --- | --- |
| `strands_decider_config.json` | The model configuration: the base model name, the readout type and size, the window (`max_length`) and the fitted temperatures. A checkpoint saved before the rename, such as the published v19, holds `hobson_config.json` instead; `load` reads either |
| `lora/` | The LoRA adapter (`adapter_config.json`, `adapter_model.safetensors`) |
| `slot_head.pt` or `head.safetensors` | The readout weights. `save_pretrained` writes `slot_head.pt`. An export from `strands_decider.hf_export` holds `head.safetensors` instead. |
| `tokenizer.json`, `tokenizer_config.json` | The tokenizer |
| `chat_template.jinja` | The chat template, saved with the tokenizer. Inference does not read it. |

Training also writes `train_config.json` and `history.json`. Inference does not read them.

The base weights are not in the checkpoint. At first use, `load` downloads
`Qwen/Qwen3.5-2B-Base` (about 4.5 GB) from Hugging Face. The loader passes no revision, so
it gets the current `main` of that repository. A later change to that repository changes
what a fresh install loads.

`python -m strands_decider.hf_export export CKPT OUT --run-id RUN` writes a checkpoint as a Hugging
Face-format folder. Without `--run-id`, the exporter takes the run id from a
`results/<run-id>/` part of the CKPT path, and it stops if the path has none. The folder
holds the readout as `head.safetensors`, so it has no pickle files. It also holds the PEFT
adapter, the tokenizer and configuration files, and a model card (`README.md`).
`provenance.json`, `LICENSE.md` with the licence status and `MANIFEST.sha256` complete it.
`python -m strands_decider.hf_export verify DIR` checks a folder. `StrandsDeciderModel.load` reads the
folder as it reads a checkpoint, and the base weights still download separately from
Hugging Face. Where a command takes a checkpoint path (`strands-decider serve`, `strands-decider ask`,
`strands-decider info`), a Hub model repo id such as `StrandsAgents/strands-decider-2B-hobson-v19` also works: the loader
downloads the repo to the Hub cache. `strands-decider calibrate` writes into the checkpoint, so it
needs a local directory. `provenance.json` records the base-model revision, but the loader does not
read it. OUT can be a local directory or an `s3://` URI. The exporter does not upload to
the Hugging Face Hub.

## Ask

There are three question types, and all three are one mechanism read back three ways.

- A **noul** question is a yes/no question. It has two options and returns the
  probability of "true".
- A **choice** question has any number of options. It returns the top option, a
  probability for each option, and a confidence: the top probability, rescaled so that an
  even split gives 0 and a certain answer gives 1, whatever the number of options.
- A **score** question has ordered levels. It returns an expected value on that scale,
  the distribution, and a confidence.

On the command line, a choice or a score is written as `question=option,option,...`. The
confidence is computed from the distribution, not predicted.

The output below was recorded from a v19 checkpoint, which the v19 reference recipe
trains. The command below names `checkpoints/hobson-2b-recipe`, the path
`training/recipe.sh` (or `bash training/recipe.sh`) writes; pass
`StrandsAgents/strands-decider-2B-hobson-v19` instead to run the published weights. On a Mac,
add `--device mps`. A retrained checkpoint gives somewhat different numbers with the same
answers: a later v19 retrain gives 0.809, `technical` at 0.783 and a score confidence
of 0.472 on CPU.

```bash
strands-decider ask checkpoints/hobson-2b-recipe \
  --state "Help! My payouts have been failing for 3 days." \
  --noul "Does this convey urgency?" \
  --choice "Which team should handle this?=billing,technical,sales" \
  --score "How frustrated is the writer?=calm,frustrated,very angry"
```

```
noul_0 noul = 0.801
choice_0 -> technical (confidence 0.622)
    technical                0.748
    billing                  0.234
    sales                    0.018
score_0 score = 1.24 (confidence 0.578)
    0: calm                                     0.093
    1: frustrated                               0.578
    2: very angry                               0.330
```

That is real output from v19, not an illustration — and it is
worth reading the confidence, not just the answer. Whether failing payouts are
`billing` or `technical` is arguable, and the model leans but says so: 0.622 is in the
"confirm" band of the [routing convention](../docs/architecture.md#the-routing-convention), well short
of the 0.9 at which that convention acts without a check.

## Serve

```bash
strands-decider serve checkpoints/hobson-2b-recipe --port 8000
```

`POST /v1/systemone` takes a state and typed questions. JevBench's `typesafe` adapter
targets this endpoint and runs against it unmodified: 231 of 231 tasks attempted and
schema validity 1.000 ([External benchmark](../evaluation/jevbench.md#external-benchmark-jevbench-v1-public-set)).
Compatibility with the Jev API itself is not verified. `GET /health` returns the model
name, the checkpoint path, the base model, the device and the calibration temperature.

The server binds to `127.0.0.1` by default and has no authentication. Its behaviour under
concurrent requests is not verified. Use it for local experiments only
([Security](../README.md#security)).

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Help! My payouts have been failing for 3 days.",
  "model": "hobson-latest",
  "questions": {
    "is_urgent":  {"type": "noul",   "instructions": "Does this convey urgency?"},
    "department": {"type": "choice", "instructions": "Route it.",
                   "criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}},
    "frustration":{"type": "score",  "instructions": "How frustrated?",
                   "criteria": ["Calm", "Frustrated", "Very angry"]}
  }
}'
```

A hybrid torso such as v14's shares the state across questions like any other; a single
question is encoded in one pass ([Asking many questions is nearly free](#asking-many-questions-is-nearly-free)).

The `state` may be empty when the question carries the whole task. An option description
may be a string, structured data (rendered as JSON, like a structured `state`), or `null`
for a bare label.

By default a prompt longer than the checkpoint's window is shortened to fit: the state is
cut, and the question keeps its options. `--strict-window` refuses such a prompt instead,
with HTTP 422 and a message that names the context window, for an evaluation that forbids
truncation. `--max-batch N` (default 32) sets how many questions one forward pass encodes;
lower it when a very long state with many questions does not fit in GPU memory.

## Ollama-compatible API

The same server also answers a subset of the [Ollama API](https://github.com/ollama/ollama/blob/main/docs/api.md):
`POST /api/chat`, `POST /api/generate`, `GET /api/tags`, `POST /api/show` and
`GET /api/version`. It is a translation onto the same engine as `/v1/systemone`, not a
language model: the decider generates no text. A request must carry a JSON schema in
Ollama's `format` field. Each top-level property of that schema becomes one typed
question, and the reply's content is a JSON string that matches the schema.

Ollama clients default to `http://127.0.0.1:11434`, so serve on that port to repoint one
without changing it:

```bash
strands-decider serve checkpoints/hobson-2b-recipe --port 11434
```

That port clashes with a running Ollama. To keep both, serve on another port and point
the client at it, with `OLLAMA_HOST=127.0.0.1:8000` or `ollama.Client(host="http://127.0.0.1:8000")`.
The `model` field of a request is accepted and not checked, so a client hard-coded to a
name such as `llama3` still works. Replies always report the served model name.

### From schema to questions

| Property in `format` | Question |
| --- | --- |
| `{"type": "boolean"}` | `noul` |
| `{"enum": [...]}` | `choice`; the enum values are the option names |
| `{"enum": [...], "x-strands-decider": {"type": "score"}}` | `score`; the enum is read in ascending order, the first value is the low end |

The question's instructions are the property's `description`, else its `title`, else the
property name. The optional `x-strands-decider` object takes two keys, `type` (`noul`,
`choice` or `score`) and `criteria`. Any other key is refused, to catch typos.

- On a boolean, `criteria` is `{"true": "...", "false": "..."}`, as in a `noul` question.
- On a choice, `criteria` maps option names to descriptions (a string or `null`). An
  option without an entry is a bare label.
- On a score, `criteria` is a list of level texts, one per enum value, low to high. It
  lets a numeric enum such as `[1, 2, 3, 4, 5]` carry a rubric. Without it the level texts
  are the enum values.

Enum values must be strings or numbers, distinct as text. A choice takes 2 to 255
options and a score 2 to 10 levels. A local `$ref` into the schema's `$defs` or
`definitions` is resolved, which covers the schema `Model.model_json_schema()` emits for a
Pydantic model with a Python `Enum` field. Every property is answered: `required`,
`additionalProperties` and `title` at the top level are ignored. Anything else, such as a
string with no enum, a number, an array, an object, `anyOf` or `const`, is refused with a
400 that names the property.

### From conversation to state

On `/api/chat`, the messages become the `state` in request order, system messages
included. Messages whose content is blank are dropped. A single remaining message is the
state verbatim, so a one-message chat sends the model exactly what `/v1/systemone` would.
Two or more are joined as `role: content` blocks separated by a blank line:

```
system: You triage tickets.

user: My payouts fail.

assistant: Since when?

user: 3 days.
```

On `/api/generate`, the state is the `prompt`, verbatim, or `system: ...` and
`user: ...` blocks when `system` is set. The model was calibrated on plain-text states,
so put the instructions in each property's `description` rather than in a system message.

### From answers to content

- A boolean is `true` when P(true) is at least 0.5.
- A choice is the most probable option, returned as the enum value with its JSON type (an
  integer enum gives an integer).
- A score is the enum value at the expected level, rounded half up. The score answer stays
  an expected value over the levels, as on `/v1/systemone`.

The full answers, with probabilities, confidences and the exact expected score, are in a
top-level `strands_decider` field. It has the same shape as the `answers` of
`/v1/systemone`, plus `usage` and `latency_ms`. Standard Ollama clients ignore it.

```bash
curl -s localhost:11434/api/chat -H 'content-type: application/json' -d '{
  "model": "strands-decider",
  "stream": false,
  "messages": [{"role": "user", "content": "Help! My payouts have been failing for 3 days."}],
  "format": {
    "type": "object",
    "properties": {
      "is_urgent":  {"type": "boolean", "description": "Does this convey urgency?"},
      "department": {"type": "string", "enum": ["billing", "technical", "sales"],
                     "description": "Which team should handle this?",
                     "x-strands-decider": {"criteria": {"billing": "money", "technical": "bugs", "sales": "pricing"}}},
      "frustration":{"type": "string", "enum": ["calm", "frustrated", "very angry"],
                     "description": "How frustrated is the writer?",
                     "x-strands-decider": {"type": "score"}}
    },
    "required": ["is_urgent", "department", "frustration"]
  }
}'
```

An illustrative reply follows. It shows the shape, not recorded output: the numbers are
made up.

```json
{
  "model": "hobson-2b-recipe",
  "created_at": "2025-01-01T12:00:00.123456Z",
  "message": {"role": "assistant",
              "content": "{\"is_urgent\": true, \"department\": \"technical\", \"frustration\": \"frustrated\"}"},
  "done": true,
  "done_reason": "stop",
  "total_duration": 152000000,
  "load_duration": 0,
  "prompt_eval_count": 120,
  "prompt_eval_duration": 150000000,
  "eval_count": 3,
  "eval_duration": 0,
  "strands_decider": {
    "answers": {
      "is_urgent": {"type": "noul", "noul": 0.80},
      "department": {"type": "choice", "choice": "technical",
                     "probabilities": {"billing": 0.23, "technical": 0.75, "sales": 0.02},
                     "confidence": 0.62},
      "frustration": {"type": "score", "score": 1.24,
                      "legend": {"0": "calm", "1": "frustrated", "2": "very angry"},
                      "probabilities": {"0": 0.09, "1": 0.58, "2": 0.33}, "confidence": 0.58}
    },
    "usage": {"input_tokens": 120, "output_tokens": 3},
    "latency_ms": 150.0
  }
}
```

All durations are integer nanoseconds. `prompt_eval_count` is the input token count and
`eval_count` is one decision per question. `load_duration` and `eval_duration` are 0: the
model loads at startup, and no tokens are generated. A client that divides by
`eval_duration` to report tokens per second has nothing to measure. `/api/generate`
returns the same keys, with the content in `response` instead of `message`.

### Streaming

`stream` defaults to `true`, as in Ollama. A streamed reply is `application/x-ndjson` with
exactly one line: the full reply above, which already has `"done": true`. A client that
joins the content of all chunks gets the full JSON. The answer is computed before the
stream starts, so an error is still a plain 400 JSON body.

### Errors and ignored fields

Every caller error is HTTP 400 with `{"error": "..."}`, the key Ollama clients read:

- `format` missing, not a schema, or `"json"` alone: the message explains that the decider
  needs a JSON schema of questions.
- A property the decider cannot answer, or a malformed `x-strands-decider`: the message
  names the property.
- `images`, on a chat message or a generate request: refused, the model is text-only.
- `tools` on a chat request: refused, the decider does not call tools.
- A body that is not JSON, a field of the wrong type, or a question the model cannot take,
  such as more options than its slots or a prompt over the window under `--strict-window`.

`options`, `keep_alive`, `think`, `raw`, `template`, `context`, `suffix`, a message's
`tool_calls` and `thinking`, and unknown fields are ignored.

### Discovery

`GET /api/tags` lists the one served model. `POST /api/show` describes it, whatever name
is asked for, with the checkpoint, base model, slot count and window in `model_info`.
`GET /api/version` returns `{"version": "0.5.0", "strands_decider": "<package version>"}`.
`version` is the Ollama API level emulated, not the package version: structured outputs
arrived in Ollama 0.5.0, and a client that checks the version would treat a lower one as
too old.

The Ollama routes are always on and have the same exposure as `/v1/systemone`: the server
binds to `127.0.0.1` by default and has no authentication. Use it for local experiments
only ([Security](../README.md#security)).

## Asking many questions is nearly free

*The latency table below was measured on v7, a Qwen3 torso. The cache now also forks the
recurrent state of the hybrid Qwen3.5 torso
([Hybrid torsos share the prefix too](../docs/architecture.md#design-decisions-worth-knowing)).
The v14 figures at the end of this section use it.*

The state is encoded **once** and its KV cache is broadcast across the question batch;
only the short question suffixes are forwarded per question.

```
naive:  N x (state + question)     tokens
strands-decider: state + N x question       tokens
```

Measured against the served 1.7B checkpoint, one ~640-token state:

| questions | latency | input tokens |
| --- | --- | --- |
| 1 | 219.4 ms | 666 |
| 6 | 217.4 ms | 966 |

Five extra questions cost roughly nothing. The naive path would have encoded ~4,000
tokens instead of 966. The questions genuinely evaluate independently — no question
can see another's text — which is what keeps the answers decomposable.

`--no-prefix-cache` falls back to plain batched encoding. The two paths are
*numerically identical in fp32* (pinned by `test_prefix_cache_is_exact_in_fp32`, and for
hybrid torsos by `tests/test_prefix_cache.py`); in bf16 they differ by ~2e-3 from
accumulation order alone, far below any threshold worth routing on. On v14 on the RTX 3090, eight
questions over a ~2,000-token state take 369 ms through the shared prefix against
1,964 ms batched, and sixteen take 445 ms against 4,086 ms. A request with one question
skips the prefix path: it would pay a second forward for nothing.

## Serving on a Mac

Measured with v19 on an M3 Pro (36 GB) running macOS 26.6: torch 2.7.1, transformers
5.17.0, bf16 on MPS.
**The answers match the RTX 3090 run, but the latency does not.** Under 300 tokens, a v19
question takes a median 153 ms warm and 310 ms on the first request of its length. Across
JevBench the warm median is 234 ms and the 95th percentile is 2,628 ms. These figures are
for v19 on this Mac only. They are not a latency claim for other hardware or other
checkpoints.

The accuracy and latency measurements are in
[Serving on a Mac: accuracy and latency](../evaluation/results.md#serving-on-a-mac-accuracy-and-latency).

**One kernel had to be replaced.** Without `fla`, the reference chunk rule for the 18
Gated DeltaNet layers was 806 of 1,739 ms of a 1,024-token forward. Nearly all of it
was `torch.linalg.solve_triangular`, about 20 ms a call on MPS. `mps_kernels.py` inverts
the same unit-lower-triangular system by recursive block inversion, which is six levels
of batched matmuls, and applies the inverse to both right-hand sides. The engine
installs it for MPS only, and the forward is 1.7x faster (1,024 tokens: 1,750 to
1,029 ms). The obvious inverse, the finite Neumann product (I-N)(I+N²)(I+N⁴)..., passed
random-input tests and was wrong by up to 5e8 on real activations. Correlated,
l2-normalised keys make the powers of N huge before they cancel.
`tests/test_mps_kernels.py` builds its inputs that way and fails 14 cases against that
version.

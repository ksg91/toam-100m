# TOAM-100M

**TOAM-100M** is a 102.5M-parameter language model for **tool use and agent control**, trained from
scratch. You give it tool schemas and a request, and it returns the next action as JSON: call a tool
(or several), ask the user for a missing value or for confirmation, or say that no tool fits.

The question behind it: *how much agent-control ability can a very small model learn if it is
trained to interact with the world rather than to know it?* The target is useful capability per unit
of memory and latency, not the highest benchmark score.

- **Weights:** [ksg91/toam-100m-v0.2](https://huggingface.co/ksg91/toam-100m-v0.2) (safetensors + GGUF), Apache-2.0
- **Technical report:** [REPORT.md](REPORT.md), covering what was built, what worked and what didn't
- **All results:** [results/](results/README.md)

> **Research release.** TOAM is a research artefact, not a production model. Validate every tool call
> before it runs, and never let it trigger irreversible actions without human review.

## Results (v0.2)

BFCL v4 single-turn, same local harness for both models ([details](results/README.md)):

| Model | Params | Non-live AST | Live AST | Irrelevance | Latency p50 | Peak process RAM |
|---|---:|---:|---:|---:|---:|---:|
| **TOAM-100M v0.2** | 102.5M | 28.0 % | 26.2 % | 71.6 % | 207 ms | 1.67 GB |
| FunctionGemma-270M | 268.1M | 43.6 % | 26.6 % | 72.2 % | 731 ms | 2.06 GB |

With 38 % of FunctionGemma's parameters, TOAM is about level on live (real-user) requests and on
recognising when no tool fits, with 3.5× lower median latency in this harness. It is **clearly behind
on non-live requests**, mostly those that need several calls at once (parallel 16 % vs 36 %). On 84
tool servers held out from all training, it picks the right tool 81.7 % of the time (an in-house
test, [not independently verifiable](#what-is-not-released)).

All numbers come from single runs, on one machine (RTX 5080), measured on the FP32 training
checkpoint. The released FP16 weights score 27.6 / 26.2 / 71.9 % with the same harness (97 % of
outputs identical) at 286 ms p50 ([details](results/README.md#reproducing-section-1-from-the-released-weights)).

## Quickstart

```bash
pip install -r requirements.txt
python examples/quickstart.py
```

```python
from toam.inference import TinyAgent

agent = TinyAgent("ksg91/toam-100m-v0.2")
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]

agent.decide(tools, "What's the weather in Pune right now?")
# {'type': 'tool_call', 'tool': 'get_weather', 'arguments': {'city': 'Pune'}}
```

TOAM uses its own raw prompt format (no chat template). `toam/inference.py` builds the prompt,
decodes greedily and parses the action. The context is 2,048 tokens, so keep menus to about 10 tools.

### What it gets right and wrong

`examples/quickstart.py` runs 11 requests I wrote for this repository against a menu of 5 tools.
These are the real outputs (CPU, FP32), not a selection:

| Request | Output | |
|---|---|---|
| What's the weather in Pune right now? | `get_weather(city="Pune")` | ✓ |
| How warm is it in Chicago? Fahrenheit please. | `get_weather(city="Chicago", unit="Fahrenheit")` | ~ enum is lowercase `fahrenheit` |
| Weather in Paris and in Berlin, please. | two parallel `get_weather` calls | ✓ |
| Email priya@example.com with subject 'Lunch' and say see you at 1pm. | `send_email(to=…, subject="Lunch", body="see you at 1pm")` | ✓ |
| Send an email to my manager. | asks: "Which to should I use?" | ✓ right action, clumsy wording |
| How much is 250 US dollars in euros? | says no tool fits | ✗ `convert_currency` fits |
| Put 'Dentist' in my calendar tomorrow at 9:30. (state: today = 2026-10-02) | `create_calendar_event(date="tomorrow", …)` | ✗ date not resolved |
| Add a 1-hour sync with the design team on 2026-10-14 at 15:00. | says no tool fits | ✗ |
| Delete /reports/old_draft.docx (policy: confirm before deleting) | says no tool fits | ✗ should ask for confirmation |
| Book me a flight to Tokyo. | says no tool fits | ✓ |
| Who won the football world cup in 2010? | says no tool fits | ✓ |

6 right, 1 partly right, 4 wrong. Benchmarks give the averages; these show what the mistakes look like.

## Repository

| Path | What |
|---|---|
| [`toam/`](toam) | `model.py` (native PyTorch model), `inference.py` (prompt format, decoding, action parsing) |
| [`examples/`](examples) | quickstart |
| [`eval/bfcl/`](eval/bfcl) | BFCL v4 harness: prepare → generate → score with BFCL's own checker |
| [`pretrain/`](pretrain) | pretraining loop (WSD schedule, resumable sessions) and tokenizer training |
| [`release/`](release) | conversion to Hugging Face Llama format, FP16, GGUF, with equivalence checks |
| [`results/`](results/README.md) | all reported numbers and how they were measured |

## Reproduce the BFCL numbers

BFCL's scorer pins old dependencies (`numpy==1.26.4`), so it needs its own environment:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
python -m venv .venv-bfcl && .venv-bfcl/bin/pip install -r eval/bfcl/requirements.txt
PY=.venv/bin/python BFCL_PY=.venv-bfcl/bin/python bash eval/bfcl/run.sh
```

On Windows: `powershell -ExecutionPolicy Bypass -File eval\bfcl\run.ps1 -Py .venv\Scripts\python.exe -BfclPy .venv-bfcl\Scripts\python.exe`.

On an RTX 5080 the TOAM run takes about 15 minutes and the FunctionGemma run about 45. FunctionGemma requires accepting
Google's Gemma terms on Hugging Face first. Expected TOAM result on the released weights: 27.6 %
non-live, 26.2 % live, 71.9 % irrelevance. The published 28.0 / 26.2 / 71.6 % come from the FP32
training checkpoint, which is not released. A CPU run in FP32 gives slightly different text on a few
prompts.

## What is not released

- **Training data.** The fine-tuning data (generated in-house from real-world tool schemas, plus
  public datasets), its generators and the exact mixture are not released. Fine-tuning cannot be
  reproduced from this repository.
- **In-house evaluation sets.** The held-out-server and dialogue results in [results/](results/README.md)
  section 3 cannot be checked independently. The BFCL results can.
- **Pretraining corpus scripts.** `pretrain/pretrain.py` documents the token-shard format it reads.
  The corpus came from public datasets (listed below), decontaminated against the held-out tool servers
  but **not** against BFCL.

## How it was built

- **Architecture:** Llama-style decoder (RMSNorm, SwiGLU, RoPE, grouped-query attention with 12 query /
  4 KV heads), d=768, 12 layers, tied embeddings, 32k byte-level BPE with single-digit splitting,
  2,048-token context. Loads as a standard `LlamaForCausalLM`.
- **Pretraining:** 10B tokens of public web, educational, code, structured-data and encyclopedic text.
  Warmup-stable-decay schedule, peak LR 1e-3, 262k tokens per step. Ran for 34 GPU-hours in 8
  sessions on a single RTX 5080 (16 GB).
- **Fine-tuning:** one epoch on about 137k rows: in-house data built from real MCP tool schemas
  (about 55 %), plus public function-calling and human-written dialogue data. Every label was
  checked automatically: schema-valid, arguments traceable to the request, the state or a schema
  default.

The report has the reasoning, the dead ends and the corrections.

## Versions

| Version | Date | |
|---|---|---|
| **v0.2** | 2026-10 | Full 10B-token pretraining, audited fine-tuning data ([weights](https://huggingface.co/ksg91/toam-100m-v0.2)) |
| v0.1 | 2026-09 | Pretraining stopped early at 3.25B tokens ([weights](https://huggingface.co/ksg91/toam-100m-v0.1)) |

Versions are git tags. Larger TOAM models will get their own repositories.

## Citation

```bibtex
@software{gor_toam100m_2026,
  author  = {Gor, Kishan},
  title   = {{TOAM-100M}: a tiny language model for tool use and agent control},
  year    = {2026},
  version = {0.2},
  url     = {https://github.com/ksg91/toam-100m}
}
```

## Acknowledgements

Fine-tuning data includes openly licensed datasets from Salesforce and Argilla (APIGen / xLAM),
Team-ACE (ToolACE), NVIDIA (When2Call), Google (Schema-Guided Dialogue, CC BY-SA 4.0) and Amazon
(MASSIVE, CC BY 4.0). Pretraining data includes FineWeb-Edu, Cosmopedia (SmolLM corpus), CodeParrot,
StarCoderData, CodeSearchNet, Wikipedia, the APIs.guru OpenAPI directory and SchemaStore. Evaluation
uses the Berkeley Function-Calling Leaderboard (BFCL v4). GGUF conversion uses llama.cpp.

## License

Apache-2.0 for the code and weights. The datasets keep their own licenses.

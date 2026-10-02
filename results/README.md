# Results

All numbers are single runs (one fine-tuning seed, greedy decoding). Differences of 1–2 points
between versions are within what a different seed could produce. TOAM numbers in sections 1 and 2
were measured on the FP32 training checkpoints. For the same benchmark on the released FP16
weights, see the [last section](#reproducing-section-1-from-the-released-weights).

## 1. BFCL v4, single-turn (independent benchmark)

BFCL prompts and functions were not written by me. Every model was run locally with the harness in
[`eval/bfcl/`](../eval/bfcl) and scored with `bfcl_eval`'s own AST checker (`bfcl-eval==2026.3.23`).
This is not an official leaderboard submission. The FunctionGemma-270M numbers match its model
card, which is the check that the harness is correct.

| Model | Params | Non-live AST | Live AST | Irrelevance | Relevance (n=16) |
|---|---:|---:|---:|---:|---:|
| **TOAM-100M v0.2** | 102.5M | 28.0 % | 26.2 % | 71.6 % | 62.5 % |
| TOAM-100M v0.1 | 102.5M | 22.8 % | 22.9 % | 68.1 % | 56.2 % |
| FunctionGemma-270M | 268.1M | 43.6 % | 26.6 % | 72.2 % | 81.2 % |

### By category

| Category | n | TOAM v0.2 | TOAM v0.1 | FunctionGemma-270M |
|---|---:|---:|---:|---:|
| simple_python | 400 | 46.8 | 41.0 | 62.7 |
| simple_java | 100 | 28.0 | 23.0 | 37.0 |
| simple_javascript | 50 | 36.0 | 36.0 | 46.0 |
| multiple | 200 | 51.5 | 40.0 | 59.0 |
| parallel | 200 | 16.0 | 11.5 | 36.0 |
| parallel_multiple | 200 | 7.5 | 6.5 | 31.0 |
| irrelevance | 240 | 67.1 | 62.1 | 64.2 |
| live_simple | 258 | 33.7 | 27.5 | 32.9 |
| live_multiple | 1,053 | 24.9 | 22.3 | 25.3 |
| live_parallel | 16 | 12.5 | 18.8 | 18.8 |
| live_parallel_multiple | 24 | 12.5 | 0.0 | 25.0 |
| live_irrelevance | 884 | 76.0 | 74.1 | 80.2 |
| live_relevance | 16 | 62.5 | 56.2 | 81.2 |

Summary columns follow `bfcl_eval`: Non-live AST = unweighted mean of simple (average of
Python/Java/JavaScript), multiple, parallel and parallel_multiple; Live AST = count-weighted mean of the
four live categories; Irrelevance = unweighted mean of irrelevance and live_irrelevance.

**Context length.** 38 of the 3,641 prompts do not fit TOAM's 2,048-token context (22 live_multiple,
1 live_parallel_multiple, 15 live_irrelevance). They are scored as failures in every category,
including irrelevance, where BFCL's own rule ("no call = pass") would have counted them as correct.

**Where TOAM loses.** Most of the Non-live gap is requests that need several calls at once
(parallel 16.0 vs 36.0, parallel_multiple 7.5 vs 31.0) and maths/science-style functions.

## 2. Resources

Same harness, one request at a time, GPU otherwise idle.

| Model | Latency p50 / p95 | Peak VRAM | Peak process RAM |
|---|---:|---:|---:|
| TOAM-100M v0.2 | 207 / 542 ms | 626 MB | 1.67 GB |
| TOAM-100M v0.1 | 202 / 509 ms | 626 MB | 1.66 GB |
| FunctionGemma-270M | 731 / 1,996 ms | 578 MB | 2.06 GB |

| | |
|---|---|
| Hardware | NVIDIA RTX 5080 (16 GB), Windows 11 |
| Software | PyTorch 2.14 (CUDA 13.0), transformers 5.17 |
| TOAM | FP32 weights, BF16 autocast, plain decoding loop **without** KV cache, max 256 new tokens |
| FunctionGemma | BF16, transformers `generate` **with** KV cache, max 512 new tokens |
| Latency | Wall time per request, tokenization to decoded text |
| Peak RAM | Peak working set of the process, including the PyTorch/CUDA runtime |

These numbers compare two models in one harness, not optimised deployments. TOAM's peak VRAM is
higher than its size suggests because the harness keeps FP32 weights under BF16 autocast.

**CPU (llama.cpp, GGUF, 16 threads, `llama-server`):** median ≈ 280 ms (F16) and ≈ 330 ms (Q8_0) per
request on the same machine. The GGUF files tokenize identically to the transformers model on 39/39
prompts and give identical greedy outputs on 35/35 prompts (llama.cpp build b11223).

**Energy was not measured.**

## 3. In-house held-out evaluation (not independently verifiable)

These use evaluation sets that are not released, so they cannot be checked from this repository.

| Test | v0.1 | v0.2 |
|---|---:|---:|
| 84 real-world MCP tool servers held out from all training, including pretraining: tool correct | 78.2 % | 81.7 % |
| same, exact action (tool + all arguments) | – | 64.5 % |
| Human-written dialogues (Schema-Guided Dialogue) on services absent from training: right action type | – | 86.7 % |

The held-out servers were split off by whole server before any data was built. Their tool names,
identifiers and description text were filtered out of the pretraining corpus. As a control, blanking
the user request drops tool accuracy to ≤ 2 %, so the model's choices depend on the request.

## Reproducing section 1 from the released weights

The numbers in sections 1 and 2 were measured on the FP32 training checkpoint, running in the native
model (`toam/model.py`). The released Hugging Face weights are stored in FP16 and run through the
transformers Llama implementation. Running this repository's harness on the released weights
(`ksg91/toam-100m-v0.2`, same GPU, BF16 autocast) gives:

| | Training checkpoint (published) | Released weights, this harness |
|---|---:|---:|
| Non-live AST | 28.0 % | 27.6 % |
| Live AST | 26.2 % | 26.2 % |
| Irrelevance | 71.6 % | 71.9 % |
| Relevance | 62.5 % | 62.5 % |
| Raw outputs identical to the published run | | 3,545 / 3,641 (97.4 %) |
| Latency p50 / p95 | 207 / 542 ms | 286 / 731 ms |
| Peak VRAM / peak process RAM | 626 MB / 1.67 GB | 615 MB / 2.01 GB |

Per category on the released weights: simple_python 46.2, simple_java 28.0, simple_javascript 34.0,
multiple 51.5, parallel 16.0, parallel_multiple 7.0, irrelevance 67.5, live_simple 33.7,
live_multiple 24.9, live_parallel 12.5, live_parallel_multiple 12.5, live_irrelevance 76.4,
live_relevance 62.5.

The 96 differing outputs come from numerical differences: FP16 vs FP32 weights and a different
implementation of the same architecture. I did not separate the two effects. The higher latency and
RAM come from running through transformers rather than the plain PyTorch model. A CPU run in FP32
differs from both on a few prompts (36 of 39 identical in a 3-per-category smoke test).

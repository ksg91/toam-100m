# TOAM-100M: technical report

Kishan Gor · v0.2 · October 2026

## Summary

TOAM-100M is a 102.5M-parameter decoder trained from scratch for one job: given tool schemas and a
request, choose the next action. It was pretrained on 10B tokens on a single 16 GB GPU (34 GPU-hours)
and fine-tuned for one epoch on about 137k verified examples. On BFCL v4 it scores 28.0 % non-live,
26.2 % live and 71.6 % irrelevance, against 43.6 / 26.6 / 72.2 % for FunctionGemma-270M in the same
harness, at 3.5× lower median latency. It is behind on non-live requests, mostly those that need
several calls at once.

The most useful parts of this report are probably the negative results: an in-house metric that
looked good while the model ignored the request, and a stronger base model that made the benchmark
score worse because it copied label errors more faithfully.

## 1. The question

A general LLM spends much of its capacity storing knowledge. An agent controller can get most of the
knowledge it needs at run time, from tool schemas, tool results and state. So how small can a model
be when its main job is to **decide the next action**: call, call several, ask, confirm, refuse,
escalate? The aim is capability per unit of memory and latency, not the top score.

Every comparable small function-calling model I know of is a fine-tune of a general pretrained base.
TOAM is trained from scratch, so the whole pipeline (tokenizer, corpus, pretraining, fine-tuning,
evaluation) is under control and can be varied.

## 2. Model

Llama-style decoder: RMSNorm, SwiGLU, RoPE, grouped-query attention, tied embeddings, 2,048-token
context, 32k byte-level BPE tokenizer. Digits are split into single tokens, because IDs, amounts and
timestamps are arguments the model must copy exactly. The prompt markers (`<|tools|>`, `<|goal|>`,
…) are single special tokens.

**Wide or deep?** Two shapes of the same size were pretrained on 200M tokens with identical data
order, schedule and seed:

| | Deep | **Wide (chosen)** |
|---|---|---|
| Shape | d576 · 24 layers · 9 / 3 heads | d768 · 12 layers · 12 / 4 heads |
| Params | 106.5M | 102.5M |
| Validation loss (mixture) | 2.9450 | 2.9444 |
| Loss on structured text (JSON/YAML/schemas) | **2.125** | 2.144 |
| GPU prefill, 1,222-token prompt | 12.4 ms | **6.7 ms** |
| CPU prefill, 16 threads | 447 ms | **320 ms** |
| Training throughput | 57k tok/s | **74k tok/s** |

The deep model won every structured-text slice, by about 1 % relative loss, and the gap grew during
training. The wide one was 1.85× faster at GPU prefill. Agent prompts are long (≈1.2k tokens) and
outputs short (≈30 tokens), so prefill dominates latency, and wide was chosen for the latency claim.
The cost of that choice is a possible loss in structured-text skill, which I didn't test further.
(I had expected depth to cost little latency because decoding is memory-bound. That was wrong for
prefill, which is what matters here.)

## 3. Pretraining

**Corpus.** 19.6B tokens of public data: web and educational text (≈60 % of what the model saw), code
(≈25 %), structured data and API specifications (≈12 %) and encyclopedic text (≈5 %). Before
tokenization, documents mentioning the 84 held-out tool servers (names, identifiers, description
shingles) were removed. The corpus was **not** decontaminated against BFCL.

**Schedule.** Peak LR 1e-3 (a short sweep: 5e-4 led early and lost by 286 steps, 2e-3 trailed
throughout), 262k tokens per step, warmup-stable-decay (WSD) with 2 % warmup and a 20 % linear decay.
WSD was chosen because it allows early stopping. v0.1 was released from a short early decay at 3.25B
tokens, and the 10B run then continued from the checkpoint saved before that decay.

**Running it on one consumer GPU.**

- Ran in sessions of up to 7 hours, around normal use of the machine. Stopping is soft (finish the
  step, validate, save atomically). Resume refuses to start if any setting that changes the training
  trajectory differs from the original run.
- Windows silently pages GPU memory that overflows into system RAM. There is no OOM error, only a
  stall at 100 % utilisation. A hard per-process VRAM cap turns this into a normal OOM.
- `torch.compile` plus micro-batch 8 × accumulation 16 (instead of 4 × 32) gave +19 % throughput
  (86.5k vs 72.4k tok/s). It uses the same 128 sequences per step in the same order, so the trajectory
  is unchanged: compiled and eager loss matched, and gradient norms agreed within 0.05 %.
- At step 17,229 I found the loss had been computed on BF16 logits. It is computed in FP32 from that
  step on.

The final validation loss was 2.198 at 10B tokens, against 2.311 for the v0.1 base at 3.25B. The
whole run took 34.05 GPU-hours over 8 sessions.

## 4. Fine-tuning data

This part is described only at a high level, because the data and its generators are not released.

- About 55 % in-house data built from 17,266 real MCP tool schemas: single-step requests, minimal
  pairs that differ in one detail and flip the right action (call vs ask vs confirm), and multi-step
  episodes in simulated tool environments.
- About 45 % public function-calling data and human-written dialogue and intent data, with 25 %
  typo/noise augmentation.
- **Every label is checked by code, not trusted.** Calls must be schema-valid. Every argument must be
  traceable to the request, the state or a schema default; if it isn't, the correct action is to ask.
- Rows overlapping BFCL (function names, 8-gram overlap) or any held-out tool are removed.

## 5. Evaluation design

Three tiers, reported separately so that close-to-training numbers can't stand in for generalisation:

1. **Independent:** BFCL v4 single-turn, scored with `bfcl_eval`'s own checker. The harness was
   validated by reproducing FunctionGemma-270M's published numbers before any TOAM number was
   trusted.
2. **Held-out in-house:** 84 real tool servers split off by whole server, before any data was built,
   and kept out of pretraining as well. Results are stratified by group and never taken from the first
   N rows of a file. (That mistake happened once early on, on a file sorted by group.)
3. **Close to training:** unseen tools from training servers, simulated episodes. Read only as a
   sanity check.

Two **controls** run with every evaluation: blank out the request (tool accuracy should collapse)
and rename the tools (it should not). Section 6.1 explains why.

## 6. What I learned

### 6.1 A strong in-house number hid a model that ignored the request

The first fine-tune used simulated multi-step episodes only. Every model got the same fine-tune and
was tested on 4 held-out task families, 100 rows each. (These families came from training-split
servers, which were not removed from pretraining.)

| | From scratch | 0.2B pretraining tokens | 1B pretraining tokens |
|---|---:|---:|---:|
| Tool selection on unseen tools | 18.2 % | 67.2 % | 72.5 % |
| Valid JSON | 45.8 % | 93.0 % | 97.0 % |

The same model scored **0.0 % non-live and 0.2 % live on BFCL**. It answered almost everything with
generic `search_…` calls, and only 3.8 % of its calls named a tool that was in the prompt. Input
ablations showed why:

- blanking the request left tool selection **unchanged** (66 % → 64 %);
- renaming the tools dropped it to **0 %**.

The model had learned a fixed episode script over a few dozen tools. The request was redundant in that
data, so it never learned to read it. The 72.5 % was real, but it measured the wrong thing. The base
model alone (1B tokens), prompted with three examples, scored 0 % on BFCL. At this size the skill has
to come from the fine-tuning data.

The fix was data that makes the request necessary: many tools per menu from many servers,
near-duplicate distractors, the same menu with different requests leading to different tools, and
arguments taken from prose. On the **same 1B-token checkpoint**, BFCL went from 0.0 / 0.2 % to
**18.2 / 11.3 %**, 98.4 % of calls named an offered tool, and blanking the request dropped held-out
tool accuracy to 2 %. Since then the two controls have been part of every evaluation.

*Lesson: a metric needs a control that shows the model uses the input you think it uses.*

### 6.2 Generated data teaches structure; public data teaches request language

Three fine-tunes of equal size (~74.7k rows) on the same checkpoint:

| Data | Held-out servers, tool | BFCL non-live |
|---|---:|---:|
| In-house generated only | 64.8 % | 8.7 % |
| Public only | 29.5 % | 21.2 % |
| Mix | 64.2 % | 16.4 % |

Neither source is enough alone. The generated data's BFCL failures traced back to its phrasing: values
quoted or written as "set to X", too few numeric arguments, and missing-value requests that named the
slot. This taught the model to ask whenever a value wasn't in its usual format. The in-house test
flatters in-house data (both come from the same pipeline), which is why BFCL is the neutral judge.
The release mix added human-written requests and noise augmentation.

### 6.3 A better base copies label errors more faithfully

Same fine-tuning data, 3.25B → 10B pretraining tokens:

| | 3.25B (v0.1) | 10B, same data |
|---|---:|---:|
| Held-out servers, tool | 78.2 % | 83.4 % |
| BFCL non-live | 22.8 % | 27.6 % |
| BFCL live | 22.9 % | **20.5 %** |

Live got *worse*. Of the model's requests for a value on BFCL live, 95 % (141 of 149) asked for an
argument the user had already given. An audit of the training labels found the same mistake in both
public and in-house data: an estimated 25–35 % of one public dataset's "ask the user" labels asked
for stated values, often against schemas that required nothing. The stronger base learned that
habit more faithfully.

On the candidate trained with the minimal pairs added (live 20.4 %), removing about 1,800 of those
labels cut value requests on live from 13.9 % to 3.4 % of answers and raised live to 25.8 %. Two
smaller changes went into the same step: 524 leaked rows were removed (section 6.4) and about 2,300
public direct-call rows were added. The released v0.2, which also has cleaner generated data, has
2.2 % and **26.2 %**.

*Lesson: as the base improves, label errors cost more, not less. At this point the limit looks like
data quality, not model capacity. That is a hypothesis for the next version, not a measured result.*

### 6.4 Keeping the evaluation honest

- **Leak.** While preparing v0.2 I found that one data addition had skipped a hold-out filter: 524
  fine-tuning rows targeted 146 of the 235 tools in one in-house "unseen tools" test, which inflated
  that test for two candidate models. BFCL and the held-out-server test were not affected. The rows
  were removed, and the released model has none.
- **Selection rule.** Before the three v0.2 candidates finished, I fixed a rule: highest mean of the
  three BFCL scores, provided held-out tool accuracy stays within 2 points of the baseline. The rule
  picked a candidate by 0.6 points, but that candidate's live score (20.4 %) was below v0.1's
  (22.9 %). I overrode the rule and released the candidate that improves on v0.1 on every headline
  number. The override and its reason are part of the record.

## 7. Limitations

- **Several calls in one request** is the main weakness (BFCL parallel 16 %, parallel_multiple 7.5 %).
  On long multi-part requests the model can drop a sub-task, or change a value it should keep when it
  repeats a call.
- **Value binding:** with several values in one request, it sometimes puts one in the wrong field.
- Weak on maths/science-style functions, which make up much of BFCL non-live. Its questions to the user
  can be clumsy. English only, 2,048-token context.
- **Evidence:** single fine-tuning seed, one machine, no energy measurement. The in-house results can't
  be verified independently. The pretraining corpus was not decontaminated against BFCL.

## 8. Next

Every next experiment changes only the fine-tuning data on the frozen 10B checkpoint: label
verification at scale, natural wording, and more multi-call examples. Each change is measured on BFCL
and on a human-written development set, before any change to model size.

"""
Stage 3 of the BFCL harness (run in the BFCL environment): score raw outputs
with BFCL's own checker functions, unmodified.

Per entry we call bfcl_eval's `_evaluate_single_ast_entry` /
`_evaluate_single_relevance_entry` — the same functions `bfcl evaluate` calls —
passing a thin handler whose only job is `decode_ast`:

  toam          TOAM's output is one JSON action. `tool_call` and `retry` carry
                a tool + arguments and decode to [{tool: arguments}];
                `tool_calls` decodes to one entry per call; every other action
                type (ask_user, finish, escalate, wait) is "no function call"
                and decodes to []. Invalid JSON raises, exactly like a failed
                AST decode.
  functiongemma BFCL's FunctionGemmaHandler.decode_ast.

Summary columns follow bfcl_eval.eval_checker.eval_runner_helper:
  Non-live AST = unweighted mean(simple[avg py/java/js], multiple, parallel, parallel_multiple)
  Live AST     = count-weighted mean(live_simple, live_multiple, live_parallel, live_parallel_multiple)
  Irrelevance  = unweighted mean(irrelevance, live_irrelevance)
  Relevance    = live_relevance

Deviation (conservative): an entry whose prompt does not fit a model's context
window is scored as a FAILURE in every category, including irrelevance, where
BFCL's rule ("no call decoded = pass") would otherwise reward it.

    python eval/bfcl/score.py toam_100m_v0.2 functiongemma_270m
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from bfcl_eval.constants.enums import Language, ReturnFormat
from bfcl_eval.eval_checker.eval_runner import (
    _evaluate_single_ast_entry,
    _evaluate_single_relevance_entry,
)
from bfcl_eval.model_handler.local_inference.functiongemma import FunctionGemmaHandler
from bfcl_eval.utils import load_dataset_entry, load_ground_truth_entry

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Only used by ast_checker.convert_func_name to look up `underscore_to_dot`.
# FunctionGemma's registry entry has underscore_to_dot=False, i.e. function
# names are compared verbatim, which is also right for TOAM (it emits the
# name exactly as it appears in the tool list).
CHECKER_MODEL_NAME = "google/functiongemma-270m-it-FC"

NON_LIVE_AST = ["simple_python", "simple_java", "simple_javascript",
                "multiple", "parallel", "parallel_multiple"]
LIVE_AST = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
RELEVANCE = ["irrelevance", "live_irrelevance", "live_relevance"]


class ToamDecoder:
    def decode_ast(self, result, language=None, has_tool_call_tag=False):
        action = json.loads(result)  # raises on invalid JSON -> decoder failure
        if not isinstance(action, dict):
            raise ValueError("action is not a JSON object")
        if action.get("type") in ("tool_call", "retry"):
            args = action.get("arguments", {})
            if not isinstance(args, dict):
                raise ValueError("arguments is not an object")
            return [{action["tool"]: args}]
        if action.get("type") == "tool_calls":  # several calls in one action (parallel categories)
            out = []
            for c in action.get("calls") or []:
                if not isinstance(c, dict) or not isinstance(c.get("arguments", {}), dict):
                    raise ValueError("malformed call")
                out.append({c["tool"]: c.get("arguments", {})})
            if not out:
                raise ValueError("tool_calls without calls")
            return out
        return []


def decoder_for(meta: dict):
    if meta["kind"] == "toam":
        return ToamDecoder()
    return FunctionGemmaHandler(meta["hf_model"], 0.001, CHECKER_MODEL_NAME, True)


def language_of(cat: str):
    if cat.endswith("java"):
        return Language.JAVA, ReturnFormat.JAVA
    if cat.endswith("javascript"):
        return Language.JAVASCRIPT, ReturnFormat.JAVASCRIPT
    return Language.PYTHON, ReturnFormat.PYTHON


def score_run(name: str) -> dict:
    run = RESULTS / name
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
    if meta.get("limit"):
        print(f"WARNING: {name} is a --limit {meta['limit']} smoke run, not a full result")
    dec = decoder_for(meta)
    per_cat, errors = {}, {}
    for cat in NON_LIVE_AST + LIVE_AST + RELEVANCE:
        path = run / f"{cat}.jsonl"
        if not path.exists():
            continue
        outs = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        prompts = {e["id"]: e for e in load_dataset_entry(
            cat, include_prereq=False, include_language_specific_hint=False)}
        truth = {} if cat in RELEVANCE else {g["id"]: g for g in load_ground_truth_entry(cat)}
        lang, fmt = language_of(cat)
        ok, over, err_types = 0, 0, Counter()
        for o in outs:
            if o.get("overlength"):
                over += 1
                err_types["overlength:prompt_exceeds_context"] += 1
                continue
            if cat in RELEVANCE:
                r = _evaluate_single_relevance_entry(dec, o["id"], o["raw"], prompts[o["id"]],
                                                     CHECKER_MODEL_NAME, cat)
            else:
                r = _evaluate_single_ast_entry(dec, o["id"], o["raw"], truth[o["id"]]["ground_truth"],
                                               prompts[o["id"]], CHECKER_MODEL_NAME, cat,
                                               language=lang, return_format=fmt)
            if r["valid"]:
                ok += 1
            else:
                err_types[r.get("error_type", "?")] += 1
        per_cat[cat] = {"accuracy": ok / len(outs), "n": len(outs), "correct": ok, "overlength": over}
        errors[cat] = dict(err_types.most_common())

    def acc(c):
        return per_cat[c]["accuracy"] if c in per_cat else None

    summary = {}
    if all(c in per_cat for c in NON_LIVE_AST):
        simple = (acc("simple_python") + acc("simple_java") + acc("simple_javascript")) / 3
        summary["non_live_simple"] = simple
        summary["non_live_ast"] = (simple + acc("multiple") + acc("parallel") + acc("parallel_multiple")) / 4
    if all(c in per_cat for c in LIVE_AST):
        n = sum(per_cat[c]["n"] for c in LIVE_AST)
        summary["live_ast"] = sum(per_cat[c]["correct"] for c in LIVE_AST) / n
    if "irrelevance" in per_cat and "live_irrelevance" in per_cat:
        summary["irrelevance"] = (acc("irrelevance") + acc("live_irrelevance")) / 2
    if "live_relevance" in per_cat:
        summary["relevance"] = acc("live_relevance")

    # Latency / tokens across all generated entries.
    lat, tin, tout = [], [], []
    for p in run.glob("*.jsonl"):
        for l in p.read_text(encoding="utf-8").splitlines():
            o = json.loads(l)
            if not o.get("overlength"):
                lat.append(o["latency_s"]); tin.append(o["input_tokens"]); tout.append(o["output_tokens"])
    lat.sort()
    res = {
        "name": name, "meta": meta, "summary": summary, "per_category": per_cat,
        "error_types": errors,
        "latency": {"n": len(lat), "mean_s": sum(lat) / len(lat), "p50_s": lat[len(lat) // 2],
                    "p95_s": lat[int(len(lat) * 0.95)],
                    "mean_input_tokens": sum(tin) / len(tin), "mean_output_tokens": sum(tout) / len(tout)},
    }
    (run / "score.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main():
    names = sys.argv[1:] or sorted(p.name for p in RESULTS.iterdir() if (p / "meta.json").exists())
    rows = [score_run(n) for n in names]
    cats = NON_LIVE_AST + ["irrelevance"] + LIVE_AST + ["live_irrelevance", "live_relevance"]
    w = 26
    print("\n" + "category".ljust(w) + "".join(r["name"][:22].rjust(24) for r in rows))
    for c in cats:
        print(c.ljust(w) + "".join(
            (f"{r['per_category'][c]['accuracy']*100:6.1f}%  (n={r['per_category'][c]['n']:4})"
             if c in r["per_category"] else "—").rjust(24) for r in rows))
    print("-" * (w + 24 * len(rows)))
    for k in ["non_live_ast", "live_ast", "irrelevance", "relevance"]:
        print(k.ljust(w) + "".join(
            (f"{r['summary'][k]*100:6.1f}%" if k in r["summary"] else "—").rjust(24) for r in rows))
    print("-" * (w + 24 * len(rows)))
    for k, f in [("params (M)", lambda r: f"{r['meta']['parameters']/1e6:.1f}"),
                 ("peak VRAM (MB)", lambda r: f"{r['meta']['peak_vram_mb']:.0f}"
                  if r["meta"]["peak_vram_mb"] is not None else "— (cpu)"),
                 ("peak RAM (MB)", lambda r: f"{r['meta']['peak_rss_mb']:.0f}"),
                 ("latency p50 (ms)", lambda r: f"{r['latency']['p50_s']*1000:.0f}"),
                 ("latency p95 (ms)", lambda r: f"{r['latency']['p95_s']*1000:.0f}"),
                 ("overlength entries", lambda r: str(sum(v['overlength'] for v in r['per_category'].values())))]:
        print(k.ljust(w) + "".join(f(r).rjust(24) for r in rows))


if __name__ == "__main__":
    main()

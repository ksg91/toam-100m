"""
Stage 1 of the BFCL harness (run in the BFCL environment).

Loads BFCL v4 single-turn entries exactly as `bfcl generate` does (with the
language-specific hints appended to function docs), and renders the
FunctionGemma prompt with BFCL's own FunctionGemmaHandler, so that model is
prompted byte-identically to the official leaderboard run.

TOAM prompts are not rendered here: generate.py renders them with
`toam.inference.format_prompt`, in the main environment.

    python eval/bfcl/prepare.py
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from bfcl_eval.model_handler.local_inference.functiongemma import FunctionGemmaHandler
from bfcl_eval.utils import load_dataset_entry

# Every single-turn category in BFCL v4. Multi-turn, memory, web-search and
# format-sensitivity need an executable environment or >2k context and are
# out of scope for this harness.
CATEGORIES = [
    "simple_python", "simple_java", "simple_javascript", "multiple",
    "parallel", "parallel_multiple", "irrelevance",
    "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
    "live_irrelevance", "live_relevance",
]

OUT = Path(__file__).resolve().parent / "prepared"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    fg = FunctionGemmaHandler(
        "google/functiongemma-270m-it", temperature=0.001,
        registry_name="google/functiongemma-270m-it-FC", is_fc_model=True,
    )
    for cat in CATEGORIES:
        entries = load_dataset_entry(cat)  # generation-time view: hints included
        n = 0
        with (OUT / f"{cat}.jsonl").open("w", encoding="utf-8") as f:
            for e in entries:
                assert len(e["question"]) == 1, f"{e['id']} is multi-turn"
                e_fg = copy.deepcopy(e)
                data = fg._pre_query_processing_prompting(e_fg)
                data = fg.add_first_turn_message_prompting(data, e_fg["question"][0])
                prompt = fg._format_prompt(data["message"], data["function"])
                f.write(json.dumps({
                    "id": e["id"],
                    "category": cat,
                    "messages": e["question"][0],
                    "function": e["function"],
                    "functiongemma_prompt": prompt,
                }, ensure_ascii=False) + "\n")
                n += 1
        print(f"{cat:24} {n:5}")


if __name__ == "__main__":
    main()

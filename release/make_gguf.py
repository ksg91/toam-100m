"""
Convert an exported TOAM HF folder to GGUF with llama.cpp's converter (unmodified).

Needs a llama.cpp source checkout (tested with build b11223) at $LLAMA_CPP or
../llama.cpp, plus its Python requirements.

Two adaptations, applied by patching at runtime:
  * TOAM loads as a Llama model but its tokenizer is a byte-level BPE (tokenizer.json), not
    SentencePiece: go straight to the GPT-2-style BPE vocab path.
  * llama.cpp identifies BPE pre-tokenizers by a hash of a test encoding; ours is new. Its
    pre-tokenizer is Sequence[Digits(individual), ByteLevel(GPT-2 regex)], which is exactly
    llama.cpp's "starcoder" pre-tokenizer ("\\p{N}" then the GPT-2 regex).

    python release/make_gguf.py release/toam-hf release/toam-100m-f16.gguf f16
    python release/make_gguf.py release/toam-hf release/toam-100m-q8_0.gguf q8_0
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LCPP = Path(os.environ.get("LLAMA_CPP", ROOT.parent / "llama.cpp"))
sys.path.insert(0, str(LCPP))
sys.path.insert(1, str(LCPP / "gguf-py"))

import conversion.base as base  # noqa: E402
import conversion.llama as llama  # noqa: E402

_orig_pre = base.TextModel.get_vocab_base_pre


def _pre(self, tokenizer) -> str:
    try:
        return _orig_pre(self, tokenizer)
    except NotImplementedError:
        return "starcoder"


base.TextModel.get_vocab_base_pre = _pre
llama.LlamaModel.set_vocab = lambda self: self._set_vocab_gpt2()


def main():
    src, out, outtype = sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "bf16"
    spec = importlib.util.spec_from_file_location("convert_hf_to_gguf", LCPP / "convert_hf_to_gguf.py")
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)
    sys.argv = ["convert_hf_to_gguf.py", src, "--outfile", out, "--outtype", outtype]
    conv.main()


if __name__ == "__main__":
    main()

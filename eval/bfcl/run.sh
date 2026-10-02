#!/usr/bin/env bash
# Full BFCL v4 single-turn run: TOAM-100M v0.2 and FunctionGemma-270M, same harness.
# Run from anywhere, with the GPU otherwise idle so latency is not disturbed.
# Resumable: finished categories are skipped.
#
# Two Python environments, because bfcl-eval pins numpy==1.26.4 and other exact versions:
#   PY       main environment   (pip install -r requirements.txt)
#   BFCL_PY  BFCL environment   (pip install -r eval/bfcl/requirements.txt)
#
#   PY=.venv/bin/python BFCL_PY=.venv-bfcl/bin/python bash eval/bfcl/run.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=${PY:-python}
BFCL_PY=${BFCL_PY:-python}
export PYTHONUNBUFFERED=1 PYTHONIOENCODING=utf-8

"$BFCL_PY" eval/bfcl/prepare.py
"$PY" eval/bfcl/generate.py --kind toam --model ksg91/toam-100m-v0.2 --name toam_100m_v0.2
"$PY" eval/bfcl/generate.py --kind hf --hf-model google/functiongemma-270m-it --name functiongemma_270m --max-new 512
"$BFCL_PY" eval/bfcl/score.py toam_100m_v0.2 functiongemma_270m

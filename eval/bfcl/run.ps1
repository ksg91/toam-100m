# Full BFCL v4 single-turn run: TOAM-100M v0.2 and FunctionGemma-270M, same harness.
# Run with the GPU otherwise idle so latency is not disturbed. Resumable: finished
# categories are skipped.
#
# Two Python environments, because bfcl-eval pins numpy==1.26.4 and other exact versions:
#   -Py      main environment   (pip install -r requirements.txt)
#   -BfclPy  BFCL environment   (pip install -r eval\bfcl\requirements.txt)
#
#   powershell -ExecutionPolicy Bypass -File eval\bfcl\run.ps1 -Py .venv\Scripts\python.exe -BfclPy .venv-bfcl\Scripts\python.exe

param(
    [string]$Py = "python",
    [string]$BfclPy = "python"
)

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..\..")
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

function Run([string]$exe, [string[]]$argv) {
    & $exe @argv
    if ($LASTEXITCODE -ne 0) { throw "failed: $exe $($argv -join ' ')" }
}

Run $BfclPy @("eval\bfcl\prepare.py")
Run $Py @("eval\bfcl\generate.py", "--kind", "toam", "--model", "ksg91/toam-100m-v0.2", "--name", "toam_100m_v0.2")
Run $Py @("eval\bfcl\generate.py", "--kind", "hf", "--hf-model", "google/functiongemma-270m-it", "--name", "functiongemma_270m", "--max-new", "512")
Run $BfclPy @("eval\bfcl\score.py", "toam_100m_v0.2", "functiongemma_270m")

"""
Pre-push check: fails if anything that must stay private would be published.

Scans every file git would commit (tracked + untracked, minus .gitignore) and
every commit message, for:
  - data and weight files (.jsonl, .parquet, .pt, .safetensors, .gguf, ...),
    and any file over 1 MB;
  - JSON files that look like data rows (lists of objects);
  - denylisted terms (private pipeline names, local paths), read from
    scripts/denylist.txt if present (kept out of git, one regex per line),
    plus the built-in list below.

    python scripts/check_public.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

BLOCKED_SUFFIXES = {".jsonl", ".parquet", ".pt", ".pth", ".ckpt", ".safetensors", ".gguf", ".bin",
                    ".npy", ".npz", ".arrow", ".gz", ".zip", ".docx", ".log"}
MAX_BYTES = 1_000_000

BUILTIN_TERMS = [
    r"[A-Za-z]:[\\/]+kAI",          # local project paths
    r"[\\/]Users[\\/]",             # home directories
]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def main() -> int:
    terms = list(BUILTIN_TERMS)
    local = ROOT / "scripts" / "denylist.txt"
    if local.exists():
        terms += [l.strip() for l in local.read_text(encoding="utf-8").splitlines()
                  if l.strip() and not l.startswith("#")]
    pattern = re.compile("|".join(f"(?:{t})" for t in terms), re.IGNORECASE)

    files = sorted(set(git("ls-files", "--cached", "--others", "--exclude-standard").split("\n")) - {""})
    problems: list[str] = []
    for rel in files:
        if rel == "scripts/denylist.txt":
            problems.append(f"{rel}: the denylist itself must not be committed")
            continue
        p = ROOT / rel
        if not p.is_file():
            continue
        if p.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f"{rel}: blocked file type")
            continue
        if p.stat().st_size > MAX_BYTES:
            problems.append(f"{rel}: larger than 1 MB")
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            if p.suffix.lower() not in {".png", ".jpg", ".svg"}:
                problems.append(f"{rel}: binary file")
            continue
        if p.suffix.lower() == ".json":
            try:
                obj = json.loads(text)
                if isinstance(obj, list) and len(obj) > 3 and all(isinstance(x, dict) for x in obj):
                    problems.append(f"{rel}: looks like data rows")
            except json.JSONDecodeError:
                pass
        for i, line in enumerate(text.splitlines(), 1):
            m = pattern.search(line)
            if m:
                problems.append(f"{rel}:{i}: denylisted term {m.group(0)!r}")

    try:
        log = git("log", "--all", "--format=%an <%ae>%n%cn <%ce>%n%B")
    except subprocess.CalledProcessError:
        log = ""  # no commits yet
    for line in log.splitlines():
        m = pattern.search(line) or re.search(r"co-authored-by", line, re.IGNORECASE)
        if m:
            problems.append(f"commit message/author: {line.strip()!r}")

    print(f"checked {len(files)} files")
    if problems:
        print("\n".join(problems))
        print(f"\nFAILED: {len(problems)} problem(s)")
        return 1
    print("OK: nothing private found")
    return 0


if __name__ == "__main__":
    sys.exit(main())

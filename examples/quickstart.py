"""
Run TOAM-100M on a few hand-written requests.

    python examples/quickstart.py                         # downloads ksg91/toam-100m-v0.2
    python examples/quickstart.py path/to/local/model     # or a local HF folder

Runs on CPU in a few seconds per request; uses CUDA if available.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from toam.examples import REQUESTS, TOOLS  # noqa: E402
from toam.inference import TinyAgent  # noqa: E402


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "ksg91/toam-100m-v0.2"
    agent = TinyAgent(model)
    print("Tools:", ", ".join(t["function"]["name"] for t in TOOLS), "\n")
    for request, policy, state in REQUESTS:
        action = agent.decide(TOOLS, request, policy=policy, state=state)
        print(f"> {request}")
        if policy:
            print(f"  (policy: {policy})")
        print("  " + json.dumps(action, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

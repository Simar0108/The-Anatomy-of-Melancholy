#!/usr/bin/env python3
"""
Single entrypoint to run the full pipeline (Phases 1–7) in order.
Useful for graders / reproducibility.

Run from project root with venv activated:
  source .venv/bin/activate
  python run_pipeline.py

Optional: --through N runs only through phase N (e.g. --through 2 for data + features).
          --skip 1,2 skips phases 1 and 2 (e.g. if you already have corpus and features).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def run_phase(phase: int) -> None:
    script = PROJECT_ROOT / f"run_phase{phase}.py"
    if not script.exists():
        raise FileNotFoundError(script)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    p = argparse.ArgumentParser(description="Run The Anatomy of Melancholy pipeline (Phases 1–7).")
    p.add_argument("--through", type=int, default=7, metavar="N",
                   help="Run through phase N (default: 7)")
    p.add_argument("--skip", type=str, default="", metavar="1,2,3",
                   help="Comma-separated phase numbers to skip (e.g. 1,2)")
    args = p.parse_args()

    skip = {int(x.strip()) for x in args.skip.split(",") if x.strip()}
    for phase in range(1, args.through + 1):
        if phase in skip:
            print(f"[ Skipping phase {phase} ]")
            continue
        print(f"\n{'='*60}\nPhase {phase}\n{'='*60}")
        run_phase(phase)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()

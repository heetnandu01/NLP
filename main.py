"""Root-level launcher for the opinion mining project."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent / "opinion-mining-project"
PROJECT_MAIN = PROJECT_DIR / "main.py"


def main() -> None:
    """Run the actual project entrypoint from the nested project folder."""
    if not PROJECT_MAIN.exists():
        raise FileNotFoundError(f"Could not find project entrypoint: {PROJECT_MAIN}")

    sys.path.insert(0, str(PROJECT_DIR))
    runpy.run_path(str(PROJECT_MAIN), run_name="__main__")


if __name__ == "__main__":
    main()
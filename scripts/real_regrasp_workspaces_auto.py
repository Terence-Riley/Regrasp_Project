#!/usr/bin/env python3
"""Fully automatic workspace regrasp entry point.

This script forwards to scripts/real_regrasp_workspaces.py with:
    --auto --yes

It does not ask for Enter between motion segments and does not ask for the
initial REALREGRASP confirmation. Use only after verifying the same parameters
with --dry-run and a one-cup non-auto test.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from scripts.real_regrasp_workspaces import main as real_regrasp_main  # noqa: E402


def main():
    for flag in ("--auto", "--yes"):
        if flag not in sys.argv:
            sys.argv.append(flag)
    return real_regrasp_main()


if __name__ == "__main__":
    raise SystemExit(main())

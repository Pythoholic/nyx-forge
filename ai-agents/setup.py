"""Idempotent, agent-safe setup for NyxForge.

This helper installs repository dependencies only. It deliberately does not
start services, download model checkpoints, or change backend configuration.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV = ROOT / ".venv"
FRONTEND = ROOT / "frontend"


def _run(command: list[str], *, cwd: Path = ROOT) -> None:
    print(f"\n> {' '.join(command)}")
    subprocess.run(command, cwd=cwd, check=True)


def _venv_python() -> Path:
    if sys.platform == "win32":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def _npm() -> str:
    executable = shutil.which("npm.cmd" if sys.platform == "win32" else "npm")
    if not executable:
        raise SystemExit(
            "npm was not found. Install Node.js 22 LTS (or another version "
            "supported by frontend/package.json), reopen the terminal, and retry."
        )
    return executable


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Install NyxForge repository dependencies without starting services."
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="install requirements-dev.txt instead of runtime-only requirements.txt",
    )
    parser.add_argument(
        "--skip-frontend",
        action="store_true",
        help="skip frontend dependency installation and build",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="install frontend dependencies but do not build the frontend",
    )
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        raise SystemExit("Python 3.10 or newer is required; Python 3.12 is recommended.")

    if not _venv_python().exists():
        print(f"Creating virtual environment at {VENV}")
        _run([sys.executable, "-m", "venv", str(VENV)])
    else:
        print(f"Using existing virtual environment at {VENV}")

    python = str(_venv_python())
    requirements = "requirements-dev.txt" if args.dev else "requirements.txt"
    _run([python, "-m", "pip", "install", "--upgrade", "pip"])
    _run([python, "-m", "pip", "install", "-r", requirements])

    if not args.skip_frontend:
        npm = _npm()
        _run([npm, "ci"], cwd=FRONTEND)
        if not args.skip_build:
            _run([npm, "run", "build"], cwd=FRONTEND)

    relative_python = (
        ".venv\\Scripts\\python.exe"
        if sys.platform == "win32"
        else ".venv/bin/python"
    )
    print("\nSetup complete. No application or model backend was started.")
    print(f"Run diagnostics: {relative_python} ai-agents/doctor.py")
    print(f"Start the app:    {relative_python} run.py start")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

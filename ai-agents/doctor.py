"""Read-only environment diagnostics for NyxForge."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VENV_PYTHON = (
    ROOT / ".venv" / "Scripts" / "python.exe"
    if sys.platform == "win32"
    else ROOT / ".venv" / "bin" / "python"
)


@dataclass(frozen=True)
class Check:
    status: str
    name: str
    detail: str
    required: bool = True


def _command_version(command: str, argument: str = "--version") -> str | None:
    executable = shutil.which(command)
    if not executable:
        return None
    try:
        result = subprocess.run(
            [executable, argument],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (result.stdout or result.stderr).strip().splitlines()[0]


def _node_supported(version: str | None) -> bool:
    if not version:
        return False
    match = re.search(r"(\d+)\.(\d+)", version)
    if not match:
        return False
    major, minor = (int(value) for value in match.groups())
    return major > 22 or (major == 22 and minor >= 12) or (major == 20 and minor >= 19)


def _missing_modules(python: Path, modules: tuple[str, ...]) -> list[str]:
    script = (
        "import importlib.util, sys; "
        "sys.stdout.write('\\n'.join(name for name in sys.argv[1:] "
        "if importlib.util.find_spec(name) is None))"
    )
    try:
        result = subprocess.run(
            [str(python), "-c", script, *modules],
            capture_output=True,
            check=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return list(modules)
    return [name for name in result.stdout.splitlines() if name]


def _service(name: str, url: str) -> Check:
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "nyx-forge-doctor"})
        with urllib.request.urlopen(request, timeout=2) as response:
            return Check("OK", name, f"reachable (HTTP {response.status})", required=False)
    except (OSError, urllib.error.URLError, TimeoutError):
        return Check("INFO", name, "not reachable; optional for code-only work", required=False)


def main() -> int:
    checks: list[Check] = []

    python_ok = sys.version_info >= (3, 10)
    checks.append(
        Check(
            "OK" if python_ok else "FAIL",
            "Python",
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
            + ("" if python_ok else " (3.10+ required)"),
        )
    )

    git_version = _command_version("git")
    checks.append(Check("OK" if git_version else "FAIL", "Git", git_version or "not found"))

    node_version = _command_version("node")
    node_ok = _node_supported(node_version)
    checks.append(
        Check(
            "OK" if node_ok else "FAIL",
            "Node.js",
            (node_version or "not found")
            + ("" if node_ok else " (Node 20.19+ or 22.12+ required)"),
        )
    )

    npm_version = _command_version("npm.cmd" if sys.platform == "win32" else "npm")
    checks.append(Check("OK" if npm_version else "FAIL", "npm", npm_version or "not found"))

    checks.append(
        Check(
            "OK" if VENV_PYTHON.exists() else "FAIL",
            "Virtual environment",
            str(VENV_PYTHON.relative_to(ROOT)) if VENV_PYTHON.exists() else "missing; run setup.py",
        )
    )

    modules = ("fastapi", "uvicorn", "requests", "PIL", "qrcode", "multipart", "websocket", "tokenizers")
    package_python = VENV_PYTHON if VENV_PYTHON.exists() else Path(sys.executable)
    missing_modules = _missing_modules(package_python, modules)
    checks.append(
        Check(
            "OK" if not missing_modules else "FAIL",
            "Python packages",
            "available" if not missing_modules else "missing: " + ", ".join(missing_modules),
        )
    )

    node_modules = ROOT / "frontend" / "node_modules"
    checks.append(
        Check(
            "OK" if node_modules.is_dir() else "FAIL",
            "Frontend packages",
            "installed" if node_modules.is_dir() else "missing; run setup.py",
        )
    )
    frontend_build = ROOT / "frontend" / "dist" / "index.html"
    checks.append(
        Check(
            "OK" if frontend_build.is_file() else "WARN",
            "Frontend build",
            "available" if frontend_build.is_file() else "not built; setup.py builds it by default",
            required=False,
        )
    )

    checks.extend(
        (
            _service("NyxForge", "http://127.0.0.1:8000/api/instance"),
            _service("Forge/reForge", "http://127.0.0.1:7860/sdapi/v1/sd-models"),
            _service("ComfyUI", "http://127.0.0.1:8188/system_stats"),
        )
    )

    print("NyxForge environment doctor\n")
    for check in checks:
        print(f"[{check.status}] {check.name}: {check.detail}")

    failures = [check for check in checks if check.required and check.status == "FAIL"]
    if failures:
        print("\nRequired setup is incomplete. Run: python ai-agents/setup.py --dev")
        return 1

    print("\nRequired development environment checks passed.")
    print("External model services remain user-managed and may be offline.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

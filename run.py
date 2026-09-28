"""Build and run the complete NyxForge application with one command."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.util import find_spec
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from backend import backend_manager
from backend.database import BackendConfig, configured_backends

ROOT_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = ROOT_DIR / "frontend"
FRONTEND_LOCK_FILE = FRONTEND_DIR / "package-lock.json"
FRONTEND_INSTALL_STAMP = FRONTEND_DIR / "node_modules" / ".nyx-forge-package-lock.sha256"
PID_FILE = ROOT_DIR / ".run.pid"
APP_URL = "http://127.0.0.1:8000"
APP_HOST = "127.0.0.1"
APP_PORT = 8000
DEFAULT_FORGE_URL = "http://127.0.0.1:7860"
FORGE_DISCOVERY_PORTS = range(7860, 7870)
BACKEND_IMPORTS = ("fastapi", "uvicorn", "requests", "PIL", "qrcode", "multipart")
BACKEND_START_TIMEOUT = 300.0
APP_START_TIMEOUT = 60.0


def existing_server() -> str:
    """Return ``app``, ``occupied``, or ``free`` for the configured port."""
    try:
        with urlopen(f"{APP_URL}/api/instance", timeout=0.8) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, dict) and payload.get("app") == "nyx-forge":
            return "app"
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        pass
    try:
        with socket.create_connection((APP_HOST, APP_PORT), timeout=0.3):
            return "occupied"
    except OSError:
        return "free"


def npm_command() -> str:
    """Return the platform-appropriate npm executable."""
    executable = shutil.which("npm") or shutil.which("npm.cmd")
    if executable is None:
        raise RuntimeError("Node.js/npm is required. Install it from https://nodejs.org/")
    return executable


def forge_api_available(url: str) -> bool:
    """Return whether one local URL exposes reForge's authenticated-free API."""
    # sd-models rather than cmd-flags: some reForge builds return 500 from
    # cmd-flags while the rest of the API works, and probing it would reject a
    # perfectly healthy instance.
    try:
        with urlopen(f"{url}/sdapi/v1/sd-models", timeout=2.0) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return isinstance(payload, list)
    except (OSError, URLError, ValueError, json.JSONDecodeError):
        return False


def discover_forge_url() -> str:
    """Find the Stability Matrix reForge port unless explicitly configured."""
    configured = os.getenv("FORGE_BASE_URL", "").strip().rstrip("/")
    if configured:
        return configured
    if forge_api_available(DEFAULT_FORGE_URL):
        return DEFAULT_FORGE_URL
    for port in FORGE_DISCOVERY_PORTS:
        candidate = f"http://127.0.0.1:{port}"
        if forge_api_available(candidate):
            return candidate
    return DEFAULT_FORGE_URL


def ensure_backend_dependencies() -> None:
    """Install missing packages into the interpreter that launches Uvicorn."""
    missing = [module for module in BACKEND_IMPORTS if find_spec(module) is None]
    if not missing:
        return
    print(f"Installing backend dependencies ({', '.join(missing)})…")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(ROOT_DIR / "requirements.txt")],
        cwd=ROOT_DIR,
        check=True,
    )


def ensure_frontend_dependencies(npm: str) -> None:
    """Install the exact lockfile dependencies when the dependency tree is stale."""
    lock_digest = hashlib.sha256(FRONTEND_LOCK_FILE.read_bytes()).hexdigest()
    try:
        installed_digest = FRONTEND_INSTALL_STAMP.read_text(encoding="utf-8").strip()
    except OSError:
        installed_digest = ""
    if installed_digest == lock_digest:
        return
    print("Synchronizing frontend dependencies…")
    subprocess.run([npm, "ci"], cwd=FRONTEND_DIR, check=True)
    FRONTEND_INSTALL_STAMP.write_text(lock_digest, encoding="utf-8")


def build_frontend() -> None:
    """Synchronize frontend packages and create the production bundle."""
    npm = npm_command()
    ensure_frontend_dependencies(npm)
    print("Building React frontend…")
    subprocess.run([npm, "run", "build"], cwd=FRONTEND_DIR, check=True)


def start_server() -> subprocess.Popen[bytes]:
    """Start Uvicorn as a child process supervised by this runner."""
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "backend.main:app",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
        "--log-level",
        "info",
    ]
    environment = os.environ.copy()
    environment["FORGE_BASE_URL"] = discover_forge_url()
    print(f"ForgeAI is connected to reForge at {environment['FORGE_BASE_URL']}")
    process = subprocess.Popen(command, cwd=ROOT_DIR, env=environment)
    PID_FILE.write_text(str(process.pid))
    return process


def wait_for_app_ready(
    server: subprocess.Popen[bytes], timeout: float = APP_START_TIMEOUT
) -> None:
    """Wait until the child identifies itself before opening the browser."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        return_code = server.poll()
        if return_code is not None:
            raise RuntimeError(
                f"NyxForge exited during startup (code {return_code})."
            )
        if existing_server() == "app":
            return
        time.sleep(0.15)
    raise RuntimeError(
        f"NyxForge did not become ready at {APP_URL} within {timeout:.0f} seconds."
    )


def stop_server(server: subprocess.Popen[bytes]) -> None:
    """Stop Uvicorn promptly, escalating when a request prevents shutdown."""
    PID_FILE.unlink(missing_ok=True)
    if server.poll() is not None:
        return
    server.terminate()
    try:
        server.wait(timeout=3.0)
    except subprocess.TimeoutExpired:
        server.kill()
        server.wait(timeout=2.0)


def stop_by_pid_file() -> bool:
    """Stop a server started by a prior, separate invocation of this script.

    ``run.py`` supervises Uvicorn as a child of its own process, so a second
    invocation (e.g. run in a new terminal, or after the first one's terminal
    was closed) has no handle on that child - only the PID file, written by
    whichever invocation is currently supervising the server, bridges that
    gap. Returns True if a running server was found and stopped.
    """
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
    except ValueError:
        PID_FILE.unlink(missing_ok=True)
        return False
    try:
        # On Windows, SIGTERM maps to an immediate TerminateProcess - there is
        # no graceful-then-forceful escalation to do here (no SIGKILL exists).
        os.kill(pid, signal.SIGTERM)
    except OSError:
        # Process is already gone; stale PID file from a prior crash/kill.
        pass
    PID_FILE.unlink(missing_ok=True)
    return True


def do_stop() -> None:
    """Stop whatever instance is running, started by this script or another."""
    if stop_by_pid_file():
        print("NyxForge stopped.")
        return
    if existing_server() in ("app", "occupied"):
        print(
            "NyxForge appears to be running, but it wasn't started by "
            "this script (no .run.pid on record) - stop it manually.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print("NyxForge is not running.")


def do_start() -> None:
    """Build React, open the browser, and supervise the FastAPI process."""
    server_state = existing_server()
    if server_state == "app":
        print(f"NyxForge is already running at {APP_URL}")
        webbrowser.open(APP_URL)
        return
    if server_state == "occupied":
        print(
            f"Port {APP_PORT} is already used by another application. "
            "Close it or change the NyxForge port before starting.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    try:
        ensure_backend_dependencies()
        build_frontend()
    except KeyboardInterrupt:
        print("\nStartup cancelled.")
        raise SystemExit(130) from None
    except (RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"Could not prepare the application: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print(f"NyxForge is starting at {APP_URL}")
    server = start_server()
    shutdown_requested = threading.Event()

    def request_shutdown(_signum: int, _frame: object) -> None:
        """Ask the supervision loop to stop after Ctrl+C or termination."""
        shutdown_requested.set()

    previous_sigint = signal.signal(signal.SIGINT, request_shutdown)
    previous_sigterm = signal.signal(signal.SIGTERM, request_shutdown)
    previous_sigbreak = None
    if sys.platform == "win32" and hasattr(signal, "SIGBREAK"):
        # Git Bash/MSYS may deliver Ctrl+C as CTRL_BREAK on Windows.
        previous_sigbreak = signal.signal(signal.SIGBREAK, request_shutdown)
    try:
        print("Waiting for NyxForge to become ready...")
        wait_for_app_ready(server)
        if shutdown_requested.is_set():
            print("\nStopping NyxForge...")
            stop_server(server)
            print("NyxForge stopped.")
            return
        print(f"NyxForge is ready at {APP_URL}")
        webbrowser.open(APP_URL)
        # Polling stays responsive in PowerShell, cmd.exe, and Git Bash.
        while server.poll() is None and not shutdown_requested.is_set():
            time.sleep(0.15)
        if shutdown_requested.is_set():
            print("\nStopping NyxForge...")
            stop_server(server)
            print("NyxForge stopped.")
            return
        return_code = server.returncode or 0
    except KeyboardInterrupt:
        print("\nStopping NyxForge…")
        stop_server(server)
        print("NyxForge stopped.")
        return
    except RuntimeError as exc:
        print(f"Could not start the application: {exc}", file=sys.stderr)
        return_code = server.poll() or 1
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)
        if previous_sigbreak is not None:
            signal.signal(signal.SIGBREAK, previous_sigbreak)
        stop_server(server)
    if return_code:
        raise SystemExit(return_code)


class _ArgumentParser(argparse.ArgumentParser):
    """Keep the launcher's historical exit code for invalid commands."""

    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(1, f"{self.prog}: error: {message}\n")


def _parser() -> argparse.ArgumentParser:
    parser = _ArgumentParser(
        description="Build, run, and manage NyxForge and its Forge backends."
    )
    commands = parser.add_subparsers(dest="command", metavar="COMMAND")
    for command, help_text in (
        ("start", "start the app (the default command)"),
        ("stop", "stop the app"),
        ("restart", "restart the app"),
    ):
        command_parser = commands.add_parser(command, help=help_text)
        command_parser.add_argument(
            "--with-backends",
            action="store_true",
            help="also manage all configured Forge backends",
        )

    backends = commands.add_parser("backends", help="manage configured Forge backends")
    backend_commands = backends.add_subparsers(
        dest="backend_command", metavar="ACTION", required=True
    )
    backend_commands.add_parser("status", help="show status for every backend")
    backend_ids = tuple(item.id for item in configured_backends())
    for action in ("start", "stop", "restart"):
        action_parser = backend_commands.add_parser(
            action, help=f"{action} one backend, or all when ID is omitted"
        )
        action_parser.add_argument(
            "backend_id",
            nargs="?",
            choices=backend_ids,
            metavar="ID",
            help=f"backend id ({', '.join(backend_ids)}; default: all)",
        )
    return parser


def _print_backend_statuses(statuses: list[backend_manager.BackendStatus]) -> None:
    headers = ("id", "label", "port", "running", "reachable", "loaded checkpoint")
    rows = [
        (
            item.id,
            item.label,
            str(item.port),
            "yes" if item.running else "no",
            "yes" if item.reachable else "no",
            item.loaded_checkpoint or "-",
        )
        for item in statuses
    ]
    widths = [
        max(len(header), *(len(row[index]) for row in rows))
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _selected_backends(backend_id: str | None = None) -> tuple[BackendConfig, ...]:
    return tuple(
        backend
        for backend in configured_backends()
        if backend_id is None or backend.id == backend_id
    )


def _manage_backends(action: str, backend_id: str | None = None) -> None:
    failed = False
    operation = getattr(backend_manager, action)
    for backend in _selected_backends(backend_id):
        try:
            result = operation(backend.id)
            print(
                f"{backend.label}: running={'yes' if result.running else 'no'}, "
                f"reachable={'yes' if result.reachable else 'no'}"
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            failed = True
            print(f"Could not {action} {backend.label}: {exc}", file=sys.stderr)
    if failed:
        raise SystemExit(1)


def _start_backends_and_wait(action: str = "start") -> None:
    """Start the default SD/SDXL runner without reserving VRAM for FLUX.

    Forge Neo is intentionally demand-started when a submitted model profile
    explicitly targets it. Starting both runners here leaves two checkpoints
    competing for one GPU before the user has selected a FLUX model.
    """
    operation = getattr(backend_manager, action)
    backends = configured_backends()
    if any(backend.id == "forge_neo" for backend in backends):
        # A normal application launch is a non-FLUX baseline. Clear a Neo
        # process left behind by an earlier FLUX session before reForge loads.
        backend_manager.stop("forge_neo")
    default_backends = tuple(
        backend for backend in backends if backend.id != "forge_neo"
    )
    for backend in default_backends:
        try:
            initial = operation(backend.id)
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            print(f"Could not {action} {backend.label}: {exc}", file=sys.stderr)
            continue
        if initial.reachable:
            print(f"{backend.label} is reachable at {backend.base_url}")
            continue
        print(f"Waiting up to 5 minutes for {backend.label} at {backend.base_url}...")
        try:
            final = backend_manager.wait_until_reachable(
                backend.id,
                timeout=BACKEND_START_TIMEOUT,
                on_progress=lambda item, elapsed: print(
                    f"Still waiting for {item.label} ({elapsed:.0f}s elapsed)..."
                ),
            )
        except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as exc:
            print(f"Could not check {backend.label}: {exc}", file=sys.stderr)
            continue
        if final.reachable:
            print(f"{backend.label} is ready at {backend.base_url}")
        else:
            detail = final.error or "timed out waiting for the model API"
            print(f"{backend.label} did not become reachable: {detail}", file=sys.stderr)


def _wait_for_app_port() -> None:
    for _ in range(30):
        if existing_server() == "free":
            break
        time.sleep(0.2)


def main(argv: list[str] | None = None) -> None:
    """Dispatch app and opt-in backend lifecycle commands."""
    parsed = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    action = parsed.command or "start"
    with_backends = getattr(parsed, "with_backends", False)
    if action == "backends":
        if parsed.backend_command == "status":
            _print_backend_statuses(backend_manager.statuses())
        else:
            _manage_backends(parsed.backend_command, parsed.backend_id)
    elif action == "stop":
        do_stop()
        if with_backends:
            _manage_backends("stop")
    elif action == "restart":
        do_stop()
        _wait_for_app_port()
        if with_backends:
            _start_backends_and_wait("restart")
        do_start()
    elif action == "start":
        if with_backends:
            _start_backends_and_wait()
        do_start()


if __name__ == "__main__":
    main()

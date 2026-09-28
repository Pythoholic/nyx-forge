"""Health checks and local process control for configured Forge backends."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
from threading import Lock
from time import monotonic, sleep
from typing import Callable

import requests

from .database import BackendConfig, backend_config, configured_backends


@dataclass(frozen=True, slots=True)
class BackendStatus:
    id: str
    label: str
    port: int
    running: bool
    reachable: bool
    loaded_checkpoint: str | None
    error: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


_PROCESS_LOCK = Lock()


def _normalized_path(value: str) -> str:
    return os.path.normcase(os.path.normpath(value)).replace("/", "\\")


def _processes() -> list[tuple[int, str]]:
    """Return process ids and command lines using only the standard library."""
    if os.name == "nt":
        command = (
            "Get-CimInstance Win32_Process | "
            "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=4,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=True,
        )
        if not completed.stdout.strip():
            return []
        decoded = json.loads(completed.stdout)
        rows = decoded if isinstance(decoded, list) else [decoded]
        return [
            (int(row["ProcessId"]), str(row.get("CommandLine") or ""))
            for row in rows
            if isinstance(row, dict) and row.get("ProcessId") is not None
        ]
    completed = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        capture_output=True,
        text=True,
        timeout=4,
        check=True,
    )
    processes: list[tuple[int, str]] = []
    for line in completed.stdout.splitlines():
        fields = line.strip().split(maxsplit=1)
        if fields and fields[0].isdigit():
            processes.append((int(fields[0]), fields[1] if len(fields) > 1 else ""))
    return processes


def _matching_pids(backend: BackendConfig, processes: list[tuple[int, str]]) -> list[int]:
    if not backend.package_dir:
        return []
    package_dir = _normalized_path(backend.package_dir)
    return [
        pid
        for pid, command_line in processes
        if package_dir in _normalized_path(command_line)
        and "launch.py" in command_line.casefold()
    ]


def _health(backend: BackendConfig) -> tuple[bool, str | None, str | None]:
    try:
        response = requests.get(
            f"{backend.base_url}/sdapi/v1/sd-models", timeout=(0.25, 1.5)
        )
        response.raise_for_status()
        if not isinstance(response.json(), list):
            raise ValueError("sd-models did not return a JSON list")
    except (requests.RequestException, ValueError) as exc:
        return False, None, str(exc)[:300]

    checkpoint = None
    try:
        response = requests.get(
            f"{backend.base_url}/sdapi/v1/options", timeout=(0.25, 0.75)
        )
        response.raise_for_status()
        options = response.json()
        if isinstance(options, dict) and options.get("sd_model_checkpoint"):
            checkpoint = str(options["sd_model_checkpoint"])
    except (requests.RequestException, ValueError):
        pass
    return True, checkpoint, None


def _status_with_processes(
    backend: BackendConfig,
    processes: list[tuple[int, str]],
    process_error: str | None = None,
) -> BackendStatus:
    reachable, checkpoint, health_error = _health(backend)
    if not reachable and not backend.package_dir:
        health_error = "Package folder is not configured."
    return BackendStatus(
        id=backend.id,
        label=backend.label,
        port=backend.port,
        running=bool(_matching_pids(backend, processes)),
        reachable=reachable,
        loaded_checkpoint=checkpoint,
        error=health_error or process_error,
    )


def status(backend_id: str) -> BackendStatus:
    backend = backend_config(backend_id)
    try:
        processes = _processes()
        process_error = None
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        processes = []
        process_error = f"Could not inspect local processes: {exc}"[:300]
    return _status_with_processes(backend, processes, process_error)


def statuses() -> list[BackendStatus]:
    """Check all backends concurrently so one dead port cannot delay the rest."""
    backends = configured_backends()
    try:
        processes = _processes()
        process_error = None
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        processes = []
        process_error = f"Could not inspect local processes: {exc}"[:300]
    with ThreadPoolExecutor(max_workers=max(1, len(backends))) as executor:
        return list(
            executor.map(
                lambda item: _status_with_processes(item, processes, process_error),
                backends,
            )
        )


def wait_until_reachable(
    backend_id: str,
    *,
    timeout: float = 300.0,
    poll_interval: float = 2.0,
    progress_interval: float = 10.0,
    on_progress: Callable[[BackendConfig, float], None] | None = None,
) -> BackendStatus:
    """Wait for one backend's model API, returning its final status.

    Progress callbacks are deliberately less frequent than health polls so CLI
    callers can show that a slow-loading Forge process is alive without
    flooding the terminal.
    """
    backend = backend_config(backend_id)
    started_at = monotonic()
    deadline = started_at + max(0.0, timeout)
    next_progress = started_at + max(0.0, progress_interval)
    while True:
        reachable, _checkpoint, _error = _health(backend)
        if reachable:
            final = status(backend_id)
            if final.reachable:
                return final
        now = monotonic()
        if now >= deadline:
            return status(backend_id)
        if on_progress is not None and now >= next_progress:
            on_progress(backend, now - started_at)
            next_progress = now + max(0.0, progress_interval)
        sleep(min(max(0.05, poll_interval), deadline - now))


def _python_executable(backend: BackendConfig) -> Path:
    if not backend.package_dir:
        raise FileNotFoundError(
            f"Configure the package folder for {backend.label} before starting it"
        )
    package_dir = Path(backend.package_dir)
    venv_python = package_dir / "venv" / (
        "Scripts/python.exe" if os.name == "nt" else "bin/python"
    )
    if venv_python.is_file():
        return venv_python
    data_dir = package_dir.parent.parent
    candidates = [data_dir / "Assets" / "Python310" / "python.exe"]
    candidates.extend(
        sorted((data_dir / "Assets" / "Python").glob("*/python.exe"), reverse=True)
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No Python interpreter found for {backend.label}")


def start(backend_id: str) -> BackendStatus:
    backend = backend_config(backend_id)
    if not backend.package_dir:
        raise RuntimeError(
            f"Configure the package folder for {backend.label} before starting it"
        )
    with _PROCESS_LOCK:
        try:
            processes = _processes()
        except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Could not inspect local processes: {exc}") from exc
        if _matching_pids(backend, processes):
            return _status_with_processes(backend, processes)
        package_dir = Path(backend.package_dir)
        launch_script = package_dir / "launch.py"
        if not launch_script.is_file():
            raise FileNotFoundError(f"launch.py was not found in {package_dir}")
        # --api is not optional: without it the web UI still comes up but
        # /sdapi/v1/* is never mounted, so every call this app makes 404s and
        # the backend looks running-but-unreachable. Stability Matrix passes
        # it too. Skip it only if the configured args already name it.
        model_store = package_dir.parent.parent / "Models"
        extra_args = [
            argument.replace("{model_store}", str(model_store))
            for argument in backend.launch_args
        ]
        command = [
            str(_python_executable(backend)),
            str(launch_script),
            "--port",
            str(backend.port),
        ]
        if not any(arg == "--api" for arg in extra_args):
            command.append("--api")
        command.extend(extra_args)
        creationflags = 0
        start_new_session = os.name != "nt"
        if os.name == "nt":
            creationflags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        subprocess.Popen(
            command,
            cwd=package_dir,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            creationflags=creationflags,
            start_new_session=start_new_session,
        )
    return status(backend_id)


def _terminate_pid(pid: int, *, force: bool) -> None:
    if os.name == "nt":
        command = ["taskkill", "/PID", str(pid), "/T"]
        if force:
            command.append("/F")
        subprocess.run(
            command,
            capture_output=True,
            timeout=3,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=False,
        )
    else:
        os.kill(pid, signal.SIGKILL if force else signal.SIGTERM)


def stop(backend_id: str) -> BackendStatus:
    backend = backend_config(backend_id)
    with _PROCESS_LOCK:
        processes = _processes()
        pids = _matching_pids(backend, processes)
        for pid in pids:
            try:
                _terminate_pid(pid, force=False)
            except (OSError, subprocess.SubprocessError):
                pass
        deadline = monotonic() + 3.0
        remaining = pids
        while remaining and monotonic() < deadline:
            sleep(0.15)
            remaining = _matching_pids(backend, _processes())
        for pid in remaining:
            try:
                _terminate_pid(pid, force=True)
            except (OSError, subprocess.SubprocessError):
                pass
    return status(backend_id)


def restart(backend_id: str) -> BackendStatus:
    stop(backend_id)
    return start(backend_id)

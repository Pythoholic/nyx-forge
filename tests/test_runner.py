from __future__ import annotations

import unittest
import sys
import os
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import run
from backend.backend_manager import BackendStatus
from backend.database import BackendConfig


class RunnerTests(unittest.TestCase):
    """Keep the one-command launcher from creating competing servers."""

    def test_existing_ai_quick_gen_is_reused(self) -> None:
        with (
            patch.object(run, "existing_server", return_value="app"),
            patch.object(run, "build_frontend") as build_frontend,
            patch.object(run.webbrowser, "open") as open_browser,
        ):
            run.main([])
        build_frontend.assert_not_called()
        open_browser.assert_called_once_with(run.APP_URL)

    def test_unrelated_port_owner_stops_startup(self) -> None:
        with (
            patch.object(run, "existing_server", return_value="occupied"),
            patch.object(run, "build_frontend") as build_frontend,
            self.assertRaises(SystemExit) as raised,
        ):
            run.main([])
        self.assertEqual(raised.exception.code, 1)
        build_frontend.assert_not_called()

    def test_app_readiness_waits_for_identifying_endpoint(self) -> None:
        server = MagicMock()
        server.poll.return_value = None
        with (
            patch.object(run, "existing_server", side_effect=["free", "occupied", "app"]),
            patch.object(run.time, "sleep") as sleep,
        ):
            run.wait_for_app_ready(server, timeout=5)
        self.assertEqual(sleep.call_count, 2)

    def test_browser_opens_only_after_the_app_is_ready(self) -> None:
        server = MagicMock()
        server.poll.return_value = 0
        server.returncode = 0
        order = []
        with (
            patch.object(run, "existing_server", return_value="free"),
            patch.object(run, "ensure_backend_dependencies"),
            patch.object(run, "build_frontend"),
            patch.object(run, "start_server", return_value=server),
            patch.object(run, "stop_server"),
            patch.object(run, "wait_for_app_ready", side_effect=lambda child: order.append(("ready", child))),
            patch.object(run.webbrowser, "open", side_effect=lambda url: order.append(("browser", url))),
        ):
            run.do_start()
        self.assertEqual(order, [("ready", server), ("browser", run.APP_URL)])

    def test_unknown_command_exits_with_an_error(self) -> None:
        with self.assertRaises(SystemExit) as raised:
            run.main(["bogus"])
        self.assertEqual(raised.exception.code, 1)

    def test_stop_with_no_pid_file_and_nothing_listening_is_a_no_op(self) -> None:
        with (
            patch.object(run, "stop_by_pid_file", return_value=False),
            patch.object(run, "existing_server", return_value="free"),
        ):
            run.main(["stop"])  # Must not raise.

    def test_restart_stops_then_starts(self) -> None:
        with (
            patch.object(run, "do_stop") as do_stop,
            patch.object(run, "do_start") as do_start,
            patch.object(run, "existing_server", return_value="free"),
        ):
            run.main(["restart"])
        do_stop.assert_called_once()
        do_start.assert_called_once()

    def test_plain_app_commands_do_not_manage_backends(self) -> None:
        for arguments in (["start"], ["stop"], ["restart"]):
            with (
                self.subTest(arguments=arguments),
                patch.object(run, "do_start"),
                patch.object(run, "do_stop"),
                patch.object(run, "_wait_for_app_port"),
                patch.object(run.backend_manager, "start") as backend_start,
                patch.object(run.backend_manager, "stop") as backend_stop,
                patch.object(run.backend_manager, "restart") as backend_restart,
            ):
                run.main(arguments)
            backend_start.assert_not_called()
            backend_stop.assert_not_called()
            backend_restart.assert_not_called()

    def test_start_with_backends_waits_then_starts_app_after_timeout(self) -> None:
        backend = BackendConfig("reforge", "reForge", 7860, "package")
        neo = BackendConfig("forge_neo", "Forge Neo", 7861, "neo-package")
        unavailable = BackendStatus(
            id=backend.id,
            label=backend.label,
            port=backend.port,
            running=True,
            reachable=False,
            loaded_checkpoint=None,
            error="connection refused",
        )
        with (
            patch.object(run, "configured_backends", return_value=(backend, neo)),
            patch.object(run.backend_manager, "start", return_value=unavailable),
            patch.object(run.backend_manager, "stop") as backend_stop,
            patch.object(
                run.backend_manager,
                "wait_until_reachable",
                return_value=unavailable,
            ) as wait_until_reachable,
            patch.object(run, "do_start") as do_start,
        ):
            run.main(["start", "--with-backends"])
        wait_until_reachable.assert_called_once()
        self.assertEqual(wait_until_reachable.call_args.args[0], "reforge")
        backend_stop.assert_called_once_with("forge_neo")
        do_start.assert_called_once()

    def test_start_with_unconfigured_backend_explains_first_run_setup(self) -> None:
        backend = BackendConfig("reforge", "reForge", 7860, "")
        stderr = StringIO()
        with (
            patch.object(run, "configured_backends", return_value=(backend,)),
            patch.object(run.backend_manager, "start") as backend_start,
            patch.object(run, "do_start") as do_start,
            redirect_stderr(stderr),
        ):
            run.main(["start", "--with-backends"])

        backend_start.assert_not_called()
        do_start.assert_called_once()
        self.assertIn("Account > Forge backends", stderr.getvalue())
        self.assertIn("--with-backends", stderr.getvalue())

    def test_restart_with_backends_uses_backend_manager_restart(self) -> None:
        backend = BackendConfig("reforge", "reForge", 7860, "package")
        reachable = BackendStatus(
            backend.id, backend.label, backend.port, True, True, "model.safetensors", None
        )
        with (
            patch.object(run, "configured_backends", return_value=(backend,)),
            patch.object(run, "do_stop"),
            patch.object(run, "_wait_for_app_port"),
            patch.object(run, "do_start") as do_start,
            patch.object(
                run.backend_manager, "restart", return_value=reachable
            ) as backend_restart,
        ):
            run.main(["restart", "--with-backends"])
        backend_restart.assert_called_once_with("reforge")
        do_start.assert_called_once()

    def test_backend_action_without_id_manages_every_configured_backend(self) -> None:
        backends = (
            BackendConfig("reforge", "reForge", 7860, "one"),
            BackendConfig("forge_neo", "Forge Neo", 7861, "two"),
        )

        def stopped(backend_id: str) -> BackendStatus:
            backend = next(item for item in backends if item.id == backend_id)
            return BackendStatus(
                backend.id,
                backend.label,
                backend.port,
                False,
                False,
                None,
                "connection refused",
            )

        with (
            patch.object(run, "configured_backends", return_value=backends),
            patch.object(run.backend_manager, "stop", side_effect=stopped) as stop,
        ):
            run.main(["backends", "stop"])
        self.assertEqual(
            [call.args[0] for call in stop.call_args_list],
            ["reforge", "forge_neo"],
        )

    def test_backend_status_always_returns_normally(self) -> None:
        stopped = BackendStatus(
            "reforge", "reForge", 7860, False, False, None, "connection refused"
        )
        with (
            patch.object(run.backend_manager, "statuses", return_value=[stopped]),
            patch.object(run, "_print_backend_statuses") as print_statuses,
        ):
            run.main(["backends", "status"])
        print_statuses.assert_called_once_with([stopped])

    def test_missing_backend_package_is_installed_for_launcher_python(self) -> None:
        def available(module: str):
            return None if module == "qrcode" else object()

        with (
            patch.object(run, "find_spec", side_effect=available),
            patch.object(run.subprocess, "run") as subprocess_run,
        ):
            run.ensure_backend_dependencies()

        subprocess_run.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                str(run.ROOT_DIR / "requirements.txt"),
            ],
            cwd=run.ROOT_DIR,
            check=True,
        )

    def test_frontend_dependencies_are_resynced_when_lockfile_changes(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            frontend_dir = Path(temporary_directory)
            lock_file = frontend_dir / "package-lock.json"
            install_stamp = frontend_dir / "node_modules" / ".nyx-forge-package-lock.sha256"
            lock_file.write_text('{"lockfileVersion": 3}', encoding="utf-8")
            install_stamp.parent.mkdir()
            with (
                patch.object(run, "FRONTEND_DIR", frontend_dir),
                patch.object(run, "FRONTEND_LOCK_FILE", lock_file),
                patch.object(run, "FRONTEND_INSTALL_STAMP", install_stamp),
                patch.object(run.subprocess, "run") as subprocess_run,
            ):
                run.ensure_frontend_dependencies("npm.cmd")

            subprocess_run.assert_called_once_with(
                ["npm.cmd", "ci"], cwd=frontend_dir, check=True
            )
            self.assertEqual(
                install_stamp.read_text(encoding="utf-8"),
                run.hashlib.sha256(lock_file.read_bytes()).hexdigest(),
            )

    def test_frontend_dependencies_are_reused_when_lockfile_matches(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            frontend_dir = Path(temporary_directory)
            lock_file = frontend_dir / "package-lock.json"
            install_stamp = frontend_dir / "node_modules" / ".nyx-forge-package-lock.sha256"
            lock_file.write_text('{"lockfileVersion": 3}', encoding="utf-8")
            install_stamp.parent.mkdir()
            install_stamp.write_text(
                run.hashlib.sha256(lock_file.read_bytes()).hexdigest(),
                encoding="utf-8",
            )
            with (
                patch.object(run, "FRONTEND_DIR", frontend_dir),
                patch.object(run, "FRONTEND_LOCK_FILE", lock_file),
                patch.object(run, "FRONTEND_INSTALL_STAMP", install_stamp),
                patch.object(run.subprocess, "run") as subprocess_run,
            ):
                run.ensure_frontend_dependencies("npm.cmd")

            subprocess_run.assert_not_called()

    def test_forge_discovery_uses_the_active_stability_matrix_port(self) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(
                run,
                "forge_api_available",
                side_effect=lambda url: url == "http://127.0.0.1:7861",
            ),
        ):
            self.assertEqual(run.discover_forge_url(), "http://127.0.0.1:7861")

    def test_configured_forge_url_overrides_port_discovery(self) -> None:
        with (
            patch.dict(os.environ, {"FORGE_BASE_URL": "http://127.0.0.1:7999/"}, clear=True),
            patch.object(run, "forge_api_available") as available,
        ):
            self.assertEqual(run.discover_forge_url(), "http://127.0.0.1:7999")
        available.assert_not_called()


if __name__ == "__main__":
    unittest.main()

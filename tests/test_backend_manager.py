from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import requests

from backend import backend_manager, database
from backend.database import BackendConfig


class BackendConfigurationTests(unittest.TestCase):
    def test_public_defaults_require_user_package_folders(self) -> None:
        self.assertTrue(database.DEFAULT_BACKENDS)
        self.assertTrue(
            all(not backend.package_dir for backend in database.DEFAULT_BACKENDS)
        )

    def test_saved_registry_may_contain_an_unconfigured_optional_runner(self) -> None:
        entries = [
            {
                "id": "reforge",
                "label": "reForge",
                "port": 7860,
                "package_dir": r"C:\Forge\reforge",
                "launch_args": ["--skip-install"],
            },
            {
                "id": "forge_neo",
                "label": "Forge Neo",
                "port": 7861,
                "package_dir": "",
                "launch_args": [],
            },
        ]
        with TemporaryDirectory() as temporary_directory:
            config_path = Path(temporary_directory) / "storage-config.json"
            config_path.write_text(json.dumps({"backends": entries}), encoding="utf-8")
            with patch.object(database, "STORAGE_CONFIG_PATH", config_path):
                configured = database.configured_backends()

        self.assertEqual(configured[0].package_dir, r"C:\Forge\reforge")
        self.assertEqual(configured[1].package_dir, "")

    def test_unconfigured_runner_fails_before_process_inspection(self) -> None:
        backend = BackendConfig("reforge", "reForge", 7860, "")
        with (
            patch.object(backend_manager, "backend_config", return_value=backend),
            patch.object(backend_manager, "_processes") as processes,
            self.assertRaisesRegex(RuntimeError, "Configure the package folder"),
        ):
            backend_manager.start("reforge")
        processes.assert_not_called()

    def test_health_hides_low_level_connection_errors(self) -> None:
        backend = BackendConfig("reforge", "reForge", 7860, r"C:\Forge\reforge")
        low_level_error = (
            "HTTPConnectionPool(host='127.0.0.1', port=7860): Max retries exceeded"
        )
        with patch.object(
            backend_manager.requests,
            "get",
            side_effect=requests.ConnectionError(low_level_error),
        ):
            reachable, checkpoint, error = backend_manager._health(backend)

        self.assertFalse(reachable)
        self.assertIsNone(checkpoint)
        self.assertEqual(error, "reForge is not running on port 7860.")
        self.assertNotIn("HTTPConnectionPool", error)

    def test_health_explains_slow_start_without_transport_details(self) -> None:
        backend = BackendConfig("reforge", "reForge", 7860, r"C:\Forge\reforge")
        with patch.object(
            backend_manager.requests,
            "get",
            side_effect=requests.Timeout("socket timeout"),
        ):
            _reachable, _checkpoint, error = backend_manager._health(backend)

        self.assertEqual(
            error,
            "reForge did not respond on port 7860. It may still be starting.",
        )
        self.assertNotIn("socket", error)


if __name__ == "__main__":
    unittest.main()

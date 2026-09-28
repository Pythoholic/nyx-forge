from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

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


if __name__ == "__main__":
    unittest.main()

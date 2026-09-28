"""Regression coverage for deterministic named quality tiers."""

from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

from PIL import Image

from backend import database, main
from backend.forge import GenerationSettings, Txt2ImgPayload


class QualityModeTests(unittest.TestCase):
    def test_every_mode_maps_to_documented_portrait_and_landscape_size(self) -> None:
        expected_portrait = {
            "normal": (832, 1216),
            "high": (1248, 1824),
            "super": (1664, 2432),
            "4k": (2496, 3648),
            "8k": (6656, 9728),
            "12k": (9984, 14592),
        }
        for mode, portrait in expected_portrait.items():
            with self.subTest(mode=mode, orientation="portrait"):
                self.assertEqual(main._quality_target_size(832, 1216, mode), portrait)
            with self.subTest(mode=mode, orientation="landscape"):
                self.assertEqual(
                    main._quality_target_size(1216, 832, mode),
                    (portrait[1], portrait[0]),
                )

    def test_legacy_booleans_resolve_to_the_original_three_tiers(self) -> None:
        self.assertEqual(main._resolve_quality_mode(None, False, False), "normal")
        self.assertEqual(main._resolve_quality_mode(None, True, False), "high")
        self.assertEqual(main._resolve_quality_mode(None, False, True), "super")
        self.assertEqual(main._resolve_quality_mode(None, True, True), "super")

    def test_original_three_modes_keep_the_same_forge_payload(self) -> None:
        payload = Txt2ImgPayload("subject", "negative", 832, 1216)
        for mode, old_high, old_super in (
            ("normal", False, False),
            ("high", True, False),
            ("super", True, True),
        ):
            with self.subTest(mode=mode):
                high_res, super_res = main._quality_flags(mode)
                self.assertEqual(
                    payload.as_request(high_res=high_res, super_res=super_res),
                    payload.as_request(high_res=old_high, super_res=old_super),
                )

    def test_12k_resume_after_first_upscale_does_not_regenerate_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = (
                patch.object(database, "DATABASE_PATH", root / "test.db"),
                patch.object(database, "GENERATED_DIR", root / "generated"),
                patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
                patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
                patch.object(main, "GENERATED_DIR", root / "generated"),
            )
            for active_patch in patches:
                active_patch.start()
            try:
                database.initialize_database()
                job_id = database.create_job(
                    "generate", {"quality_mode": "12k"}, None, None
                )
                artifact_dir = root / "generated" / "quality-jobs" / f"job-{job_id}"
                artifact_dir.mkdir(parents=True)
                (artifact_dir / "detail-passed.png").write_bytes(b"detail")
                (artifact_dir / "upscale-1.png").write_bytes(b"eight-k")
                database.update_job_request_payload(
                    job_id,
                    quality_stage="UPSCALE_1_DONE",
                    quality_seed=77,
                    quality_diffusion_seconds=12.0,
                    quality_upscale_seconds=4.0,
                )
                forge = Mock()
                forge.upscale_image_bytes.return_value = b"twelve-k"
                # The pipeline picks its host with for_model now, so the stub
                # has to answer that too, not just construction.
                client_stub = Mock(return_value=forge)
                client_stub.for_model.return_value = forge
                with patch.object(main, "ForgeApiClient", client_stub):
                    result = main._run_quality_pipeline(
                        Txt2ImgPayload("subject", "negative", 832, 1216),
                        "12k",
                        job_id=job_id,
                        on_stage=Mock(),
                    )
                forge.generate_image.assert_not_called()
                forge.upscale_image_bytes.assert_called_once()
                self.assertEqual(
                    forge.upscale_image_bytes.call_args.kwargs["target_width"], 9984
                )
                self.assertEqual(
                    forge.upscale_image_bytes.call_args.kwargs["target_height"], 14592
                )
                self.assertEqual(result[0], b"twelve-k")
                self.assertEqual(
                    database.get_job(job_id).request_payload["quality_stage"],
                    "UPSCALE_2_DONE",
                )
            finally:
                for active_patch in reversed(patches):
                    active_patch.stop()

    def test_resolved_mode_is_persisted_on_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            patches = (
                patch.object(database, "DATABASE_PATH", root / "test.db"),
                patch.object(database, "GENERATED_DIR", root / "generated"),
                patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
                patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
            )
            for active_patch in patches:
                active_patch.start()
            try:
                database.initialize_database()
                image = BytesIO()
                Image.new("RGB", (16, 16), "black").save(image, "PNG")
                record = database.save_generation(
                    image.getvalue(),
                    Txt2ImgPayload("subject", "negative", 16, 16),
                    GenerationSettings(
                        16, 16, "test-model", "Safe", "Photoreal", True,
                        super_res=True, quality_mode="8k"
                    ),
                )
                self.assertEqual(record.quality_mode, "8k")
                self.assertEqual(database.get_generation(record.id).quality_mode, "8k")
            finally:
                for active_patch in reversed(patches):
                    active_patch.stop()


if __name__ == "__main__":
    unittest.main()

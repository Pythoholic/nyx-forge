"""Regression coverage for gallery-sourced ForgeIMG transformations."""

from __future__ import annotations

import base64
from io import BytesIO
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import subprocess
import unittest

import requests
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image, PngImagePlugin
from pydantic import ValidationError

from backend import auth, database, detect_masks, main
from backend.identity import IdentityAnalysis
from backend.forge import (
    GenerationSettings,
    build_img2img_payload,
    build_instantid_payload,
    build_payload,
)


RAGNAROK = "sd\\juggernautXL_ragnarok.safetensors"
PONY = "sd\\ponyDiffusionV6XL.safetensors"


class DetectorMaskGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        inference_mask = np.zeros((448, 640), dtype=bool)
        inference_mask[40:430, 160:500] = True
        self.source_size = (1248, 1824)
        self.person = detect_masks.source_sized_mask(inference_mask, self.source_size)

    def test_body_detector_has_lower_coverage_than_person(self) -> None:
        face_boxes = np.array([[480, 290, 760, 620]], dtype=float)

        body = detect_masks.protect_face_regions(self.person, face_boxes)

        self.assertLess(np.count_nonzero(body), np.count_nonzero(self.person))
        self.assertGreater(np.count_nonzero(body), 0)
        self.assertFalse(body[800, 620], "neck and collar transition must be protected")
        self.assertTrue(body[1050, 430], "torso must not be removed by box-based hair guesses")

    def test_detector_mask_matches_source_dimensions_exactly(self) -> None:
        encoded = detect_masks.encode_mask(self.person, self.source_size)

        with Image.open(BytesIO(base64.b64decode(encoded))) as decoded:
            self.assertEqual(decoded.size, self.source_size)
        self.assertEqual(self.person.shape, (self.source_size[1], self.source_size[0]))

    def test_no_face_falls_back_to_full_person_mask(self) -> None:
        body = detect_masks.protect_face_regions(self.person, np.empty((0, 4)))

        np.testing.assert_array_equal(body, self.person)


class InstalledDetectorCompositionTests(unittest.TestCase):
    def test_body_coverage_stays_bounded_across_reference_compositions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        detector_python = Path(os.getenv(
            "FORGE_PYTHON",
            r"D:\FUSION\StabilityMatrix-win-x64\Data\Packages\reforge\venv\Scripts\python.exe",
        ))
        detector_models = Path(os.getenv(
            "FORGE_ADETAILER_MODELS_DIR",
            r"D:\FUSION\StabilityMatrix-win-x64\Data\Packages\reforge\models\diffusers\models--Bingsu--adetailer",
        ))
        sources = [root / "generated" / f"generation-{generation}.png" for generation in (193, 194, 196)]
        if not detector_python.is_file() or not detector_models.is_dir() or not all(source.is_file() for source in sources):
            self.skipTest("Reference generations or the installed Forge detector runtime are unavailable.")
        script = """
import base64, json, sys
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, sys.argv[1])
from backend.detect_masks import load_yolo, model_path, segmented_body, segmented_people
models = Path(sys.argv[2])
measurements = {}
for source_name in sys.argv[3:]:
    source = Path(source_name)
    coverage = {}
    body_mask = None
    person_mask = None
    for name, detector in (("person", segmented_people), ("body", segmented_body)):
        proposal = detector(source, models)[0]
        image = Image.open(BytesIO(base64.b64decode(proposal["mask"])))
        binary = np.asarray(image) > 0
        coverage[name] = float(binary.mean())
        if name == "body":
            body_mask = binary
        else:
            person_mask = binary
    face_result = load_yolo(model_path(models, "face_yolov8n.pt"))(str(source), verbose=False, device="cpu")[0]
    boxes = face_result.boxes.xyxy.cpu().numpy() if face_result.boxes is not None else np.empty((0, 4))
    editable_face_pixels = 0
    for x1, y1, x2, y2 in boxes:
        left, right = max(0, int(np.floor(x1))), min(body_mask.shape[1], int(np.ceil(x2)))
        top, bottom = max(0, int(np.floor(y1))), min(body_mask.shape[0], int(np.ceil(y2)))
        editable_face_pixels += int(body_mask[top:bottom, left:right].sum())
    rows = np.any(person_mask, axis=1).nonzero()[0]
    protected = person_mask & ~body_mask
    protected_rows = np.any(protected, axis=1).nonzero()[0]
    measurements[source.stem] = {
        "ratio": coverage["body"] / coverage["person"],
        "editable_face_pixels": editable_face_pixels,
        # Where the protection landed, as a fraction down the person's own
        # bounding box. A head sits at the top; 0.5 would mean the protected
        # region drifted to the torso.
        "protected_centre": (
            float((protected_rows.mean() - rows[0]) / max(1, rows[-1] - rows[0]))
            if protected_rows.size and rows.size else 1.0
        ),
    }
print(json.dumps(measurements))
"""
        completed = subprocess.run(
            [str(detector_python), "-c", script, str(root), str(detector_models), *(str(source) for source in sources)],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
        )
        measurements = json.loads(next(line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")))
        for generation, measurement in measurements.items():
            ratio = measurement["ratio"]
            self.assertEqual(measurement["editable_face_pixels"], 0, f"{generation} face box must never be editable")
            # No lower bound on ratio: on a close-up the head legitimately
            # occupies most of the person, and a floor there would trade away
            # face protection to hit a number.
            self.assertLessEqual(ratio, 0.95, f"{generation} body mask protected too little of the identity region")
            self.assertLess(
                measurement["protected_centre"],
                0.4,
                f"{generation} protected region sits too low - it must cover the head, not the torso",
            )


class FaceIdentityArchitectureTests(unittest.TestCase):
    """FaceID is an SDXL adapter and cannot condition other architectures."""

    def test_only_sdxl_profiles_advertise_face_identity(self) -> None:
        from backend.forge import MODEL_PROFILES

        for key, profile in MODEL_PROFILES.items():
            expected = profile.family.startswith("sdxl")
            self.assertEqual(
                profile.supports_face_identity,
                expected,
                f"{key} ({profile.family}) advertises the wrong FaceID support",
            )


class InstantIdWeightTests(unittest.TestCase):
    """The two InstantID units must not share a weight."""

    def test_keypoint_weight_stays_below_the_adapter(self) -> None:
        # Measured on a fixed seed: both units at 0.8 renders the image as
        # noise texture with no detectable face, which fails the whole
        # character pipeline. The keypoint unit must stay low.
        payload = build_instantid_payload(
            "x", "", _settings(), "REF", adapter_model="a", controlnet_model="c"
        )
        units = payload.as_request(high_res=False)["alwayson_scripts"]["ControlNet"]["args"]
        adapter, keypoint = units[0], units[1]
        self.assertEqual(adapter["module"], "InsightFace (InstantID)")
        self.assertEqual(keypoint["module"], "instant_id_face_keypoints")
        self.assertLessEqual(keypoint["weight"], 0.45)
        self.assertGreater(adapter["weight"], keypoint["weight"])


class ControlNetModuleNameTests(unittest.TestCase):
    """The FaceID preprocessor name must match what Forge actually registers."""

    def test_face_id_module_is_registered_by_forge(self) -> None:
        # A wrong name does not fail the request: Forge resolves the
        # preprocessor to None, every ControlNet hook raises into the log, and
        # an image still comes back - one FaceID never conditioned. Only the
        # live module list can catch that.
        try:
            response = requests.get(
                "http://127.0.0.1:7860/controlnet/module_list", timeout=5
            )
            modules = response.json()["module_list"]
        except Exception:  # noqa: BLE001 - Forge is optional in CI
            self.skipTest("Forge is not running")
        payload = build_img2img_payload(
            "x", "", _settings(), "SRC", 0.8, face_reference="FACE"
        )
        unit = payload.as_request()["alwayson_scripts"]["ControlNet"]["args"][0]
        self.assertIn(unit["module"], modules)


def _settings(model: str = RAGNAROK, cfg_scale: float | None = None) -> GenerationSettings:
    return GenerationSettings(
        width=832,
        height=1216,
        model=model,
        rating="Safe",
        style="Anime" if "ponydiffusion" in model.lower() else "Photoreal",
        high_res=False,
        cfg_scale=cfg_scale,
    )


def _png_bytes(size: tuple[int, int] = (832, 1216)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, "gray").save(buffer, format="PNG")
    return buffer.getvalue()


class Img2ImgPayloadTests(unittest.TestCase):
    def test_cfg_floor_holds(self) -> None:
        payload = build_img2img_payload("adult woman portrait", "", _settings(cfg_scale=1.0), "c291cmNl", 0.4)
        self.assertEqual(payload.cfg_scale, 3.0)

    def test_face_identity_adds_lora_and_controlnet_without_a_mask(self) -> None:
        payload = build_img2img_payload(
            "same character in a snowy street",
            "",
            _settings(),
            "c291cmNl",
            0.8,
            face_reference="ZmFjZQ==",
        )
        request = payload.as_request()

        self.assertIn("<lora:ip-adapter-faceid-plusv2_sdxl_lora:0.7>", payload.prompt)
        unit = request["alwayson_scripts"]["ControlNet"]["args"][0]
        self.assertEqual(unit["module"], "InsightFace+CLIP-H (IPAdapter)")
        self.assertEqual(unit["model"], "ip-adapter-faceid-plusv2_sdxl [187cb962]")
        self.assertEqual(unit["weight"], 1.2)
        self.assertEqual(unit["image"], "ZmFjZQ==")

    def test_face_identity_keeps_hand_adetailer_but_removes_face_pass(self) -> None:
        request = build_img2img_payload(
            "same character in a snowy street",
            "",
            _settings(),
            "c291cmNl",
            0.8,
            face_reference="ZmFjZQ==",
        ).as_request()

        args = request["alwayson_scripts"]["ADetailer"]["args"]
        models = [
            item["ad_model"] for item in args
            if isinstance(item, dict) and item.get("ad_model")
        ]
        self.assertEqual(models, ["hand_yolov8n.pt"])

    def test_instantid_units_keep_required_order_and_disable_face_adetailer(self) -> None:
        payload = build_instantid_payload(
            "same woman in a snowy street",
            "",
            _settings(),
            "ZmFjZQ==",
            adapter_model="ip-adapter_instant_id_sdxl [adapter]",
            controlnet_model="control_instant_id_sdxl [control]",
        )

        request = payload.as_request(high_res=False)
        units = request["alwayson_scripts"]["ControlNet"]["args"]
        self.assertEqual(units[0]["module"], "InsightFace (InstantID)")
        self.assertEqual(units[0]["model"], "ip-adapter_instant_id_sdxl [adapter]")
        self.assertEqual(units[1]["module"], "instant_id_face_keypoints")
        self.assertEqual(units[1]["model"], "control_instant_id_sdxl [control]")
        self.assertIn(
            "waist-up environmental portrait, entire head visible, face clearly visible, subject fills the frame",
            request["prompt"],
        )
        self.assertIn(
            "cropped head, headless, face out of frame, distant subject, tiny face",
            request["negative_prompt"],
        )
        self.assertNotIn("ADetailer", request["alwayson_scripts"])

    def test_face_identity_request_has_no_mask(self) -> None:
        request = build_img2img_payload(
            "same character", "", _settings(), "c291cmNl", 0.8,
            face_reference="ZmFjZQ==",
        ).as_request()

        self.assertNotIn("mask", request)

    def test_regular_img2img_has_no_faceid_components(self) -> None:
        payload = build_img2img_payload(
            "adult woman portrait", "", _settings(), "c291cmNl", 0.4,
        )
        request = payload.as_request()

        self.assertNotIn("ip-adapter-faceid", payload.prompt)
        self.assertNotIn("ControlNet", request.get("alwayson_scripts", {}))

    def test_cfg_floor_holds_on_face_identity_path(self) -> None:
        request = build_img2img_payload(
            "same character", "", _settings(cfg_scale=1.0), "c291cmNl", 0.8,
            face_reference="ZmFjZQ==",
        ).as_request()

        self.assertEqual(request["cfg_scale"], 3.0)

    def test_denoise_bounds(self) -> None:
        for value in (0.0, 1.5):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                main.Img2ImgRequest(source_generation_id=1, denoising_strength=value)

    def test_inpaint_padding_bounds(self) -> None:
        for value in (-1, 257):
            with self.subTest(value=value), self.assertRaises(ValidationError):
                main.Img2ImgRequest(
                    source_generation_id=1,
                    inpaint_full_res_padding=value,
                )

    def test_adetailer_carries_over(self) -> None:
        request = build_img2img_payload("adult woman portrait", "", _settings(), "c291cmNl", 0.4).as_request()
        args = request["alwayson_scripts"]["ADetailer"]["args"]
        hand = next(item for item in args if isinstance(item, dict) and str(item.get("ad_model", "")).startswith("hand"))
        self.assertEqual(hand["ad_denoising_strength"], 0.4)
        self.assertEqual(hand["ad_inpaint_width"], 1024)

    def test_pony_opts_out_of_adetailer(self) -> None:
        request = build_img2img_payload("adult woman portrait", "", _settings(PONY), "c291cmNl", 0.4).as_request()
        self.assertNotIn("alwayson_scripts", request)

    def test_img2img_has_no_high_resolution_fields(self) -> None:
        request = build_img2img_payload("adult woman portrait", "", _settings(), "c291cmNl", 0.4).as_request()
        self.assertNotIn("enable_hr", request)
        self.assertFalse(any(key.startswith("hr_") for key in request))

    def test_masked_payload_contains_verified_inpainting_fields(self) -> None:
        request = build_img2img_payload(
            "adult woman portrait", "", _settings(), "c291cmNl", 0.75,
            mask="bWFzaw==", mask_blur=8, inpaint_full_res=True,
        ).as_request()
        self.assertEqual(request["mask"], "bWFzaw==")
        self.assertEqual(request["inpainting_fill"], 1)
        self.assertTrue(request["inpaint_full_res"])
        self.assertEqual(request["inpaint_full_res_padding"], 32)
        self.assertEqual(request["mask_blur"], 8)

    def test_masked_payload_accepts_per_request_context_padding(self) -> None:
        request = build_img2img_payload(
            "red silk dress", "", _settings(), "c291cmNl", 0.75,
            mask="bWFzaw==", inpaint_full_res_padding=64,
        ).as_request()

        self.assertEqual(request["inpaint_full_res_padding"], 64)

    def test_identity_preserving_mask_skips_all_adetailer_passes(self) -> None:
        request = build_img2img_payload(
            "red silk dress", "", _settings(), "c291cmNl", 0.75,
            mask="bWFzaw==", region_only="region",
            preserve_unmasked_regions=True,
        ).as_request()

        self.assertNotIn("ADetailer", request.get("alwayson_scripts", {}))

    def test_inverted_mask_sets_forge_invert_flag(self) -> None:
        request = build_img2img_payload(
            "adult woman portrait", "", _settings(), "c291cmNl", 0.8,
            mask="bWFzaw==", invert_mask=True,
        ).as_request()
        self.assertEqual(request["inpainting_mask_invert"], 1)

    def test_cfg_floor_holds_on_masked_path(self) -> None:
        request = build_img2img_payload(
            "adult woman portrait", "", _settings(cfg_scale=1.0), "c291cmNl", 0.8,
            mask="bWFzaw==",
        ).as_request()
        self.assertEqual(request["cfg_scale"], 3.0)

    def test_soft_inpainting_uses_live_forge_defaults(self) -> None:
        request = build_img2img_payload(
            "adult woman portrait", "", _settings(), "c291cmNl", 0.8,
            mask="bWFzaw==", soft_inpainting=True,
        ).as_request()
        self.assertEqual(
            request["alwayson_scripts"]["soft inpainting"]["args"],
            [True, 1, 0.5, 4, 0, 0.5, 2],
        )


class Img2ImgEndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        database_path = root / "img2img.db"

        def refine_image(
            image_bytes: bytes,
            *,
            target_width: int,
            target_height: int,
            **_options,
        ) -> bytes:
            output = BytesIO()
            with Image.open(BytesIO(image_bytes)) as source:
                source.convert("RGB").resize((target_width, target_height)).save(
                    output, format="PNG"
                )
            return output.getvalue()

        self.patches = (
            patch.object(database, "DATABASE_PATH", database_path),
            patch.object(database, "GENERATED_DIR", root / "generated"),
            patch.object(database, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(database, "RECOVERY_DIR", root / "generated" / "recovery"),
            patch.object(database, "IDENTITY_PROFILES_DIR", root / "generated" / "identity-profiles"),
            patch.object(database, "UPLOADS_DIR", root / "uploads"),
            patch.object(auth, "DATABASE_PATH", database_path),
            patch.object(auth, "PASSWORD_ITERATIONS", 10_000),
            patch.object(main, "GENERATED_DIR", root / "generated"),
            patch.object(main, "THUMBNAILS_DIR", root / "generated" / "thumbnails"),
            patch.object(main, "IDENTITY_PROFILES_DIR", root / "generated" / "identity-profiles"),
            patch.object(main, "UPLOADS_DIR", root / "uploads"),
            patch.object(
                main.ForgeApiClient,
                "restore_and_upscale_image_bytes",
                side_effect=refine_image,
            ),
        )
        for active_patch in self.patches:
            active_patch.start()
        database.initialize_database()
        auth.initialize_auth_database()
        self.account = auth.create_account("img@example.com", "IMG Admin", "long-password")
        self.session_token = auth.create_session(self.account.id)
        auth.set_session_workspace(self.session_token, "forgeimg")
        self.client_context = TestClient(main.app)
        self.client = self.client_context.__enter__()
        self.client.cookies.set("aiqg_session", self.session_token)
        source_payload = build_payload("adult woman portrait", "", _settings())
        self.source = database.save_generation(_png_bytes(), source_payload, _settings(), actual_seed=123)

    def tearDown(self) -> None:
        self.client_context.__exit__(None, None, None)
        for active_patch in reversed(self.patches):
            active_patch.stop()
        self.temporary_directory.cleanup()

    def _body(self) -> dict[str, object]:
        return {
            "source_generation_id": self.source.id,
            "prompt": "restore natural detail",
            "negative_prompt": "",
            "model": RAGNAROK,
            "style": "Photoreal",
            "content_rating": "Safe",
            "preset": "restore",
            "denoising_strength": 0.35,
        }

    def _upload(self, data: bytes, content_type: str = "image/png"):
        return self.client.post(
            "/api/uploads",
            files={"file": ("source.bin", data, content_type)},
        )

    def _mask(self, size: tuple[int, int]) -> str:
        return base64.b64encode(_png_bytes(size)).decode("ascii")

    def _remote_response(self, chunks: list[bytes], content_type: str = "image/png") -> Mock:
        response = Mock()
        response.status_code = 200
        response.headers = {"content-type": content_type}
        response.iter_content.return_value = iter(chunks)
        response.raise_for_status.return_value = None
        return response

    def test_upload_rejects_spoofed_non_image(self) -> None:
        response = self._upload(b"this is not a png", "image/png")
        self.assertEqual(response.status_code, 422)

    def test_upload_rejects_file_larger_than_25_mb(self) -> None:
        response = self._upload(b"x" * (25 * 1024 * 1024 + 1), "image/png")
        self.assertEqual(response.status_code, 413)

    def test_upload_rejects_axis_larger_than_8192(self) -> None:
        response = self._upload(_png_bytes((8193, 1)))
        self.assertEqual(response.status_code, 422)

    def test_upload_reencode_strips_exif(self) -> None:
        source = BytesIO()
        exif = Image.Exif()
        exif[0x010E] = "sensitive metadata"
        Image.new("RGB", (64, 48), "navy").save(source, format="JPEG", exif=exif)

        response = self._upload(source.getvalue(), "image/jpeg")

        self.assertEqual(response.status_code, 200)
        stored = database.UPLOADS_DIR / f"{response.json()['upload_id']}.png"
        with Image.open(stored) as sanitized:
            self.assertEqual(sanitized.format, "PNG")
            self.assertEqual(len(sanitized.getexif()), 0)

    @patch("backend.main._detected_face_box", return_value=[200, 250, 630, 720])
    def test_face_id_crop_suggestion_is_source_bounded(self, _detect) -> None:
        response = self.client.post(
            "/api/identity-profiles/suggest-crop",
            json={"source_generation_id": self.source.id},
        )
        self.assertEqual(response.status_code, 200, response.text)
        crop = response.json()["crop"]
        self.assertGreater(crop["width"], 0)
        self.assertGreater(crop["height"], 0)
        self.assertLessEqual(crop["x"] + crop["width"], 1.0)
        self.assertLessEqual(crop["y"] + crop["height"], 1.0)

    @patch("backend.main.InsightFaceLocalIdentityProvider")
    def test_face_id_can_be_validated_saved_and_listed(self, provider_type) -> None:
        provider_type.return_value.analyze.return_value = IdentityAnalysis(
            provider="insightface-buffalo-l",
            face_count=1,
            bbox=(70.0, 65.0, 330.0, 335.0),
            det_score=0.98,
            pose=(1.0, -2.0, 0.5),
            image_size=(400, 400),
            sharpness=120.0,
            brightness=128.0,
            dark_clip_ratio=0.01,
            light_clip_ratio=0.01,
            embedding=b"identity-vector",
        )
        body = {
            "source_generation_id": self.source.id,
            "crop": {"x": 0.2, "y": 0.25, "width": 0.5, "height": 0.35},
        }
        validated = self.client.post("/api/identity-profiles/validate", json=body)
        self.assertEqual(validated.status_code, 200, validated.text)
        self.assertTrue(validated.json()["usable"])
        created = self.client.post(
            "/api/identity-profiles", json={**body, "name": "  Studio reference  "}
        )
        self.assertEqual(created.status_code, 201, created.text)
        self.assertEqual(created.json()["name"], "Studio reference")
        profiles = self.client.get("/api/identity-profiles")
        self.assertEqual([item["id"] for item in profiles.json()], [created.json()["id"]])
        thumbnail = self.client.get(created.json()["thumbnail_url"])
        self.assertEqual(thumbnail.status_code, 200)
        self.assertEqual(thumbnail.headers["content-type"], "image/jpeg")

    @patch("backend.main.InsightFaceLocalIdentityProvider")
    def test_face_id_rejects_multiple_faces(self, provider_type) -> None:
        provider_type.return_value.analyze.return_value = IdentityAnalysis(
            provider="insightface-buffalo-l",
            face_count=2,
            bbox=(70.0, 65.0, 330.0, 335.0),
            det_score=0.98,
            pose=(0.0, 0.0, 0.0),
            image_size=(400, 400),
            sharpness=120.0,
            brightness=128.0,
            dark_clip_ratio=0.0,
            light_clip_ratio=0.0,
            embedding=b"identity-vector",
        )
        response = self.client.post(
            "/api/identity-profiles/validate",
            json={
                "source_generation_id": self.source.id,
                "crop": {"x": 0.2, "y": 0.25, "width": 0.5, "height": 0.35},
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["usable"])
        self.assertIn("exactly one face", " ".join(response.json()["reasons"]))

    def test_upload_lands_only_in_uploads_directory(self) -> None:
        response = self._upload(_png_bytes((80, 60)))
        self.assertEqual(response.status_code, 200)
        filename = f"{response.json()['upload_id']}.png"
        self.assertTrue((database.UPLOADS_DIR / filename).is_file())
        self.assertFalse((database.GENERATED_DIR / filename).exists())

    def test_upload_is_downscaled_to_fit_2048(self) -> None:
        response = self._upload(_png_bytes((3000, 1500)))
        self.assertEqual(response.status_code, 200)
        self.assertEqual((response.json()["width"], response.json()["height"]), (2048, 1024))

    def test_generation_vault_serves_deletes_and_preserves_live_directories(self) -> None:
        secret_prompt = "private prompt embedded by Forge"
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("parameters", secret_prompt)
        source = BytesIO()
        Image.new("RGB", (40, 30), "teal").save(
            source, format="PNG", pnginfo=pnginfo
        )
        settings = _settings()
        record = database.save_generation(
            source.getvalue(),
            build_payload(secret_prompt, "", settings),
            settings,
        )

        self.assertRegex(record.filename, r"^[0-9a-f]{32}$")
        self.assertEqual(Path(record.filename).suffix, "")
        image_path = database.GENERATED_DIR / record.filename
        thumbnail_path = database.THUMBNAILS_DIR / record.filename
        self.assertTrue(image_path.is_file())
        self.assertTrue(thumbnail_path.is_file())
        self.assertNotIn(secret_prompt.encode(), image_path.read_bytes())
        with Image.open(image_path) as stored:
            stored.load()
            self.assertNotIn("parameters", stored.info)
            self.assertEqual(getattr(stored, "text", {}), {})

        full = self.client.get(f"/api/images/{record.id}")
        thumbnail = self.client.get(f"/api/images/{record.id}/thumbnail")
        self.assertEqual(full.status_code, 200, full.text)
        self.assertEqual(full.headers["content-type"], "image/png")
        self.assertIn(
            f'filename="generation-{record.id}.png"',
            full.headers["content-disposition"],
        )
        self.assertEqual(thumbnail.status_code, 200, thumbnail.text)
        self.assertEqual(thumbnail.headers["content-type"], "image/jpeg")

        orphan_image = database.GENERATED_DIR / "unindexed-image"
        orphan_thumbnail = database.THUMBNAILS_DIR / "unindexed-thumbnail"
        orphan_image.write_bytes(b"orphan")
        orphan_thumbnail.write_bytes(b"orphan")
        sentinels = []
        for directory_name in (
            "recovery",
            "quality-jobs",
            "character-jobs",
            "identity-profiles",
        ):
            sentinel = database.GENERATED_DIR / directory_name / "keep.bin"
            sentinel.parent.mkdir(parents=True, exist_ok=True)
            sentinel.write_bytes(b"live")
            sentinels.append(sentinel)
        database.initialize_database()
        # Startup recovery must never delete user artifacts merely because a
        # concurrent database snapshot has not indexed them yet.
        self.assertTrue(orphan_image.exists())
        self.assertTrue(orphan_thumbnail.exists())
        self.assertTrue(all(sentinel.is_file() for sentinel in sentinels))

        deleted = self.client.delete(f"/api/images/{record.id}")
        self.assertEqual(deleted.status_code, 204, deleted.text)
        self.assertIsNone(database.get_generation(record.id))
        self.assertFalse(image_path.exists())
        self.assertFalse(thumbnail_path.exists())
        self.assertEqual(self.client.delete(f"/api/images/{record.id}").status_code, 404)

    @patch("backend.main.requests.get")
    def test_url_upload_rejects_non_https_scheme(self, fetch) -> None:
        response = self.client.post("/api/uploads/from-url", json={"url": "http://example.com/image.png"})

        self.assertEqual(response.status_code, 422)
        fetch.assert_not_called()

    @patch("backend.main.socket.getaddrinfo")
    @patch("backend.main.requests.get")
    def test_url_upload_rejects_private_or_loopback_host(self, fetch, resolve) -> None:
        resolve.return_value = [(2, 1, 6, "", ("127.0.0.1", 443))]

        response = self.client.post("/api/uploads/from-url", json={"url": "https://localhost/image.png"})

        self.assertEqual(response.status_code, 422)
        self.assertIn("Private or local", response.json()["detail"])
        fetch.assert_not_called()

    @patch("backend.main.socket.getaddrinfo")
    @patch("backend.main.requests.get")
    def test_url_upload_stream_is_capped_at_25_mb(self, fetch, resolve) -> None:
        resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        fetch.return_value = self._remote_response(
            [b"x" * main.UPLOAD_CHUNK_BYTES] * 26,
            "image/png",
        )

        response = self.client.post("/api/uploads/from-url", json={"url": "https://example.com/large.png"})

        self.assertEqual(response.status_code, 413)
        fetch.assert_called_once()
        self.assertFalse(fetch.call_args.kwargs["allow_redirects"])

    @patch("backend.main.socket.getaddrinfo")
    @patch("backend.main.requests.get")
    def test_url_upload_rejects_non_image_payload_with_image_type(self, fetch, resolve) -> None:
        resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        fetch.return_value = self._remote_response([b"not really a png"], "image/png")

        response = self.client.post("/api/uploads/from-url", json={"url": "https://example.com/fake.png"})

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"], "The uploaded image is invalid.")

    @patch("backend.main.socket.getaddrinfo")
    @patch("backend.main.requests.get")
    def test_url_upload_reencodes_png_and_strips_exif(self, fetch, resolve) -> None:
        resolve.return_value = [(2, 1, 6, "", ("93.184.216.34", 443))]
        source = BytesIO()
        exif = Image.Exif()
        exif[0x010E] = "private remote metadata"
        Image.new("RGB", (80, 60), "navy").save(source, format="JPEG", exif=exif)
        fetch.return_value = self._remote_response([source.getvalue()], "image/jpeg")

        response = self.client.post("/api/uploads/from-url", json={"url": "https://example.com/source.jpg"})

        self.assertEqual(response.status_code, 200)
        stored = database.UPLOADS_DIR / f"{response.json()['upload_id']}.png"
        with Image.open(stored) as sanitized:
            self.assertEqual(sanitized.format, "PNG")
            self.assertEqual(sanitized.size, (80, 60))
            self.assertEqual(len(sanitized.getexif()), 0)

    def test_stale_upload_sweep_keeps_recent_inputs(self) -> None:
        stale = database.UPLOADS_DIR / f"{'a' * 32}.png"
        recent = database.UPLOADS_DIR / f"{'b' * 32}.png"
        stale.write_bytes(_png_bytes((8, 8)))
        recent.write_bytes(_png_bytes((8, 8)))
        now = time.time()
        os.utime(stale, (now - 8 * 24 * 60 * 60, now - 8 * 24 * 60 * 60))

        removed = database.sweep_stale_uploads(now=now)

        self.assertEqual(removed, 1)
        self.assertFalse(stale.exists())
        self.assertTrue(recent.exists())

    def test_img2img_requires_exactly_one_source(self) -> None:
        uploaded = self._upload(_png_bytes((80, 60))).json()["upload_id"]
        both = self.client.post("/api/img2img", json={**self._body(), "upload_id": uploaded})
        neither_body = self._body()
        neither_body.pop("source_generation_id")
        neither = self.client.post("/api/img2img", json=neither_body)
        self.assertEqual(both.status_code, 422)
        self.assertEqual(neither.status_code, 422)

    def test_upload_source_requires_authenticated_account(self) -> None:
        uploaded = self._upload(_png_bytes((80, 60))).json()["upload_id"]
        self.client.cookies.clear()
        body = {**self._body(), "upload_id": uploaded}
        body.pop("source_generation_id")
        response = self.client.post("/api/img2img", json=body)
        self.assertIn(response.status_code, {401, 403})

    def test_mask_dimensions_must_match_source(self) -> None:
        response = self.client.post(
            "/api/img2img",
            json={**self._body(), "mask": self._mask((32, 32))},
        )
        self.assertEqual(response.status_code, 422)

    def test_manual_and_automatic_masks_are_mutually_exclusive(self) -> None:
        response = self.client.post(
            "/api/img2img",
            json={**self._body(), "mask": self._mask((832, 1216)), "auto_mask": "body"},
        )
        self.assertEqual(response.status_code, 422)

    @patch("backend.main._automatic_mask")
    @patch("backend.main.ForgeApiClient.transform_image")
    def test_server_side_body_auto_mask_is_applied(self, transform, automatic_mask) -> None:
        automatic_mask.return_value = self._mask((832, 1216))
        transform.return_value = SimpleNamespace(
            image_bytes=_png_bytes(), seed=999,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )

        response = self.client.post("/api/img2img", json={**self._body(), "auto_mask": "body"})
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        automatic_mask.assert_called_once()
        self.assertEqual(automatic_mask.call_args.args[1], "body")
        self.assertIsNotNone(transform.call_args.args[0].mask)

    @patch("backend.main.ForgeApiClient.transform_image")
    def test_outfit_preset_preserves_unmasked_regions_and_forwards_padding(self, transform) -> None:
        transform.return_value = SimpleNamespace(
            image_bytes=_png_bytes(), seed=999,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )
        response = self.client.post(
            "/api/img2img",
            json={
                **self._body(),
                "preset": "outfit",
                "region_only": "region",
                "mask": self._mask((832, 1216)),
                "inpaint_full_res_padding": 64,
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        request = transform.call_args.args[0].as_request()
        self.assertEqual(request["inpaint_full_res_padding"], 64)
        self.assertNotIn("ADetailer", request.get("alwayson_scripts", {}))

    @patch("backend.main._face_reference", return_value=base64.b64encode(_png_bytes()).decode("ascii"))
    @patch("backend.main.InsightFaceLocalIdentityProvider")
    @patch("backend.main.ForgeApiClient.resolve_controlnet_model")
    @patch("backend.main.ForgeApiClient.generate_image")
    def test_face_identity_uses_resumable_instantid_character_pipeline(
        self, generate, resolve_model, identity_provider, _face_reference
    ) -> None:
        generate.return_value = SimpleNamespace(
            image_bytes=_png_bytes(), seed=999,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )
        resolve_model.side_effect = lambda prefix: f"{prefix} [hash]"
        provider = identity_provider.return_value
        provider.readiness_error.return_value = None
        provider.verify.return_value = SimpleNamespace(
            score=0.93, threshold=0.6, provider="buffalo_l", passed=True,
        )

        def lock(source, target, output):
            output.write_bytes(target.read_bytes())
            return SimpleNamespace(
                provider="inswapper_128", similarity_before=0.5, similarity_after=0.93,
            )

        provider.lock.side_effect = lock
        response = self.client.post(
            "/api/img2img",
            json={
                **self._body(),
                "preset": "variation",
                "denoising_strength": 0.8,
                "face_identity": True,
                "mask": self._mask((832, 1216)),
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        self.assertEqual(job["kind"], "character")
        payload = generate.call_args.args[0]
        request = payload.as_request(high_res=False)
        self.assertNotIn("mask", request)
        self.assertEqual(
            request["alwayson_scripts"]["ControlNet"]["args"][0]["module"],
            "InsightFace (InstantID)",
        )
        self.assertEqual(
            request["alwayson_scripts"]["ControlNet"]["args"][1]["module"],
            "instant_id_face_keypoints",
        )
        self.assertEqual(job["request_payload"]["character_stage"], "COMPLETED")
        self.assertEqual(
            job["request_payload"]["character_selected_generator"],
            "instantid-sdxl",
        )
        self.assertGreaterEqual(job["request_payload"]["character_identity_score"], 0.6)
        self.assertTrue(job["request_payload"]["character_refinement_selected"])
        self.assertEqual(
            job["request_payload"]["character_candidate_a_refinement_upscaler"],
            "4x_NMKD-Siax_200k",
        )
        result = database.get_generation(job["generation_id"])
        self.assertEqual((result.width, result.height), (1248, 1824))

    @patch("backend.main._face_reference", return_value=base64.b64encode(_png_bytes()).decode("ascii"))
    @patch("backend.main.InsightFaceLocalIdentityProvider")
    @patch("backend.main.ForgeApiClient.resolve_controlnet_model")
    @patch("backend.main.ForgeApiClient.generate_image")
    def test_character_refinement_falls_back_to_verified_raw_swap(
        self, generate, resolve_model, identity_provider, _face_reference
    ) -> None:
        generate.return_value = SimpleNamespace(
            image_bytes=_png_bytes(), seed=444,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )
        resolve_model.side_effect = lambda prefix: f"{prefix} [hash]"
        provider = identity_provider.return_value
        provider.readiness_error.return_value = None
        provider.verify.side_effect = [
            SimpleNamespace(score=1.0, threshold=0.6, provider="buffalo_l", passed=True),
            SimpleNamespace(score=0.4, threshold=0.6, provider="buffalo_l", passed=False),
            SimpleNamespace(score=0.93, threshold=0.6, provider="buffalo_l", passed=True),
            SimpleNamespace(score=0.52, threshold=0.6, provider="buffalo_l", passed=False),
        ]

        def lock(source, target, output):
            output.write_bytes(target.read_bytes())
            return SimpleNamespace(
                provider="inswapper_128", similarity_before=0.4, similarity_after=0.93,
            )

        provider.lock.side_effect = lock
        response = self.client.post(
            "/api/img2img",
            json={**self._body(), "preset": "variation", "face_identity": True},
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        self.assertFalse(job["request_payload"]["character_refinement_selected"])
        self.assertEqual(
            job["request_payload"]["character_candidate_a_refined_identity_score"],
            0.52,
        )
        result = database.get_generation(job["generation_id"])
        self.assertEqual((result.width, result.height), (832, 1216))

    @patch("backend.main._face_reference", return_value=base64.b64encode(_png_bytes()).decode("ascii"))
    @patch("backend.main.InsightFaceLocalIdentityProvider")
    @patch("backend.main.ForgeApiClient.resolve_controlnet_model")
    @patch("backend.main.ForgeApiClient.generate_image")
    def test_character_pipeline_accepts_saved_face_id_without_full_source(
        self, generate, resolve_model, identity_provider, _face_reference
    ) -> None:
        image_name = "identity-test.png"
        thumb_name = "identity-test.jpg"
        (database.IDENTITY_PROFILES_DIR / image_name).write_bytes(_png_bytes((400, 400)))
        (database.IDENTITY_PROFILES_DIR / thumb_name).write_bytes(b"thumbnail")
        identity = database.create_identity_profile(
            account_id=self.account.id,
            name="Saved identity",
            filename=image_name,
            thumbnail_filename=thumb_name,
            embedding=b"vector",
            detector="buffalo_l",
            embedding_model="buffalo_l",
            quality={"status": "ready"},
        )
        generate.return_value = SimpleNamespace(
            image_bytes=_png_bytes(), seed=222,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )
        resolve_model.side_effect = lambda prefix: f"{prefix} [hash]"
        provider = identity_provider.return_value
        provider.readiness_error.return_value = None
        provider.verify.return_value = SimpleNamespace(
            score=0.92, threshold=0.6, provider="buffalo_l", passed=True,
        )

        def lock(source, target, output):
            output.write_bytes(target.read_bytes())
            return SimpleNamespace(
                provider="inswapper_128", similarity_before=0.4, similarity_after=0.92,
            )

        provider.lock.side_effect = lock
        response = self.client.post(
            "/api/img2img",
            json={
                **self._body(),
                "source_generation_id": None,
                "identity_profile_id": identity.id,
                "preset": "variation",
                "face_identity": True,
            },
        )
        self.assertEqual(response.status_code, 202, response.text)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        self.assertEqual(job["request_payload"]["identity_profile_id"], identity.id)
        self.assertEqual(job["request_payload"]["character_source_size"], [400, 400])
        refreshed = database.get_identity_profile(identity.id, account_id=self.account.id)
        self.assertIsNotNone(refreshed.last_used_at)

    @patch("backend.main._face_reference")
    @patch("backend.main.InsightFaceLocalIdentityProvider")
    @patch("backend.main.ForgeApiClient.resolve_controlnet_model")
    @patch("backend.main.ForgeApiClient.generate_image")
    def test_character_pipeline_falls_back_to_verified_plain_sdxl(
        self, generate, resolve_model, identity_provider, face_reference
    ) -> None:
        face_reference.return_value = base64.b64encode(_png_bytes()).decode("ascii")
        generate.side_effect = [
            SimpleNamespace(
                image_bytes=_png_bytes((32, 32)), seed=111,
                diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
            ),
            SimpleNamespace(
                image_bytes=_png_bytes((32, 32)), seed=222,
                diffusion_duration_seconds=1.5, upscale_duration_seconds=0.0,
            ),
        ]
        resolve_model.side_effect = lambda prefix: f"{prefix} [hash]"
        provider = identity_provider.return_value
        provider.readiness_error.return_value = None
        provider.verify.side_effect = [
            SimpleNamespace(score=1.0, threshold=0.6, provider="buffalo_l", passed=True),
            main.IdentityProviderError("No face detected in the result image."),
            SimpleNamespace(score=0.31, threshold=0.6, provider="buffalo_l", passed=False),
            SimpleNamespace(score=0.91, threshold=0.6, provider="buffalo_l", passed=True),
            SimpleNamespace(score=0.90, threshold=0.6, provider="buffalo_l", passed=True),
        ]

        def lock(source, target, output):
            output.write_bytes(target.read_bytes())
            return SimpleNamespace(
                provider="inswapper_128", similarity_before=0.31, similarity_after=0.91,
            )

        provider.lock.side_effect = lock
        response = self.client.post(
            "/api/img2img",
            json={
                **self._body(),
                "preset": "variation",
                "denoising_strength": 0.8,
                "face_identity": True,
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(provider.lock.call_count, 1)
        first_request = generate.call_args_list[0].args[0].as_request(high_res=False)
        second_request = generate.call_args_list[1].args[0].as_request(high_res=False)
        self.assertIn("ControlNet", first_request["alwayson_scripts"])
        self.assertNotIn("ControlNet", second_request.get("alwayson_scripts", {}))
        self.assertEqual(job["request_payload"]["character_selected_candidate"], "B")
        self.assertEqual(
            job["request_payload"]["character_selected_generator"],
            "sdxl-identity-lock",
        )
        self.assertIn(
            "No face detected",
            job["request_payload"]["character_candidate_a_failure"],
        )

    @patch("backend.main._face_reference", return_value=base64.b64encode(_png_bytes()).decode("ascii"))
    @patch("backend.main.InsightFaceLocalIdentityProvider")
    @patch("backend.main.ForgeApiClient.resolve_controlnet_model")
    @patch("backend.main.ForgeApiClient.generate_image")
    def test_character_pipeline_rejects_nonpositive_prelock_identity(
        self, generate, resolve_model, identity_provider, _face_reference
    ) -> None:
        generate.return_value = SimpleNamespace(
            image_bytes=_png_bytes((32, 32)), seed=333,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )
        resolve_model.side_effect = lambda prefix: f"{prefix} [hash]"
        provider = identity_provider.return_value
        provider.readiness_error.return_value = None
        provider.verify.side_effect = [
            SimpleNamespace(score=1.0, threshold=0.6, provider="buffalo_l", passed=True),
            SimpleNamespace(score=-0.076, threshold=0.6, provider="buffalo_l", passed=False),
            SimpleNamespace(score=0.0, threshold=0.6, provider="buffalo_l", passed=False),
        ]

        response = self.client.post(
            "/api/img2img",
            json={
                **self._body(),
                "preset": "variation",
                "face_identity": True,
            },
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)

        self.assertEqual(job["status"], "failed")
        self.assertEqual(generate.call_count, 2)
        provider.lock.assert_not_called()
        self.assertEqual(
            job["request_payload"]["character_candidate_a_prelock_score"],
            -0.076,
        )
        self.assertIn(
            "not positive",
            job["request_payload"]["character_candidate_b_failure"],
        )

    def test_face_identity_rejects_non_sdxl_checkpoint(self) -> None:
        response = self.client.post(
            "/api/img2img",
            json={
                **self._body(),
                "model": PONY,
                "style": "Anime",
                "preset": "variation",
                "face_identity": True,
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("SDXL checkpoint", response.json()["detail"])

    @patch("backend.main.subprocess.run")
    @patch("backend.main._mask_detector_runtime")
    def test_mask_proposals_use_authorized_source_and_detector(self, runtime, run) -> None:
        runtime.return_value = (Path("forge-python.exe"), Path("adetailer-models"))
        run.return_value = SimpleNamespace(
            stdout='{"proposals":[{"id":"person","label":"Detected person","mask":"bWFzaw=="}]}\n'
        )

        response = self.client.post(
            "/api/img2img/mask-proposals",
            json={"source_generation_id": self.source.id, "detector": "person"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["proposals"][0]["id"], "person")
        self.assertIn("person", run.call_args.args[0])

    def test_mask_proposals_reject_unauthorized_source(self) -> None:
        self.client.cookies.clear()
        response = self.client.post(
            "/api/img2img/mask-proposals",
            json={"source_generation_id": self.source.id, "detector": "features"},
        )
        self.assertEqual(response.status_code, 403)

    def test_curated_transform_suggestion_matches_requested_field(self) -> None:
        response = self.client.post(
            "/api/img2img/suggestion",
            json={"source_generation_id": self.source.id, "kind": "background", "prompt_engine": "curated"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["engine"], "curated")
        self.assertIn(response.json()["suggestion"], main._TRANSFORM_SUGGESTIONS["background"])

    def test_curated_scene_suggestion_is_distinct_from_outfit_suggestions(self) -> None:
        response = self.client.post(
            "/api/img2img/suggestion",
            json={"source_generation_id": self.source.id, "kind": "scene", "prompt_engine": "curated"},
        )
        self.assertEqual(response.status_code, 200)
        suggestion = response.json()["suggestion"]
        self.assertIn(suggestion, main._TRANSFORM_SUGGESTIONS["scene"])
        self.assertNotIn(suggestion, main._TRANSFORM_SUGGESTIONS["outfit"])

    @patch("backend.main.OllamaPromptClient.suggest_transform")
    def test_ollama_transform_suggestion_is_returned(self, suggest) -> None:
        suggest.return_value = "a moonlit botanical garden with glass lanterns"
        response = self.client.post(
            "/api/img2img/suggestion",
            json={"source_generation_id": self.source.id, "kind": "background", "prompt_engine": "ollama", "ollama_model": "qwen-test"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["suggestion"], suggest.return_value)
        self.assertEqual(response.json()["engine"], "ollama:qwen-test")
        self.assertTrue(suggest.call_args.kwargs["image_base64"])

    def test_img2img_json_body_is_capped_at_32_mb(self) -> None:
        response = self.client.post(
            "/api/img2img",
            content=b"{}",
            headers={
                "content-type": "application/json",
                "content-length": str(main.MAX_IMG2IMG_BODY_BYTES + 1),
            },
        )
        self.assertEqual(response.status_code, 413)

    @patch("backend.main.ForgeApiClient.transform_image")
    def test_large_source_is_downscaled_to_native_resolution(self, transform) -> None:
        transform.return_value = SimpleNamespace(
            image_bytes=_png_bytes(), seed=999,
            diffusion_duration_seconds=1.0, upscale_duration_seconds=0.0,
        )
        large_settings = GenerationSettings(
            width=1664, height=2432, model=RAGNAROK, rating="Safe",
            style="Photoreal", high_res=True,
        )
        large_payload = build_payload("adult woman portrait", "", large_settings)
        large_source = database.save_generation(
            _png_bytes((1664, 2432)), large_payload, large_settings, actual_seed=123
        )
        response = self.client.post(
            "/api/img2img",
            json={**self._body(), "source_generation_id": large_source.id},
        )
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        self.assertEqual(job["status"], "succeeded", job.get("error_message"))
        payload = transform.call_args.args[0]
        self.assertEqual((payload.width, payload.height), (832, 1216))

    @patch("backend.main.ForgeApiClient.transform_image")
    def test_completed_transform_preserves_lineage(self, transform) -> None:
        transform.return_value = SimpleNamespace(
            image_bytes=_png_bytes(),
            seed=456,
            diffusion_duration_seconds=1.0,
            upscale_duration_seconds=0.0,
        )
        response = self.client.post("/api/img2img", json=self._body())
        self.assertEqual(response.status_code, 202)
        job_id = response.json()["job_id"]
        for _ in range(100):
            job = self.client.get(f"/api/jobs/{job_id}").json()
            if job["status"] == "succeeded":
                self.assertEqual(job["generation"]["source_generation_id"], self.source.id)
                break
            if job["status"] == "failed":
                self.fail(job.get("error_message"))
            time.sleep(0.01)
        else:
            self.fail("Img2img job did not finish in time.")

    @patch("backend.main.ForgeApiClient.transform_image")
    def test_unauthorized_source_returns_403_without_rendering(self, transform) -> None:
        self.client.cookies.clear()
        response = self.client.post("/api/img2img", json=self._body())
        self.assertEqual(response.status_code, 403)
        transform.assert_not_called()


if __name__ == "__main__":
    unittest.main()

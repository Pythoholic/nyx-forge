from __future__ import annotations

import tempfile
import re
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from backend import auth, database, main
from backend.comfy import ComfyApiClient, VIDEO_PIXEL_BUDGET, build_workflow, derive_i2v_size, derive_upscale_size
from backend.video_prompt import VIDEO_PROMPT_PROFILES, curated_video_prompt, ollama_video_prompt, strip_sdxl_tags, validate_video_prompt


class VideoWorkflowTests(unittest.TestCase):
    def test_t2v_mutates_only_allowlisted_inputs(self) -> None:
        graph = build_workflow("t2v", prompt="waves move", negative_prompt="blur", seed=123)
        self.assertEqual(graph["6"]["inputs"]["text"], "waves move")
        self.assertEqual(graph["7"]["inputs"]["text"], "blur")
        self.assertEqual(graph["3"]["inputs"]["seed"], 123)
        self.assertEqual(graph["40"]["inputs"]["length"], 49)
        self.assertEqual(graph["37"]["inputs"]["unet_name"], "wan2.1_t2v_1.3B_fp16.safetensors")

    def test_i2v_requires_source_and_rejects_camera_outside_allowlist(self) -> None:
        with self.assertRaises(ValueError):
            build_workflow("i2v", prompt="move", negative_prompt="", seed=1)
        with self.assertRaises(ValueError):
            build_workflow("i2v", prompt="move", negative_prompt="", seed=1, source_filename="a.png", camera_motion="custom node")

    def test_quality_graph_keeps_all_rife_frames_and_adds_upscale(self) -> None:
        graph = build_workflow("t2v", prompt="waves move", negative_prompt="blur", seed=3, interpolation=4, output_resolution="1080p")
        interpolate = next(node for node in graph.values() if node["class_type"] == "FrameInterpolate")
        create = next(node for node in graph.values() if node["class_type"] == "CreateVideo")
        resize = next(node for node in graph.values() if node["class_type"] == "ImageScale")
        self.assertEqual(interpolate["inputs"]["multiplier"], 4)
        self.assertEqual(create["inputs"]["fps"], 64.0)
        self.assertEqual((resize["inputs"]["width"], resize["inputs"]["height"]), derive_upscale_size(832, 480, "1080p"))

    def test_final_frame_is_saved_before_interpolation_and_upscale(self) -> None:
        graph = build_workflow("t2v", prompt="waves move", negative_prompt="blur", seed=3, interpolation=4, output_resolution="1080p")
        decoder_id = next(key for key, node in graph.items() if node["class_type"] == "VAEDecode")
        selector_id, selector = next((key, node) for key, node in graph.items() if node["class_type"] == "ImageFromBatch")
        saver = next(node for node in graph.values() if node.get("_meta", {}).get("title") == "Save final native frame")
        self.assertEqual(selector["inputs"], {"image": [decoder_id, 0], "batch_index": 48, "length": 1})
        self.assertEqual(saver["inputs"]["images"], [selector_id, 0])

    def test_i2v_size_matches_source_aspect_within_budget_and_stride(self) -> None:
        portrait = derive_i2v_size(832, 1216)
        landscape = derive_i2v_size(1216, 832)
        square = derive_i2v_size(1000, 1000)
        self.assertLess(portrait[0], portrait[1])
        self.assertGreater(landscape[0], landscape[1])
        self.assertEqual(square, (512, 512))
        for size in (portrait, landscape, square, derive_i2v_size(10000, 100), derive_i2v_size(100, 10000)):
            self.assertEqual(size[0] % 16, 0)
            self.assertEqual(size[1] % 16, 0)
            self.assertLessEqual(size[0] * size[1], VIDEO_PIXEL_BUDGET)

    def test_post_upscale_preserves_portrait_aspect(self) -> None:
        native = derive_i2v_size(832, 1216)
        upscaled = derive_upscale_size(*native, "1080p")
        self.assertLess(upscaled[0], upscaled[1])
        self.assertAlmostEqual(upscaled[0] / upscaled[1], native[0] / native[1], delta=0.02)

    def test_video_prompts_use_motion_prose_and_strip_sdxl_tags(self) -> None:
        cleaned = strip_sdxl_tags("score_9, source_photo, rating_safe, a woman walking, \\(medium\\)")
        self.assertNotIn("score_9", cleaned)
        self.assertNotIn("source_photo", cleaned)
        self.assertNotIn("rating_safe", cleaned)
        prompt = curated_video_prompt(mode="t2v", source_prompt=cleaned)
        motion_words = {"walks", "turns", "looks", "reaches", "camera", "move", "changes"}
        self.assertTrue(any(word in prompt.lower() for word in motion_words))

    def test_engine_video_job_rejects_appearance_only_or_checkpoint_tags(self) -> None:
        client = TestClient(main.app)
        payload = {
            "prompt": "score_9, source_photo, rating_safe, woman in a tailored suit, portrait",
            "prompt_engine": "curated",
        }
        with patch.object(main, "_require_workspace"):
            response = client.post("/api/video/t2v", json=payload)
        self.assertEqual(response.status_code, 422)
        self.assertIn("motion prompt", response.json()["detail"].lower())

        motion_prompt = curated_video_prompt(mode="t2v", source_prompt=payload["prompt"])
        self.assertEqual(validate_video_prompt(motion_prompt), motion_prompt)
        self.assertNotIn("score_9", motion_prompt)
        self.assertNotIn("source_photo", motion_prompt)

    @patch("backend.video_prompt.requests.post")
    def test_ollama_i2v_prompt_sends_the_source_image_to_vision_model(self, post) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"response": "The subject turns slowly while the camera tracks forward."}
        post.return_value = response

        prompt = ollama_video_prompt("gemma4:12b", context="score_9, portrait", image_base64="encoded-frame")

        self.assertIn("turns slowly", prompt)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["images"], ["encoded-frame"])
        self.assertIn("Study this starting frame", payload["prompt"])
        self.assertIn("Analyze the supplied starting frame", payload["system"])

    @patch("backend.comfy.requests.get")
    @patch("backend.comfy.requests.post")
    @patch("backend.comfy.websocket.create_connection")
    def test_dropped_websocket_falls_back_to_completed_history(self, create_connection, post, get) -> None:
        import websocket
        socket = Mock()
        socket.recv.side_effect = websocket.WebSocketConnectionClosedException("dropped")
        create_connection.return_value = socket
        submitted = Mock(); submitted.json.return_value = {"prompt_id": "pid", "node_errors": {}}; submitted.raise_for_status.return_value = None
        post.return_value = submitted
        def response(url, **_kwargs):
            item = Mock(); item.raise_for_status.return_value = None
            if "/view?" in url:
                is_poster = "poster.png" in url
                item.content = b"poster" if is_poster else b"video"
                item.headers = {"content-type": "image/png" if is_poster else "video/mp4"}
                return item
            item.json.return_value = {"pid": {"status": {"completed": True}, "outputs": {
                "1": {"images": [{"filename": "x.mp4", "subfolder": "video", "type": "output"}]},
                "2": {"images": [{"filename": "poster.png", "subfolder": "video", "type": "output"}]},
            }}}
            return item
        get.side_effect = response
        result = ComfyApiClient().generate_video({
            "1": {"class_type": "SaveVideo", "inputs": {}},
            "2": {"class_type": "SaveImage", "inputs": {}, "_meta": {"title": "Save final native frame"}},
        })
        self.assertEqual(result.content, b"video")
        self.assertEqual(result.poster_content, b"poster")

    @patch("backend.main.ComfyApiClient.health", side_effect=__import__("requests").ConnectionError("offline"))
    def test_status_reports_comfy_unreachable(self, _health) -> None:
        response = TestClient(main.app).get("/api/video/status")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"connected": False, "backend": "ComfyUI"})


class VideoPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.patches = [
            patch.object(database, "DATABASE_PATH", root / "prompts.db"),
            patch.object(database, "VIDEOS_DIR", root / "videos"),
            patch.object(main, "VIDEOS_DIR", root / "videos"),
        ]
        for item in self.patches: item.start()
        database.initialize_database()

    def tearDown(self) -> None:
        for item in reversed(self.patches): item.stop()
        self.temp.cleanup()

    def test_video_bytes_are_stored_on_disk_and_range_streams(self) -> None:
        record = database.save_video_generation(content=b"0123456789", prompt="p", negative_prompt="n",
            mode="t2v", source_generation_id=None, width=832, height=480, frame_count=49,
            fps=16, seed=5, duration_seconds=3.0625, camera_motion=None, comfy_prompt_id="pid")
        with database._connection() as connection:
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(video_generations)")}
        self.assertNotIn("video_bytes", columns)
        account = SimpleNamespace(id=1, is_admin=True)
        with patch("backend.main._require_authenticated_account", return_value=account):
            response = TestClient(main.app).get(f"/api/video/{record.id}/file", headers={"Range":"bytes=2-5"})
        self.assertEqual(response.status_code, 206)
        self.assertEqual(response.content, b"2345")
        self.assertEqual(response.headers["content-range"], "bytes 2-5/10")

    def test_native_final_frame_is_persisted_and_served(self) -> None:
        record = database.save_video_generation(
            content=b"video",
            poster_content=b"\x89PNG\r\n\x1a\nposter",
            prompt="camera tracks the subject as she walks",
            negative_prompt="blur",
            mode="i2v",
            source_generation_id=None,
            width=512,
            height=512,
            frame_count=49,
            fps=16,
            seed=5,
            duration_seconds=3.0625,
            camera_motion="Zoom In",
            comfy_prompt_id="pid",
        )
        self.assertIsNotNone(record.poster_filename)
        self.assertEqual((database.VIDEOS_DIR / record.poster_filename).read_bytes(), b"\x89PNG\r\n\x1a\nposter")
        account = SimpleNamespace(id=1, is_admin=True)
        with patch("backend.main._require_authenticated_account", return_value=account):
            response = TestClient(main.app).get(f"/api/video/{record.id}/poster")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        self.assertEqual(response.content, b"\x89PNG\r\n\x1a\nposter")

    def test_running_video_job_is_requeued_after_restart(self) -> None:
        job_id = database.create_job("video_t2v", {"prompt":"p", "negative_prompt":"n", "comfy_prompt_id":"pid"}, None, None)
        database.update_job_progress(job_id, status="running")
        database.recover_stalled_jobs()
        self.assertEqual(database.get_job(job_id).status, "queued")
        self.assertEqual(database.get_job(job_id).request_payload["comfy_prompt_id"], "pid")


if __name__ == "__main__":
    unittest.main()

class CameraMotionTests(unittest.TestCase):
    def test_frontend_and_backend_camera_lists_match(self) -> None:
        """The two lists drifted apart once and ComfyUI rejected the prompt.

        A camera_pose outside WanCameraEmbedding's options fails validation for
        the whole graph, so the UI must never offer one.
        """
        from backend.comfy import CAMERA_MOTIONS

        source = Path(__file__).resolve().parents[1] / "frontend" / "src" / "ForgeVIDPage.jsx"
        block = re.search(r"const CAMERAS = \[(.*?)\];", source.read_text(encoding="utf-8"), re.S)
        self.assertIsNotNone(block, "CAMERAS list not found in ForgeVIDPage.jsx")
        frontend = re.findall(r'"([^"]+)"', block.group(1))
        self.assertEqual(list(CAMERA_MOTIONS), frontend)

class FrameAnalysisTests(unittest.TestCase):
    """The vision analysis decides what can move and which camera fits."""

    def _payload(self, **over):
        import json as _json

        data = {
            "pose": "lying", "facing": "right",
            "motion_candidates": [{"action": "hair swaying in water", "intensity": "moderate"}],
            "secondary_motion": ["water rippling"],
            "camera": {"recommended": "Zoom In", "reason": "close portrait"},
        }
        data.update(over)
        return _json.dumps(data)

    def test_parses_and_composes_a_motion_only_prompt(self) -> None:
        from backend.video_prompt import _parse_analysis, compose_i2v_prompt

        analysis = _parse_analysis(self._payload())
        prompt = compose_i2v_prompt(analysis)
        self.assertIn("hair swaying in water", prompt.lower())
        self.assertIn("pushes in", prompt.lower())
        # the image already carries appearance; the prompt must not restate it
        self.assertNotIn("woman", prompt.lower())

    def test_rejects_a_camera_the_pose_cannot_support(self) -> None:
        from backend.video_prompt import _parse_analysis

        analysis = _parse_analysis(
            self._payload(camera={"recommended": "Anti Clockwise (ACW)", "reason": "x"})
        )
        self.assertNotEqual(analysis["camera"], "Anti Clockwise (ACW)")

    def test_recommended_camera_is_always_a_real_comfy_pose(self) -> None:
        from backend.comfy import CAMERA_MOTIONS
        from backend.video_prompt import CAMERA_POSES, _parse_analysis

        self.assertEqual(list(CAMERA_POSES), list(CAMERA_MOTIONS))
        analysis = _parse_analysis(self._payload(camera={"recommended": "Tilt Down", "reason": "x"}))
        self.assertIn(analysis["camera"], CAMERA_MOTIONS)

    def test_unusable_vision_output_raises_rather_than_guessing(self) -> None:
        from backend.video_prompt import _parse_analysis

        for junk in ("no json at all", '{"pose": "lying"}'):
            with self.assertRaises(ValueError):
                _parse_analysis(junk)

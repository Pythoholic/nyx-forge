"""Tests for the memory-aware Forge generation pipeline."""

from __future__ import annotations

import base64
import unittest
from unittest.mock import Mock, patch

import requests

from backend.forge import (
    ForgeApiClient,
    Txt2ImgPayload,
    release_other_backends,
)


class ForgeClientTests(unittest.TestCase):
    """Verify quality tiers use their intended memory-aware pipelines."""

    def test_model_profiles_route_to_their_configured_backends(self) -> None:
        default_url = ForgeApiClient().base_url
        self.assertEqual(ForgeApiClient.for_model("PonyDiffusionV6XL").base_url, default_url)
        self.assertEqual(ForgeApiClient.for_model("JuggernautXL_Ragnarok").base_url, default_url)
        self.assertNotEqual(ForgeApiClient.for_model("flux1-dev-Q5_K_S.gguf").base_url, default_url)

    @patch("backend.forge.requests.post")
    def test_release_other_backends_skips_active_backend(self, post: Mock) -> None:
        response = Mock()
        post.return_value = response
        active_url = ForgeApiClient("reforge").base_url

        release_other_backends(active_url)

        post.assert_called_once_with(
            f"{ForgeApiClient('forge_neo').base_url}/sdapi/v1/unload-checkpoint",
            timeout=(0.5, 5.0),
        )
        response.raise_for_status.assert_called_once_with()

    @patch(
        "backend.forge.requests.post",
        side_effect=requests.ConnectionError("backend is stopped"),
    )
    def test_release_other_backends_ignores_unreachable_backend(self, post: Mock) -> None:
        release_other_backends(ForgeApiClient("reforge").base_url)

        self.assertEqual(post.call_count, 1)

    @patch("backend.forge.requests.get")
    def test_request_uses_the_configured_default_backend(self, get: Mock) -> None:
        response = Mock()
        get.return_value = response
        client = ForgeApiClient()

        result = client._request("get", "/docs", timeout=(1.0, 2.0))

        self.assertIs(result, response)
        get.assert_called_once_with(f"{client.base_url}/docs", timeout=(1.0, 2.0))

    @patch("backend.forge.requests.get", side_effect=requests.ReadTimeout("still rendering"))
    def test_read_timeout_is_not_retried_as_a_duplicate_job(self, get: Mock) -> None:
        with self.assertRaises(requests.ReadTimeout):
            ForgeApiClient()._request("get", "/docs", timeout=(1.0, 2.0))
        self.assertEqual(get.call_count, 1)

    @patch("backend.forge.requests.post")
    def test_dispatch_rejects_over_budget_prompt_before_http(self, post: Mock) -> None:
        with self.assertRaisesRegex(ValueError, "Positive prompt has"):
            ForgeApiClient().generate_image(
                Txt2ImgPayload(
                    "woman " * 76, "negative", 1216, 832,
                    enforce_sdxl_token_budget=True,
                ),
                high_res=False,
            )
        post.assert_not_called()

    @patch("backend.forge.requests.post")
    def test_super_res_uses_bounded_detail_pass_then_exact_pixel_upscale(self, post: Mock) -> None:
        source = b"generated-image"
        enlarged = b"upscaled-image"
        stages: list[str] = []
        txt_response = Mock()
        txt_response.json.return_value = {
            "images": [base64.b64encode(source).decode("ascii")],
            "info": {"seed": 42},
        }
        upscale_response = Mock()
        upscale_response.json.return_value = {
            "image": base64.b64encode(enlarged).decode("ascii")
        }
        post.side_effect = [txt_response, upscale_response]

        result = ForgeApiClient().generate_image(
            Txt2ImgPayload("subject", "negative", 1216, 832),
            high_res=True,
            super_res=True,
            on_stage=stages.append,
        )

        self.assertEqual(result.image_bytes, enlarged)
        self.assertEqual(result.seed, 42)
        self.assertEqual(post.call_args_list[0].args[0], f"{ForgeApiClient().base_url}/sdapi/v1/txt2img")
        generation_body = post.call_args_list[0].kwargs["json"]
        self.assertTrue(generation_body["enable_hr"])
        self.assertEqual(generation_body["hr_scale"], 1.25)
        self.assertEqual(post.call_args_list[1].args[0], f"{ForgeApiClient().base_url}/sdapi/v1/extra-single-image")
        upscale_body = post.call_args_list[1].kwargs["json"]
        self.assertEqual(upscale_body["upscaling_resize_w"], 2432)
        self.assertEqual(upscale_body["upscaling_resize_h"], 1664)
        self.assertEqual(upscale_body["upscaler_1"], "R-ESRGAN 4x+")
        self.assertEqual(stages, ["decoding", "upscaling"])

    @patch("backend.forge.requests.post")
    def test_high_res_uses_modest_refinement_then_1_5x_upscale(self, post: Mock) -> None:
        stages: list[str] = []
        response = Mock()
        response.json.return_value = {
            "images": [base64.b64encode(b"source").decode("ascii")], "info": {}
        }
        post.return_value = response

        result = ForgeApiClient().generate_image(
            Txt2ImgPayload(
                "subject", "negative", 1216, 832,
                refine_scale=1.5, high_output_scale=1.5,
            ),
            high_res=True,
            on_stage=stages.append,
        )

        generation_body = post.call_args_list[0].kwargs["json"]
        self.assertTrue(generation_body["enable_hr"])
        self.assertEqual(generation_body["hr_scale"], 1.5)
        self.assertEqual(generation_body["hr_second_pass_steps"], 8)
        self.assertEqual(generation_body["hr_cfg"], 6.0)
        self.assertEqual(generation_body["hr_upscaler"], "Lanczos")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(result.image_bytes, b"source")
        self.assertEqual(stages, ["decoding"])

    @patch("backend.forge.requests.post")
    def test_character_refinement_restores_before_nmkd_upscale(self, post: Mock) -> None:
        response = Mock()
        response.json.return_value = {
            "image": base64.b64encode(b"restored-and-upscaled").decode("ascii")
        }
        post.return_value = response

        result = ForgeApiClient().restore_and_upscale_image_bytes(
            b"identity-locked",
            target_width=1248,
            target_height=1824,
            upscaler="4x_NMKD-Siax_200k",
        )

        self.assertEqual(result, b"restored-and-upscaled")
        request = post.call_args.kwargs["json"]
        self.assertEqual(request["codeformer_visibility"], 0.6)
        self.assertEqual(request["codeformer_weight"], 0.6)
        self.assertFalse(request["upscale_first"])
        self.assertEqual(request["upscaler_1"], "4x_NMKD-Siax_200k")
        self.assertEqual(request["upscaling_resize_w"], 1248)
        self.assertEqual(request["upscaling_resize_h"], 1824)


class StalledForgeDetectionTests(unittest.TestCase):
    """A reachable Forge can still be wedged on a job it already finished."""

    def _client_returning(self, payload):
        client = ForgeApiClient()
        response = Mock()
        response.json.return_value = payload
        response.raise_for_status.return_value = None
        client._request = Mock(return_value=response)  # noqa: SLF001 - test seam
        return client

    def test_finished_job_still_held_is_reported(self) -> None:
        client = self._client_returning(
            {"progress": 1.0, "state": {"job": "extras", "sampling_step": 0}}
        )
        self.assertEqual(client.stalled_job_name(), "extras")

    def test_idle_forge_reports_nothing(self) -> None:
        client = self._client_returning({"progress": 0.0, "state": {"job": ""}})
        self.assertIsNone(client.stalled_job_name())

    def test_a_job_actually_sampling_is_not_stalled(self) -> None:
        client = self._client_returning(
            {"progress": 1.0, "state": {"job": "txt2img", "sampling_step": 12}}
        )
        self.assertIsNone(client.stalled_job_name())


if __name__ == "__main__":
    unittest.main()

"""Only the three supported checkpoints are offered to the client."""

from __future__ import annotations

import unittest

from backend.main import _is_supported_model

SUPPORTED = (
    r"sd\realvisxlV50_v50Bakedvae.safetensors",
    r"sd\cyberrealisticPony_v180Coreshift.safetensors [9d0e340f5f]",
    r"sd\juggernautXL_ragnarok.safetensors [dd08fa32f9]",
    r"sd\ponyDiffusionV6XL_v6StartWithThisOne.safetensors [67ab2fd8ec]",
    r"sd\waiIllustriousSDXL_v170.safetensors",
)

UNSUPPORTED = (
    r"sd\juggernautXL_juggXILightningByRD.safetensors [609fde646e]",
    r"sd\realisticVisionV60B1_v51HyperVAE.safetensors [f47e942ad4]",
    r"sd\juggernautXL.safetensors",
)


class SupportedModelTests(unittest.TestCase):
    def test_the_tuned_checkpoints_are_offered(self) -> None:
        for title in SUPPORTED:
            with self.subTest(title=title):
                self.assertTrue(_is_supported_model(title))

    def test_untuned_checkpoints_are_hidden(self) -> None:
        for title in UNSUPPORTED:
            with self.subTest(title=title):
                self.assertFalse(_is_supported_model(title))

    def test_a_duplicate_copy_is_not_offered_twice(self) -> None:
        self.assertFalse(
            _is_supported_model(r"sd\cyberrealisticPony_v180Coreshift (1).safetensors")
        )

    def test_the_hash_suffix_does_not_affect_the_decision(self) -> None:
        bare = r"sd\juggernautXL_ragnarok.safetensors"
        self.assertTrue(_is_supported_model(bare))


if __name__ == "__main__":
    unittest.main()

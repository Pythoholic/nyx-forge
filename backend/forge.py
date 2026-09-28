"""Typed Forge API client and Pony/SDXL prompt construction."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import random
import re
import unicodedata
from dataclasses import asdict, dataclass, replace
from time import monotonic
from typing import Any, Callable

import requests

from .clip_tokenizer import CLIPBPETokenizer

LOGGER = logging.getLogger(__name__)

SDXL_PROMPT_TOKEN_LIMIT = 75
_SDXL_TOKENIZER = CLIPBPETokenizer()
SUPER_RES_UPSCALER = "R-ESRGAN 4x+"
# Trained for photographic skin and fabric, where R-ESRGAN tends toward plastic.
DETAIL_REFINE_UPSCALER = "4x_NMKD-Siax_200k"
# High enough to add texture, low enough not to redraw faces.
DETAIL_REFINE_DENOISE = 0.30
# Diffusing beyond a 2x step costs far more time than the detail it returns.
REFINE_MAX_STEP = 2.0
FACEID_LORA_TAG = "<lora:ip-adapter-faceid-plusv2_sdxl_lora:0.7>"
FACEID_CONTROLNET_MODEL = "ip-adapter-faceid-plusv2_sdxl [187cb962]"
INSTANTID_ADAPTER_PREFIX = "ip-adapter_instant_id_sdxl"
INSTANTID_CONTROLNET_PREFIX = "control_instant_id_sdxl"
CHARACTER_FRAMING_POSITIVE = (
    "waist-up environmental portrait, entire head visible, face clearly visible, "
    "subject fills the frame"
)
CHARACTER_FRAMING_NEGATIVE = (
    "cropped head, headless, face out of frame, distant subject, tiny face"
)


def _probe_forge_api(base_url: str) -> bool:
    # sd-models rather than cmd-flags: some reForge builds return 500 from
    # cmd-flags while the rest of the API works, and probing it would reject a
    # perfectly healthy instance.
    try:
        response = requests.get(
            f"{base_url}/sdapi/v1/sd-models",
            timeout=(0.35, 2.0),
        )
        if not response.ok:
            return False
        return isinstance(response.json(), list)
    except (requests.RequestException, ValueError):
        return False


SDXL_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "Portrait (832x1216)": (832, 1216),
    "Landscape (1216x832)": (1216, 832),
    "Square (1024x1024)": (1024, 1024),
}
SD15_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "Portrait (512x768)": (512, 768),
    "Landscape (768x512)": (768, 512),
    "Square (512x512)": (512, 512),
}
FLUX_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "Portrait (832x1216)": (832, 1216),
    "Portrait (896x1152)": (896, 1152),
    "Portrait (768x1152)": (768, 1152),
}
ILLUSTRIOUS_ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "Portrait (1024x1344)": (1024, 1344),
    "Landscape (1344x1024)": (1344, 1024),
    "Square (1024x1024)": (1024, 1024),
}
# Backward-compatible export for callers/tests that mean SDXL-native ratios.
ASPECT_RATIOS = SDXL_ASPECT_RATIOS


@dataclass(frozen=True, slots=True)
class RatingRule:
    """Prompt tags associated with a content rating."""

    positive_tag: str
    negative_tag: str = ""
    negative_modifiers: str = ""
    positive_modifiers: str = ""


@dataclass(frozen=True, slots=True)
class RatingPromptProfile:
    """Rating-specific scene fragments used by the random prompt generator."""

    subjects: tuple[str, ...]
    wardrobe: tuple[str, ...]
    actions: tuple[str, ...]
    environments: tuple[str, ...]
    body_types: tuple[str, ...]
    # A rear request may use alternate wardrobe and body-description pools when
    # a profile needs orientation-specific phrasing.
    rear_wardrobe: tuple[str, ...] = ()
    rear_body_types: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StyleVariant:
    """One interchangeable rendering style for a drawn style preset."""

    name: str
    positive_modifiers: str
    style_exclusions: str


@dataclass(frozen=True, slots=True)
class StyleRule:
    """Hidden prompt tags associated with a style preset."""

    source_tag: str
    positive_modifiers: str
    style_exclusions: str
    variants: tuple[StyleVariant, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Capabilities and prompt family for a known checkpoint."""

    family: str
    styles: tuple[str, ...]
    default_style: str
    description: str
    display_name: str = "General SDXL"
    category_label: str = "General-purpose SDXL"
    resolution_family: str = "sdxl"
    sampler_name: str = "DPM++ 2M SDE"
    scheduler: str = "Karras"
    steps: int = 28
    cfg_scale: float = 6.0
    distilled_cfg_scale: float | None = None
    # Lowest CFG that still lets the prompt steer this checkpoint. Below it the
    # positive prompt barely conditions the image and the negative prompt goes
    # inert. Lightning/Hyper models are distilled for very low CFG, so this is
    # per-profile rather than one global floor.
    min_cfg_scale: float = 3.0
    clip_skip: int = 1
    prompt_token_budget: int = 100
    supports_hires_fix: bool = True
    # The reForge instance that serves this checkpoint, when it is not the
    # shared default. FLUX needs its own: its UNet, encoders and VAE are set
    # instance-wide and conflict with the SDXL checkpoints.
    forge_backend_id: str | None = None
    positive_word_budget: int = 60
    negative_word_budget: int = 42
    refine_steps: int = 8
    refine_denoising_strength: float = 0.34
    refine_scale: float = 1.25
    # Landscape canvases already have more raw pixels than portrait/square at
    # the same refine_scale, so the same factor costs meaningfully more VRAM.
    # None means "use refine_scale/refine_steps for every orientation".
    landscape_refine_scale: float | None = None
    landscape_refine_steps: int | None = None
    high_output_scale: float = 1.5
    super_output_scale: float = 2.0
    # Detect-and-inpaint repair pass for hands and faces, which are too few
    # pixels in a full-body base render for the model to place correctly.
    # Measured: a 1024 inpaint canvas is the lever that adds detail; raising
    # denoise instead makes the pass invent a distorted hand.
    detail_pass_models: tuple[str, ...] = ("hand_yolov8n.pt", "face_yolov8n.pt")
    # The IPAdapter FaceID adapter is trained for SDXL. Pony XL and SD 1.5 are
    # different architectures, so the adapter cannot condition them: ControlNet
    # contributes nothing and the face is regenerated freely. Silent, because
    # the request still returns an image.
    supports_face_identity: bool = True
    refine_upscaler: str = "R-ESRGAN 4x+"
    output_upscaler: str = SUPER_RES_UPSCALER
    # Some checkpoint creators explicitly recommend starting from a minimal
    # negative prompt (e.g. Juggernaut Ragnarok) - a fixed "lowres, blurry,
    # watermark..." quality floor applied to every model regardless of that
    # guidance works against models tuned to need little or no negative
    # prompt. False drops that quality floor for this model; anatomy/hand/eye
    # protection (NEGATIVE_ANATOMY_GUIDANCE) is never dropped, since a
    # shorter negative prompt should not mean weaker anatomy protection.
    include_negative_quality_floor: bool = True
    # A dedicated negative Textual Inversion/embedding some checkpoint
    # creators publish alongside the model (e.g. CyberRealistic Pony's own
    # negative embedding) - appended to the negative prompt when present.
    negative_embedding: str = ""
    standard_quality_summary: str = "Native model resolution - fastest"
    high_quality_summary: str = "Conservative detail pass + tiled 1.5x output"
    super_quality_summary: str = "Same detail pass + tiled 2x output"
    settings_summary: str = "28 steps - CFG 6 - DPM++ 2M SDE / Karras"


RATING_RULES = {
    "Safe": RatingRule(
        "rating_safe",
        "rating_unsafe",
        "nsfw, nude, naked, nipples, genitals",
        "safe, fully clothed, non-sexual scene",
    ),
}

STYLE_RULES = {
    "Photoreal": StyleRule(
        "source_photo",
        "RAW photograph, photorealistic, natural skin texture, realistic eyes, detailed hands, well-defined fingers, physically accurate lighting",
        "source_anime, source_cartoon, source_pony, source_furry, drawing, illustration, 3d render, artificial",
    ),
    "Anime": StyleRule(
        "source_anime",
        "",
        "",
        variants=(
            StyleVariant(
                "cel",
                "anime screenshot, cel shading, flat color, simple shading",
                r"watercolor \(medium\), traditional media, sketch, monochrome, 3d",
            ),
            StyleVariant(
                "retro90s",
                r"1990s \(style\), retro artstyle, oldschool, anime screenshot, cel shading",
                r"watercolor \(medium\), painting \(medium\), 3d, glossy",
            ),
            StyleVariant(
                "manga",
                "monochrome, greyscale, manga, lineart, screentones, hatching",
                r"colorized, flat color, cel shading, watercolor \(medium\), 3d",
            ),
            StyleVariant(
                "watercolor",
                r"watercolor \(medium\), traditional media, painting \(medium\)",
                "cel shading, flat color, anime screenshot, lineart, monochrome, 3d",
            ),
            StyleVariant(
                "graphite",
                r"graphite \(medium\), traditional media, sketch, crosshatching, monochrome",
                r"cel shading, flat color, watercolor \(medium\), colorized, 3d",
            ),
        ),
    ),
    "Illustration": StyleRule(
        "source_cartoon",
        "professional digital illustration, intricate details, refined color palette",
        "source_photo, source_anime, low-detail sketch, 3d render",
    ),
    "Cinematic": StyleRule(
        "source_photo",
        "realistic, cinematic still, filmic color grading, detailed hands, well-defined fingers, highly detailed",
        "source_anime, source_cartoon, flat lighting, 3d render",
    ),
    "Sketch": StyleRule(
        "source_cartoon",
        "expressive hand-drawn sketch, confident linework, textured paper",
        "source_photo, source_anime, photorealistic, glossy 3d render",
    ),
    "Editorial": StyleRule(
        "source_photo",
        "editorial fashion photography, natural skin texture, detailed hands, well-defined fingers, refined studio lighting",
        "source_anime, source_cartoon, source_pony, source_furry, drawing, illustration, 3d render",
    ),
    "Cartoon": StyleRule(
        "source_cartoon",
        "polished cartoon artwork, expressive shapes, clean color design",
        "source_photo, source_anime, photorealistic, 3d render",
    ),
    "Fantasy": StyleRule(
        "",
        "epic fantasy scene, atmospheric worldbuilding, dramatic light, intricate materials",
        "flat lighting, generic background, low detail",
    ),
    "Concept art": StyleRule(
        "",
        "professional concept art, production design, strong silhouette, detailed environment",
        "unfinished sketch, flat composition, low detail",
    ),
}

MODEL_PROFILES: dict[str, ModelProfile] = {
    "realvisxlv50v50bakedvae": ModelProfile(
        family="sdxl_realvis",
        styles=("Photoreal", "Cinematic", "Editorial"),
        default_style="Photoreal",
        description="Natural SDXL photography with realistic skin, faces, and everyday scenes",
        display_name="RealVisXL V5.0",
        category_label="Natural photoreal · SDXL",
        sampler_name="DPM++ SDE",
        scheduler="Karras",
        steps=32,
        cfg_scale=5.5,
        min_cfg_scale=3.0,
        prompt_token_budget=75,
        positive_word_budget=68,
        negative_word_budget=70,
        refine_steps=25,
        refine_denoising_strength=0.25,
        refine_scale=1.5,
        refine_upscaler="4x_NMKD-Siax_200k",
        output_upscaler="4x_NMKD-Siax_200k",
        # RealVis already resolves faces and skin during its creator-recommended
        # Hires pass. A generic detector-driven inpaint redraw made faces more
        # uniform and skin smoother than the checkpoint's native result.
        detail_pass_models=(),
        standard_quality_summary="832x1216 native - 32 steps - natural photographic baseline",
        high_quality_summary="1.5x NMKD-Siax detail pass - 25 steps - tiled output",
        super_quality_summary="1.5x NMKD-Siax detail pass - tiled 2x output",
        settings_summary="32 steps - CFG 5.5 - DPM++ SDE / Karras - baked VAE",
    ),
    "waiillustrioussdxlv170": ModelProfile(
        family="illustrious",
        supports_face_identity=False,
        styles=("Anime", "Illustration", "Cartoon"),
        default_style="Anime",
        description="WAI Illustrious SDXL v17 for high-detail anime and illustration",
        display_name="WAI Illustrious SDXL v17",
        category_label="Anime & illustration - Illustrious XL",
        resolution_family="illustrious",
        sampler_name="Euler a",
        scheduler="Automatic",
        # Creator guidance allows 15-30. Twenty retained the recommended
        # quality range while avoiding ~25% unnecessary latency on the 10 GB
        # validation host at native 1024x1344.
        steps=20,
        cfg_scale=6.0,
        prompt_token_budget=100,
        positive_word_budget=75,
        negative_word_budget=55,
        refine_steps=20,
        refine_denoising_strength=0.40,
        refine_scale=1.5,
        landscape_refine_scale=1.5,
        landscape_refine_steps=20,
        refine_upscaler="R-ESRGAN 4x+ Anime6B",
        output_upscaler="R-ESRGAN 4x+ Anime6B",
        detail_pass_models=(),
        # This installation's Forge Neo runner is reserved for FLUX and keeps
        # a 16-channel autoencoder loaded instance-wide. WAI is SDXL (4 latent
        # channels), so it belongs on the standard reForge SDXL runner.
        forge_backend_id=None,
        standard_quality_summary="1024x1344 native - 20 steps - recommended baseline",
        high_quality_summary="1.5x Anime6B detail pass - 20 steps - tiled output",
        super_quality_summary="1.5x Anime6B detail pass - tiled 2x output",
        settings_summary="20 steps - CFG 6 - Euler a - integrated VAE",
    ),
    # Candidate replacement for CyberRealistic Pony on lower-anatomy work,
    # which that checkpoint failed in all 25 diagnostic images. Its release
    # notes list anatomy and full-body proportion fixes. Settings follow the
    # model card: DPM++ 2M Karras, 25-30 steps, CFG 4-7.
    "realismbystableyogiponyv65": ModelProfile(
        family="pony_realism",
        supports_face_identity=False,
        styles=("Photoreal", "Cinematic", "Editorial"),
        default_style="Photoreal",
        description="Pony realism checkpoint with anatomy and proportion fixes",
        display_name="Realism by Stable Yogi Pony 6.5",
        category_label="Safe photoreal - Pony XL",
        sampler_name="DPM++ 2M",
        scheduler="Karras",
        steps=28,
        cfg_scale=5.0,
        clip_skip=2,
        prompt_token_budget=90,
        positive_word_budget=70,
        negative_word_budget=80,
        refine_steps=15,
        refine_denoising_strength=0.30,
        refine_scale=1.5,
        refine_upscaler="4x_NMKD-Siax_200k",
        high_output_scale=1.5,
        super_output_scale=2.0,
        output_upscaler="4x_NMKD-Siax_200k",
        standard_quality_summary="832x1216 native - 28 steps - fastest and lowest VRAM",
        high_quality_summary="1.5x NMKD-Siax detail pass - 15 steps - tiled 1.5x output",
        super_quality_summary="1.5x NMKD-Siax detail pass - tiled 2x output",
        settings_summary="28 steps - CFG 5 - DPM++ 2M / Karras - Clip Skip 2",
    ),
    "flux1dev": ModelProfile(
        family="flux",
        supports_face_identity=False,
        styles=("Photoreal", "Cinematic", "Editorial"),
        default_style="Photoreal",
        description="Quantized FLUX.1 Dev for detailed natural-language image generation",
        display_name="FLUX.1 Dev (Q5_K_S)",
        category_label="Quality FLUX",
        resolution_family="flux",
        sampler_name="Euler",
        scheduler="Simple",
        steps=20,
        cfg_scale=1.0,
        distilled_cfg_scale=3.5,
        min_cfg_scale=1.0,
        # FLUX reads long natural-language prose through T5, so the positive
        # budget is generous. At CFG 1.0 the negative prompt has no effect at
        # all, so its budget only has to be wide enough not to reject the
        # required fragments the shared builder always emits.
        positive_word_budget=75,
        negative_word_budget=80,
        # The FLUX instance only ships Lanczos, Nearest and ESRGAN; the
        # R-ESRGAN and NMKD models the other profiles use are installed in a
        # different reForge directory and 500 here.
        refine_upscaler="Lanczos",
        output_upscaler="Lanczos",
        supports_hires_fix=False,
        # The FLUX instance has no ADetailer extension, and sending an
        # alwayson_scripts entry for a script Forge does not have is a hard
        # 422. FLUX's own hand and face coherence is what the detail pass
        # existed to patch on Pony anyway.
        detail_pass_models=(),
        forge_backend_id="forge_neo",
        standard_quality_summary="Validated native portrait resolution - 20 steps - lowest VRAM",
        settings_summary="20 steps - CFG 1 - Distilled CFG 3.5 - Euler / Simple",
    ),
    "cyberrealisticpony": ModelProfile(
        family="pony_realism",
        supports_face_identity=False,
        styles=("Photoreal", "Cinematic", "Editorial"),
        default_style="Photoreal",
        description="Recommended for Safe photorealism, expressive poses, and cinematic photography",
        display_name="CyberRealistic Pony 18 CoreShift",
        category_label="Recommended Safe photoreal · Pony XL",
        sampler_name="DPM++ SDE",
        scheduler="Karras",
        steps=30,
        cfg_scale=5.0,
        clip_skip=2,
        prompt_token_budget=90,
        positive_word_budget=70,
        negative_word_budget=80,
        refine_steps=15,
        refine_denoising_strength=0.30,
        refine_scale=1.5,
        refine_upscaler="4x_NMKD-Siax_200k",
        high_output_scale=1.5,
        super_output_scale=2.0,
        output_upscaler="4x_NMKD-Siax_200k",
        negative_embedding="embedding:CyberRealistic_Negative_Pony",
        standard_quality_summary="832×1216 native · 30 steps · fastest and lowest VRAM",
        high_quality_summary="1.5× NMKD-Siax detail pass · 15 steps · tiled 1.5× output",
        super_quality_summary="1.5× NMKD-Siax detail pass · tiled 2× output · same diffusion peak as High",
        settings_summary="30 steps - CFG 5 - DPM++ SDE / Karras - Clip Skip 2",
    ),
    "juggernautxljuggxilightning": ModelProfile(
        family="sdxl_lightning",
        styles=("Photoreal", "Cinematic", "Fantasy", "Concept art"),
        default_style="Cinematic",
        description="Fast Juggernaut SDXL Lightning: low-step cinematic photorealism",
        display_name="Juggernaut XI Lightning",
        category_label="Fast SDXL",
        sampler_name="DPM++ SDE",
        scheduler="Karras",
        steps=6,
        cfg_scale=1.75,
        min_cfg_scale=1.0,
        prompt_token_budget=75,
        positive_word_budget=52,
        negative_word_budget=80,
        refine_steps=5,
        refine_denoising_strength=0.30,
        settings_summary="6 steps - CFG 1.75 - DPM++ SDE / Karras",
    ),
    "juggernautxlragnarok": ModelProfile(
        family="sdxl",
        styles=("Photoreal", "Cinematic", "Editorial", "Fantasy", "Concept art"),
        # Photoreal, not Cinematic: the Cinematic modifiers ("8k uhd", "filmic
        # color grading") push this checkpoint toward a stylized look, while
        # its strength is unstyled photographic realism.
        default_style="Photoreal",
        description="Juggernaut Ragnarok: maximum-detail SDXL realism and prompt control",
        display_name="Juggernaut Ragnarok",
        category_label="Quality SDXL",
        sampler_name="DPM++ 2M SDE",
        scheduler="Karras",
        steps=35,
        # 4.5 measured against 4.0 and 5.0 on a fixed prompt+seed: 4.0 softens
        # hands, 5.0 sharpens background but mottles skin.
        cfg_scale=4.5,
        min_cfg_scale=3.0,
        prompt_token_budget=75,
        # 60 starved the scene: the environment was trimmed from 119 of 120
        # prompts, leaving the subject with nothing to stand in and the model
        # inventing furniture to fuse her into. Verified at 68 on a fixed
        # prompt+seed; SDXL chunks past 75 tokens rather than truncating, so
        # the longer prompt degrades gracefully.
        positive_word_budget=68,
        negative_word_budget=80,
        # Creator's HiRes recipe for Ragnarok: 4xNMKD-Siax_200k, 15 steps,
        # 0.3 denoise, 1.5x upscale.
        refine_steps=15,
        refine_denoising_strength=0.30,
        refine_scale=1.5,
        refine_upscaler="4x_NMKD-Siax_200k",
        output_upscaler="4x_NMKD-Siax_200k",
        # The creator's own guidance is to start with a minimal negative
        # prompt - the generic quality floor applied here the same as every
        # other SDXL model works against that tuning. Anatomy/hand/eye
        # protection and rating-safety exclusions still apply regardless.
        include_negative_quality_floor=False,
        standard_quality_summary="832×1216 native · 35 steps · fastest and lowest VRAM",
        high_quality_summary="1.5× NMKD-Siax detail pass · 15 steps · tiled 1.5× output",
        super_quality_summary="1.5× NMKD-Siax detail pass · tiled 2× output · same diffusion peak as High",
        settings_summary="35 steps - CFG 4.5 - DPM++ 2M SDE / Karras",
    ),
    "juggernautxl": ModelProfile(
        family="sdxl",
        styles=("Photoreal", "Cinematic", "Fantasy", "Concept art"),
        default_style="Cinematic",
        description="Cinematic SDXL photorealism, fantasy scenes, and production concept art",
        display_name="Juggernaut XL",
        category_label="Quality SDXL",
        steps=35,
        cfg_scale=4.5,
        prompt_token_budget=75,
        positive_word_budget=60,
        negative_word_budget=80,
        refine_steps=13,
        refine_denoising_strength=0.38,
        settings_summary="35 steps - CFG 4.5 - DPM++ 2M SDE / Karras",
    ),
    "ponydiffusionv6xl": ModelProfile(
        family="pony",
        supports_face_identity=False,
        styles=("Anime", "Illustration", "Cartoon"),
        default_style="Anime",
        description="Tag-driven anime, cartoon, furry, and pony illustration",
        display_name="Pony Diffusion V6 XL",
        category_label="Anime & illustration",
        sampler_name="Euler a",
        scheduler="Automatic",
        steps=25,
        cfg_scale=7.0,
        clip_skip=2,
        prompt_token_budget=100,
        positive_word_budget=75,
        negative_word_budget=80,
        refine_steps=16,
        refine_denoising_strength=0.34,
        refine_scale=1.7,
        landscape_refine_scale=1.5,
        landscape_refine_steps=14,
        refine_upscaler="R-ESRGAN 4x+ Anime6B",
        # YOLO hand/face models are trained on photographs; running them over
        # illustration pulls those regions toward realism and off-style.
        detail_pass_models=(),
        high_quality_summary="1.7× Anime6B detail pass (1.5× on Landscape) · tiled 1.5× output",
        super_quality_summary="1.7× Anime6B detail pass (1.5× on Landscape) · tiled 2× output · same diffusion peak as High",
        settings_summary="25 steps - CFG 7 - Euler a - Clip Skip 2",
    ),
    "realisticvisionv60b1v51hypervae": ModelProfile(
        family="sd15_hyper",
        styles=("Photoreal", "Cinematic", "Editorial"),
        default_style="Photoreal",
        description="Fast SD 1.5 Hyper realism: efficient low-step photographic generation",
        display_name="Realistic Vision V6 Hyper",
        category_label="Fast SD 1.5",
        resolution_family="sd15",
        supports_face_identity=False,
        sampler_name="DPM++ SDE",
        scheduler="Karras",
        steps=6,
        cfg_scale=2.0,
        min_cfg_scale=1.0,
        clip_skip=2,
        prompt_token_budget=75,
        # Subject, wardrobe, orientation and scene are all required now; 55
        # left no room for them beside the deterministic tail.
        positive_word_budget=62,
        negative_word_budget=80,
        refine_steps=5,
        refine_denoising_strength=0.30,
        settings_summary="6 steps - CFG 2 - DPM++ SDE / Karras - Clip Skip 2",
    ),
}

FALLBACK_MODEL_PROFILE = ModelProfile(
    family="sdxl",
    styles=("Photoreal", "Cinematic", "Illustration"),
    default_style="Photoreal",
    description="General SDXL checkpoint",
)


def model_profile_for(model: str) -> ModelProfile:
    """Infer a checkpoint capability profile from its Forge title."""
    normalized = model.lower().replace("_", "").replace("-", "").replace(" ", "")
    for marker, profile in MODEL_PROFILES.items():
        if marker in normalized:
            return profile
    return FALLBACK_MODEL_PROFILE


def aspect_ratios_for(model: str) -> dict[str, tuple[int, int]]:
    """Return native, VRAM-conscious dimensions for a checkpoint family."""
    profile = model_profile_for(model)
    if profile.resolution_family == "sd15":
        return SD15_ASPECT_RATIOS
    if profile.resolution_family == "flux":
        return FLUX_ASPECT_RATIOS
    if profile.resolution_family == "illustrious":
        return ILLUSTRIOUS_ASPECT_RATIOS
    return SDXL_ASPECT_RATIOS


def nearest_native_resolution(model: str, width: int, height: int) -> tuple[int, int]:
    """Return this model's native size whose aspect ratio best matches width/height.

    The detail-pass upscale re-renders from a source image's own dimensions,
    which can already be far larger than any native size (e.g. after a prior
    pixel upscale). Re-running base diffusion at that size - then applying
    the same 1.5x/2x tiers on top - compounds resolution every time instead
    of adding detail, and pushes VRAM into a tiled-VAE fallback that is a
    known source of anatomy/structure artifacts. Anchoring back to native
    keeps every detail-pass run the same bounded cost regardless of how
    large the source has already grown, while preserving orientation.
    """
    options = aspect_ratios_for(model)
    target_ratio = width / height if height else 1.0
    return min(
        options.values(),
        key=lambda size: abs((size[0] / size[1]) - target_ratio),
    )


POSITIVE_QUALITY_PREFIX = "score_9, score_8_up, score_7_up, score_6_up"
NEGATIVE_QUALITY_PREFIX = "score_6, score_5, score_4"
LEGACY_NEGATIVE_QUALITY_PREFIX = "score_4, score_5, score_6"
FEMALE_ONLY_POSITIVE = "1woman, solo, clearly adult woman, female focus"
MALE_EXCLUSION_CORE = "man, men, male, boy, boys, 1boy"
MALE_EXCLUSION_NEGATIVE = f"{MALE_EXCLUSION_CORE}, male focus, mixed-gender group"
MALE_SUBJECT_PATTERN = re.compile(
    r"(?<![a-z])(?:\d+)?(?:man|men|males?|boys?|gentlemen|gentleman|husbands?|"
    r"boyfriends?|fathers?|brothers?|sons?|uncles?|nephews?|grooms?|princes?|kings?)\b",
    flags=re.IGNORECASE,
)
POLICY_CONFUSABLES = str.maketrans(
    {
        "а": "a",  # Cyrillic lookalikes used to evade simple ASCII matching.
        "е": "e",
        "о": "o",
        "с": "c",
        "р": "p",
        "х": "x",
        "у": "y",
        "к": "k",
        "м": "m",
        "т": "t",
        "α": "a",
        "ε": "e",
        "ο": "o",
    }
)
# Split so a model can opt out of the generic quality floor (some checkpoint
# creators recommend starting from few or no negative tags) while every model
# still keeps anatomy/hand/eye protection regardless of that preference.
NEGATIVE_QUALITY_FLOOR = "lowres, blurry, watermark, text, signature"
NEGATIVE_ANATOMY_GUIDANCE = (
    "bad anatomy, bad hands, deformed eyes, asymmetrical eyes, "
    "extra fingers, extra limbs"
)
# Counterweight to the model's default centred, front-facing composition.
NEGATIVE_POSE_REPETITION = (
    "front-facing passport pose, symmetrical stance, rigid posture, static pose"
)
# Only applied when the user asked for a rear view; these would fight a
# front-facing request.
NEGATIVE_REAR_ORIENTATION = "front view, facing viewer, looking at viewer"
NEGATIVE_ANATOMY_BASE = (
    f"{NEGATIVE_QUALITY_FLOOR}, {NEGATIVE_ANATOMY_GUIDANCE}, "
    f"{NEGATIVE_POSE_REPETITION}"
)

RATING_PROMPT_PROFILES = {
    "Safe": RatingPromptProfile(
        subjects=(
            "28-year-old adult woman",
            "42-year-old adult lady, composed expression",
            "35-year-old adult woman, elegant features",
            "24-year-old adult woman, curious expression",
            "33-year-old adult woman, focused expression",
            "38-year-old adult lady, relaxed confidence",
            "46-year-old mature woman",
            "52-year-old mature lady",
        ),
        body_types=(
            "athletic build with balanced proportions",
            "softly rounded build with a natural stance",
            "hourglass build with balanced proportions",
            "petite build with gentle curves",
            "tall, statuesque build with long lines",
            "full-figured build with natural proportions",
            "toned build with strong posture",
            "broad-shouldered build with a grounded stance",
            "lean build with relaxed posture",
            "pear-shaped build with balanced proportions",
        ),
        wardrobe=(
            "wearing a tailored tactical coat and leather gloves",
            "wearing ornate silver armor with engraved details",
            "wearing a modern streetwear outfit with layered fabrics",
            "wearing an elegant evening dress with subtle embroidery",
            "wearing a structured linen suit with sculptural tailoring",
            "wearing a heritage trench coat over a fine knit dress",
            "wearing an embroidered utility jumpsuit with rolled sleeves",
            "wearing a wool turtleneck and flowing pleated skirt",
        ),
        actions=(
            "standing at the center of the scene",
            "walking toward the camera",
            "looking over one shoulder",
            "resting one hand on a railing",
            "examining a delicate mechanical object",
            "adjusting the cuff of her tailored jacket",
            "pausing mid-step beneath an architectural arch",
            "reading a handwritten note beside a window",
        ),
        environments=(
            "neon-lit Tokyo street after rain, reflective pavement",
            "grand marble hall with towering windows",
            "misty pine forest at dawn",
            "minimalist studio with a textured charcoal backdrop",
            "glass conservatory filled with silver-green foliage",
            "coastal railway platform beneath a stormy sky",
            "quiet contemporary gallery before opening hours",
            "mountaintop observatory with panoramic windows",
        ),
    ),
}
# Pose is split across four axes rather than baked into the subject line.
# Combining them multiplies variety without spending more words than the old
# "full body shot" phrasing did.
#
# Two thirds of BODY_ORIENTATIONS turn away from or across the camera. The
# previous pools forced a forward gaze twice per prompt - once in the subject
# and again in the action - so every generation faced the lens.
ORIENTATION_GROUPS = {
    "front": (
        "body squarely facing lens",
        "frontal stance, weight asymmetrical",
        "direct frontal view, chin lifted",
        "open stance toward viewer",
        "frontal diagonal standing stance",
        "frontal pose, hips offset",
        "torso square to lens",
        "direct frontal kneeling view",
        "straight-on full standing view",
        "direct frontal low crouch",
        "front-facing asymmetrical seated pose",
        "frontal recline, eyes forward",
        "chest square, hips angled",
        "symmetrical full frontal silhouette",
        "direct view, shoulders level",
        "frontal lunge toward lens",
        "straight-on grounded low stance",
        "full frontal body line",
        "frontal twist, face centered",
        "forward-facing classical contrapposto stance",
    ),
    "side": (
        "clean full left profile",
        "clean full right profile",
        "strict full lateral silhouette",
        "body aligned across frame",
        "lateral stance, face averted",
        "clean profile kneeling silhouette",
        "sideways torso, gaze distant",
        "lateral seated body line",
        "profile crouch, spine curved",
        "strict profile reclining pose",
        "side-on walking body silhouette",
        "lateral lunge across frame",
        "profile deep backbend silhouette",
        "sideways stance, shoulders stacked",
        "lateral twist, face obscured",
        "profile squat, heels grounded",
        "side-on lean beneath light",
        "lateral reach, body elongated",
        "profile arch, chin elevated",
        "sideways stride, gaze forward",
    ),
    "back": (
        "rearward diagonal body view",
        "back squarely toward lens",
        "direct rear view, head averted",
        "walking away, face unseen",
        "backward-facing upright kneeling pose",
        "rearward crouch, spine arched",
        "turned away, shoulder glance",
        "back presented, hips offset",
        "rearward asymmetrical seated silhouette",
        "away-facing classical contrapposto stance",
        "direct full back silhouette",
        "rearward lunge, torso lowered",
        "back toward viewer, head bowed",
        "retreating stride, shoulders turned",
        "rearward recline, face obscured",
        "turned-away standing body arch",
        "backward-facing grounded deep squat",
        "rear silhouette, arms raised",
        "departing stance, gaze hidden",
        "away-facing elongated body line",
    ),
}
ORIENTATION_CHOICES = ("mixed", "front", "side", "back")


def orientation_pool(preference: str) -> tuple[str, ...]:
    """Return the orientations a preference allows, mixed meaning all of them."""
    if preference in ORIENTATION_GROUPS:
        return ORIENTATION_GROUPS[preference]
    return tuple(item for group in ORIENTATION_GROUPS.values() for item in group)

POSE_ACTIONS = (
    "contrapposto, weight on one hip",
    "arms raised, deep back arch",
    "one hand tangled in hair",
    "leaning backward against rough stone",
    "mid-stride, hips swinging naturally",
    "kneeling upright, thighs apart, chin raised",
    "reclining sideways, torso twisted toward light",
    "seated backward on chair, spine elongated",
    "crouching low, one knee extended sideways",
    "bending at waist, hands braced forward",
    "standing tiptoe, reaching toward overhead light",
    "one leg lifted onto low ledge",
    "sitting on heels, shoulders rolled back",
    "lying supine, knees raised, spine curved",
    "lying prone, feet lifted behind",
    "balancing sideways, arms forming diagonal line",
    "twisting mid-turn, hair and fabric airborne",
    "descending stairs, torso counter-rotated",
    "climbing ladder, body stretched vertically",
    "arching over curved metal rail",
    "squatting deeply, elbows resting on knees",
    "half-kneeling, front leg planted strongly",
    "seated high, legs dangling asymmetrically",
    "folded forward, hair cascading downward",
    "standing wide-legged, hands behind neck",
    "pivoting on one foot, shoulders trailing",
    "leaning sideways beneath a low arch",
    "reaching backward, chest lifted upward",
    "one knee raised, forearms loosely crossed",
    "walking through shallow water, splashes frozen",
    "pressing palms against translucent wall",
    "resting forearms on high railing",
    "curled sideways, knees drawn toward chest",
    "shoulder stand, legs angled overhead",
    "bridge pose across a narrow platform",
    "side lunge, torso stretched over thigh",
    "standing split, foot braced overhead",
    "kneeling lunge, back leg extended",
    "hanging from low bar, torso elongated",
    "floating supine, limbs drifting apart",
    "emerging from water, hands sweeping hair",
    "stepping through curtain, leading with shoulder",
    "ducking beneath beam, spine curved gracefully",
    "seated on floor, one leg folded",
    "cross-legged, torso leaning far forward",
    "back against column, pelvis shifted outward",
    "wrapping both arms around narrow pillar",
    "stretching horizontally across two surfaces",
    "turning from sudden gust, balance disrupted",
    "walking uphill, body pitched into wind",
    "kneeling over reflective surface, fingers grazing water",
    "reaching down, knees bent unevenly",
    "perched on narrow edge, toes pointed",
    "rolling one shoulder, hips counterbalanced",
    "standing beneath cascade, head thrown back",
    "lifting translucent sheet into the wind",
    "crawling slowly across mirrored floor",
    "leaning over railing, one heel raised",
    "spinning sharply, arms opening outward",
    "folding into deep asymmetrical backbend",
)

CAMERA_ANGLES = (
    "eye-level candid editorial framing",
    "low-angle full-body editorial shot",
    "high-angle full-body editorial shot",
    "waist-height lateral camera position",
    "ground-level steep upward view",
    "ceiling-level steep downward view",
    "off-axis asymmetrical editorial framing",
    "wide immersive environmental portrait",
    "close balanced full-figure framing",
    "distant figure within architecture",
    "dramatic Dutch angle composition",
    "compressed long-lens telephoto perspective",
    "wide-angle controlled near-far distortion",
    "low floor reflection viewpoint",
    "shooting through rippled glass",
    "shooting through hanging chains",
    "layered foreground obstruction framing",
    "direct overhead geometric composition",
    "worm's-eye upward architectural view",
    "low lateral tracking angle",
    "high diagonal balcony angle",
    "tight off-center medium-full crop",
    "extreme wide establishing shot",
    "long-lens distant observational viewpoint",
    "mirror-split doubled figure composition",
    "waterline half-submerged camera viewpoint",
    "tilted horizon fashion framing",
    "centered architectural one-point perspective",
    "asymmetrical expansive negative-space composition",
    "silhouette against distant vista",
    "close foreground, distant body",
    "camera inside translucent enclosure",
    "view through circular aperture",
    "reflection-only indirect camera view",
    "shadow-first indirect figure composition",
    "panoramic wide lateral composition",
    "diagonal overhead crane shot",
    "low rear tracking shot",
    "high frontal plunge angle",
    "layered environmental deep-focus composition",
)

RANDOM_LIGHTING = (
    "hard cyan-red opposing crosslight",
    "single sodium-vapor industrial backlight",
    "moving searchlight beams, fog",
    "lightning flashes through smoke",
    "underwater caustics across skin",
    "projector stripes, deep shadows",
    "moonlight through perforated metal",
    "green aurora rim light",
    "furnace glow, cold ambient fill",
    "prismatic sunlight, rainbow fractures",
    "eclipse light, silver corona",
    "bioluminescent blue aquatic underlighting",
    "rotating distant lighthouse beam",
    "strobing magenta industrial lights",
    "candle grid, monumental shadows",
    "golden airborne dust shafts",
    "infrared-red monochrome scene lighting",
    "cold skylight, warm floor bounce",
    "mirror-ball fragments across body",
    "firelight reflected from black water",
    "ultraviolet glow, fluorescent pigments",
    "storm-green daylight, amber practicals",
    "razor-thin white rim light",
    "dappled light through moving water",
)
RANDOM_LENSES = (
    "35mm lens, environmental composition",
    "50mm lens, documentary composition",
    "85mm lens, shallow depth of field",
    "105mm lens, creamy background separation",
)

# Framing is selected before the lens so a landscape prompt does not randomly
# ask a portrait-oriented composition (and vice versa).
ASPECT_PROMPT_RULES: dict[str, tuple[tuple[str, ...], tuple[str, ...], str]] = {
    "Portrait (832x1216)": (
        (
            "vertical composition, full body framing, centered subject",
            "portrait orientation, head-to-toe composition, strong vertical lines",
            "vertical editorial framing, subject dominant in frame",
        ),
        (
            "50mm lens, natural perspective",
            "85mm portrait lens, shallow depth of field",
            "105mm lens, compressed background",
        ),
        "cropped feet, cut off head, awkward vertical framing",
    ),
    "Landscape (1216x832)": (
        (
            "wide establishing shot, environmental composition",
            "landscape orientation, subject placed using rule of thirds",
            "wide cinematic framing, layered foreground and background",
        ),
        (
            "24mm wide-angle lens, controlled perspective",
            "35mm lens, environmental composition",
            "50mm lens, wide scene coverage",
        ),
        "cramped composition, accidental portrait crop, empty side gutters",
    ),
    "Square (1024x1024)": (
        (
            "square composition, balanced visual weight",
            "symmetrical square framing, centered focal point",
            "balanced medium-wide composition, clean negative space",
        ),
        (
            "35mm lens, balanced perspective",
            "50mm lens, natural perspective",
            "85mm lens, controlled background separation",
        ),
        "awkward crop, unbalanced composition, excessive empty space",
    ),
}


def aspect_prompt_rules_for(
    aspect_ratio: str,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Resolve framing rules by orientation across SDXL and SD 1.5 sizes."""
    direct = ASPECT_PROMPT_RULES.get(aspect_ratio)
    if direct is not None:
        return direct
    orientation = aspect_ratio.partition(" ")[0]
    for label, rules in ASPECT_PROMPT_RULES.items():
        if label.startswith(f"{orientation} "):
            return rules
    raise ValueError(f"Unsupported aspect-ratio orientation: {aspect_ratio}.")


FAMILY_PROMPT_MODIFIERS = {
    "illustrious": (
        "clean silhouette, coherent anatomy, refined illustration detail",
        "muddy linework, broken silhouette, malformed anatomy",
    ),
    "flux": (
        "natural detail, coherent composition",
        "",
    ),
    "pony_realism": (
        "anatomically coherent pose, natural facial detail, realistic material response",
        "waxy skin, uncanny face",
    ),
    "pony": (
        "anatomically coherent pose, clean silhouette, polished character detail",
        "muddy linework, inconsistent character design, broken silhouette",
    ),
    "sdxl": (
        "anatomically coherent pose, coherent scene geometry, physically plausible materials",
        "incoherent geometry, implausible materials, accidental composition",
    ),
    "sdxl_lightning": (
        "anatomically coherent pose, clear subject, coherent composition",
        "incoherent geometry, oversaturated highlights, artificial texture",
    ),
    "sd15_hyper": (
        "anatomically coherent pose, RAW photo, natural light, photographic detail",
        "cgi, 3d, digital art, airbrushed skin, waxy texture",
    ),
}

FAMILY_ANATOMY_GUIDANCE = {
    "flux": "symmetrical eyes, anatomically coherent pose",
    "sdxl_realvis": "symmetrical eyes, anatomically coherent pose",
    "pony_realism": "symmetrical eyes, anatomically coherent pose",
    "pony": "symmetrical eyes, clear iris detail, anatomically coherent pose",
    "sdxl": "symmetrical eyes, anatomically coherent pose",
    "sdxl_lightning": "symmetrical eyes, anatomically coherent pose",
    "sd15_hyper": "symmetrical eyes, anatomically coherent pose",
    "illustrious": "symmetrical eyes, anatomically coherent pose",
}

HIGH_RES_PROMPT_MODIFIERS = {
    "flux": "crisp natural detail",
    "sdxl_realvis": "natural skin microtexture, crisp focal detail",
    "pony_realism": "crisp focal detail",
    "pony": "precise edges, refined texture",
    "sdxl": "controlled micro-contrast, crisp focal detail",
    "sdxl_lightning": "crisp focal detail",
    "sd15_hyper": "fine photographic detail, clean focal plane",
    "illustrious": "precise linework, refined color detail",
}
HIGH_RES_NEGATIVE_MODIFIERS = "edge halos, tiled texture, upscaling artifacts"
SDXL_STYLE_EXCLUSIONS = {
    "Photoreal": (
        "cgi, 3d render, illustration, anime, cartoon, airbrushed, waxy skin, "
        "plastic skin, overly smooth skin, beauty filter, doll-like face"
    ),
    "Cinematic": "cgi, 3d render, anime, cartoon, flat lighting, waxy skin, plastic skin",
    "Fantasy": "flat lighting, generic background, low detail, plastic materials",
    "Concept art": "unfinished sketch, flat composition, low detail, generic design",
    "Editorial": "cgi, 3d render, anime, cartoon, airbrushed skin, waxy skin, beauty filter",
    "Illustration": "photographic artifacts, unfinished sketch, flat composition, low detail",
}
RANDOM_NEGATIVE_FRAGMENTS = (
    "cropped subject",
    "duplicate objects",
    "distorted hands",
    "extra fingers",
    "missing fingers",
    "poorly drawn hands",
    "harsh shadows",
    "flat lighting",
    "oversaturated colors",
    "plastic texture",
    "cluttered background",
    "text and watermark",
    "fisheye distortion",
    "overexposed highlights",
)


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """User-selected settings used to build a Forge request."""

    width: int
    height: int
    model: str
    rating: str
    style: str
    high_res: bool
    super_res: bool = False
    # None means "use the checkpoint profile's tuned CFG". A user override
    # only ever replaces this one value; steps/sampler stay profile-driven.
    cfg_scale: float | None = None
    quality_mode: str = "normal"


@dataclass(frozen=True, slots=True)
class Txt2ImgPayload:
    """Strongly typed request body for Forge's txt2img endpoint."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    sampler_name: str = "DPM++ 2M SDE"
    scheduler: str = "Karras"
    steps: int = 28
    cfg_scale: float = 6.0
    distilled_cfg_scale: float | None = None
    seed: int = -1
    model_checkpoint: str | None = None
    clip_skip: int = 1
    refine_steps: int = 8
    refine_denoising_strength: float = 0.28
    refine_scale: float = 1.25
    high_output_scale: float = 1.5
    super_output_scale: float = 2.0
    refine_upscaler: str = "Lanczos"
    output_upscaler: str = SUPER_RES_UPSCALER
    detail_pass_models: tuple[str, ...] = ()
    controlnet_units: tuple[dict[str, Any], ...] = ()
    style_variant: str | None = None
    enforce_sdxl_token_budget: bool = False
    supports_hires_fix: bool = True

    def as_request(self, *, high_res: bool, super_res: bool = False) -> dict[str, Any]:
        """Serialize the payload and conditionally add high-res fields."""
        request_data: dict[str, Any] = asdict(self)
        if request_data["distilled_cfg_scale"] is None:
            request_data.pop("distilled_cfg_scale")
        model_checkpoint = request_data.pop("model_checkpoint")
        clip_skip = request_data.pop("clip_skip")
        refine_steps = request_data.pop("refine_steps")
        refine_denoising_strength = request_data.pop("refine_denoising_strength")
        refine_scale = request_data.pop("refine_scale")
        request_data.pop("high_output_scale")
        request_data.pop("super_output_scale")
        refine_upscaler = request_data.pop("refine_upscaler")
        request_data.pop("output_upscaler")
        detail_pass_models = request_data.pop("detail_pass_models")
        controlnet_units = request_data.pop("controlnet_units")
        request_data.pop("style_variant")
        request_data.pop("enforce_sdxl_token_budget")
        supports_hires_fix = request_data.pop("supports_hires_fix")
        override_settings: dict[str, Any] = {
            "CLIP_stop_at_last_layers": clip_skip,
        }
        if model_checkpoint:
            override_settings["sd_model_checkpoint"] = model_checkpoint
        request_data["override_settings"] = override_settings
        request_data["override_settings_restore_afterwards"] = False
        # Both enhanced tiers share one conservative latent pass, keeping peak
        # VRAM stable on 10 GB cards. Their only difference is the final tiled
        # pixel output size, which does not allocate a giant diffusion latent.
        # FLUX has no working hires-fix pass on this reForge build: any
        # enable_hr request fails with "'NoneType' object is not iterable"
        # regardless of upscaler or step settings. It generates at its native
        # resolution instead, so the enhanced tiers just run the base pass.
        if high_res and supports_hires_fix:
            request_data.update(
                enable_hr=True,
                hr_scale=refine_scale,
                hr_upscaler=refine_upscaler,
                denoising_strength=refine_denoising_strength,
                hr_second_pass_steps=refine_steps,
                hr_cfg=request_data["cfg_scale"],
                hr_sampler_name=request_data["sampler_name"],
                hr_scheduler=request_data["scheduler"],
            )
        if detail_pass_models:
            request_data["alwayson_scripts"] = {
                "ADetailer": {"args": _adetailer_args(tuple(detail_pass_models))}
            }
        if controlnet_units:
            request_data.setdefault("alwayson_scripts", {})["ControlNet"] = {
                "args": controlnet_units
            }
        return request_data


@dataclass(frozen=True, slots=True)
class Img2ImgPayload:
    """Strongly typed request body for Forge's img2img endpoint."""

    prompt: str
    negative_prompt: str
    width: int
    height: int
    init_images: list[str]
    denoising_strength: float
    mask: str | None = None
    mask_blur: int = 8
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = 32
    invert_mask: bool = False
    soft_inpainting: bool = False
    resize_mode: int = 1
    sampler_name: str = "DPM++ 2M SDE"
    scheduler: str = "Karras"
    steps: int = 28
    cfg_scale: float = 6.0
    seed: int = -1
    model_checkpoint: str | None = None
    clip_skip: int = 1
    detail_pass_models: tuple[str, ...] = ()
    face_reference: str = ""
    style_variant: str | None = None

    def as_request(self) -> dict[str, Any]:
        """Serialize img2img without txt2img-only high-resolution fields."""
        request_data: dict[str, Any] = asdict(self)
        model_checkpoint = request_data.pop("model_checkpoint")
        clip_skip = request_data.pop("clip_skip")
        detail_pass_models = request_data.pop("detail_pass_models")
        face_reference = request_data.pop("face_reference")
        request_data.pop("style_variant")
        mask = request_data.pop("mask")
        mask_blur = request_data.pop("mask_blur")
        inpaint_full_res = request_data.pop("inpaint_full_res")
        inpaint_full_res_padding = request_data.pop("inpaint_full_res_padding")
        invert_mask = request_data.pop("invert_mask")
        soft_inpainting = request_data.pop("soft_inpainting")
        override_settings: dict[str, Any] = {
            "CLIP_stop_at_last_layers": clip_skip,
        }
        if model_checkpoint:
            override_settings["sd_model_checkpoint"] = model_checkpoint
        request_data["override_settings"] = override_settings
        request_data["override_settings_restore_afterwards"] = False
        if detail_pass_models:
            request_data["alwayson_scripts"] = {
                "ADetailer": {"args": _adetailer_args(tuple(detail_pass_models))}
            }
        if face_reference:
            request_data.setdefault("alwayson_scripts", {})["ControlNet"] = {
                "args": [{
                    "enabled": True,
                    # Use reForge's native mapping-based IPAdapter preprocessor.
                    # The similarly named legacy "_variant" preprocessor emits
                    # FaceIdPlusInput, which this IPAdapter patcher cannot
                    # expand as **cond; Forge then returns an unconditioned
                    # image despite recording ControlNet in PNG metadata.
                    "module": "InsightFace+CLIP-H (IPAdapter)",
                    "model": FACEID_CONTROLNET_MODEL,
                    "weight": 1.2,
                    "image": face_reference,
                    "resize_mode": "Crop and Resize",
                    "control_mode": "Balanced",
                    "guidance_start": 0.0,
                    "guidance_end": 1.0,
                    "pixel_perfect": True,
                }]
            }
        if mask is not None:
            request_data.update(
                mask=mask,
                inpainting_fill=1,
                inpaint_full_res=inpaint_full_res,
                inpaint_full_res_padding=inpaint_full_res_padding,
                mask_blur=mask_blur,
                inpainting_mask_invert=1 if invert_mask else 0,
            )
            if soft_inpainting:
                request_data.setdefault("alwayson_scripts", {})["soft inpainting"] = {
                    "args": [True, 1, 0.5, 4, 0, 0.5, 2]
                }
        return request_data


HAND_REPAIR_POSITIVE = "a hand with exactly five fingers, correct human anatomy"
HAND_REPAIR_NEGATIVE = (
    "extra fingers, six fingers, fused fingers, malformed hand, extra digit"
)


def _adetailer_args(models: tuple[str, ...]) -> list[Any]:
    """Build ADetailer alwayson_scripts args for a hand/face repair pass.

    Denoise stays at 0.4: measured at 0.55 the pass stops repairing the hand
    underneath it and reinvents one, with drifted colour and warped fingers.
    """
    args: list[Any] = [True, False]
    args.extend(
        {
            "ad_model": model,
            "ad_tab_enable": True,
            # The repair pass inherits no prompt of its own, so without this it
            # has no instruction about digit count and will faithfully sharpen
            # a six-fingered hand from the base render. This biases the repair
            # without guaranteeing a correct count - an open palm facing camera
            # is still the hardest case.
            "ad_prompt": HAND_REPAIR_POSITIVE if model.startswith("hand") else "",
            "ad_negative_prompt": (
                HAND_REPAIR_NEGATIVE if model.startswith("hand") else ""
            ),
            "ad_confidence": 0.3,
            "ad_denoising_strength": 0.4,
            "ad_inpaint_only_masked": True,
            # 64px of surrounding context so the repair matches the wrist and
            # skin tone it attaches to.
            "ad_inpaint_only_masked_padding": 64,
            "ad_use_inpaint_width_height": True,
            "ad_inpaint_width": 1024,
            "ad_inpaint_height": 1024,
            "ad_use_steps": True,
            "ad_steps": 40,
        }
        for model in models
    )
    return args


@dataclass(frozen=True, slots=True)
class ForgeGeneration:
    """Decoded Forge output plus the actual seed used for reproducibility."""

    image_bytes: bytes
    seed: int
    diffusion_duration_seconds: float
    upscale_duration_seconds: float


def assert_sdxl_dispatch_budget(request_data: dict[str, Any]) -> tuple[int, int]:
    """Reject an over-budget prompt at the last boundary before Forge.

    ``request_data`` is the fully serialized request body, so this measures the
    exact positive and negative strings that reForge will send to SDXL after
    every application wrapper and adapter has run.
    """
    positive = str(request_data.get("prompt") or "")
    negative = str(request_data.get("negative_prompt") or "")
    positive_tokens = _SDXL_TOKENIZER.count(positive)
    negative_tokens = _SDXL_TOKENIZER.count(negative)
    if positive_tokens > SDXL_PROMPT_TOKEN_LIMIT:
        raise ValueError(
            f"Positive prompt has {positive_tokens} SDXL tokens; "
            f"maximum is {SDXL_PROMPT_TOKEN_LIMIT}."
        )
    if negative_tokens > SDXL_PROMPT_TOKEN_LIMIT:
        raise ValueError(
            f"Negative prompt has {negative_tokens} SDXL tokens; "
            f"maximum is {SDXL_PROMPT_TOKEN_LIMIT}."
        )
    return positive_tokens, negative_tokens


def join_prompt_parts(*parts: str) -> str:
    """Join non-empty prompt fragments with Forge-friendly comma separation."""
    cleaned = (part.strip(" ,.;\n\t") for part in parts)
    return ", ".join(part for part in cleaned if part)


def validate_female_only_prompt(value: str) -> None:
    """Reject positive prompts that contradict the app's female-only policy."""
    normalized = unicodedata.normalize("NFKC", value).casefold().translate(
        POLICY_CONFUSABLES
    )
    normalized = "".join(
        character
        for character in normalized
        if unicodedata.category(character) != "Cf"
    )
    match = MALE_SUBJECT_PATTERN.search(normalized)
    if match is not None:
        raise ValueError(
            f"Positive prompts cannot request male subjects ('{match.group(0)}')."
        )


def _semantic_fragment_key(fragment: str) -> str:
    """Return a conservative concept key for common prompt synonyms."""
    normalized = re.sub(r"[^a-z0-9_]+", " ", fragment.lower()).strip()
    if re.search(
        r"\b(natural|lifelike|refined|realistic) skin (texture|pores|detail)\b",
        normalized,
    ):
        return "skin-detail"
    if re.search(r"\b(natural|fine|detailed) facial detail\b", normalized):
        return "face-detail"
    if normalized in {"highly detailed", "8k uhd", "crisp focal detail"}:
        return "overall-detail"
    if re.search(r"\b(bad|malformed|blurry|distorted) hands?\b", normalized):
        return "hand-defect"
    if normalized in {"male presence", "male person"}:
        return "male"
    if normalized in {"clothed clothing", "fully clothed", "clothing items"}:
        return "clothing"
    return normalized


def compile_prompt(
    slots: tuple[tuple[str, bool] | tuple[str, bool, int], ...],
    maximum_words: int | None,
) -> str:
    """Compile prompt slots with required retention and optional priority."""
    candidates: list[tuple[str, bool, int, str, int]] = []
    for slot in slots:
        value, required = slot[:2]
        priority = slot[2] if len(slot) == 3 else 0
        for fragment in value.split(","):
            cleaned = fragment.strip(" ,.;\n\t")
            if not cleaned:
                continue
            candidates.append(
                (
                    cleaned,
                    required,
                    priority,
                    _semantic_fragment_key(cleaned),
                    len(re.findall(r"[A-Za-z0-9_]+", cleaned)),
                )
            )

    # If an AI-supplied optional detail repeats a deterministic safety or model
    # rule that occurs later, preserve the deterministic rule and drop the copy.
    required_keys = {key for _, required, _, key, _ in candidates if required}
    unique: list[tuple[str, bool, int, int]] = []
    seen: set[str] = set()
    for fragment, required, priority, key, word_count in candidates:
        if key in seen or (not required and key in required_keys):
            continue
        seen.add(key)
        unique.append((fragment, required, priority, word_count))

    if maximum_words is None:
        return join_prompt_parts(*(fragment for fragment, _, _, _ in unique))

    required_words = sum(
        word_count for _, required, _, word_count in unique if required
    )
    if required_words > maximum_words:
        raise ValueError(
            f"Required prompt fragments need {required_words} words, "
            f"exceeding the {maximum_words}-word budget."
        )
    remaining_words = maximum_words - required_words
    selected_indexes = {
        index for index, (_, required, _, _) in enumerate(unique) if required
    }

    # Vary optional admission independently of presentation order. Otherwise a
    # tight budget always preserves the earliest optional slots and erases the
    # entropy supplied by later scene, lighting, palette, and mood fragments.
    priorities = sorted(
        {priority for _, required, priority, _ in unique if not required},
        reverse=True,
    )
    for priority in priorities:
        optional_indexes = [
            index
            for index, (_, required, item_priority, _) in enumerate(unique)
            if not required and item_priority == priority
        ]
        random.shuffle(optional_indexes)
        for index in optional_indexes:
            word_count = unique[index][3]
            if word_count <= remaining_words:
                selected_indexes.add(index)
                remaining_words -= word_count

    return join_prompt_parts(
        *(fragment for index, (fragment, _, _, _) in enumerate(unique) if index in selected_indexes)
    )


def rating_positive_for_subject(subject: str, rating: str) -> str:
    """Add rating intent only when the creative prompt does not already state it."""
    return RATING_RULES[rating].positive_modifiers


def select_style_variant(
    rule: StyleRule, compiled_positive_prompt: str = ""
) -> StyleVariant | None:
    """Resolve one variant, preserving it when a compiled prompt is resubmitted."""
    if not rule.variants:
        return None
    for variant in rule.variants:
        if variant.positive_modifiers in compiled_positive_prompt:
            return variant
    return random.choice(rule.variants)


def build_pony_prompts(
    subject: str,
    exclusions: str,
    settings: GenerationSettings,
    *,
    positive_context: tuple[str, ...] = (),
    prioritized_positive_context: tuple[tuple[str, int], ...] = (),
    required_positive_context: tuple[str, ...] = (),
    negative_context: tuple[str, ...] = (),
    required_negative_context: tuple[str, ...] = (),
    enforce_budget: bool = False,
    style_variant: StyleVariant | None = None,
    rating_subject: str | None = None,
) -> tuple[str, str]:
    """Build Pony/SDXL prompts in the required semantic tag order."""
    profile = model_profile_for(settings.model)
    rating = RATING_RULES[settings.rating]
    style = STYLE_RULES[settings.style]
    variant = style_variant or select_style_variant(style, subject)
    positive_modifiers = variant.positive_modifiers if variant else style.positive_modifiers
    style_exclusions = variant.style_exclusions if variant else style.style_exclusions
    rating_positive = rating_positive_for_subject(
        rating_subject if rating_subject is not None else subject, settings.rating
    )
    anatomy_base = (
        NEGATIVE_ANATOMY_BASE
        if profile.include_negative_quality_floor
        else f"{NEGATIVE_ANATOMY_GUIDANCE}, {NEGATIVE_POSE_REPETITION}"
    )
    if profile.negative_embedding:
        anatomy_base = f"{anatomy_base}, {profile.negative_embedding}"
    return (
        compile_prompt(
            (
                (POSITIVE_QUALITY_PREFIX, True),
                (style.source_tag, True),
                (rating.positive_tag, True),
                (FEMALE_ONLY_POSITIVE, True),
                (rating_positive, True),
                (subject, not enforce_budget),
                (FAMILY_ANATOMY_GUIDANCE[profile.family], True),
                *((item, True) for item in required_positive_context),
                *((item, False, priority) for item, priority in prioritized_positive_context),
                *((item, False) for item in positive_context),
                (positive_modifiers, True),
            ),
            profile.positive_word_budget if enforce_budget else None,
        ),
        compile_prompt(
            (
                (NEGATIVE_QUALITY_PREFIX, True),
                (style_exclusions, False),
                (anatomy_base, True),
                (rating.negative_tag, True),
                (rating.negative_modifiers, True),
                *((item, True) for item in required_negative_context),
                *((item, False) for item in negative_context),
                (exclusions, not enforce_budget),
                (MALE_EXCLUSION_NEGATIVE, True),
            ),
            profile.negative_word_budget if enforce_budget else None,
        ),
    )


def build_sdxl_prompts(
    subject: str,
    exclusions: str,
    settings: GenerationSettings,
    *,
    positive_context: tuple[str, ...] = (),
    prioritized_positive_context: tuple[tuple[str, int], ...] = (),
    required_positive_context: tuple[str, ...] = (),
    negative_context: tuple[str, ...] = (),
    required_negative_context: tuple[str, ...] = (),
    enforce_budget: bool = False,
    style_variant: StyleVariant | None = None,
    rating_subject: str | None = None,
) -> tuple[str, str]:
    """Build natural-language/tag hybrid prompts for non-Pony SDXL models."""
    profile = model_profile_for(settings.model)
    rating = RATING_RULES[settings.rating]
    style = STYLE_RULES[settings.style]
    variant = style_variant or select_style_variant(style, subject)
    positive_modifiers = variant.positive_modifiers if variant else style.positive_modifiers
    style_exclusions = (
        variant.style_exclusions
        if variant
        else SDXL_STYLE_EXCLUSIONS.get(settings.style, style.style_exclusions)
    )
    rating_positive = rating_positive_for_subject(
        rating_subject if rating_subject is not None else subject, settings.rating
    )
    quality_floor = "low quality, lowres" if profile.include_negative_quality_floor else ""
    anatomy_base = (
        NEGATIVE_ANATOMY_BASE
        if profile.include_negative_quality_floor
        else f"{NEGATIVE_ANATOMY_GUIDANCE}, {NEGATIVE_POSE_REPETITION}"
    )
    if profile.negative_embedding:
        anatomy_base = f"{anatomy_base}, {profile.negative_embedding}"
    return (
        compile_prompt(
            (
                (subject, not enforce_budget),
                (FEMALE_ONLY_POSITIVE, True),
                (rating_positive, True),
                (FAMILY_ANATOMY_GUIDANCE[profile.family], True),
                *((item, True) for item in required_positive_context),
                *((item, False, priority) for item, priority in prioritized_positive_context),
                *((item, False) for item in positive_context),
                (positive_modifiers, True),
            ),
            profile.positive_word_budget if enforce_budget else None,
        ),
        compile_prompt(
            (
                (quality_floor, True),
                (style_exclusions, False),
                (anatomy_base, True),
                (rating.negative_modifiers, True),
                *((item, True) for item in required_negative_context),
                *((item, False) for item in negative_context),
                (exclusions, not enforce_budget),
                (MALE_EXCLUSION_NEGATIVE, True),
            ),
            profile.negative_word_budget if enforce_budget else None,
        ),
    )


def _effective_cfg_scale(requested: float | None, profile: ModelProfile) -> float:
    """Resolve a request's CFG against the checkpoint's tuned floor."""
    if requested is None:
        return profile.cfg_scale
    return max(float(requested), profile.min_cfg_scale)


def build_payload(
    subject: str,
    exclusions: str,
    settings: GenerationSettings,
    *,
    seed: int = -1,
    prebuilt: bool = False,
) -> Txt2ImgPayload:
    """Construct the same typed Forge payload used by the Streamlit app.

    Pass prebuilt=True when the caller already holds a finished prompt pair, as
    the prompt engine does. Without it this rebuilds the prompt and appends its
    own scaffolding, which silently pushes the result past the token budget the
    caller was compiling against.
    """
    validate_female_only_prompt(subject)
    profile = model_profile_for(settings.model)
    if settings.style not in profile.styles:
        raise ValueError(
            f"Style '{settings.style}' is not supported by {settings.model}."
        )
    # A prebuilt engine prompt already contains its selected style vocabulary.
    # Inventing a second legacy variant here does not alter the prompt, but it
    # does write false metadata (for example "manga" beside a cel-shaded WAI
    # prompt), which corrupts gallery/analytics labels.
    style_variant = (
        None if prebuilt else select_style_variant(STYLE_RULES[settings.style], subject)
    )
    if profile.family in {"pony", "pony_realism"}:
        positive, negative = build_pony_prompts(
            subject, exclusions, settings, style_variant=style_variant
        )
        if prebuilt or subject.strip().startswith(f"{POSITIVE_QUALITY_PREFIX}, source_"):
            positive = subject.strip()
        if prebuilt or exclusions.strip().startswith(
            (NEGATIVE_QUALITY_PREFIX, LEGACY_NEGATIVE_QUALITY_PREFIX)
        ):
            negative = exclusions.strip()
    else:
        positive, negative = build_sdxl_prompts(
            subject, exclusions, settings, style_variant=style_variant
        )
        # Surprise Me returns a finished SDXL prompt pair. Keep payload
        # construction idempotent when that pair is submitted for generation.
        if prebuilt or FEMALE_ONLY_POSITIVE in subject:
            positive = subject.strip()
        if prebuilt or exclusions.strip().startswith(
            ("worst quality, low quality, lowres", "low quality, lowres")
        ):
            negative = exclusions.strip()
    if not prebuilt and FEMALE_ONLY_POSITIVE not in positive:
        positive = join_prompt_parts(positive, FEMALE_ONLY_POSITIVE)
    if not prebuilt and MALE_EXCLUSION_CORE not in negative:
        negative = join_prompt_parts(negative, MALE_EXCLUSION_NEGATIVE)
    is_landscape = settings.width > settings.height
    refine_scale = (
        profile.landscape_refine_scale
        if is_landscape and profile.landscape_refine_scale is not None
        else profile.refine_scale
    )
    refine_steps = (
        profile.landscape_refine_steps
        if is_landscape and profile.landscape_refine_steps is not None
        else profile.refine_steps
    )
    return Txt2ImgPayload(
        prompt=positive,
        negative_prompt=negative,
        width=settings.width,
        height=settings.height,
        sampler_name=profile.sampler_name,
        scheduler=profile.scheduler,
        steps=profile.steps,
        cfg_scale=_effective_cfg_scale(settings.cfg_scale, profile),
        distilled_cfg_scale=profile.distilled_cfg_scale,
        seed=seed,
        model_checkpoint=None if settings.model == "Forge default" else settings.model,
        clip_skip=profile.clip_skip,
        refine_steps=refine_steps,
        refine_denoising_strength=profile.refine_denoising_strength,
        refine_scale=refine_scale,
        high_output_scale=profile.high_output_scale,
        super_output_scale=profile.super_output_scale,
        refine_upscaler=profile.refine_upscaler,
        output_upscaler=profile.output_upscaler,
        supports_hires_fix=profile.supports_hires_fix,
        detail_pass_models=profile.detail_pass_models,
        style_variant=style_variant.name if style_variant else None,
        enforce_sdxl_token_budget=prebuilt,
    )


def build_img2img_payload(
    subject: str,
    exclusions: str,
    settings: GenerationSettings,
    init_image: str,
    denoising_strength: float,
    *,
    seed: int = -1,
    mask: str | None = None,
    mask_blur: int = 8,
    inpaint_full_res: bool = True,
    inpaint_full_res_padding: int = 32,
    invert_mask: bool = False,
    soft_inpainting: bool = False,
    # "scene" also negates people; "region" keeps the prompt focused without
    # that, for a garment or repair mask that sits on the subject.
    region_only: str = "",
    face_reference: str = "",
    preserve_unmasked_regions: bool = False,
) -> Img2ImgPayload:
    """Build img2img through the canonical prompt/profile payload path."""
    base = build_payload(subject, exclusions, settings, seed=seed)
    if region_only:
        # Inside a mask only that region is generated, so the subject, rating
        # and anatomy scaffolding build_payload adds describes pixels that are
        # preserved rather than drawn - and competes with the description of
        # what the region should actually become. Keep the quality prefix and
        # the caller's own text.
        profile = model_profile_for(settings.model)
        floor = NEGATIVE_QUALITY_FLOOR if profile.include_negative_quality_floor else ""
        base = replace(
            base,
            prompt=join_prompt_parts(POSITIVE_QUALITY_PREFIX, subject),
            negative_prompt=join_prompt_parts(
                NEGATIVE_QUALITY_PREFIX,
                "person, people, human, figure, portrait" if region_only == "scene" else "",
                floor,
            ),
        )
    if face_reference:
        base = replace(base, prompt=join_prompt_parts(base.prompt, FACEID_LORA_TAG))
    detail_pass_models = base.detail_pass_models
    if preserve_unmasked_regions:
        # ADetailer runs after the main masked inpaint with masks of its own.
        # Running it here would modify a protected face or hand even though the
        # caller's garment/background mask deliberately left those pixels out.
        detail_pass_models = ()
    elif face_reference:
        # The installed ADetailer defaults its nested pass to ControlNet=None,
        # so a face pass would overwrite the FaceID-conditioned face with an
        # unconditioned 0.4-denoise inpaint. Hand repair remains useful because
        # FaceID regenerates the entire scene and does not condition anatomy.
        detail_pass_models = tuple(
            model for model in detail_pass_models
            if not model.lower().startswith("face")
        )
    return Img2ImgPayload(
        prompt=base.prompt,
        negative_prompt=base.negative_prompt,
        width=base.width,
        height=base.height,
        init_images=[init_image],
        denoising_strength=denoising_strength,
        mask=mask,
        mask_blur=mask_blur,
        inpaint_full_res=inpaint_full_res,
        inpaint_full_res_padding=inpaint_full_res_padding,
        invert_mask=invert_mask,
        soft_inpainting=soft_inpainting,
        sampler_name=base.sampler_name,
        scheduler=base.scheduler,
        steps=base.steps,
        cfg_scale=base.cfg_scale,
        seed=base.seed,
        model_checkpoint=base.model_checkpoint,
        clip_skip=base.clip_skip,
        detail_pass_models=detail_pass_models,
        face_reference=face_reference,
        style_variant=base.style_variant,
    )


def build_instantid_payload(
    subject: str,
    exclusions: str,
    settings: GenerationSettings,
    face_reference: str,
    *,
    adapter_model: str,
    controlnet_model: str,
    seed: int = -1,
) -> Txt2ImgPayload:
    """Build an SDXL txt2img request with InstantID's ordered two-unit pair.

    InstantID's IP-Adapter unit supplies the face embedding consumed by the
    following keypoint ControlNet unit, so changing this order silently removes
    identity conditioning. A face ADetailer pass is intentionally omitted: it
    would repaint the conditioned face without InstantID after diffusion.
    """
    if not face_reference:
        raise ValueError("InstantID requires a reference face image.")
    if not adapter_model or not controlnet_model:
        raise ValueError("InstantID ControlNet models are not available.")
    base = build_payload(subject, exclusions, settings, seed=seed)
    base = replace(
        base,
        # Put framing first so it survives the long model/rating scaffold and
        # controls composition before a garment phrase can bias toward a crop.
        prompt=join_prompt_parts(CHARACTER_FRAMING_POSITIVE, base.prompt),
        negative_prompt=join_prompt_parts(
            CHARACTER_FRAMING_NEGATIVE,
            base.negative_prompt,
        ),
    )
    common = {
        "enabled": True,
        "image": face_reference,
        "resize_mode": "Crop and Resize",
        "control_mode": "Balanced",
        "guidance_start": 0.0,
        "guidance_end": 1.0,
        "pixel_perfect": True,
    }
    return replace(
        base,
        # ADetailer's nested img2img pass does not reproduce InstantID's paired
        # conditioning reliably in reForge. Run anatomy repair only after the
        # identity-verified image in a future conservative post-pass.
        detail_pass_models=(),
        controlnet_units=(
            {
                **common,
                "module": "InsightFace (InstantID)",
                "model": adapter_model,
                # The two units are not interchangeable and must not share a
                # weight. Measured on a fixed seed and reference: the keypoint
                # ControlNet at 0.8 destroys the image into noise texture, with
                # no face left for the verifier to detect. At 0.3 the image is
                # clean. Identity comes from the adapter, so it carries the
                # weight - it peaks at 1.0 and falls off above that.
                "weight": 1.0,
            },
            {
                **common,
                "module": "instant_id_face_keypoints",
                "model": controlnet_model,
                "weight": 0.3,
            },
        ),
    )


def generate_random_prompt_pair(
    model: str,
    style: str,
    rating: str,
    aspect_ratio: str,
    high_res: bool,
    creative_brief: tuple[str, ...] | None = None,
    orientation: str = "mixed",
) -> tuple[str, str]:
    """Generate prompts from checkpoint, style, rating, framing, and HR context."""
    rating_profile = RATING_PROMPT_PROFILES[rating]
    rear = orientation == "back"
    wardrobe_pool = (
        rating_profile.rear_wardrobe
        if rear and rating_profile.rear_wardrobe
        else rating_profile.wardrobe
    )
    body_pool = (
        rating_profile.rear_body_types
        if rear and rating_profile.rear_body_types
        else rating_profile.body_types
    )
    environment = (
        random.choice(rating_profile.environments)
        if creative_brief is None
        else (creative_brief[1] if len(creative_brief) > 1 else creative_brief[0])
    )
    optional_brief = (
        creative_brief
        if creative_brief is None
        else tuple(
            part
            for index, part in enumerate(creative_brief)
            if index != (1 if len(creative_brief) > 1 else 0)
        )
    )
    wardrobe = random.choice(wardrobe_pool)
    creative_parts = (
        random.choice(rating_profile.subjects),
        random.choice(body_pool),
        random.choice(CAMERA_ANGLES),
        random.choice(POSE_ACTIONS),
        *(
            optional_brief
            if optional_brief is not None
            else (random.choice(RANDOM_LIGHTING),)
        ),
    )
    # Two contextual exclusions are enough beside the deterministic model,
    # anatomy, rating, framing, and female-only negatives. More dilutes CLIP.
    exclusions = tuple(random.sample(RANDOM_NEGATIVE_FRAGMENTS, k=2))
    return build_contextual_prompt_pair(
        model,
        style,
        rating,
        aspect_ratio,
        high_res,
        creative_parts,
        exclusions,
        orientation=orientation,
        orientation_fragment=random.choice(orientation_pool(orientation)),
        environment=environment,
        wardrobe=wardrobe,
    )


def build_contextual_prompt_pair(
    model: str,
    style: str,
    rating: str,
    aspect_ratio: str,
    high_res: bool,
    creative_parts: tuple[str, ...],
    creative_exclusions: tuple[str, ...],
    orientation: str = "mixed",
    orientation_fragment: str = "",
    environment: str = "",
    wardrobe: str = "",
) -> tuple[str, str]:
    """Normalize creative scene ideas through deterministic model rules."""
    model_profile = model_profile_for(model)
    if style not in model_profile.styles:
        raise ValueError(f"Style '{style}' is not supported by {model}.")
    aspect_ratios = aspect_ratios_for(model)
    if aspect_ratio not in aspect_ratios:
        raise ValueError(f"Unsupported aspect ratio: {aspect_ratio}.")

    width, height = aspect_ratios[aspect_ratio]
    compositions, lenses, composition_exclusions = aspect_prompt_rules_for(aspect_ratio)
    family_positive, family_negative = FAMILY_PROMPT_MODIFIERS[model_profile.family]
    settings = GenerationSettings(
        width=width,
        height=height,
        model=model,
        rating=rating,
        style=style,
        high_res=high_res,
    )
    composition = random.choice(compositions)
    framing, _, optional_composition = composition.partition(",")
    action = creative_parts[3] if len(creative_parts) > 3 else ""
    # Subject (age and maturity cue) and body type are both chosen deliberately
    # by the caller, so they outrank camera and lighting filler. Left at the
    # bottom tier the model falls back on whatever its training prefers.
    subject_descriptor = creative_parts[0] if creative_parts else ""
    body_descriptor = creative_parts[1] if len(creative_parts) > 1 else ""
    supporting_creative_parts = tuple(
        part for index, part in enumerate(creative_parts) if index not in {0, 1, 3}
    )
    # ponytail: four priority-3 fragments do not all fit the 52-word Lightning
    # budget, so that checkpoint still drops roughly a third of them. Fine for
    # the 70/68-word models; give Lightning its own tier order if it matters.
    prioritized_positive_context = (
        # Wardrobe is a deliberate choice, so it ranks with the other selected
        # concepts rather than below them.
        (wardrobe, 3),
        (environment, 3),
        (subject_descriptor, 3),
        (body_descriptor, 3),
        (action, 2),
        *((part, 1) for part in supporting_creative_parts),
    )
    positive_context = (
        optional_composition,
        random.choice(lenses),
        family_positive,
        HIGH_RES_PROMPT_MODIFIERS[model_profile.family] if high_res else "",
    )
    negative_context = (
        family_negative,
        composition_exclusions,
    )
    # Framing and orientation are correctness constraints. Novel scene content
    # remains optional, but its priority lets it consume the optional budget
    # before fixed family, wardrobe, composition, lens, and high-res filler.
    required_positive_context = (
        orientation_fragment,
        framing,
    )
    required_negative_context = (
        HIGH_RES_NEGATIVE_MODIFIERS if high_res else "",
        NEGATIVE_REAR_ORIENTATION if orientation == "back" else "",
    )
    exclusions = join_prompt_parts(*creative_exclusions)
    subject = ""
    if model_profile.family in {"pony", "pony_realism"}:
        return build_pony_prompts(
            subject,
            exclusions,
            settings,
            positive_context=positive_context,
            prioritized_positive_context=prioritized_positive_context,
            required_positive_context=required_positive_context,
            negative_context=negative_context,
            required_negative_context=required_negative_context,
            enforce_budget=True,
            rating_subject=join_prompt_parts(*creative_parts, wardrobe),
        )
    return build_sdxl_prompts(
        subject,
        exclusions,
        settings,
        positive_context=positive_context,
        prioritized_positive_context=prioritized_positive_context,
        required_positive_context=required_positive_context,
        negative_context=negative_context,
        required_negative_context=required_negative_context,
        enforce_budget=True,
        rating_subject=join_prompt_parts(*creative_parts, wardrobe),
    )


def release_other_backends(active_base_url: str) -> None:
    """Best-effort release of checkpoints held by every inactive backend."""
    try:
        # Lazy import preserves the database/Forge import cycle during startup.
        from .database import configured_backends

        backends = configured_backends()
    except Exception as exc:  # The active generation must never depend on this cleanup.
        LOGGER.info("Could not inspect inactive Forge backends: %s", exc)
        return

    active_url = active_base_url.rstrip("/")
    for backend in backends:
        if backend.base_url.rstrip("/") == active_url:
            continue
        try:
            ForgeApiClient(backend.id).unload_checkpoint(timeout=(0.5, 5.0))
        except Exception as exc:  # Best effort: an inactive backend may be stopped.
            LOGGER.info(
                "Could not release checkpoint from inactive backend %s: %s",
                backend.id,
                exc,
            )


class ForgeApiClient:
    """HTTP client isolating all Forge communication."""

    # Generation and upscale calls now run on the background job worker, not
    # inside a browser-facing request - nothing times out client-side while
    # this waits, so it can afford real headroom for GPU contention (a job
    # observed taking 9m12s under a second concurrent job) while still
    # eventually giving up if Forge is genuinely hung rather than just busy.
    GENERATION_READ_TIMEOUT_SECONDS = 1800.0

    def __init__(self, backend_id: str | None = None) -> None:
        """Talk to one reForge instance, the shared default unless pinned.

        A model whose profile names its own host gets that host: FLUX lives on
        a separate instance with its own UNet, text encoders and VAE, and those
        modules are instance-wide, so it cannot share with the SDXL
        checkpoints. Callers that pass nothing use the first registry entry.
        """
        self._backend_id = backend_id

    @property
    def base_url(self) -> str:
        """The instance this client talks to right now."""
        # Lazy import avoids a database/Forge type-import cycle during startup.
        from .database import backend_config

        return backend_config(self._backend_id).base_url

    @classmethod
    def for_model(cls, model: str | None) -> "ForgeApiClient":
        """Return a client bound to the host that serves this checkpoint."""
        if not model:
            return cls()
        return cls(model_profile_for(model).forge_backend_id)

    def _request(self, method: str, path: str, **kwargs: Any):
        """Send a request to this client's configured local backend."""
        request_method = getattr(requests, method)
        return request_method(f"{self.base_url}{path}", **kwargs)

    def is_reachable(self) -> bool:
        """Return whether Forge's API documentation responds quickly."""
        try:
            return self._request("get", "/docs", timeout=(1.0, 2.0)).ok
        except requests.RequestException:
            return False

    def stalled_job_name(self) -> str | None:
        """Return the job Forge still holds after finishing, if any.

        A crashed or abandoned request can leave Forge reporting a completed
        job forever. /docs keeps answering, so is_reachable stays true while
        every new request queues behind a phantom job. Only the progress
        endpoint exposes it: progress pinned at 1.0 with a job name still set.
        """
        try:
            response = self._request("get", "/sdapi/v1/progress", timeout=(1.0, 3.0))
            response.raise_for_status()
            snapshot = response.json()
        except (requests.RequestException, ValueError):
            return None
        state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
        job = str(state.get("job") or "")
        progress = float(snapshot.get("progress") or 0.0)
        # A real in-flight job reports progress below 1.0 or advancing steps.
        if job and progress >= 1.0 and not int(state.get("sampling_step") or 0):
            return job
        return None

    def available_models(self) -> tuple[str, ...]:
        """Return checkpoint titles reported by Forge."""
        response = self._request("get", "/sdapi/v1/sd-models", timeout=(1.5, 5.0))
        response.raise_for_status()
        models = response.json()
        if not isinstance(models, list):
            raise ValueError("Forge returned an invalid model list.")
        titles = tuple(
            str(model["title"])
            for model in models
            if isinstance(model, dict) and model.get("title")
        )
        return titles or ("Forge default",)

    def controlnet_models(self) -> tuple[str, ...]:
        """Return the ControlNet model names registered by the running reForge."""
        response = self._request("get", "/controlnet/model_list", timeout=(1.5, 8.0))
        response.raise_for_status()
        body = response.json()
        values = body.get("model_list") if isinstance(body, dict) else None
        if not isinstance(values, list):
            raise ValueError("Forge returned an invalid ControlNet model list.")
        return tuple(str(value) for value in values if isinstance(value, str))

    def resolve_controlnet_model(self, prefix: str) -> str:
        """Resolve a stable filename prefix to reForge's hash-suffixed title."""
        normalized = prefix.casefold()
        for model in self.controlnet_models():
            if model.casefold().startswith(normalized):
                return model
        raise ValueError(
            f"Required ControlNet model '{prefix}' is not registered in reForge."
        )

    def memory(self) -> dict[str, Any]:
        """Return reForge's process/VRAM memory snapshot."""
        response = self._request("get", "/sdapi/v1/memory", timeout=(1.5, 8.0))
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Forge returned an invalid memory response.")
        return body

    def unload_checkpoint(
        self, *, timeout: tuple[float, float] = (2.0, 120.0)
    ) -> None:
        """Ask the process that owns CUDA memory to release its checkpoint."""
        response = self._request("post", "/sdapi/v1/unload-checkpoint", timeout=timeout)
        response.raise_for_status()

    def reload_checkpoint(self) -> None:
        """Ask reForge to restore the configured checkpoint after a local stage."""
        response = self._request("post", "/sdapi/v1/reload-checkpoint", timeout=(2.0, 300.0))
        response.raise_for_status()

    def generate_image(
        self,
        payload: Txt2ImgPayload,
        *,
        high_res: bool,
        super_res: bool = False,
        on_stage: Callable[[str], None] | None = None,
    ) -> ForgeGeneration:
        """Generate with Forge and use a memory-aware upscale for Super-res."""
        diffusion_started_at = monotonic()
        request_data = payload.as_request(high_res=high_res, super_res=super_res)
        if payload.enforce_sdxl_token_budget:
            assert_sdxl_dispatch_budget(request_data)
        response = self._request(
            "post",
            "/sdapi/v1/txt2img",
            json=request_data,
            timeout=(5.0, self.GENERATION_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        if on_stage is not None:
            on_stage("decoding")
        try:
            response_body = response.json()
            encoded = response_body["images"][0]
        except (requests.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("Forge returned a response without an image.") from exc
        image_bytes = self._decode_image(encoded)
        info = response_body.get("info", {})
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except json.JSONDecodeError:
                info = {}
        seed = (
            info.get("seed", payload.seed) if isinstance(info, dict) else payload.seed
        )
        try:
            actual_seed = int(seed)
        except (TypeError, ValueError):
            actual_seed = payload.seed
        diffusion_duration_seconds = monotonic() - diffusion_started_at
        upscale_duration_seconds = 0.0
        # When Hires.fix already produced the requested High dimensions, do not
        # run the same neural upscaler again at 1.0x. That redundant Extras pass
        # softened skin pores and fabric without adding pixels. Super still
        # needs its final tiled resize, as do profiles whose output scale is
        # larger than their latent refine scale.
        needs_output_upscale = super_res or (
            high_res and (
                not payload.supports_hires_fix
                or abs(payload.high_output_scale - payload.refine_scale) > 1e-6
            )
        )
        if needs_output_upscale:
            if on_stage is not None:
                on_stage("upscaling")
            scale = (
                payload.super_output_scale if super_res else payload.high_output_scale
            )
            upscale_started_at = monotonic()
            image_bytes = self._upscale_image(
                image_bytes,
                target_width=round(payload.width * scale),
                target_height=round(payload.height * scale),
                upscaler=payload.output_upscaler,
            )
            upscale_duration_seconds = monotonic() - upscale_started_at
        return ForgeGeneration(
            image_bytes=image_bytes,
            seed=actual_seed,
            diffusion_duration_seconds=diffusion_duration_seconds,
            upscale_duration_seconds=upscale_duration_seconds,
        )

    def transform_image(
        self,
        payload: Img2ImgPayload,
        *,
        on_stage: Callable[[str], None] | None = None,
    ) -> ForgeGeneration:
        """Transform one source image through Forge's img2img endpoint."""
        diffusion_started_at = monotonic()
        request_data = payload.as_request()
        response = self._request(
            "post",
            "/sdapi/v1/img2img",
            json=request_data,
            timeout=(5.0, self.GENERATION_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        if on_stage is not None:
            on_stage("decoding")
        try:
            response_body = response.json()
            encoded = response_body["images"][0]
        except (requests.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("Forge returned a response without an image.") from exc
        image_bytes = self._decode_image(encoded)
        info = response_body.get("info", {})
        if isinstance(info, str):
            try:
                info = json.loads(info)
            except json.JSONDecodeError:
                info = {}
        seed = info.get("seed", payload.seed) if isinstance(info, dict) else payload.seed
        try:
            actual_seed = int(seed)
        except (TypeError, ValueError):
            actual_seed = payload.seed
        return ForgeGeneration(
            image_bytes=image_bytes,
            seed=actual_seed,
            diffusion_duration_seconds=monotonic() - diffusion_started_at,
            upscale_duration_seconds=0.0,
        )

    @staticmethod
    def _decode_image(encoded: object) -> bytes:
        """Decode an A1111/Forge base64 image field."""
        if not isinstance(encoded, str) or not encoded:
            raise ValueError("Forge returned invalid image data.")
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError("Forge returned malformed base64 image data.") from exc

    def upscale_image_bytes(
        self,
        image_bytes: bytes,
        *,
        target_width: int,
        target_height: int,
        upscaler: str = SUPER_RES_UPSCALER,
    ) -> bytes:
        """Resize an already-generated image via Forge's tiled pixel upscaler.

        Pure pixel resize: no diffusion, no seed involved, works on any
        existing image regardless of how it was originally rendered.
        """
        return self._upscale_image(
            image_bytes,
            target_width=target_width,
            target_height=target_height,
            upscaler=upscaler,
        )

    def refine_and_upscale_image_bytes(
        self,
        image_bytes: bytes,
        *,
        prompt: str,
        negative_prompt: str,
        target_width: int,
        target_height: int,
        sampler_name: str,
        scheduler: str,
        steps: int,
        cfg_scale: float,
        seed: int,
        multiplier: float,
        denoising_strength: float = DETAIL_REFINE_DENOISE,
        upscaler: str = DETAIL_REFINE_UPSCALER,
        tile_size: int = 1024,
        tile_overlap: int = 64,
    ) -> bytes:
        """Enlarge an image, running real diffusion over it tile by tile.

        A plain pixel upscale can only interpolate what is already there, so
        detail is spread thinner as the canvas grows. The "SD upscale" script
        upscales and then denoises each tile at native resolution, which adds
        genuine texture at a VRAM cost of one tile rather than the whole image.

        Refinement is capped at REFINE_MAX_STEP per pass; anything beyond it is
        finished with a pixel resize, since diffusing a very large canvas costs
        far more time than the detail it returns.
        """
        refine_scale = min(multiplier, REFINE_MAX_STEP)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self._request(
            "post",
            "/sdapi/v1/img2img",
            json={
                "init_images": [encoded],
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "denoising_strength": denoising_strength,
                "sampler_name": sampler_name,
                "scheduler": scheduler,
                "steps": steps,
                "cfg_scale": cfg_scale,
                "seed": seed,
                "width": tile_size,
                "height": tile_size,
                "script_name": "sd upscale",
                # Positional: [info html, tile overlap, upscaler, scale factor]
                "script_args": ["", tile_overlap, upscaler, refine_scale],
                "save_images": False,
            },
            timeout=(5.0, self.GENERATION_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        try:
            refined = self._decode_image(response.json()["images"][0])
        except (requests.JSONDecodeError, KeyError, IndexError, TypeError) as exc:
            raise ValueError("Forge returned no image from the refine pass.") from exc
        if refine_scale >= multiplier:
            return refined
        # Cover the remaining scale with a plain resize.
        return self._upscale_image(
            refined,
            target_width=target_width,
            target_height=target_height,
            upscaler=upscaler,
        )

    def restore_and_upscale_image_bytes(
        self,
        image_bytes: bytes,
        *,
        target_width: int,
        target_height: int,
        upscaler: str,
        codeformer_visibility: float = 0.6,
        codeformer_weight: float = 0.6,
    ) -> bytes:
        """Restore faces before a pixel upscale in one Forge Extras request."""
        return self._upscale_image(
            image_bytes,
            target_width=target_width,
            target_height=target_height,
            upscaler=upscaler,
            codeformer_visibility=codeformer_visibility,
            codeformer_weight=codeformer_weight,
            upscale_first=False,
        )

    def _upscale_image(
        self,
        image_bytes: bytes,
        *,
        target_width: int,
        target_height: int,
        upscaler: str,
        codeformer_visibility: float = 0.0,
        codeformer_weight: float = 0.0,
        upscale_first: bool = True,
    ) -> bytes:
        """Run Forge Extras restoration/upscaling with explicit operation order."""
        encoded = base64.b64encode(image_bytes).decode("ascii")
        response = self._request(
            "post",
            "/sdapi/v1/extra-single-image",
            json={
                "resize_mode": 1,
                "show_extras_results": True,
                "gfpgan_visibility": 0,
                "codeformer_visibility": codeformer_visibility,
                "codeformer_weight": codeformer_weight,
                "upscaling_resize_w": target_width,
                "upscaling_resize_h": target_height,
                "upscaling_crop": False,
                "upscaler_1": upscaler,
                "upscaler_2": "None",
                "extras_upscaler_2_visibility": 0,
                "upscale_first": upscale_first,
                "image": encoded,
            },
            timeout=(5.0, self.GENERATION_READ_TIMEOUT_SECONDS),
        )
        response.raise_for_status()
        try:
            body = response.json()
            return self._decode_image(body["image"])
        except (requests.JSONDecodeError, KeyError, TypeError) as exc:
            raise ValueError(
                "Forge returned no image from the Super-res upscaler."
            ) from exc

    def generation_progress(self) -> dict[str, Any]:
        """Return Forge's live A1111-compatible generation progress snapshot."""
        response = self._request(
            "get",
            "/sdapi/v1/progress",
            params={"skip_current_image": "false"},
            timeout=(1.0, 4.0),
        )
        response.raise_for_status()
        body = response.json()
        if not isinstance(body, dict):
            raise ValueError("Forge returned an invalid progress response.")
        return body

    def interrupt_generation(self) -> None:
        """Ask Forge to interrupt its currently active diffusion request."""
        response = self._request("post", "/sdapi/v1/interrupt", timeout=(1.0, 5.0))
        response.raise_for_status()

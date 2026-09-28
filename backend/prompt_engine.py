"""Deterministic, model-aware prompt AST compiler.

This module intentionally lives beside :mod:`backend.forge`; adoption by the
HTTP request path is a separate concern.  The keyword bundle is the semantic
source of truth and rendered strings exist only at the adapter boundary.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Collection, Mapping, Protocol, Sequence

from .clip_tokenizer import CLIPBPETokenizer
from .diversity import DiversitySnapshot, DiversityStore
from .batch_profile import NEUTRAL, BatchProfile
from .keywords import BY_ID, CATEGORIES


class Rating(str, Enum):
    SAFE = "SAFE"


class Orientation(str, Enum):
    FRONT = "front"
    SIDE = "side"
    REAR = "rear"


class Decision(str, Enum):
    ACCEPT = "ACCEPT"
    REPAIR = "REPAIR"
    REJECT = "REJECT"


class ReasonCode(str, Enum):
    BUDGET_UNRECOVERABLE = "BUDGET_UNRECOVERABLE"
    CONTRADICTION_HARD = "CONTRADICTION_HARD"
    MODEL_UNSUPPORTED = "MODEL_UNSUPPORTED"
    RATING_CONFLICT = "RATING_CONFLICT"
    AGE_SAFETY_CONFLICT = "AGE_SAFETY_CONFLICT"
    IDENTITY_RULE_VIOLATION = "IDENTITY_RULE_VIOLATION"
    EMPTY_CORE = "EMPTY_CORE"
    REDUNDANT_CONCEPT = "REDUNDANT_CONCEPT"
    CARDINALITY_EXCEEDED = "CARDINALITY_EXCEEDED"
    REQUIREMENT_UNSATISFIED = "REQUIREMENT_UNSATISFIED"
    ORIENTATION_CONFLICT = "ORIENTATION_CONFLICT"
    BUDGET_OVERFLOW = "BUDGET_OVERFLOW"
    QUALITY_BELOW_ACCEPT = "QUALITY_BELOW_ACCEPT"
    SUBJECT_CONTRACT_VIOLATION = "SUBJECT_CONTRACT_VIOLATION"
    POSITIVE_NEGATIVE_CONFLICT = "POSITIVE_NEGATIVE_CONFLICT"
    DUPLICATE_NEGATIVE = "DUPLICATE_NEGATIVE"
    EXPOSURE_CONFLICT = "EXPOSURE_CONFLICT"
    MATERIAL_BINDING_INVALID = "MATERIAL_BINDING_INVALID"
    STYLE_CONFLICT = "STYLE_CONFLICT"


class ExposureState(str, Enum):
    CLOTHED = "CLOTHED"


SUBJECT_CONTRACT = {
    "gender": "female",
    "age_min": 24,
    "age_max": 44,
    "adult_appearance": True,
    "build_family": ("soft_curvy", "voluptuous", "feminine_hourglass"),
}
SUBJECT_FORBIDDEN = ("male",)
AGE_GUARD_NEGATIVE = ("child", "teen", "loli", "elderly", "wrinkled skin")
PROFILE_NEGATIVE = SUBJECT_FORBIDDEN + AGE_GUARD_NEGATIVE
# Safe prompts are already positively tagged for supported checkpoints, but a
# negative safety boundary is still required. It protects against checkpoint
# drift and remains in force when the finished engine prompt is passed through
# build_payload(prebuilt=True), which deliberately does not add legacy rules.
SAFE_RATING_NEGATIVE = ("nsfw", "nude", "naked", "nipples", "genitals")
SAFE_RATING_POSITIVE = ("fully clothed", "non-sexual scene")
# Preserve the gender and age boundary when a Safe prompt's negative budget is
# tight. Body-shape preferences remain eligible but may yield to the more
# important rating controls.
SAFE_PROFILE_NEGATIVE = ("male",) + AGE_GUARD_NEGATIVE
SOLO_SUBJECT_TEXT = "1woman, solo"
MULTIPLE_SUBJECT_NEGATIVE_ID = "negative_exclusions.extra_person"
PHOTOGRAPHIC_MEDIUM_NEGATIVE = (
    "cgi", "drawing", "anime", "cartoon", "3d render", "airbrush", "illustration"
)
PHOTOGRAPHIC_STYLE_IDS = frozenset({
    "style_medium.raw_photograph", "style_medium.glamour_photograph",
    "style_medium.cinematic_photograph",
    "style_medium.fine_art_photograph", "style_medium.analog_photograph",
    "style_medium.editorial_photograph", "style_medium.fashion_photograph",
    "style_medium.documentary_photograph", "style_medium.instant_film_photograph",
    "style_medium.35mm_film_look", "style_medium.medium_format_look",
    "style_medium.kodachrome_look", "style_medium.bleach_bypass_look",
    "style_medium.cross_processed_film", "style_medium.black_and_white_photograph",
    "style_medium.infrared_photograph", "style_medium.soft_focus_photograph",
    "style_medium.high_contrast_photograph", "style_medium.cinematic_still",
}) | frozenset(
    str(record["id"]) for record in CATEGORIES["style_medium"]
    if "photographic" in record["semantic_tags"]
)
PRIMARY_MEDIUM_IDS = frozenset({
    "style_medium.documentary_photograph",
    "style_medium.fashion_photograph",
    "style_medium.cinematic_photograph",
    "style_medium.editorial_photograph",
    "style_medium.forge_style_medium_beauty_photograph",
})
RECIPE_PHOTOREAL_PRIMARY_MEDIUM_IDS = frozenset({
    "style_medium.raw_photograph",
    "style_medium.glamour_photograph",
    "style_medium.fine_art_photograph",
    "style_medium.analog_photograph",
    "style_medium.documentary_photograph",
    "style_medium.instant_film_photograph",
    "style_medium.black_and_white_photograph",
    "style_medium.soft_focus_photograph",
    "style_medium.high_contrast_photograph",
    "style_medium.forge_style_medium_beauty_photograph",
})
PHOTOREAL_PRIMARY_MEDIUM_IDS = PRIMARY_MEDIUM_IDS | RECIPE_PHOTOREAL_PRIMARY_MEDIUM_IDS
FINISH_IDS = frozenset({
    "style_medium.kodachrome_look",
    "style_medium.bleach_bypass_look",
    "style_medium.forge_style_medium_warm_film_stock_look",
    "style_medium.forge_style_medium_cool_film_stock_look",
    "style_medium.forge_style_medium_subtle_film_grain",
})


@dataclass(frozen=True)
class StyleMediumProfile:
    """Existing KB records that faithfully express one request style parent."""

    primary_ids: frozenset[str]
    finish_ids: frozenset[str] = frozenset()
    photographic: bool = True
    finish_required: bool = False


@dataclass(frozen=True)
class ImageTypeRecipe:
    """Protected semantic identity for one model and requested image type.

    The KB still selects the concrete medium, lighting, camera, finish, and
    atmosphere.  This compact foundation is the request contract that those
    optional details may vary around, but never replace.
    """

    positive: str


# This is a vocabulary classification, not a checkpoint capability list.
# Checkpoint support remains owned by forge.ModelProfile.styles. A missing
# entry is deliberate: the current KB cannot faithfully express that parent.
STYLE_MEDIUM_PROFILES: Mapping[str, StyleMediumProfile] = {
    "Photoreal": StyleMediumProfile(
        PHOTOREAL_PRIMARY_MEDIUM_IDS, FINISH_IDS,
        photographic=True, finish_required=True,
    ),
    "Cinematic": StyleMediumProfile(frozenset({
        "style_medium.cinematic_photograph",
        "style_medium.cinematic_still",
        "style_medium.forge_style_medium_cinematic_realism",
    })),
    "Editorial": StyleMediumProfile(frozenset({
        "style_medium.editorial_photograph",
        "style_medium.fashion_photograph",
        "style_medium.modern_editorial_look",
        "style_medium.forge_style_medium_editorial_realism",
        "style_medium.forge_style_medium_fashion_realism",
        "style_medium.forge_style_medium_beauty_photograph",
        "style_medium.forge_style_medium_luxury_campaign_photograph",
    })),
    "Illustration": StyleMediumProfile(frozenset({
        "style_medium.ink_illustration",
        "style_medium.digital_painting",
        "style_medium.graphic_novel_style",
        "style_medium.pin_up_illustration",
        "style_medium.airbrush_illustration",
        "style_medium.watercolor_painting",
        "style_medium.charcoal_drawing",
        "style_medium.pastel_drawing",
        "style_medium.collage_art",
    }), photographic=False),
    "Anime": StyleMediumProfile(frozenset({
        "style_medium.cel_shaded_anime",
        "style_medium.anime_key_visual",
        "style_medium.manga_linework",
        "style_medium.manga_screentone_rendering",
        "style_medium.anime_screencap",
        "style_medium.visual_novel_illustration",
        "style_medium.light_novel_illustration",
        "style_medium.retro_anime_cel_style",
        "style_medium.painterly_anime_illustration",
        "style_medium.flat_color_anime_illustration",
        "style_medium.sakuga_animation_frame",
        "style_medium.anime_game_cg",
    }), photographic=False),
    "Cartoon": StyleMediumProfile(frozenset({
        "style_medium.toon_shading",
        "style_medium.clean_cartoon_linework",
        "style_medium.flat_color_cartoon_illustration",
        "style_medium.vector_cartoon_illustration",
        "style_medium.limited_animation_cel_style",
        "style_medium.saturday_morning_cartoon",
        "style_medium.editorial_cartoon_style",
        "style_medium.webcomic_illustration",
        "style_medium.painterly_cartoon_rendering",
        "style_medium.cutout_animation_style",
        "style_medium.three_dimensional_cartoon_rendering",
        "style_medium.graphic_novel_style",
    }), photographic=False),
    "Fantasy": StyleMediumProfile(frozenset({
        "style_medium.fantasy_illustration",
        "style_medium.high_fantasy_painting",
        "style_medium.dark_fantasy_illustration",
        "style_medium.sword_and_sorcery_illustration",
        "style_medium.mythic_fantasy_painting",
        "style_medium.fairy_tale_illustration",
        "style_medium.fantasy_matte_painting",
    }), photographic=False),
    "Concept art": StyleMediumProfile(frozenset({
        "style_medium.painterly_concept_art_rendering",
        "style_medium.production_concept_art",
        "style_medium.visual_development_painting",
        "style_medium.concept_design_sketch",
        "style_medium.keyframe_concept_art",
        "style_medium.game_concept_art",
        "style_medium.film_concept_art",
    }), photographic=False),
}
STYLE_MEDIUM_PROFILE_IDS = frozenset(
    record_id
    for style_profile in STYLE_MEDIUM_PROFILES.values()
    for record_id in style_profile.primary_ids | style_profile.finish_ids
)
BUILD_CORE_IDS = frozenset({
    "appearance.build_curvy_build", "appearance.build_soft_hourglass_build",
    "appearance.build_voluptuous_build", "appearance.build_rounded_feminine_build",
    "appearance.build_lush_hourglass_build", "appearance.build_softly_rounded_build",
    "appearance.build_generously_curved_build",
}) | frozenset(
    str(record["id"]) for record in CATEGORIES["appearance"]
    if "build_core" in record["semantic_tags"] or record["subcategory"] == "build"
)
BODY_DETAIL_IDS = frozenset(
    str(record["id"]) for record in CATEGORIES["appearance"]
    if "body_detail" in record["semantic_tags"] or record["subcategory"] in {
        "hips", "waist", "body_detail",
    }
)
# Backward-compatible name for callers that enforce the subject contract.
BUILD_CONTRACT_IDS = BUILD_CORE_IDS

SKIN_TONE_IDS = frozenset(
    str(record["id"]) for record in CATEGORIES["appearance"]
    if record["subcategory"] == "skin_tone" or (
        record["subcategory"] == "skin" and
        "texture" not in str(record["canonical_text"]).casefold() and
        "freckled" not in str(record["canonical_text"]).casefold()
    )
)

# Several brown/deep tone records are useful vocabulary variants, but treating
# every alias as a full independent slot makes that visual family more likely
# simply because it has multiple names. Keep the vocabulary available while
# reducing its aggregate selection pressure. This is applied only when choosing
# the required skin-tone concept; it does not weaken a user-selected constraint.
DEEP_SKIN_TONE_IDS = frozenset({
    "appearance.forge_skin_tone_warm_brown_skin",
    "appearance.skin_dark_skin",
    "appearance.skin_deep_brown_skin",
    "appearance.skin_golden_brown_skin",
    "appearance.skin_medium_brown_skin",
}) & SKIN_TONE_IDS
DEEP_SKIN_TONE_SELECTION_WEIGHT = .25


def _skin_tone_selection_weight(record: Mapping[str, object]) -> float:
    return (
        DEEP_SKIN_TONE_SELECTION_WEIGHT
        if str(record["id"]) in DEEP_SKIN_TONE_IDS
        else 1.0
    )


@dataclass(frozen=True)
class ModelProfile:
    id: str
    positive_target: int
    positive_soft_max: int
    positive_hard_max: int
    negative_target: int
    negative_soft_max: int
    negative_hard_max: int
    syntax: str
    score_mode: bool = False
    first_sentence_budget: int = 0
    # Unconditional subject-profile guards. Kept under the historic name so
    # callers that audit age guards continue to see the dispatch requirements.
    negative_baseline: tuple[str, ...] = ()
    model_baseline: tuple[str, ...] = ()


# Activation tokens a checkpoint's author publishes for its own style. Kept
# beside the profiles rather than inside them so the engine treats them as
# controls without a schema change.
# Checkpoints that share a tokenizer and prompt syntax with another, and so
# reuse its per-record token costs and effectiveness scores.
MODEL_DATA_ALIAS: Mapping[str, str] = {
    "flux1_dev": "juggernaut_ragnarok",
    "realvisxl_v5": "juggernaut_ragnarok",
    "stable_yogi_pony": "cyberrealistic_pony",
    "wai_illustrious_v17": "juggernaut_ragnarok",
}

_WAI_DATA = json.loads(
    (Path(__file__).with_name("keyword_data") / "wai_illustrious_v17.json").read_text(
        encoding="utf-8"
    )
)


MODEL_TRIGGER_TOKENS: Mapping[str, tuple[str, ...]] = {
    "stable_yogi_pony": ("99rbsy99",),
}

MODEL_QUALITY_TOKENS: Mapping[str, tuple[str, ...]] = {
    "wai_illustrious_v17": tuple(_WAI_DATA["quality_prefix"]),
}

MODEL_RATING_TOKENS: Mapping[str, Mapping[str, str]] = {
    "wai_illustrious_v17": _WAI_DATA["rating_tags"],
}

PONY_SCORE_TOKENS = ("score_9", "score_8_up", "score_7_up")
PONY_NEGATIVE_SCORE_TOKENS = ("score_6", "score_5", "score_4")

# Required model x image-type foundations. These are deliberately concise:
# the selected image type is stable, while the existing KB remains free to
# vary compatible photographic media, lighting, lenses, composition, and
# treatments. Every entry is emitted as a required P0 AST concept.
MODEL_IMAGE_TYPE_RECIPES: Mapping[tuple[str, str], ImageTypeRecipe] = {
    **{
        ("wai_illustrious_v17", style): ImageTypeRecipe(text)
        for style, text in _WAI_DATA["style_recipes"].items()
    },
    ("realvisxl_v5", "Photoreal"): ImageTypeRecipe(
        "RAW photograph, natural skin pores, fine facial detail, realistic eyes, subtle film grain"
    ),
    ("realvisxl_v5", "Editorial"): ImageTypeRecipe(
        "editorial photograph, natural skin pores, fine facial detail, refined studio lighting"
    ),
    ("realvisxl_v5", "Cinematic"): ImageTypeRecipe(
        "cinematic photograph, natural skin pores, fine facial detail, filmic lighting, subtle film grain"
    ),
    ("cyberrealistic_pony", "Photoreal"): ImageTypeRecipe(
        "photorealistic"
    ),
    ("cyberrealistic_pony", "Editorial"): ImageTypeRecipe(
        "editorial photography, polished controlled composition"
    ),
    ("cyberrealistic_pony", "Cinematic"): ImageTypeRecipe(
        "cinematic film still"
    ),
    ("stable_yogi_pony", "Photoreal"): ImageTypeRecipe(
        "photorealistic, realistic materials"
    ),
    ("stable_yogi_pony", "Editorial"): ImageTypeRecipe(
        "editorial photography, polished controlled composition"
    ),
    ("stable_yogi_pony", "Cinematic"): ImageTypeRecipe(
        "cinematic film still"
    ),
    ("juggernaut_ragnarok", "Photoreal"): ImageTypeRecipe(
        "photographic realism, realistic rendering and photographic lighting"
    ),
    ("juggernaut_ragnarok", "Editorial"): ImageTypeRecipe(
        "editorial photography, polished controlled composition"
    ),
    ("juggernaut_ragnarok", "Cinematic"): ImageTypeRecipe(
        "cinematic film still, cinematic composition"
    ),
}

# Only checkpoint controls live here. General quality/artifact corrections in
# ModelProfile.model_baseline remain optional and are pruned after these.
MODEL_PROTECTED_NEGATIVE_TOKENS: Mapping[str, tuple[str, ...]] = {
    "realvisxl_v5": (
        "bad hands", "bad anatomy", "ugly", "deformed", "face asymmetry",
        "eyes asymmetry", "deformed eyes", "deformed mouth", "open mouth",
        "cgi", "waxy skin", "overly smooth skin",
    ),
    "cyberrealistic_pony": PONY_NEGATIVE_SCORE_TOKENS,
    "stable_yogi_pony": PONY_NEGATIVE_SCORE_TOKENS,
    "wai_illustrious_v17": tuple(_WAI_DATA["protected_negative"]),
}

# These are independent creative dimensions, not garnish on another selected
# concept.  Reserve any legal hard-cap headroom for them before spending the
# soft target on lower-level detail. The image-type recipe guarantees identity;
# a primary medium that fits still provides the requested controlled variation.
# The reservation changes admission only; it does not invent or override KB
# effectiveness/value measurements.
VARIETY_RESERVE_CATEGORIES = frozenset({
    "expression_gaze", "atmosphere_mood",
})


MODEL_PROFILES: Mapping[str, ModelProfile] = {
    "realvisxl_v5": ModelProfile(
        "realvisxl_v5", 67, 70, 75, 35, 55, 70,
        "natural_language", False, 40, PROFILE_NEGATIVE,
        ("bad hands", "bad anatomy", "ugly", "deformed", "face asymmetry",
         "eyes asymmetry", "deformed eyes", "deformed mouth", "open mouth",
         "cgi", "waxy skin", "overly smooth skin"),
    ),
    "wai_illustrious_v17": ModelProfile(
        # The final Forge dispatch boundary enforces one 75-token SDXL CLIP
        # window. Keep compilation on that same ceiling so a valid curated
        # plan cannot be rejected only after it has entered the batch queue.
        "wai_illustrious_v17", 67, 72, 75, 35, 50, 65,
        "booru_tags", False, 0, PROFILE_NEGATIVE, (),
    ),
    "cyberrealistic_pony": ModelProfile(
        "cyberrealistic_pony", 67, 70, 75, 45, 60, 75, "booru_tags", True, 0,
        PROFILE_NEGATIVE,
        ("score_6", "score_5", "score_4", "worst quality", "low quality",
         "lowres", "bad anatomy", "bad hands", "signature", "watermark"),
    ),
    "juggernaut_ragnarok": ModelProfile(
        "juggernaut_ragnarok", 67, 70, 75, 18, 24, 30,
        "natural_language", False, 40, PROFILE_NEGATIVE, (),
    ),
    # FLUX runs at CFG 1.0, where the negative prompt has no effect on the
    # image at all. The age guards are still declared: they are a safety
    # contract every profile owes, not a quality tweak, and the budget has to
    # be wide enough to actually carry them.
    "flux1_dev": ModelProfile(
        "flux1_dev", 67, 70, 75, 18, 24, 30,
        "natural_language", False, 40, PROFILE_NEGATIVE, (),
    ),
    # Same Pony syntax as CyberRealistic, plus the checkpoint's own realism
    # trigger. 99rbsy99 is model-owned like the score tags: it is not
    # descriptive content and must never be traded away for scene detail.
    # Same budgets as CyberRealistic: same family, tokenizer and required
    # content. The serializer appends 99rbsy99 last, and the allocator reserves
    # its room before considering any scene-detail clause.
    "stable_yogi_pony": ModelProfile(
        "stable_yogi_pony", 67, 70, 75, 45, 60, 75, "booru_tags", True, 0,
        PROFILE_NEGATIVE,
        ("score_6", "score_5", "score_4", "worst quality", "low quality",
         "lowres", "bad anatomy", "bad hands", "signature", "watermark"),
    ),
}


class Tokenizer(Protocol):
    """Replaceable measurement boundary for an actual pipeline tokenizer."""

    model_max_length: int

    def count(self, text: str) -> int: ...


class WordTokenizer:
    """Legacy/testing tokenizer; production defaults to :class:`CLIPBPETokenizer`."""

    model_max_length = 77
    _word = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

    def count(self, text: str) -> int:
        return len(self._word.findall(text))


_VIEW_OVERRIDES: Mapping[str, frozenset[str]] = {
    "pose_action.reclining_knees_raised_while_supine": frozenset({"front", "side"}),
    "composition_camera.strict_side_profile": frozenset({"side"}),
    "composition_camera.frontal_view": frozenset({"front"}),
    "composition_camera.three_quarter_view": frozenset({"front"}),
    "composition_camera.rear_tracking_view": frozenset({"rear"}),
    "expression_gaze.direct_gaze": frozenset({"front"}),
    "expression_gaze.steady_eye_contact": frozenset({"front"}),
    "expression_gaze.over_shoulder_gaze": frozenset({"front"}),
}

_REAR_FORBIDDEN_IDS = frozenset({
    "composition_camera.mirror_framing",
    "composition_camera.three_quarter_view",
    "composition_camera.forge_camera_framing_extreme_close_up",
    "composition_camera.forge_camera_framing_face_close_up",
    "composition_camera.forge_camera_angle_view_three_quarter_camera_angle",
    "composition_camera.forge_camera_lens_macro_lens",
    "pose_action.forge_pose_rear_view_rear_view_looking_over_shoulder",
})
_REAR_ROTATION_TERMS = (
    "over shoulder", "turning at waist", "turning while", "turning mid-step",
    "turning beside", "twisting", "face close-up", "three-quarter camera angle",
    "macro lens", "mirror framing",
)


def _rear_restriction_violation(record: Mapping[str, object]) -> bool:
    if str(record["id"]) in _REAR_FORBIDDEN_IDS:
        return True
    if str(record["category"]) != "pose_action":
        return False
    rear_text = str(record.get("view_text", {}).get("rear", record["canonical_text"]))
    return any(term in rear_text.casefold() for term in _REAR_ROTATION_TERMS)


def _allowed_views(record: Mapping[str, object]) -> frozenset[str]:
    """Return the common orientation graph metadata for every visual concept."""
    record_id = str(record["id"])
    if record_id in _VIEW_OVERRIDES:
        views = _VIEW_OVERRIDES[record_id]
    else:
        declared = record.get("allowed_views")
        if declared:
            views = frozenset(str(value) for value in declared)
        else:
            category = str(record["category"])
            subcategory = str(record.get("subcategory") or "")
            if category == "appearance" and subcategory in {"face", "eyes"}:
                views = frozenset({"front", "side"})
            elif category == "expression_gaze":
                views = frozenset({"front", "side"})
            else:
                views = frozenset(orientation.value for orientation in Orientation)
    text = str(record["canonical_text"]).casefold()
    if str(record["category"]) == "pose_action":
        if any(term in text for term in ("toward camera", "towards camera")):
            views = views & {"front"}
        elif any(term in text for term in ("walking away", "facing away", "rear-facing")):
            views = views & {"rear"}
        elif "over shoulder" in text or "over-shoulder" in text:
            views = views - {"side", "rear"}
        elif "supine" in text:
            views = views - {"rear"}
    if "rear" in views and _rear_restriction_violation(record):
        views = views - {"rear"}
    return views


def _environment_affordances(record: Mapping[str, object]) -> frozenset[str]:
    values = {str(value) for value in record.get("affordances", ())}
    text = str(record["canonical_text"]).casefold()

    def contains_any(terms: tuple[str, ...]) -> bool:
        return any(re.search(rf"\b{re.escape(term)}\b", text) for term in terms)
    indoor_terms = (
        "bedroom", "hotel", "studio", "loft", "penthouse", "spa", "indoor",
        "bathhouse", "sauna", "conservatory", "gallery", "theater", "cabaret",
        "library", "rotunda", "observatory", "wine cellar", "archive", "train car",
        "lounge",
    )
    outdoor_terms = (
        "rooftop", "dock", "beach", "cove", "waterfall", "garden", "courtyard",
        "terrace", "balcony", "yacht", "shoreline", "sea cliff", "clearing",
        "orchard", "field", "vineyard", "olive grove", "overlook", "veranda",
        "railway platform", "hot spring", "infinity pool", "cabin deck",
    )
    if any(term in text for term in indoor_terms):
        values.update(("indoor", "interior"))
    if any(term in text for term in outdoor_terms):
        values.add("outdoor")
    if "indoor" in values or "interior" in values:
        values.update(("doorway_available", "window_available"))
    if "bedroom" in text or "hotel suite" in text:
        values.add("bedside")
    if "bar" in text:
        values.add("bar")
    if "lounge" in text:
        values.add("lounge")
    if any(term in text for term in ("restaurant", "dining room")):
        values.add("restaurant")
    if "studio" in text:
        values.add("studio_compatible")
    if any(term in text for term in (
        "dressing room", "wardrobe", "hotel room", "hotel suite", "bedroom",
        "bathroom", "bathhouse", "spa", "studio",
    )):
        values.add("mirror_available")
    if "bathroom" in text:
        values.update(("bathroom", "vanity"))
    if "dressing room" in text or "wardrobe" in text:
        values.update(("dressing_room", "vanity"))
    if "cinema" in text:
        values.update(("cinema", "theater"))
    if "theater" in text:
        values.add("theater")
    if "opera house" in text:
        values.update(("opera_house", "theater"))
    if "stage" in text or "cabaret" in text:
        values.add("stage")
    if "fireplace" in text:
        values.add("fireplace")
    if any(term in text for term in (
        "conservatory", "greenhouse", "sunroom", "observatory dome",
    )):
        values.add("skylight")
    if "skyline" in text or any(term in text for term in (
        "new york", "los angeles", "sydney", "tokyo", "seoul", "hong kong",
        "london", "parisian", "berlin", "milan", "beirut",
    )):
        values.add("city")
    if "skyline" in text:
        values.add("skyline")
    if "rain" in text:
        values.add("rain")
    if "mist" in text:
        values.update(("mist", "haze"))
    if "fog" in text:
        values.update(("fog", "haze"))
    if any(term in text for term in (
        "shower", "spa", "sauna", "bathhouse", "hot spring",
    )):
        values.add("steam")
    if contains_any(("pool", "spring", "bathhouse", "cove", "waterfall", "beach")):
        values.add("water")
    # No current environment is explicitly submerged. This intentionally
    # leaves underwater caustics ineligible instead of turning a spa into a
    # swimming scene.
    return frozenset(values)


_REQUIRED_AFFORDANCES: Mapping[str, frozenset[str]] = {
    "lighting.soft_window_light": frozenset({"wall"}),
    "lighting.hard_window_light": frozenset({"wall"}),
    "lighting.warm_bedside_light": frozenset({"bedside"}),
    "lighting.cool_skylight": frozenset({"skylight"}),
    "lighting.reflected_pool_light": frozenset({"water"}),
    "lighting.underwater_caustics": frozenset({"underwater"}),
    "lighting.forge_lighting_studio_softbox": frozenset({"studio_compatible"}),
    "lighting.forge_lighting_fireplace_glow": frozenset({"fireplace"}),
    "lighting.forge_lighting_soft_skylight": frozenset({"skylight"}),
    "lighting.forge_lighting_light_rays_through_haze": frozenset({"haze"}),
    "lighting.forge_lighting_soft_haze_lighting": frozenset({"haze"}),
    "lighting.forge_lighting_mist_diffused_light": frozenset({"mist"}),
    "lighting.forge_lighting_steam_softened_light": frozenset({"steam"}),
    "lighting.forge_lighting_rain_reflected_neon": frozenset({"rain"}),
    "composition_camera.mirror_framing": frozenset({"mirror_available"}),
    "composition_camera.doorway_framing": frozenset({"doorway_available"}),
    "composition_camera.forge_camera_framing_window_framing": frozenset({
        "window_available",
    }),
}

_STUDIO_LIGHT_TERMS = (
    "softbox", "octabox", "beauty-dish", "beauty dish", "clamshell beauty",
    "ring light", "butterfly lighting",
)


def _is_studio_light(record: Mapping[str, object]) -> bool:
    text = str(record["canonical_text"]).casefold()
    return any(re.search(rf"\b{re.escape(term)}\b", text)
               for term in _STUDIO_LIGHT_TERMS)

_REQUIRED_ANY_AFFORDANCES: Mapping[str, frozenset[str]] = {
    "lighting.forge_lighting_bar_practical_lighting": frozenset({
        "bar", "lounge", "restaurant",
    }),
    "lighting.forge_lighting_cinema_marquee_glow": frozenset({
        "cinema", "marquee", "signage", "theater",
    }),
    "lighting.forge_lighting_soft_vanity_light": frozenset({
        "vanity", "bathroom", "dressing_room",
    }),
    "lighting.forge_lighting_stage_spotlight": frozenset({
        "stage", "theater", "opera_house",
    }),
    "lighting.forge_lighting_theater_spotlight": frozenset({
        "theater", "cinema", "opera_house",
    }),
    "lighting.forge_lighting_city_light_reflections": frozenset({"city", "skyline"}),
}

_DAYLIGHT_TIMES = frozenset({
    "sunrise", "early_morning", "midday", "late_afternoon", "golden_hour",
})
_NIGHT_TIMES = frozenset({"blue_hour", "after_midnight"})


def _environment_time(record: Mapping[str, object]) -> str | None:
    text = str(record["canonical_text"]).casefold()
    markers = (
        ("sunrise", "sunrise"), ("dawn", "sunrise"),
        ("early morning", "early_morning"),
        ("midday", "midday"), ("late afternoon", "late_afternoon"),
        ("golden hour", "golden_hour"), ("sunset", "golden_hour"),
        ("blue hour", "blue_hour"), ("dusk", "blue_hour"),
        ("after midnight", "after_midnight"), ("midnight", "after_midnight"),
        ("moonlit", "after_midnight"), ("at night", "after_midnight"),
        ("sunlit", "midday"),
    )
    return next((value for marker, value in markers if marker in text), None)


def _allowed_times(record: Mapping[str, object]) -> frozenset[str]:
    declared = record.get("allowed_time")
    if declared:
        return frozenset(str(value) for value in declared)
    record_id = str(record["id"])
    if record_id == "lighting.forge_lighting_sunrise_side_light":
        return frozenset({"sunrise", "early_morning"})
    text = str(record["canonical_text"]).casefold()
    if str(record["category"]) == "lighting":
        if "sunrise" in text or "morning" in text:
            return frozenset({"sunrise", "early_morning"})
        if "sunset" in text or "late-afternoon" in text or "late afternoon" in text:
            return frozenset({"late_afternoon", "golden_hour"})
        if "golden-hour" in text or "golden hour" in text:
            return frozenset({"golden_hour", "late_afternoon"})
        if "blue-hour" in text or "blue hour" in text:
            return _NIGHT_TIMES
        if "moon" in text:
            return _NIGHT_TIMES
        if any(term in text for term in ("daylight", "sunlight", "sunlit", "open-shade")):
            return _DAYLIGHT_TIMES
    return frozenset()


def _record_requirements(
    record: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    required = {str(value).removeprefix("affordance.") for value in record.get("requires", ())
                if str(value).startswith("affordance.")}
    required.update(_REQUIRED_AFFORDANCES.get(str(record["id"]), ()))
    forbidden = {str(value).removeprefix("affordance.")
                 for value in record.get("forbidden_affordances", ())}
    if str(record["category"]) == "lighting" and _is_studio_light(record):
        forbidden.add("outdoor")
    required_any = {
        str(value).removeprefix("affordance.") for value in record.get("requires_any", ())
    }
    required_any.update(_REQUIRED_ANY_AFFORDANCES.get(str(record["id"]), ()))
    return (tuple(sorted(required)), tuple(sorted(forbidden)),
            tuple(sorted(required_any)), tuple(sorted(_allowed_times(record))))


def _exposure_state(record: Mapping[str, object]) -> ExposureState:
    """Public builds only admit complete, non-sexual outfits."""
    return ExposureState.CLOTHED


_SAFE_WARDROBE_EXCLUSIONS = (
    "nude", "nudity", "nipples", "genitals", "boudoir", "sheer", "open ",
    "open-", "worn open", "lingerie", "bra", "panties", "mesh",
    "thong", "garter", "stockings", "corset", "bustier", "camisole",
    "slip", "babydoll", "fishnet", "harness", "robe", "backless",
    "high-slit", "crop top", "off-shoulder", "deep-v", "scoop-neck",
    "strapless", "mini skirt", "shorts", "swimsuit", "bikini", "bodysuit",
    "only", "implied", "provocative", "suggestive", "sensual",
)
_SAFE_COMPLETE_OUTFIT_TERMS = ("dress", "gown", "jumpsuit", "tailored suit")


def _safe_wardrobe_record(record: Mapping[str, object]) -> bool:
    """Return whether a KB wardrobe record is suitable for a Safe request."""
    text = str(record["canonical_text"]).casefold()
    semantic_tags = {
        str(value).casefold() for value in record.get("semantic_tags", ())
    }
    allowed_ratings = {
        str(value).upper() for value in record.get("allowed_ratings", ())
    }
    return (
        str(record.get("rating_floor")) == Rating.SAFE.value
        and str(record.get("subcategory")) == "garment"
        and (not allowed_ratings or Rating.SAFE.value in allowed_ratings)
        and _exposure_state(record) is ExposureState.CLOTHED
        and not bool(record.get("safety", {}).get("sexual_content"))
        and any(term in text for term in _SAFE_COMPLETE_OUTFIT_TERMS)
        and not any(term in text for term in _SAFE_WARDROBE_EXCLUSIONS)
    )


_VISIBILITY_RANK = {"face": 0, "upper_body": 1, "three_quarter": 2, "full_body": 3,
                    "any": 3}


def _required_framing(record: Mapping[str, object]) -> str:
    declared = str(record.get("required_framing") or "")
    if declared in _VISIBILITY_RANK:
        return declared
    if str(record["category"]) != "pose_action":
        return "any"
    posture = str(record.get("posture_state") or "")
    text = str(record["canonical_text"]).casefold()
    if posture in {"squatting", "walking", "climbing", "floating"} or any(
        term in text for term in ("all fours", "feet raised", "legs dangling", "knees apart")
    ):
        return "full_body"
    if posture in {"kneeling", "reclining", "bending", "stretching", "railing"} or any(
        term in text for term in ("lying on", "crouch", "perched", "ascending stairs")
    ):
        return "three_quarter"
    return "any"


def _body_visibility(record: Mapping[str, object]) -> str:
    declared = str(record.get("body_visibility") or "")
    if declared in _VISIBILITY_RANK:
        return declared
    if str(record["category"]) != "composition_camera":
        return "any"
    text = str(record["canonical_text"]).casefold()
    if any(term in text for term in ("face close-up", "extreme close-up", "close-up portrait",
                                     "head-and-shoulders", "tight portrait")):
        return "face"
    if any(term in text for term in ("bust portrait", "upper-body", "waist-up")):
        return "upper_body"
    if any(term in text for term in ("three-quarter framing", "three-quarter portrait",
                                     "cowboy shot")):
        return "three_quarter"
    if any(term in text for term in ("full-body", "wide environmental", "environmental portrait")):
        return "full_body"
    return "any"


def _framing_compatible(pose: Mapping[str, object], framing: Mapping[str, object]) -> bool:
    return _VISIBILITY_RANK[_body_visibility(framing)] >= _VISIBILITY_RANK[_required_framing(pose)]


def _realvis_wardrobe_text(record: Mapping[str, object], fallback: str) -> str:
    """Render the selected Safe wardrobe record as natural language."""
    return fallback.strip(" ,.")


@dataclass(frozen=True)
class MaterialBinding:
    concept_id: str
    target: str
    value: str


@dataclass(frozen=True)
class Concept:
    id: str
    category: str
    text: str
    priority: str
    required: bool = False
    allowed_views: frozenset[str] = frozenset({"front", "side", "rear"})
    requires: tuple[str, ...] = ()
    excludes: tuple[str, ...] = ()
    allowed_categories: tuple[str, ...] = ()
    required_affordances: tuple[str, ...] = ()
    forbidden_affordances: tuple[str, ...] = ()
    required_any_affordances: tuple[str, ...] = ()
    allowed_times: tuple[str, ...] = ()
    required_framing: str = "any"
    body_visibility: str = "any"

    @classmethod
    def from_record(
        cls, record: Mapping[str, object], model: str, *, text: str | None = None,
        required: bool = False,
    ) -> "Concept":
        overrides = record["model_text_override"]
        assert isinstance(overrides, Mapping)
        rendered = text or str(overrides.get(model, record["canonical_text"]))
        if model == "realvisxl_v5" and str(record["category"]) == "wardrobe":
            rendered = _realvis_wardrobe_text(record, rendered)
        if str(record["category"]) == "subjects":
            # Confidence/mood is optional expression detail, not part of the
            # reserved age/adult subject contract.
            rendered = re.sub(
                r",\s*(?:confident|poised|sensual|self-assured)\s*$", "", rendered,
                flags=re.IGNORECASE,
            )
            if model == "cyberrealistic_pony":
                rendered = re.sub(
                    r"\badult facial proportions\b",
                    "mature adult facial proportions",
                    rendered,
                    flags=re.IGNORECASE,
                )
        (required_affordances, forbidden_affordances, required_any_affordances,
         allowed_times) = _record_requirements(record)
        return cls(str(record["id"]), str(record["category"]), rendered,
                   str(record["priority_default"]), required,
                   _allowed_views(record),
                   tuple(str(value) for value in record.get("requires", ())),
                   tuple(str(value) for value in record.get("conflicts", ())),
                   tuple(str(value) for value in record.get("allowed_categories", ())),
                   required_affordances, forbidden_affordances,
                   required_any_affordances, allowed_times,
                   _required_framing(record), _body_visibility(record))


@dataclass(frozen=True)
class PromptAST:
    model_id: str
    rating: Rating
    orientation: Orientation
    control: tuple[Concept, ...] = ()
    image_type: tuple[Concept, ...] = ()
    subject: tuple[Concept, ...] = ()
    identity: tuple[Concept, ...] = ()
    appearance: tuple[Concept, ...] = ()
    wardrobe: tuple[Concept, ...] = ()
    performance: tuple[Concept, ...] = ()
    scene: tuple[Concept, ...] = ()
    composition: tuple[Concept, ...] = ()
    lighting: tuple[Concept, ...] = ()
    camera: tuple[Concept, ...] = ()
    rendering: tuple[Concept, ...] = ()
    atmosphere: tuple[Concept, ...] = ()
    style: tuple[Concept, ...] = ()
    finish: tuple[Concept, ...] = ()
    negative_intent: tuple[Concept, ...] = ()
    exposure_state: ExposureState | None = None
    materials: tuple[MaterialBinding, ...] = ()
    style_parent: str = "Photoreal"
    face_visible: bool = True
    hands_prominent: bool = False
    photographic: bool = True

    def positive_concepts(self) -> tuple[Concept, ...]:
        return sum((self.control, self.image_type, self.subject, self.identity, self.appearance,
                    self.wardrobe, self.performance, self.scene,
                    self.composition, self.lighting, self.camera,
                    self.rendering, self.atmosphere, self.style, self.finish), ())


@dataclass(frozen=True)
class ValidationResult:
    decision: Decision
    reason_codes: tuple[ReasonCode, ...]
    score: float
    repaired_ast: PromptAST | None = None


@dataclass(frozen=True)
class CompiledPrompt:
    ast: PromptAST
    positive: str
    negative: str
    positive_tokens: int
    negative_tokens: int
    validation: ValidationResult
    positive_fragments: tuple[Concept, ...] = ()
    negative_fragments: tuple[Concept, ...] = ()
    candidate_scores: tuple[float, ...] = ()
    winner_index: int = 0
    candidate_attempts: int = 1
    valid_candidates: int = 1
    rejected_candidates: int = 0
    repair_candidates: int = 0

    @property
    def generated_candidate_count(self) -> int:
        """Backward-compatible alias for the number of candidates attempted."""
        return self.candidate_attempts

    @property
    def positive_words(self) -> int:
        """Backward-compatible alias; the value is now exact CLIP tokens."""
        return self.positive_tokens

    @property
    def negative_words(self) -> int:
        """Backward-compatible alias; the value is now exact CLIP tokens."""
        return self.negative_tokens


class PromptEngineError(RuntimeError):
    def __init__(self, codes: Sequence[ReasonCode], message: str):
        self.reason_codes = tuple(codes)
        super().__init__(f"{message}: {', '.join(c.value for c in codes)}")


def _visual_reliability(record: Mapping[str, object], model: str,
                        requested_orientation: str | None = None) -> float:
    """Return the checkpoint/pose joint-success weight from the data overlay."""
    if record.get("category") != "pose_action":
        return 1.0
    profiles = record.get("visual_reliability")
    if not isinstance(profiles, Mapping):
        return 1.0
    profile = profiles.get(model)
    if not isinstance(profile, Mapping):
        return 1.0
    orientation_adherence = float(profile.get("orientation_adherence", 1.0))
    requested = profile.get("requested_orientation_adherence")
    if requested_orientation and isinstance(requested, Mapping):
        orientation_adherence = min(
            orientation_adherence, float(requested.get(requested_orientation, 1.0))
        )
    reliability = (float(profile.get("anatomy_success", 1.0)) *
                   float(profile.get("pose_adherence", 1.0)) *
                   orientation_adherence)
    if profile.get("quarantined") is True:
        reliability *= float(profile.get("quarantine_weight", 1.0))
    return reliability


def _record_weight(record: Mapping[str, object], model: str,
                   requested_orientation: str | None = None) -> float:
    # A checkpoint that shares another's tokenizer and syntax reuses its
    # measured costs and effectiveness rather than duplicating 1,923 records.
    data_model = MODEL_DATA_ALIAS.get(model, model)
    effectiveness = record["model_effectiveness"]
    assert isinstance(effectiveness, Mapping)
    cost = record["token_cost"]
    assert isinstance(cost, Mapping)
    token_cost = max(int(cost[data_model]), 1)
    token_efficiency_factor = max(.8, min(1.2, 1.2 - .04 * token_cost))
    return (float(record["semantic_value"]) * float(effectiveness[data_model]) *
            float(record["rarity_weight"]) * token_efficiency_factor *
            _visual_reliability(record, model, requested_orientation))


def _weighted_choice(rng: random.Random, records: Sequence[Mapping[str, object]], model: str,  # noqa: PLR0913
                     diversity: DiversitySnapshot | None = None,
                     selected: Sequence[str] = (), *, budget_sensitive: bool = False,
                     single_slot: bool = False,
                     requested_orientation: str | None = None,
                     profile_weight=None):
    if not records:
        raise PromptEngineError((ReasonCode.EMPTY_CORE,), "no compatible keyword")
    weights = [
        _record_weight(r, model, requested_orientation) *
        ((1 / max(int(r["token_cost"][MODEL_DATA_ALIAS.get(model, model)]), 1)) **
         (.5 if single_slot else 2)
         if budget_sensitive else 1.0) *
        (diversity.recency_weight(str(r["id"])) *
         diversity.combination_weight(selected, str(r["id"])) if diversity else 1.0) *
        # A run-level bias, e.g. weighting exposure poses up for one batch.
        (profile_weight(r) if profile_weight else 1.0)
        for r in records
    ]
    return rng.choices(records, weights=weights, k=1)[0]


_GARMENT_MATERIAL_TERMS = frozenset({
    "silk", "satin", "velvet", "lace", "leather", "denim",
    "linen", "wool", "chiffon", "sequins", "fabric", "mesh",
})
_ENVIRONMENT_FLOOR_TERMS = frozenset({
    "marble", "stone", "tile", "wood", "cedar", "carpet", "pavement", "plaster",
})


def _material_target(record: Mapping[str, object], environment: Mapping[str, object],
                     wardrobe: Mapping[str, object], state: ExposureState,
                     photographic: bool) -> str | None:
    text = str(record["canonical_text"]).casefold()
    affordances = _environment_affordances(environment)
    if any(term in text for term in ("skin", "pores", "sweat", "goosebumps", "water droplets")):
        return "subject.skin"
    if "hair" in text:
        return "subject.hair"
    if any(term in text for term in _GARMENT_MATERIAL_TERMS):
        if state is ExposureState.CLOTHED:
            return "garment"
        return None
    if any(term in text for term in ("filigree", "pearl luster")):
        wardrobe_text = str(wardrobe["canonical_text"]).casefold()
        return "accessory" if any(term in wardrobe_text for term in (
            "jewelry", "chain", "choker", "anklet", "mask", "heels", "gloves", "pearls"
        )) else None
    if any(term in text for term in ("steel", "brass", "bronze", "metal")):
        return "environment.fixture" if "support_surface" in affordances else None
    if "glass" in text or "mirror" in text or "condensation" in text:
        return "environment.wall" if "wall" in affordances else None
    if any(term in text for term in _ENVIRONMENT_FLOOR_TERMS):
        return "environment.floor" if "floor" in affordances else None
    if "sheets" in text:
        return "environment.bed" if "bedside" in affordances else None
    if "water" in text:
        return "environment.water" if "water" in affordances else None
    if "film grain" in text:
        return "image.surface" if photographic else None
    if "haze" in text:
        return "scene.atmosphere"
    return None


_MATERIAL_TARGET_TEXT = {
    "subject.skin": "on her skin", "subject.hair": "in her hair",
    "garment": "on the garment", "accessory": "on her accessory",
    "environment.fixture": "on an environment fixture",
    "environment.wall": "on the environment wall",
    "environment.floor": "on the environment floor",
    "environment.bed": "on the bed", "environment.water": "on the scene water",
    "image.surface": "in the photographic image", "scene.atmosphere": "in the atmosphere",
}


class ConstraintSolver:
    """Build one coherent AST; dependencies are satisfied before selection."""

    FRONT_ONLY_TERMS = (
        "facing viewer", "eye contact", "strict side profile", "frontal view",
        "toward camera", "towards camera",
    )

    def build(self, model: str, seed: int, orientation: Orientation,
              rating: Rating, diversity: DiversitySnapshot | None = None,
              style_parent: str = "Photoreal", *,
              compact_required: bool = False,
              batch_profile: BatchProfile = NEUTRAL,
              allowed_ids: Mapping[str, Collection[str]] | None = None,
              enforce_active_pools: bool = False) -> PromptAST:
        if model not in MODEL_PROFILES:
            raise PromptEngineError((ReasonCode.MODEL_UNSUPPORTED,), model)
        style_profile = STYLE_MEDIUM_PROFILES.get(style_parent)
        if style_profile is None:
            raise PromptEngineError((ReasonCode.STYLE_CONFLICT,), style_parent)
        recipe = MODEL_IMAGE_TYPE_RECIPES.get((model, style_parent))
        rng = random.Random(seed)
        selected: list[str] = []
        allowed = {
            key: frozenset(str(value) for value in values)
            for key, values in (allowed_ids or {}).items()
        }

        def active_compatible(record: Mapping[str, object]) -> bool:
            if not enforce_active_pools:
                return True
            if record.get("status") != "ACTIVE":
                return False
            if str(record["rating_floor"]) != Rating.SAFE.value:
                return False
            effectiveness = record.get("model_effectiveness")
            data_model = MODEL_DATA_ALIAS.get(model, model)
            return (
                isinstance(effectiveness, Mapping)
                and float(effectiveness.get(data_model, 0.0)) > 0
            )

        def constrain(records, facet: str):
            records = tuple(records)
            facet_ids = allowed.get(facet)
            if facet_ids is None:
                return records
            return tuple(record for record in records if str(record["id"]) in facet_ids)

        def explicitly_selected(facet: str, record: Mapping[str, object]) -> bool:
            return str(record["id"]) in allowed.get(facet, ())

        def choose(records, *, budget_sensitive: bool = False,
                   single_slot: bool = False, preserve_pool: bool = False,
                   selection_weight=None):
            records = tuple(records)
            if not records:
                raise PromptEngineError((ReasonCode.EMPTY_CORE,), "no compatible keyword")
            if (compact_required and budget_sensitive and not preserve_pool and
                    MODEL_PROFILES[model].syntax == "booru_tags"):
                minimum_cost = min(int(record["token_cost"][MODEL_DATA_ALIAS.get(model, model)]) for record in records)
                cost_slack = 0 if MODEL_TRIGGER_TOKENS.get(model) else 1
                records = tuple(
                    record for record in records
                    if int(record["token_cost"][MODEL_DATA_ALIAS.get(model, model)]) <= minimum_cost + cost_slack
                )
            def profile_weight(record):
                local_weight = selection_weight(record) if selection_weight else 1.0
                return batch_profile.pose_weight(record) * local_weight
            record = _weighted_choice(
                rng, records, model, diversity, selected,
                budget_sensitive=(budget_sensitive and
                                  MODEL_PROFILES[model].syntax == "booru_tags"),
                single_slot=single_slot,
                requested_orientation=orientation.value,
                profile_weight=profile_weight,
            )
            selected.append(str(record["id"]))
            return record

        def view_compatible(record: Mapping[str, object]) -> bool:
            return orientation.value in _allowed_views(record)

        subject = choose(constrain(
            (
                record for record in CATEGORIES["subjects"]
                if active_compatible(record)
            ),
            "age",
        ), budget_sensitive=True, single_slot=True)

        environment_pool = constrain(
            (record for record in CATEGORIES["environment"] if active_compatible(record)),
            "environment",
        )
        constrained_pose_ids = allowed.get("pose", ())
        if constrained_pose_ids:
            constrained_poses = [BY_ID[value] for value in constrained_pose_ids if value in BY_ID]

            def supports_a_constrained_pose(environment_record: Mapping[str, object]) -> bool:
                environment_affordances = _environment_affordances(environment_record)
                environment_time_value = _environment_time(environment_record)
                for pose_record in constrained_poses:
                    required, forbidden, required_any, times = _record_requirements(pose_record)
                    if not set(required).issubset(environment_affordances):
                        continue
                    if set(forbidden) & set(environment_affordances):
                        continue
                    if required_any and not set(required_any) & set(environment_affordances):
                        continue
                    if environment_time_value and times and environment_time_value not in times:
                        continue
                    if orientation.value not in _allowed_views(pose_record):
                        continue
                    return True
                return False

            environment_pool = tuple(
                record for record in environment_pool
                if supports_a_constrained_pose(record)
            )
        environment = choose(
            environment_pool, budget_sensitive=True, single_slot=True,
        )
        affordances = _environment_affordances(environment)
        environment_time = _environment_time(environment)

        def category_pool(
            category: str, *, subcategory: str | None = None,
            facet: str | None = None,
        ):
            compatible = []
            for record in CATEGORIES[category]:
                if category in {
                    "identity", "appearance", "wardrobe", "pose_action",
                    "expression_gaze", "environment", "composition_camera",
                    "lighting", "atmosphere_mood",
                } and not active_compatible(record):
                    continue
                if subcategory is not None and record["subcategory"] != subcategory:
                    continue
                (required_affordances, forbidden_affordances, required_any_affordances,
                 allowed_times) = _record_requirements(record)
                if not set(required_affordances).issubset(affordances):
                    continue
                if set(forbidden_affordances) & set(affordances):
                    continue
                if required_any_affordances and not set(required_any_affordances) & set(affordances):
                    continue
                if environment_time and allowed_times and environment_time not in allowed_times:
                    continue
                if not view_compatible(record):
                    continue
                allowed_ratings = record.get("allowed_ratings")
                if allowed_ratings and rating.value not in allowed_ratings:
                    continue
                compatible.append(record)
            # A run-level profile may narrow the Safe wardrobe pool further.
            if category == "wardrobe":
                compatible = batch_profile.wardrobe_pool(compatible)
            return list(constrain(compatible, facet)) if facet else compatible

        build = choose(
            constrain(
                (r for r in CATEGORIES["appearance"] if r["id"] in BUILD_CONTRACT_IDS and active_compatible(r)),
                "body_type",
            ),
            budget_sensitive=True,
            single_slot=True,
        )
        skin = choose(
            [r for r in CATEGORIES["appearance"] if r["id"] in SKIN_TONE_IDS],
            budget_sensitive=True,
            single_slot=True,
            # Skin tone is required identity information, not decorative copy.
            # Compact mode may prune optional detail, but must not collapse this
            # pool to whichever labels happen to tokenize most cheaply.
            preserve_pool=True,
            selection_weight=_skin_tone_selection_weight,
        )

        wardrobe_pool = [
            r for r in category_pool("wardrobe", facet="clothing")
            if _safe_wardrobe_record(r)
        ]
        wardrobe = choose(
            wardrobe_pool, budget_sensitive=True, single_slot=True,
        )
        exposure_state = _exposure_state(wardrobe)
        pose = choose(
            (
                record for record in category_pool("pose_action", facet="pose")
                if not record.get("required_exposure_states") or
                exposure_state.value in record["required_exposure_states"]
            ),
            budget_sensitive=True,
            single_slot=True,
        )

        if orientation is Orientation.REAR and allowed.get("expression"):
            raise PromptEngineError(
                (ReasonCode.ORIENTATION_CONFLICT,),
                "the selected expression cannot be guaranteed from a rear view",
            )
        gaze = (choose(category_pool("expression_gaze", facet="expression"))
                if orientation is not Orientation.REAR else None)
        selected_camera_ids = allowed.get("camera", ())
        composition_is_constrained = any(
            value in BY_ID and BY_ID[value].get("subcategory") == "composition"
            for value in selected_camera_ids
        )
        camera_is_constrained = any(
            value in BY_ID and BY_ID[value].get("subcategory") == "camera"
            for value in selected_camera_ids
        )
        composition = choose(
            record for record in category_pool(
                "composition_camera", subcategory="composition",
                facet="camera" if composition_is_constrained else None,
            )
            if _framing_compatible(pose, record)
        )
        camera = choose(category_pool(
            "composition_camera", subcategory="camera",
            facet="camera" if camera_is_constrained else None,
        ))
        lighting_pool = category_pool("lighting", facet="lighting")
        # When the selected environment has purpose-built lighting, use that
        # relationship for automatic prompts.  A generic draw such as
        # ``moonlight`` in a private cinema is individually legal but leaves
        # the checkpoint with two weak, unrelated scene tokens; the result is
        # commonly a plain portrait with neither concept visible. User-selected
        # run-level constraints still win and retain their exact selection.
        if not allowed.get("lighting"):
            scene_bound_lighting = [
                record for record in lighting_pool
                if set(_record_requirements(record)[2]) & set(affordances)
            ]
            if scene_bound_lighting:
                lighting_pool = scene_bound_lighting
        lighting = choose(
            lighting_pool, budget_sensitive=True, single_slot=True,
        )
        primary_medium_ids = style_profile.primary_ids
        if style_parent == "Photoreal":
            primary_medium_ids = (
                RECIPE_PHOTOREAL_PRIMARY_MEDIUM_IDS
                if recipe is not None else PRIMARY_MEDIUM_IDS
            )
        style = choose(
            (r for r in category_pool("style_medium")
             if r["id"] in primary_medium_ids),
            budget_sensitive=True,
            # This category contributes one primary-medium record. Preserve a
            # gentle square-root cost preference, but do not charge its
            # one-time token difference as though the category repeated.
            single_slot=True,
        )
        finish = (
            choose(r for r in category_pool("style_medium")
                   if r["id"] in style_profile.finish_ids)
            if style_profile.finish_ids else None
        )
        photographic = style_profile.photographic

        material_pool: list[tuple[Mapping[str, object], str]] = []
        for record in category_pool("texture_material"):
            target = _material_target(record, environment, wardrobe, exposure_state, photographic)
            if target is not None:
                material_pool.append((record, target))
        material_record = choose(record for record, _ in material_pool)
        material_target = next(target for record, target in material_pool
                               if record["id"] == material_record["id"])
        material_value = str(material_record["canonical_text"])
        material_text = f"{material_value} {_MATERIAL_TARGET_TEXT[material_target]}"
        material = MaterialBinding(str(material_record["id"]), material_target, material_value)

        controls = ()
        if MODEL_PROFILES[model].score_mode:
            rating_control = "control.rating_safe"
            # A declared score scheme is one model-owned P0 unit. No member may
            # be traded away to make room for descriptive detail.
            controls = tuple(
                Concept.from_record(BY_ID[f"control.{token}"], model, required=True)
                for token in PONY_SCORE_TOKENS
            )
            # Pony photographic checkpoints use this model dialect control in
            # addition to the semantic image-type recipe below.
            if photographic and "control.source_photo" in BY_ID:
                controls += (Concept.from_record(
                    BY_ID["control.source_photo"], model, required=True,
                ),)
            controls += (
                Concept.from_record(BY_ID[rating_control], model, required=True),
            )
        controls += tuple(
            Concept(
                f"model_quality.{model}.{index}", "control", token, "P0", True,
            )
            for index, token in enumerate(MODEL_QUALITY_TOKENS.get(model, ()))
        )
        rating_token = MODEL_RATING_TOKENS.get(model, {}).get(rating.value)
        if rating_token:
            controls += (
                Concept(
                    f"model_rating.{model}.{rating.value.casefold()}",
                    "control", rating_token, "P0", True,
                ),
            )
        if (
            rating is Rating.SAFE
            and not MODEL_PROFILES[model].score_mode
            and not rating_token
        ):
            controls += tuple(
                Concept(
                    f"rating.safe.positive.{index}",
                    "rating_semantics",
                    text,
                    "P0",
                    True,
                )
                for index, text in enumerate(SAFE_RATING_POSITIVE)
            )
        controls += tuple(
            Concept(
                f"model_trigger.{model}.{index}", "model_trigger", token, "P0", True,
            )
            for index, token in enumerate(MODEL_TRIGGER_TOKENS.get(model, ()))
        )

        image_type = (
            Concept(
                f"image_type.{model}.{style_parent.casefold()}",
                "image_type_requirement",
                recipe.positive,
                "P0",
                True,
            ),
        ) if recipe is not None else ()


        pose_text = str(pose["view_text"][orientation.value])
        face_visible = orientation is not Orientation.REAR
        hands_prominent = "hand" in pose_text.casefold()
        quality: list[Concept] = []
        if face_visible:
            quality.append(Concept("quality.face", "quality",
                                   "natural facial texture, realistic eyes", "P1"))
        if hands_prominent:
            quality.append(Concept("quality.hands", "quality", "detailed hands", "P1"))
        if photographic:
            quality.append(Concept("quality.photographic", "quality",
                                   "natural skin texture, physically realistic lighting", "P1"))

        positive_text = " ".join((
            str(subject["canonical_text"]), str(build["canonical_text"]),
            str(skin["canonical_text"]), str(wardrobe["canonical_text"]),
            pose_text, str(environment["canonical_text"]), str(style["canonical_text"]),
        )).casefold()
        multiple_subject_negative = BY_ID[MULTIPLE_SUBJECT_NEGATIVE_ID]
        reserved_negative = {
            value.casefold() for value in PROFILE_NEGATIVE + PHOTOGRAPHIC_MEDIUM_NEGATIVE
        } | {str(multiple_subject_negative["canonical_text"]).casefold()}
        scene_negative_pool = [
            record for record in category_pool("negative_exclusions")
            if str(record["canonical_text"]).casefold() not in reserved_negative
            and str(record["canonical_text"]).casefold() not in positive_text
        ]
        scene_negative = rng.sample(scene_negative_pool, min(4, len(scene_negative_pool)))

        appearance_detail_pool = (
            record for record in category_pool("appearance")
            if record["id"] not in (BUILD_CONTRACT_IDS | SKIN_TONE_IDS)
            # Stable Yogi's required skin-tone concept already occupies
            # appearance.skin. Records such as "freckled skin" declare that
            # category mutually exclusive and would otherwise fail only after
            # selection with CARDINALITY_EXCEEDED.
            and not (
                model == "stable_yogi_pony"
                and "appearance.skin" in record["excludes_categories"]
            )
            and not (orientation is Orientation.REAR and any(
                term in str(record["canonical_text"]).casefold()
                for term in ("breast", "nipple", "cleavage", "facing viewer", "eye contact")
            ))
        )
        selected_appearance: list[Concept] = []
        if allowed.get("face_features"):
            face_record = choose(category_pool("appearance", facet="face_features"))
            selected_appearance.append(Concept.from_record(face_record, model, required=True))
        if allowed.get("hair"):
            hair_record = choose(category_pool("appearance", facet="hair"))
            selected_appearance.append(Concept.from_record(hair_record, model, required=True))
        appearance_detail = (
            None if selected_appearance
            else Concept.from_record(choose(appearance_detail_pool), model)
        )
        skin_concept = replace(
            Concept.from_record(skin, model, required=True), priority="P0"
        )

        return PromptAST(
            model, rating, orientation, control=controls, image_type=image_type,
            subject=(Concept.from_record(subject, model, required=True),
                     Concept("subject.solo", "subject_contract",
                             SOLO_SUBJECT_TEXT, "P0", required=True)),
            identity=(Concept.from_record(
                choose(category_pool("identity", facet="heritage")), model,
                required=bool(allowed.get("heritage")),
            ),),
            appearance=(Concept.from_record(build, model, required=True),
                        skin_concept,
                        *selected_appearance,
                        *((appearance_detail,) if appearance_detail else ())),
            wardrobe=(Concept.from_record(wardrobe, model, required=True),),
            performance=(Concept.from_record(pose, model, text=pose_text, required=True),
                         *((Concept.from_record(
                             gaze, model, required=bool(allowed.get("expression")),
                         ),) if gaze else ())),
            scene=(Concept.from_record(environment, model, required=True),),
            composition=(Concept.from_record(
                composition, model,
                required=explicitly_selected("camera", composition),
            ),),
            lighting=(Concept.from_record(lighting, model, required=True),),
            camera=(Concept.from_record(
                camera, model, required=(
                    model == "realvisxl_v5" or explicitly_selected("camera", camera)
                ),
            ),),
            rendering=(Concept.from_record(material_record, model, text=material_text), *quality),
            atmosphere=(Concept.from_record(
                choose(category_pool("atmosphere_mood", facet="mood")), model,
                required=bool(allowed.get("mood")),
            ),),
            style=(replace(
                Concept.from_record(style, model, required=recipe is None),
                category="primary_medium", priority="P0" if recipe is None else "P2",
            ),),
            finish=((replace(Concept.from_record(finish, model),
                             category="finish", priority="P1"),)
                    if finish is not None else ()),
            negative_intent=(Concept.from_record(
                multiple_subject_negative, model, required=True
            ), *(Concept.from_record(r, model) for r in scene_negative)),
            exposure_state=exposure_state, materials=(material,), style_parent=style_parent,
            face_visible=face_visible, hands_prominent=hands_prominent,
            photographic=photographic,
        )


class BudgetAllocator:
    """Fill from required semantics toward a target within the hard ceiling."""

    def __init__(self, tokenizer: Tokenizer):
        self.tokenizer = tokenizer

    @staticmethod
    def _compact_required_concept(concept: Concept) -> Concept:
        """Shorten redundant wording without weakening fixed semantics."""
        text = concept.text
        if concept.category == "subjects":
            age = re.search(r"\b(\d{2})-year-old\b", text)
            if age:
                text = f"{age.group(1)}-year-old adult woman"
        elif concept.category == "identity":
            text = text.casefold()
        elif concept.id == "appearance.build_pregnant_maternity_silhouette":
            text = "pregnant, visible baby bump"
        return replace(concept, text=text) if text != concept.text else concept

    @staticmethod
    def _compact_protected_concept(concept: Concept) -> Concept:
        """Apply a tighter, semantic-preserving form at the hard ceiling only."""
        text = concept.text
        if concept.category == "subjects":
            age = re.search(r"\b(\d{2})-year-old\b", text)
            if age:
                # ``1woman, solo`` immediately follows this fragment, so the
                # numeric age and explicit adult guard remain unambiguous.
                text = f"{age.group(1)}-year-old adult"
        elif concept.category == "pose_action":
            # The orientation remains explicit; "strict" only repeats the
            # solver constraint and can cost a scarce Pony CLIP token.
            text = re.sub(r"^strict side view\b", "side view", text)
            text = re.sub(r"^rear-view\b", "rear view", text)
        elif concept.id in BUILD_CONTRACT_IDS:
            # The subject contract already establishes a woman. Preserve the
            # selected body shape without paying again for gendered wording.
            text = {
                "rounded feminine build": "rounded build",
                "rounded feminine figure": "rounded figure",
                "feminine rounded silhouette": "rounded silhouette",
                "soft full figure": "full figure",
                "plush full figure": "full figure",
                "generously curved build": "curvy build",
                "curvy feminine body": "curvy body",
                "soft curvy body": "curvy body",
                "curvy thick figure": "thick figure",
                "soft thick figure": "thick figure",
                "lush feminine figure": "lush figure",
                "lush mature curves": "lush curves",
                "soft mature curves": "soft curves",
                "softly rounded build": "rounded build",
                "hourglass curves": "hourglass",
                "petite curvy figure": "petite curvy",
                "plush curvy figure": "plush figure",
            }.get(text, text)
        return replace(concept, text=text) if text != concept.text else concept

    @staticmethod
    def _value_score(concept: Concept, model: str) -> float:
        record = BY_ID.get(concept.id)
        if record is None:
            if concept.required:
                return 100.0
            if concept.id.startswith("model."):
                return .9
            if concept.id.startswith("medium."):
                return .8
            if concept.id.startswith("quality."):
                return .75
            return .5
        data_model = MODEL_DATA_ALIAS.get(model, model)
        effectiveness = record["model_effectiveness"]
        cost = record["token_cost"]
        token_efficiency_factor = max(.8, min(1.2, 1.2 - .04 * int(cost[data_model])))
        return (float(record["semantic_value"]) * float(effectiveness[data_model]) *
                float(record["rarity_weight"]) * token_efficiency_factor)

    def allocate(self, concepts: Sequence[Concept], model: str, target: int,
                 ceiling: int, serialize) -> tuple[Concept, ...]:
        # P0 is a semantic/model contract, not an optional scoring hint.
        required = tuple(c for c in concepts if c.required or c.priority == "P0")
        if self.tokenizer.count(serialize(required)) > ceiling:
            required = tuple(self._compact_required_concept(c) for c in required)
            if self.tokenizer.count(serialize(required)) > ceiling:
                required = tuple(self._compact_protected_concept(c) for c in required)
                if self.tokenizer.count(serialize(required)) > ceiling:
                    raise PromptEngineError((ReasonCode.BUDGET_UNRECOVERABLE,),
                                            "required semantic content exceeds budget")
        priorities = {"P0": 4, "P1": 3, "P2": 2, "P3": 1}
        reserve_categories = VARIETY_RESERVE_CATEGORIES
        if model == "juggernaut_ragnarok":
            # Rotate one optional style family (or neither) through scarce
            # hard-cap space. This keeps Ragnarok's medium and finish reachable
            # without letting either monopolize every prompt's remaining room.
            rotation = (None, "primary_medium", "finish")
            rotation_key = "\x1f".join(concept.id for concept in required).encode("utf-8")
            selected_reserve = rotation[
                hashlib.sha256(rotation_key).digest()[0] % len(rotation)
            ]
            if selected_reserve is not None:
                reserve_categories = reserve_categories | {selected_reserve}
        def is_reserved(concept: Concept) -> bool:
            # Body detail is its own visual dimension even though the KB stores
            # it in the broad appearance category beside required build/skin.
            return (
                concept.category in reserve_categories
                or concept.id in BODY_DETAIL_IDS
            )
        # Required content already consumes the Pony soft target in typical
        # prompts.  Without a category reservation, expression and mood are
        # selected for every front-view AST but systematically lose before
        # serialization.  Prefer the cheapest exact serialized addition so
        # scarce hard-cap headroom covers as many independent dimensions as
        # possible.  Ordinary optional detail remains target-bound below.
        reserved = sorted(
            (c for c in concepts
             if not c.required and c.priority != "P0"
             and is_reserved(c)),
            key=lambda c: (
                self.tokenizer.count(serialize((*required, c))),
                c.category,
                c.id,
            ),
        )
        optional = sorted(
            (c for c in concepts
             if not c.required and c.priority != "P0"
             and not is_reserved(c)),
            key=lambda c: (priorities.get(c.priority, 0) * self._value_score(c, model), c.id),
            reverse=True,
        )
        admitted = list(required)
        for concept in reserved:
            trial = admitted + [concept]
            if self.tokenizer.count(serialize(trial)) <= ceiling:
                admitted.append(concept)
        for concept in optional:
            trial = admitted + [concept]
            if self.tokenizer.count(serialize(trial)) <= target:
                admitted.append(concept)
        # Preserve canonical AST order at the compiler boundary.
        admitted_by_id: dict[str, list[Concept]] = {}
        for concept in admitted:
            admitted_by_id.setdefault(concept.id, []).append(concept)
        ordered: list[Concept] = []
        for concept in concepts:
            matches = admitted_by_id.get(concept.id)
            if matches:
                ordered.append(matches.pop(0))
        if self.tokenizer.count(serialize(ordered)) > ceiling:
            raise PromptEngineError((ReasonCode.BUDGET_UNRECOVERABLE,),
                                    "allocated prompt exceeds hard ceiling")
        return tuple(ordered)


def _semantic_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _positive_negative_conflicts(positive: Sequence[Concept],
                                 negative: Sequence[Concept]) -> tuple[str, ...]:
    positive_text = _semantic_text(" ".join(concept.text for concept in positive))
    conflicts: list[str] = []
    for concept in negative:
        term = _semantic_text(concept.text)
        if term and re.search(rf"\b{re.escape(term)}\b", positive_text):
            conflicts.append(concept.id)
    return tuple(conflicts)


class PromptValidator:
    def __init__(self, tokenizer: Tokenizer | None = None):
        self.tokenizer = tokenizer or CLIPBPETokenizer()

    def validate(self, ast: PromptAST, positive: str | None = None,
                 negative: str | None = None, *, novelty: float = 1.0) -> ValidationResult:
        codes: list[ReasonCode] = []
        repair_codes: list[ReasonCode] = []
        records = [BY_ID[c.id] for c in ast.positive_concepts() if c.id in BY_ID]
        if not ast.subject or not ast.scene:
            codes.append(ReasonCode.EMPTY_CORE)
        if len(ast.performance) < 1:
            codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
        postures = {r.get("posture_state") for r in records if r["category"] == "pose_action"}
        if len(postures) > 1:
            codes.append(ReasonCode.CONTRADICTION_HARD)
        positive_concepts = ast.positive_concepts()
        selected = {c.id for c in positive_concepts}
        if any(set(r["conflicts"]) & selected for r in records):
            codes.append(ReasonCode.CONTRADICTION_HARD)
        id_counts = Counter(c.id for c in positive_concepts)
        if any(id_counts[r["id"]] > int(r["max_occurrences"]) for r in records):
            codes.append(ReasonCode.CARDINALITY_EXCEEDED)
        group_counts = Counter()
        for record in records:
            group_counts[str(record["category"])] += 1
            group_counts[f'{record["category"]}.{record["subcategory"]}'] += 1
        if any(group_counts[group] > int(record["max_occurrences"])
               for record in records for group in record["excludes_categories"]
                if not (record["id"] in STYLE_MEDIUM_PROFILE_IDS and
                        group == "style_medium")):
            codes.append(ReasonCode.CARDINALITY_EXCEEDED)
        environments = [r for r in records if r["category"] == "environment"]
        affordances = set().union(*(_environment_affordances(r) for r in environments)) \
            if environments else set()
        for concept in positive_concepts:
            needed_ids = {value for value in concept.requires
                          if not value.startswith("affordance.")}
            if not needed_ids.issubset(selected):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if not set(concept.required_affordances).issubset(affordances):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if set(concept.forbidden_affordances) & affordances:
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if (concept.required_any_affordances and
                    not set(concept.required_any_affordances) & affordances):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            scene_time = _environment_time(environments[0]) if environments else None
            if scene_time and concept.allowed_times and scene_time not in concept.allowed_times:
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if ast.orientation.value not in concept.allowed_views:
                codes.append(ReasonCode.ORIENTATION_CONFLICT)
        poses = [BY_ID[c.id] for c in ast.performance
                 if c.category == "pose_action" and c.id in BY_ID]
        framings = [BY_ID[c.id] for c in ast.composition if c.id in BY_ID]
        if any(not _framing_compatible(pose, framing)
               for pose in poses for framing in framings):
            codes.append(ReasonCode.CONTRADICTION_HARD)
        subject_records = [r for r in records if r["category"] == "subjects"]
        age = subject_records[0]["age_constraints"] if subject_records else {}
        adult = bool(age.get("unambiguous_adult"))
        age_min = age.get("min_age")
        age_max = age.get("max_age")
        builds = tuple(c for c in ast.appearance if c.id in BUILD_CORE_IDS)
        has_build = len(builds) == 1 and builds[0].required and builds[0].priority == "P0"
        subject_text = " ".join(c.text for c in ast.subject + ast.appearance).casefold()
        has_solo_contract = any(
            concept.required and _semantic_text(concept.text) == _semantic_text(SOLO_SUBJECT_TEXT)
            for concept in ast.subject
        )
        subject_ok = (
            adult and isinstance(age_min, int) and isinstance(age_max, int)
            and SUBJECT_CONTRACT["age_min"] <= age_min <= age_max <= SUBJECT_CONTRACT["age_max"]
            and "woman" in subject_text and has_build and has_solo_contract
            and not any(re.search(rf"\b{re.escape(term)}\b", subject_text)
                        for term in SUBJECT_FORBIDDEN)
        )
        if not subject_ok:
            codes.append(ReasonCode.SUBJECT_CONTRACT_VIOLATION)
        if any(r["rating_floor"] != Rating.SAFE.value for r in records):
            codes.append(ReasonCode.RATING_CONFLICT)
        text = " ".join(c.text for c in positive_concepts).casefold()
        if ast.orientation is Orientation.REAR and any(
            term in text for term in ConstraintSolver.FRONT_ONLY_TERMS
        ):
            codes.append(ReasonCode.ORIENTATION_CONFLICT)
        if ast.orientation is Orientation.REAR and (
            ast.face_visible or any(
                concept.id in BY_ID and _rear_restriction_violation(BY_ID[concept.id])
                for concept in positive_concepts
            )
        ):
            codes.append(ReasonCode.ORIENTATION_CONFLICT)

        recipe = MODEL_IMAGE_TYPE_RECIPES.get((ast.model_id, ast.style_parent))
        style_profile = STYLE_MEDIUM_PROFILES.get(ast.style_parent)
        if (style_profile is None or len(ast.style) != 1 or
                any(concept.id not in style_profile.primary_ids or
                    (recipe is None and not concept.required)
                    for concept in ast.style) or
                any(concept.id not in style_profile.finish_ids for concept in ast.finish) or
                (style_profile.finish_required and len(ast.finish) != 1) or
                (not style_profile.finish_required and ast.finish) or
                ast.photographic != style_profile.photographic):
            codes.append(ReasonCode.STYLE_CONFLICT)
        if recipe is not None and (
            len(ast.image_type) != 1
            or not ast.image_type[0].required
            or ast.image_type[0].priority != "P0"
            or _semantic_text(ast.image_type[0].text) != _semantic_text(recipe.positive)
        ):
            codes.append(ReasonCode.STYLE_CONFLICT)
        if not ast.lighting or any(not concept.required for concept in ast.lighting):
            codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)

        wardrobe_records = [BY_ID[c.id] for c in ast.wardrobe if c.id in BY_ID]
        states = {_exposure_state(record) for record in wardrobe_records}
        if len(states) != 1 or ast.exposure_state not in states:
            codes.append(ReasonCode.EXPOSURE_CONFLICT)
        if ast.rating is Rating.SAFE and any(
            not _safe_wardrobe_record(record)
            for record in wardrobe_records
        ):
            codes.append(ReasonCode.RATING_CONFLICT)

        bindings = {binding.concept_id: binding for binding in ast.materials}
        material_concepts = [c for c in ast.rendering if c.category == "texture_material"]
        if len(material_concepts) != len(ast.materials) or not environments or not wardrobe_records:
            codes.append(ReasonCode.MATERIAL_BINDING_INVALID)
        else:
            for concept in material_concepts:
                binding = bindings.get(concept.id)
                expected = _material_target(BY_ID[concept.id], environments[0],
                                            wardrobe_records[0], ast.exposure_state,
                                            ast.photographic)
                if (binding is None or expected is None or binding.target != expected or
                        _MATERIAL_TARGET_TEXT[expected] not in concept.text):
                    codes.append(ReasonCode.MATERIAL_BINDING_INVALID)

        if (not ast.face_visible and any(c.id == "quality.face" for c in ast.rendering)) or (
            not ast.hands_prominent and any(c.id == "quality.hands" for c in ast.rendering)
        ) or (not ast.photographic and any(c.id == "quality.photographic" for c in ast.rendering)):
            codes.append(ReasonCode.CONTRADICTION_HARD)

        negative_keys = [_semantic_text(c.text) for c in ast.negative_intent]
        if len(negative_keys) != len(set(negative_keys)):
            repair_codes.append(ReasonCode.DUPLICATE_NEGATIVE)
        if _positive_negative_conflicts(positive_concepts, ast.negative_intent):
            repair_codes.append(ReasonCode.POSITIVE_NEGATIVE_CONFLICT)
        profile = MODEL_PROFILES.get(ast.model_id)
        if profile is None:
            codes.append(ReasonCode.MODEL_UNSUPPORTED)
        elif positive is not None:
            if self.tokenizer.count(positive) > profile.positive_hard_max:
                codes.append(ReasonCode.BUDGET_UNRECOVERABLE)
            if profile.first_sentence_budget:
                first_sentence = positive.split(".", 1)[0] + "."
                if self.tokenizer.count(first_sentence) > profile.first_sentence_budget:
                    codes.append(ReasonCode.BUDGET_OVERFLOW)
            if any(token not in positive for token in MODEL_TRIGGER_TOKENS.get(ast.model_id, ())):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if (
                recipe is not None
                and _semantic_text(recipe.positive) not in _semantic_text(positive)
            ):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if profile.score_mode:
                if any(token not in positive for token in PONY_SCORE_TOKENS):
                    codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            elif any(token in positive for token in PONY_SCORE_TOKENS):
                codes.append(ReasonCode.STYLE_CONFLICT)
            has_model_rating_token = bool(
                MODEL_RATING_TOKENS.get(ast.model_id, {}).get(ast.rating.value)
            )
            if (
                ast.rating is Rating.SAFE
                and not profile.score_mode
                and not has_model_rating_token
                and any(
                    _semantic_text(value) not in _semantic_text(positive)
                    for value in SAFE_RATING_POSITIVE
                )
            ):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
        if profile is not None and any(
            not any(concept.required and concept.text == token for concept in ast.control)
            for token in MODEL_TRIGGER_TOKENS.get(ast.model_id, ())
        ):
            codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
        if profile is not None and negative is not None:
            if self.tokenizer.count(negative) > profile.negative_hard_max:
                codes.append(ReasonCode.BUDGET_UNRECOVERABLE)
            negative_fragments = [_semantic_text(value) for value in negative.split(",") if value.strip()]
            if len(negative_fragments) != len(set(negative_fragments)):
                repair_codes.append(ReasonCode.DUPLICATE_NEGATIVE)
            positive_text = _semantic_text(positive or "")
            if any(term and re.search(rf"\b{re.escape(term)}\b", positive_text)
                   for term in negative_fragments):
                repair_codes.append(ReasonCode.POSITIVE_NEGATIVE_CONFLICT)
            required_profile = (
                SAFE_PROFILE_NEGATIVE
                if ast.rating is Rating.SAFE
                else profile.negative_baseline
            )
            if any(_semantic_text(value) not in negative_fragments
                   for value in required_profile):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if ast.rating is Rating.SAFE and any(
                _semantic_text(value) not in negative_fragments
                for value in SAFE_RATING_NEGATIVE
            ):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
            if any(_semantic_text(value) not in negative_fragments
                   for value in MODEL_PROTECTED_NEGATIVE_TOKENS.get(ast.model_id, ())):
                codes.append(ReasonCode.REQUIREMENT_UNSATISFIED)
        hard = tuple(dict.fromkeys(codes))
        if hard:
            return ValidationResult(Decision.REJECT, hard, 0.0)
        effectiveness = [float(r["model_effectiveness"][MODEL_DATA_ALIAS.get(ast.model_id, ast.model_id)]) for r in records]
        model_fit = sum(effectiveness) / max(len(effectiveness), 1)
        coherence = 1.0
        controllability = sum(float(r["semantic_value"]) >= .5 for r in records) / max(len(records), 1)
        composition = sum(bool(slot) for slot in (ast.subject, ast.performance, ast.scene,
                                                   ast.composition, ast.lighting)) / 5
        uniqueness = max(0.0, min(1.0, novelty))
        density = min(1.0, sum(float(r["semantic_value"]) for r in records) /
                      max(len(records), 1))
        budget_health = 1.0 if positive is None else max(
            0.0, 1 - self.tokenizer.count(positive) / profile.positive_hard_max * .2)
        negative_quality = 1.0 if ast.negative_intent else .5
        score = (.22 * coherence + .18 * model_fit + .15 * controllability +
                 .12 * density + .10 * composition + .08 * budget_health +
                 .08 * uniqueness + .07 * negative_quality)
        repairs = tuple(dict.fromkeys(repair_codes))
        if repairs:
            return ValidationResult(Decision.REPAIR, repairs, round(score, 4), ast)
        return ValidationResult(Decision.ACCEPT if score >= .72 else Decision.REPAIR,
                                () if score >= .72 else (ReasonCode.QUALITY_BELOW_ACCEPT,),
                                round(score, 4), ast if score < .72 else None)

    def repair(self, ast: PromptAST) -> ValidationResult:
        """Apply legal, lossless-or-optional repairs and return an auditable result.

        Required contradictions are never guessed away; they remain hard rejects.
        """
        issues: list[ReasonCode] = []
        seen: set[str] = set()

        def clean(items: tuple[Concept, ...]) -> tuple[Concept, ...]:
            kept: list[Concept] = []
            for concept in items:
                if concept.id in seen and not concept.required:
                    issues.append(ReasonCode.REDUNDANT_CONCEPT)
                    continue
                seen.add(concept.id)
                kept.append(concept)
            return tuple(kept)

        def clean_negative(items: tuple[Concept, ...]) -> tuple[Concept, ...]:
            kept: list[Concept] = []
            semantic_seen: set[str] = set()
            for concept in items:
                key = _semantic_text(concept.text)
                if key in semantic_seen and not concept.required:
                    issues.append(ReasonCode.DUPLICATE_NEGATIVE)
                    continue
                semantic_seen.add(key)
                kept.append(concept)
            return tuple(kept)

        repaired = replace(
            ast, identity=clean(ast.identity), appearance=clean(ast.appearance),
            wardrobe=clean(ast.wardrobe), composition=clean(ast.composition),
            lighting=clean(ast.lighting), camera=clean(ast.camera),
            rendering=clean(ast.rendering), atmosphere=clean(ast.atmosphere),
            style=clean(ast.style), finish=clean(ast.finish),
            negative_intent=clean_negative(ast.negative_intent),
        )
        conflicts = set(_positive_negative_conflicts(
            repaired.positive_concepts(), repaired.negative_intent
        ))
        if conflicts:
            if any(c.required and c.id in conflicts for c in repaired.negative_intent):
                return ValidationResult(Decision.REJECT,
                                        (ReasonCode.POSITIVE_NEGATIVE_CONFLICT,), 0.0)
            repaired = replace(
                repaired,
                negative_intent=tuple(c for c in repaired.negative_intent
                                      if c.id not in conflicts),
            )
            issues.append(ReasonCode.POSITIVE_NEGATIVE_CONFLICT)
        if len(repaired.performance) > 2:
            optional_tail = tuple(c for c in repaired.performance[2:] if c.required)
            if optional_tail:
                return self.validate(repaired)
            repaired = replace(repaired, performance=repaired.performance[:2])
            issues.append(ReasonCode.CARDINALITY_EXCEEDED)
        view_slots = ("identity", "wardrobe", "performance", "composition", "lighting",
                      "camera", "rendering", "atmosphere", "style", "finish")
        updates: dict[str, tuple[Concept, ...]] = {}
        for slot in view_slots:
            concepts = getattr(repaired, slot)
            if any(c.required and ast.orientation.value not in c.allowed_views for c in concepts):
                return ValidationResult(Decision.REJECT,
                                        (ReasonCode.ORIENTATION_CONFLICT,), 0.0)
            filtered = tuple(c for c in concepts
                             if ast.orientation.value in c.allowed_views)
            if filtered != concepts:
                updates[slot] = filtered
                issues.append(ReasonCode.ORIENTATION_CONFLICT)
        if updates:
            repaired = replace(repaired, **updates)
        final = self.validate(repaired)
        if final.decision is Decision.REJECT:
            return final
        return ValidationResult(Decision.REPAIR if issues else final.decision,
                                tuple(dict.fromkeys(issues)), final.score,
                                repaired if issues else final.repaired_ast)


class BaseAdapter:

    def negative_concepts(self, ast: PromptAST,
                          targeted: Sequence[Concept]) -> tuple[Concept, ...]:
        protected_values = MODEL_PROTECTED_NEGATIVE_TOKENS.get(ast.model_id, ())
        protected = tuple(
            Concept(
                f"model.protected.{i}", "negative_model_control", text, "P0", True,
            )
            for i, text in enumerate(protected_values)
        )
        rating = tuple(
            Concept(
                f"rating.safe.{i}", "negative_rating_control", text, "P0", True,
            )
            for i, text in enumerate(SAFE_RATING_NEGATIVE)
        ) if ast.rating is Rating.SAFE else ()
        required_profile = (
            frozenset(SAFE_PROFILE_NEGATIVE)
            if ast.rating is Rating.SAFE
            else frozenset(self.profile.negative_baseline)
        )
        profile = tuple(
            Concept(
                f"profile.{i}",
                "negative_profile",
                text,
                "P0" if text in required_profile else "P1",
                text in required_profile,
            )
            for i, text in enumerate(self.profile.negative_baseline)
        )
        model = tuple(Concept(f"model.{i}", "negative_model", text, "P1")
                      for i, text in enumerate(self.profile.model_baseline)
                      if text not in protected_values)
        medium = tuple(
            Concept(f"medium.{i}", "negative_medium", text, "P1")
            for i, text in enumerate(PHOTOGRAPHIC_MEDIUM_NEGATIVE)
        ) if ast.photographic else ()
        ordered = protected + rating + profile + model + medium + tuple(targeted)
        unique: list[Concept] = []
        seen: set[str] = set()
        for concept in ordered:
            key = _semantic_text(concept.text)
            if key in seen:
                continue
            seen.add(key)
            unique.append(concept)
        return tuple(unique)
    def __init__(self, tokenizer: Tokenizer, profile: ModelProfile):
        self.tokenizer = tokenizer
        self.profile = profile

    def positive_concepts(self, concepts: Sequence[Concept]) -> tuple[Concept, ...]:
        return tuple(concepts)

    def serialized_positive_concepts(
        self, concepts: Sequence[Concept],
    ) -> tuple[Concept, ...]:
        """Keep checkpoint activation tokens last without making them optional."""
        bound = self.positive_concepts(concepts)
        return tuple(c for c in bound if c.category != "model_trigger") + tuple(
            c for c in bound if c.category == "model_trigger"
        )

    def positive(self, concepts: Sequence[Concept]) -> str:
        raise NotImplementedError

    def negative(self, concepts: Sequence[Concept]) -> str:
        return ", ".join(c.text.strip(" ,.") for c in concepts if c.text)


class CyberRealisticPonyAdapter(BaseAdapter):
    _PROMPT_ORDER = {
        # Keep checkpoint dialect first, then establish what/where/how before
        # spending the later CLIP positions on appearance detail.  This is
        # especially important at Pony's 75-token ceiling: FAI-4971 placed its
        # environment and light at the tail and produced an unrelated studio-
        # like close-up even though Forge received the correct prompt.
        "control": 0,
        "image_type_requirement": 1,
        "subjects": 2,
        "subject_contract": 3,
        "pose_action": 4,
        "environment": 5,
        "lighting": 6,
        "composition_camera": 7,
        "identity": 8,
        "appearance": 9,
        "wardrobe": 10,
        "expression_gaze": 11,
        "atmosphere_mood": 12,
        "primary_medium": 13,
        "finish": 14,
        "texture_material": 15,
        "quality": 16,
    }

    def serialized_positive_concepts(
        self, concepts: Sequence[Concept],
    ) -> tuple[Concept, ...]:
        """Put scene-defining Pony tags before lower-priority appearance tags."""
        bound = self.positive_concepts(concepts)
        triggers = tuple(c for c in bound if c.category == "model_trigger")
        descriptive = tuple(c for c in bound if c.category != "model_trigger")
        ordered = tuple(sorted(
            enumerate(descriptive),
            key=lambda item: (self._PROMPT_ORDER.get(item[1].category, 99), item[0]),
        ))
        return tuple(concept for _, concept in ordered) + triggers

    def positive(self, concepts: Sequence[Concept]) -> str:
        # KB overrides may evolve independently while retaining booru separators.
        return ", ".join(
            c.text.strip(" ,.").replace(" ", "_")
            if c.category == "control" and self.profile.id != "wai_illustrious_v17"
            else c.text.strip(" ,.")
            for c in self.serialized_positive_concepts(concepts) if c.text
        )



class JuggernautRagnarokAdapter(BaseAdapter):
    def first_sentence(self, concepts: Sequence[Concept]) -> str:
        concepts = self.positive_concepts(concepts)
        groups: dict[str, list[str]] = {}
        for c in concepts:
            groups.setdefault(c.category, []).append(c.text.strip(" ,."))
        subject = groups.get("subjects", [])
        action = groups.get("pose_action", [])[:1]
        scene = groups.get("environment", [])
        style = [c.text.strip(" ,.") for c in concepts
                 if c.category == "primary_medium"][:1]
        core = " ".join(subject + action + (["in"] + scene if scene else []))
        recipe = groups.get("image_type_requirement", [])[:1]
        if recipe:
            return f"{recipe[0].capitalize()}: {core}."
        return ((style[0].capitalize() if style else "Photograph") + " of " + core).rstrip() + "."

    def positive(self, concepts: Sequence[Concept]) -> str:
        concepts = self.serialized_positive_concepts(concepts)
        has_recipe = any(c.category == "image_type_requirement" for c in concepts)
        used = {"subjects", "pose_action", "environment", "control", "image_type_requirement"}
        if not has_recipe:
            used.add("primary_medium")
        rest = [c.text.strip(" ,.") for c in concepts
                if c.category not in used]
        return self.first_sentence(concepts) + (" " + ", ".join(rest) + "." if rest else "")


class PromptEngine:
    def __init__(self, tokenizer: Tokenizer | None = None, *,
                 candidate_count: int | None = None, target_valid: int = 8,
                 max_attempts: int = 32,
                 diversity_store: DiversityStore | None = None,
                 diversity_enabled: bool = False):
        if candidate_count is not None:
            target_valid = candidate_count
        if not 4 <= target_valid <= 10:
            raise ValueError("target_valid must be between 4 and 10")
        if max_attempts < target_valid:
            raise ValueError("max_attempts must be at least target_valid")
        self.tokenizer = tokenizer or CLIPBPETokenizer()
        self.target_valid = target_valid
        self.candidate_count = target_valid
        self.max_attempts = max_attempts
        self.solver = ConstraintSolver()
        self.allocator = BudgetAllocator(self.tokenizer)
        self.validator = PromptValidator(self.tokenizer)
        self.diversity_store = diversity_store
        self.diversity_enabled = diversity_enabled

    def generate_candidates(self, model_id: str, seed: int, orientation: Orientation,
                             rating: Rating,
                             diversity: DiversitySnapshot | None = None,
                             style_parent: str = "Photoreal",
                             batch_profile: BatchProfile = NEUTRAL,
                             allowed_ids: Mapping[str, Collection[str]] | None = None,
                             enforce_active_pools: bool = False,
                             ) -> tuple[PromptAST, ...]:
        """Build a deterministic candidate batch from one request seed sequence."""
        rng = random.Random(seed)
        seeds = (seed,) + tuple(rng.getrandbits(63) for _ in range(self.target_valid - 1))
        return tuple(self.solver.build(model_id, candidate_seed, orientation, rating,
                                       diversity, style_parent,
                                       batch_profile=batch_profile,
                                       allowed_ids=allowed_ids,
                                       enforce_active_pools=enforce_active_pools)
                     for candidate_seed in seeds)

    def _repair_and_revalidate(
        self, ast: PromptAST,
    ) -> tuple[PromptAST, ValidationResult, bool]:
        validation = self.validator.validate(ast)
        repaired_candidate = validation.decision is Decision.REPAIR
        if validation.decision is Decision.REPAIR:
            repaired = self.validator.repair(ast)
            if repaired.decision is Decision.REJECT:
                return ast, repaired, repaired_candidate
            ast = repaired.repaired_ast or ast
            validation = self.validator.validate(ast)
        return ast, validation, repaired_candidate

    def _compile_candidate(self, ast: PromptAST, profile: ModelProfile,
                           adapter: BaseAdapter) -> tuple[str, str, tuple[Concept, ...],
                                                         tuple[Concept, ...], ValidationResult,
                                                         bool]:
        positive_concepts = self.allocator.allocate(
            ast.positive_concepts(), ast.model_id, profile.positive_target,
            profile.positive_hard_max, adapter.positive)
        targeted = ast.negative_intent
        # Every profile's baseline applies, not just Pony's. Gating this on the
        # adapter type meant Ragnarok silently dropped its own baseline, so the
        # age guards never reached it.
        targeted = adapter.negative_concepts(ast, targeted)
        negative_concepts = self.allocator.allocate(
            targeted, ast.model_id, profile.negative_target,
            profile.negative_hard_max, adapter.negative)
        positive = adapter.positive(positive_concepts)
        negative = adapter.negative(negative_concepts)
        validation = self.validator.validate(ast, positive, negative)
        repaired_candidate = validation.decision is Decision.REPAIR
        if validation.decision is Decision.REPAIR:
            repaired = self.validator.repair(ast)
            if repaired.decision is Decision.REJECT:
                validation = repaired
            else:
                repaired_ast = repaired.repaired_ast or ast
                validation = self.validator.validate(repaired_ast, positive, negative)
        return (positive, negative, positive_concepts, negative_concepts,
                validation, repaired_candidate)

    def _rank(self, ast: PromptAST, positive: str, validation: ValidationResult,
              uniqueness: float, emitted: Sequence[Concept]) -> float:
        records = [BY_ID[c.id] for c in emitted if c.id in BY_ID]
        profile = MODEL_PROFILES[ast.model_id]
        semantic_value = sum(float(r["semantic_value"]) for r in records) / max(len(records), 1)
        model_effectiveness = sum(float(r["model_effectiveness"][MODEL_DATA_ALIAS.get(ast.model_id, ast.model_id)])
                                  for r in records) / max(len(records), 1)
        rarity = sum(float(r["rarity_weight"]) for r in records) / max(len(records), 1)
        costs = [int(r["token_cost"][MODEL_DATA_ALIAS.get(ast.model_id, ast.model_id)]) for r in records]
        token_efficiency_factor = sum(max(.8, min(1.2, 1.2 - .04 * cost))
                                      for cost in costs) / max(len(costs), 1)
        compatibility = validation.score
        score = (semantic_value * model_effectiveness * compatibility *
                 uniqueness * rarity * token_efficiency_factor)
        # Coherent information density wins. Length is not a quality proxy;
        # only a small penalty applies once the soft maximum is exceeded.
        overflow_penalty = max(0, self.tokenizer.count(positive) - profile.positive_soft_max)
        return round(score - .005 * overflow_penalty, 6)

    def generate(self, model_id: str, seed: int, *, orientation: str | Orientation = "front",
                 rating: str | Rating = "SAFE",
                 style_parent: str = "Photoreal",
                 batch_profile: BatchProfile = NEUTRAL,
                 allowed_ids: Mapping[str, Collection[str]] | None = None,
                 enforce_active_pools: bool = False,
                 ) -> CompiledPrompt:
        orientation = Orientation(orientation)
        rating = Rating(rating)
        profile = MODEL_PROFILES.get(model_id)
        if profile is None:
            raise PromptEngineError((ReasonCode.MODEL_UNSUPPORTED,), model_id)
        store = self.diversity_store
        if self.diversity_enabled and store is None:
            store = DiversityStore()
        diversity = store.load() if store else None
        # Choose by declared syntax, not by model id: a second Pony checkpoint
        # must compile as booru tags rather than falling through to prose.
        adapter: BaseAdapter = (CyberRealisticPonyAdapter(self.tokenizer, profile)
                                if profile.syntax == "booru_tags"
                                else JuggernautRagnarokAdapter(self.tokenizer, profile))
        compiled = []
        failures: list[ReasonCode] = []
        rng = random.Random(seed)
        candidate_seeds = (seed,) + tuple(
            rng.getrandbits(63) for _ in range(self.max_attempts - 1)
        )
        candidate_attempts = 0
        rejected_candidates = 0
        repair_candidates = 0
        for candidate_seed in candidate_seeds:
            candidate_attempts += 1
            try:
                candidate = self.solver.build(
                    model_id, candidate_seed, orientation, rating, diversity, style_parent,
                    compact_required=candidate_attempts > self.max_attempts // 2,
                    batch_profile=batch_profile,
                    allowed_ids=allowed_ids,
                    enforce_active_pools=enforce_active_pools,
                )
            except PromptEngineError as exc:
                failures.extend(exc.reason_codes)
                rejected_candidates += 1
                continue
            candidate, semantic, semantic_repaired = self._repair_and_revalidate(candidate)
            repair_candidates += int(semantic_repaired)
            if semantic.decision is Decision.REJECT:
                failures.extend(semantic.reason_codes)
                rejected_candidates += 1
                continue
            try:
                (positive, negative, pos_concepts, neg_concepts, validation,
                 compile_repaired) = self._compile_candidate(candidate, profile, adapter)
                repair_candidates += int(compile_repaired and not semantic_repaired)
            except PromptEngineError as exc:
                failures.extend(exc.reason_codes)
                rejected_candidates += 1
                continue
            if validation.decision is not Decision.ACCEPT:
                failures.extend(validation.reason_codes)
                rejected_candidates += 1
                continue
            # Diversity must describe what the user can actually see.  Using
            # the pre-allocation AST here made dropped expression/mood records
            # look recent and made otherwise identical serialized prompts look
            # different to candidate ranking.
            concept_ids = tuple(c.id for c in pos_concepts
                                if c.category != "control")
            similarity = diversity.maximum_similarity(concept_ids) if diversity else 0.0
            uniqueness = diversity.uniqueness_score(concept_ids) if diversity else 1.0
            rank = self._rank(candidate, positive, validation, uniqueness, pos_concepts)
            variety_coverage = len({
                concept.category for concept in pos_concepts
                if concept.category in VARIETY_RESERVE_CATEGORIES
            })
            compiled.append((rank, similarity, variety_coverage,
                             concept_ids, candidate, positive, negative,
                             pos_concepts, neg_concepts, validation))
            if len(compiled) >= self.target_valid:
                break
        if len(compiled) < min(4, self.target_valid):
            raise PromptEngineError(tuple(dict.fromkeys(failures)) or
                                    (ReasonCode.QUALITY_BELOW_ACCEPT,),
                                    "fewer than four candidates passed validation")
        scores = tuple(item[0] for item in compiled)
        variety_coverages = tuple(item[2] for item in compiled)
        eligible = [i for i, item in enumerate(compiled)
                    if not diversity or item[1] < diversity.config.similarity_threshold]
        if eligible:
            # Prefer candidates which actually serialize the independent face
            # and mood dimensions.  Keep the measured quality score as the
            # tie-breaker instead of fabricating a numeric semantic bonus.
            winner_index = max(
                eligible,
                key=lambda index: (
                    variety_coverages[index], scores[index], -index,
                ),
            )
        else:
            # A small compatible pool must still produce: select minimum
            # similarity, then visible category coverage and quality.
            winner_index = min(range(len(compiled)),
                               key=lambda index: (
                                   compiled[index][1],
                                   -variety_coverages[index],
                                   -scores[index],
                                   index,
                               ))
        (_, _, _, concept_ids, ast, positive, negative, positive_concepts,
         negative_concepts, validation) = compiled[winner_index]
        positive_tokens = self.tokenizer.count(positive)
        negative_tokens = self.tokenizer.count(negative)
        if positive_tokens > profile.positive_hard_max or negative_tokens > profile.negative_hard_max:
            raise PromptEngineError((ReasonCode.BUDGET_UNRECOVERABLE,),
                                    "final tokenizer check exceeded hard cap")
        if store:
            store.record(concept_ids, model_id, orientation.value)
        return CompiledPrompt(ast, positive, negative, positive_tokens,
                              negative_tokens, validation, positive_concepts,
                              negative_concepts, scores, winner_index,
                              candidate_attempts, len(compiled),
                              rejected_candidates, repair_candidates)


def generate_prompt(model_id: str, seed: int, *, orientation: str = "front",
                    rating: str = "SAFE", tokenizer: Tokenizer | None = None) -> CompiledPrompt:
    """Convenience API for callers which do not need a reusable engine."""
    return PromptEngine(tokenizer).generate(model_id, seed, orientation=orientation, rating=rating)

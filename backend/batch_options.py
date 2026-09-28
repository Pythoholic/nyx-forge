"""UI-safe ForgeBAT facets derived directly from the keyword knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Collection, Mapping

from .batch_common import RATING_LABELS, normalize_content_rating
from .engine_bridge import engine_model_for
from .forge import model_profile_for
from .keywords import CATEGORIES, MANIFEST
from .prompt_engine import BUILD_CONTRACT_IDS, MODEL_DATA_ALIAS, MODEL_PROFILES, _allowed_views


@dataclass(frozen=True, slots=True)
class FacetDefinition:
    key: str
    label: str
    category: str
    subcategories: frozenset[str] = frozenset()


FACETS = (
    FacetDefinition("age", "Age", "subjects"),
    FacetDefinition("heritage", "Country / Region Heritage", "identity", frozenset({"heritage", "nationality"})),
    FacetDefinition("body_type", "Body Type", "appearance", frozenset({"build", "build_core"})),
    FacetDefinition("face_features", "Face / Features", "appearance", frozenset({"face", "face_features", "eyes"})),
    FacetDefinition("hair", "Hair", "appearance", frozenset({
        "hair", "hair_color", "hair_texture", "hair_length", "haircut_layering",
        "hairstyle_ponytail_updo", "hairstyle_braid_protective",
        "hair_styling_finish", "hair_part_bangs", "hair_natural_texture_style",
        "hair_styling_loose",
    })),
    FacetDefinition("clothing", "Clothing / Outfit", "wardrobe"),
    FacetDefinition("pose", "Pose / Action", "pose_action"),
    FacetDefinition("expression", "Facial Expression", "expression_gaze"),
    FacetDefinition("environment", "Environment / Location", "environment"),
    FacetDefinition("camera", "Camera / Framing", "composition_camera"),
    FacetDefinition("lighting", "Lighting", "lighting"),
    FacetDefinition("mood", "Scene Mood", "atmosphere_mood"),
)

FACET_BY_KEY = {facet.key: facet for facet in FACETS}
FACET_FOR_CATEGORY: dict[str, tuple[str, ...]] = {}
for _facet in FACETS:
    FACET_FOR_CATEGORY.setdefault(_facet.category, ())
    FACET_FOR_CATEGORY[_facet.category] += (_facet.key,)

_ORIENTATION_ALIASES = {"front": "front", "side": "side", "back": "rear", "rear": "rear"}


def _friendly_label(record: Mapping[str, Any]) -> str:
    """Return a concise label without exposing the keyword id."""
    text = str(record["canonical_text"]).strip().strip(",.")
    text = re.sub(r"\s+", " ", text)
    return text[:1].upper() + text[1:]


def _model_key(model: str | None) -> str | None:
    if not model:
        return None
    if model in MODEL_PROFILES:
        return MODEL_DATA_ALIAS.get(model, model)
    return engine_model_for(model)


def record_is_available(
    record: Mapping[str, Any], *, model: str | None = None,
    content_rating: str = "Safe", orientation: str = "front",
) -> bool:
    """Apply active/model/rating/view filters without changing the KB."""
    if record.get("status") != "ACTIVE":
        return False
    rating = normalize_content_rating(content_rating)
    if str(record["rating_floor"]) != rating:
        return False
    allowed_ratings = record.get("allowed_ratings")
    if allowed_ratings and rating not in allowed_ratings:
        return False
    resolved_orientation = _ORIENTATION_ALIASES.get(orientation)
    if resolved_orientation is None or resolved_orientation not in _allowed_views(record):
        return False
    model_key = _model_key(model)
    if model and model_key is None:
        return False
    if model_key is not None:
        effectiveness = record.get("model_effectiveness", {})
        if float(effectiveness.get(model_key, 0.0)) <= 0:
            return False
    return True


def facet_records(
    facet_key: str, *, model: str | None = None, content_rating: str = "Safe",
    orientation: str = "front", orientations: Collection[str] | None = None,
    search: str | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Return compatible records for one UI facet from the immutable KB."""
    try:
        facet = FACET_BY_KEY[facet_key]
    except KeyError as exc:
        raise ValueError(f"Unknown batch facet: {facet_key}") from exc
    needle = (search or "").strip().casefold()
    selected_orientations = tuple(dict.fromkeys(orientations or (orientation,)))
    if not selected_orientations or any(value not in _ORIENTATION_ALIASES for value in selected_orientations):
        raise ValueError("Select at least one supported orientation.")
    records = []
    for record in CATEGORIES[facet.category]:
        if facet.subcategories and str(record["subcategory"]) not in facet.subcategories:
            continue
        if facet_key == "body_type" and str(record["id"]) not in BUILD_CONTRACT_IDS:
            continue
        if not all(record_is_available(
            record, model=model, content_rating=content_rating, orientation=value,
        ) for value in selected_orientations):
            continue
        label = _friendly_label(record)
        if needle and needle not in label.casefold() and needle not in str(record["subcategory"]).casefold():
            continue
        records.append(record)
    return tuple(records)


def get_batch_options(
    *, facet: str | None = None, model: str | None = None,
    content_rating: str = "Safe", style: str = "Photoreal",
    orientation: str = "front", orientations: Collection[str] | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    """Build a compact, UI-focused options document.

    Option bodies are returned only for the requested facet. This keeps the
    initial page from downloading most of the 1,923-record knowledge base.
    """
    rating = normalize_content_rating(content_rating)
    supported_styles = model_profile_for(model or "Forge default").styles
    if style not in supported_styles:
        raise ValueError(
            f"Style {style!r} is not supported by the selected model. "
            f"Choose from: {', '.join(supported_styles)}."
        )
    selected_orientations = tuple(dict.fromkeys(orientations or (orientation,)))
    if not selected_orientations or any(value not in _ORIENTATION_ALIASES for value in selected_orientations):
        raise ValueError("Select at least one supported orientation.")
    if model and _model_key(model) is None:
        raise ValueError("The selected model is not supported by the structured prompt engine.")
    if facet is not None and facet not in FACET_BY_KEY:
        raise ValueError(f"Unknown batch facet: {facet}")

    facets: list[dict[str, Any]] = []
    for definition in FACETS:
        records = facet_records(
            definition.key,
            model=model,
            content_rating=rating,
            orientations=selected_orientations,
            search=search if definition.key == facet else None,
        )
        facets.append({
            "key": definition.key,
            "label": definition.label,
            "category": definition.category,
            "description": {
                "heritage": "Choose broad regional heritage or a more specific country / nationality.",
                "expression": "What the subject's face communicates, such as smiling, composed, or intense.",
                "mood": "The overall emotional tone created by the setting, lighting, and color.",
            }.get(definition.key, ""),
            "option_count": len(records),
            "options": [
                {
                    "id": str(record["id"]),
                    "label": _friendly_label(record),
                    "group": {
                        "heritage": "Region / Heritage",
                        "nationality": "Country / Nationality",
                    }.get(str(record["subcategory"]), str(record["subcategory"]).replace("_", " ").title()),
                }
                for record in records
            ] if definition.key == facet else [],
        })

    ages = sorted({
        int(match.group(1))
        for record in CATEGORIES["subjects"]
        if record.get("status") == "ACTIVE"
        for match in [re.search(r"\b(\d{2})-year-old\b", str(record["canonical_text"]))]
        if match and 24 <= int(match.group(1)) <= 44
    })
    return {
        "kb_version": str(MANIFEST["kb_version"]),
        "facets": facets,
        "capabilities": {
            "gender_presentation": [{"id": "woman", "label": "Woman", "locked": True}],
            "ages": ages,
            "styles": list(supported_styles),
            "quality_modes": ["normal", "high", "super", "4k", "8k", "12k"],
            "ratings": [
                {"value": label, "engine_value": value}
                for value, label in RATING_LABELS.items()
            ],
            "orientations": ["front", "side", "back"],
            "distributions": ["balanced", "random"],
        },
    }

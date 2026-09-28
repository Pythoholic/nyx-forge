"""Validated, immutable access to the versioned keyword knowledge base."""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .clip_tokenizer import DEFAULT_TOKENIZER

DATA_DIR = Path(__file__).with_name("keyword_data")
POSE_RELIABILITY_FILE = "pose_reliability.json"
MODELS = frozenset({"cyberrealistic_pony", "juggernaut_ragnarok"})
PRIORITIES = frozenset({"P0", "P1", "P2", "P3"})
RATINGS = frozenset({"SAFE"})
STATUSES = frozenset({"ACTIVE", "EXPERIMENTAL", "DEPRECATED"})
POSITIONS = frozenset({"FRONT", "EARLY", "ANY", "LATE"})
POSTURE_STATES = frozenset({"standing", "sitting", "kneeling", "squatting", "reclining", "walking", "climbing", "floating", "leaning", "railing", "stretching", "bending"})
VIEWS = frozenset({"front", "side", "rear"})
TIMES = frozenset({
    "sunrise", "early_morning", "midday", "late_afternoon", "golden_hour",
    "blue_hour", "after_midnight",
})
AGE_AMBIGUOUS_TERMS = frozenset({"teen", "teenager", "young girl", "girl", "loli", "schoolgirl", "school girl", "barely legal", "youthful", "petite girl", "milf", "mature face", "mature woman"})


class KeywordValidationError(ValueError):
    """The on-disk bundle violates the keyword schema or compatibility model."""


def _tokens(value: str) -> int:
    return DEFAULT_TOKENIZER.count(value)


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _need(record: dict[str, Any], field: str, kind: type) -> Any:
    value = record.get(field)
    if not isinstance(value, kind):
        raise KeywordValidationError(f"{record.get('id', '<unknown>')}: {field} must be {kind.__name__}")
    return value


def validate_bundle(manifest: dict[str, Any], records: list[dict[str, Any]]) -> None:
    """Validate schema, references, budgets, safety and compatibility invariants."""
    ids = [_need(item, "id", str) for item in records]
    duplicates = [item for item, count in Counter(ids).items() if count > 1]
    if duplicates:
        raise KeywordValidationError(f"duplicate ids: {duplicates}")
    known = set(ids)
    budgets = manifest["category_token_budgets"]
    environments = [item for item in records if item["category"] == "environment"]
    available_affordances = {f"affordance.{aff}" for env in environments for aff in env["affordances"]}
    for item in records:
        item_id = item["id"]
        category = _need(item, "category", str)
        if not item_id.startswith(f"{category}."):
            raise KeywordValidationError(f"{item_id}: id/category mismatch")
        for field in ("subcategory", "canonical_text", "priority_default", "rating_floor", "version", "status"):
            _need(item, field, str)
        for field in ("aliases", "semantic_tags", "requires", "conflicts", "excludes_categories"):
            _need(item, field, list)
        if "requires_any" in item:
            _need(item, "requires_any", list)
        if "forbidden_affordances" in item:
            _need(item, "forbidden_affordances", list)
        for field in ("token_cost", "safety", "age_constraints", "identity_metadata", "model_effectiveness", "model_text_override", "model_position_bias", "quality_stats"):
            _need(item, field, dict)
        if item["priority_default"] not in PRIORITIES or item["rating_floor"] not in RATINGS or item["status"] not in STATUSES:
            raise KeywordValidationError(f"{item_id}: invalid enum")
        if item["safety"].get("sexual_content"):
            raise KeywordValidationError(f"{item_id}: sexual content is not allowed in the public bundle")
        if "allowed_time" in item and (
            not isinstance(item["allowed_time"], list) or
            not item["allowed_time"] or
            not set(item["allowed_time"]) <= TIMES
        ):
            raise KeywordValidationError(f"{item_id}: invalid allowed_time")
        if set(item["token_cost"]) != MODELS or set(item["model_effectiveness"]) != MODELS or not set(item["model_text_override"]) <= MODELS or set(item["model_position_bias"]) != MODELS:
            raise KeywordValidationError(f"{item_id}: incomplete model metadata")
        for model in MODELS:
            text = item["model_text_override"].get(model, item["canonical_text"])
            if item["token_cost"][model] != _tokens(text) or item["token_cost"][model] < 1:
                raise KeywordValidationError(f"{item_id}: stale measured token cost for {model}")
            if item["model_position_bias"][model] not in POSITIONS or not 0 <= item["model_effectiveness"][model] <= 1:
                raise KeywordValidationError(f"{item_id}: invalid model metadata for {model}")
        if max(_tokens(item["canonical_text"]), *(item["token_cost"].values())) > budgets[category]:
            raise KeywordValidationError(f"{item_id}: exceeds {category} token budget")
        searchable = json.dumps(item, ensure_ascii=False).casefold()
        for term in AGE_AMBIGUOUS_TERMS:
            if re.search(rf"(?<![a-z]){re.escape(term)}(?![a-z])", searchable):
                raise KeywordValidationError(f"{item_id}: prohibited age-ambiguous vocabulary: {term}")
        requirements = set(item["requires"])
        conflicts = set(item["conflicts"])
        if requirements & conflicts:
            raise KeywordValidationError(f"{item_id}: requirement is also a conflict")
        unresolved = {req for req in requirements if not req.startswith("affordance.") and req not in known}
        if unresolved:
            raise KeywordValidationError(f"{item_id}: unresolved requirements: {sorted(unresolved)}")
        missing_affordances = {req for req in requirements if req.startswith("affordance.")} - available_affordances
        if missing_affordances:
            raise KeywordValidationError(f"{item_id}: unsatisfiable affordances: {sorted(missing_affordances)}")
        required_any = set(item.get("requires_any", ()))
        if required_any and (
            any(not req.startswith("affordance.") for req in required_any) or
            not required_any & available_affordances
        ):
            raise KeywordValidationError(
                f"{item_id}: unsatisfiable requires_any: {sorted(required_any)}"
            )
        forbidden_affordances = set(item.get("forbidden_affordances", ()))
        if any(not value.startswith("affordance.") for value in forbidden_affordances):
            raise KeywordValidationError(
                f"{item_id}: malformed forbidden affordance: "
                f"{sorted(forbidden_affordances)}"
            )
        unresolved_conflicts = conflicts - known
        if unresolved_conflicts:
            raise KeywordValidationError(f"{item_id}: unresolved conflicts: {sorted(unresolved_conflicts)}")
        if category == "subjects" and not (item["age_constraints"].get("unambiguous_adult") and 24 <= item["age_constraints"]["min_age"] <= item["age_constraints"]["max_age"] <= 44):
            raise KeywordValidationError(f"{item_id}: subject is not unambiguously 24-44")
        if category == "identity" and item["identity_metadata"].get("stereotype_autofill") is not False:
            raise KeywordValidationError(f"{item_id}: identity may not autofill appearance")
        if category == "pose_action":
            if item.get("posture_state") not in POSTURE_STATES or set(item.get("allowed_views", ())) != set(item.get("view_text", {})) or not set(item["allowed_views"]) <= VIEWS:
                raise KeywordValidationError(f"{item_id}: invalid posture/view declaration")
            if "pose_action.posture" not in item["excludes_categories"]:
                raise KeywordValidationError(f"{item_id}: posture does not exclude a second posture")
            rear = item["view_text"].get("rear", "").casefold()
            if any(term in rear for term in ("breast", "nipple", "facing viewer", "eye contact")):
                raise KeywordValidationError(f"{item_id}: front-only anatomy in rear realization")
        if category == "wardrobe":
            allowed_views = item.get("allowed_views")
            if not isinstance(allowed_views, list) or not allowed_views or not set(allowed_views) <= VIEWS:
                raise KeywordValidationError(f"{item_id}: invalid wardrobe view declaration")
            if "rear" in allowed_views:
                texts = [item["canonical_text"], *item["model_text_override"].values()]
                if any(term in text.casefold() for text in texts for term in ("breast", "nipple", "topless")):
                    raise KeywordValidationError(f"{item_id}: front-only anatomy in rear realization")


def _reliability_value(entry: Any, label: str) -> float:
    if not isinstance(entry, dict) or set(entry) != {"value", "provenance"}:
        raise KeywordValidationError(f"{label}: expected value and provenance")
    value = entry["value"]
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not 0 <= value <= 1:
        raise KeywordValidationError(f"{label}: reliability value must be between 0 and 1")
    if not isinstance(entry["provenance"], str) or not entry["provenance"]:
        raise KeywordValidationError(f"{label}: reliability provenance is required")
    return float(value)


def _apply_pose_reliability(
    manifest: dict[str, Any], records: list[dict[str, Any]], payload: dict[str, Any],
) -> None:
    """Validate and overlay the separately writable pose-efficacy dataset."""
    if payload.get("schema_version") != "1.0" or payload.get("kb_version") != manifest["kb_version"]:
        raise KeywordValidationError("pose reliability version mismatch")
    if not isinstance(payload.get("data_version"), str):
        raise KeywordValidationError("pose reliability data_version is required")
    defaults = _need(payload, "defaults", dict)
    families = _need(payload, "families", dict)
    metrics = ("anatomy_success", "pose_adherence", "orientation_adherence")
    default_metrics = {
        metric: _reliability_value(defaults.get(metric), f"defaults.{metric}")
        for metric in metrics
    }
    requested_defaults = _need(defaults, "requested_orientation_adherence", dict)
    if set(requested_defaults) != VIEWS:
        raise KeywordValidationError("pose reliability defaults require every orientation")
    default_requested = {
        view: _reliability_value(requested_defaults[view], f"defaults.requested.{view}")
        for view in VIEWS
    }
    if not isinstance(defaults.get("quarantined"), bool):
        raise KeywordValidationError("pose reliability default quarantine must be boolean")
    quarantine_weight = _reliability_value(
        defaults.get("quarantine_weight"), "defaults.quarantine_weight"
    )

    by_id = {record["id"]: record for record in records}
    pose_ids = {record["id"] for record in records if record["category"] == "pose_action"}
    for pose_id in pose_ids:
        by_id[pose_id]["pose_families"] = []
        by_id[pose_id]["visual_reliability"] = {
            model: {
                **default_metrics,
                "requested_orientation_adherence": dict(default_requested),
                "quarantined": bool(defaults["quarantined"]),
                "quarantine_weight": quarantine_weight,
            }
            for model in MODELS
        }

    for family_name, family in families.items():
        if not isinstance(family_name, str) or not isinstance(family, dict):
            raise KeywordValidationError("pose reliability families must be named objects")
        family_pose_ids = _need(family, "pose_ids", list)
        if not family_pose_ids or len(family_pose_ids) != len(set(family_pose_ids)):
            raise KeywordValidationError(f"{family_name}: pose_ids must be unique and non-empty")
        unknown = set(family_pose_ids) - pose_ids
        if unknown:
            raise KeywordValidationError(f"{family_name}: unknown pose ids: {sorted(unknown)}")
        checkpoints = _need(family, "checkpoints", dict)
        if not set(checkpoints) <= MODELS:
            raise KeywordValidationError(f"{family_name}: unknown checkpoint")
        quarantine = _need(family, "quarantine", dict)
        if set(quarantine) != MODELS or any(not isinstance(value, bool)
                                            for value in quarantine.values()):
            raise KeywordValidationError(f"{family_name}: incomplete checkpoint quarantine data")
        for model, checkpoint in checkpoints.items():
            if not isinstance(checkpoint, dict):
                raise KeywordValidationError(f"{family_name}.{model}: invalid checkpoint data")
            unknown_fields = set(checkpoint) - set(metrics) - {"requested_orientation_adherence"}
            if unknown_fields:
                raise KeywordValidationError(
                    f"{family_name}.{model}: unknown fields: {sorted(unknown_fields)}"
                )
            for metric in metrics:
                if metric in checkpoint:
                    _reliability_value(checkpoint[metric], f"{family_name}.{model}.{metric}")
            requested = checkpoint.get("requested_orientation_adherence", {})
            if not isinstance(requested, dict) or not set(requested) <= VIEWS:
                raise KeywordValidationError(f"{family_name}.{model}: invalid orientation data")
            for view, entry in requested.items():
                _reliability_value(entry, f"{family_name}.{model}.{view}")

        for pose_id in family_pose_ids:
            record = by_id[pose_id]
            record["pose_families"].append(family_name)
            for model in MODELS:
                profile = record["visual_reliability"][model]
                checkpoint = checkpoints.get(model, {})
                for metric in metrics:
                    if metric in checkpoint:
                        profile[metric] = min(
                            profile[metric], _reliability_value(
                                checkpoint[metric], f"{family_name}.{model}.{metric}"
                            )
                        )
                for view, entry in checkpoint.get("requested_orientation_adherence", {}).items():
                    profile["requested_orientation_adherence"][view] = min(
                        profile["requested_orientation_adherence"][view],
                        _reliability_value(entry, f"{family_name}.{model}.{view}"),
                    )
                profile["quarantined"] = profile["quarantined"] or quarantine[model]


def load_keyword_base(data_dir: Path = DATA_DIR) -> tuple[Mapping[str, Any], tuple[Mapping[str, Any], ...]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    records: list[dict[str, Any]] = []
    for category, entry in manifest["files"].items():
        payload = json.loads((data_dir / entry["path"]).read_text(encoding="utf-8"))
        if payload["category"] != category or payload["kb_version"] != manifest["kb_version"] or len(payload["records"]) != entry["count"]:
            raise KeywordValidationError(f"{category}: manifest mismatch")
        records.extend(payload["records"])
    validate_bundle(manifest, records)
    reliability_entry = manifest.get("auxiliary_files", {}).get("pose_reliability", {})
    reliability_path = reliability_entry.get("path", POSE_RELIABILITY_FILE)
    reliability = json.loads((data_dir / reliability_path).read_text(encoding="utf-8"))
    if (reliability.get("schema_version") != reliability_entry.get("schema_version") or
            reliability.get("data_version") != reliability_entry.get("data_version")):
        raise KeywordValidationError("pose reliability manifest mismatch")
    _apply_pose_reliability(manifest, records, reliability)
    return _freeze(manifest), tuple(_freeze(item) for item in records)


MANIFEST, KEYWORDS = load_keyword_base()
POSE_RELIABILITY: Mapping[str, Any] = _freeze(json.loads(
    (DATA_DIR / POSE_RELIABILITY_FILE).read_text(encoding="utf-8")
))
BY_ID: Mapping[str, Mapping[str, Any]] = MappingProxyType({item["id"]: item for item in KEYWORDS})
CATEGORIES: Mapping[str, tuple[Mapping[str, Any], ...]] = MappingProxyType({
    category: tuple(item for item in KEYWORDS if item["category"] == category)
    for category in MANIFEST["files"]
})


def get_keyword(keyword_id: str) -> Mapping[str, Any]:
    return BY_ID[keyword_id]


def keywords_for(category: str) -> tuple[Mapping[str, Any], ...]:
    return CATEGORIES.get(category, ())

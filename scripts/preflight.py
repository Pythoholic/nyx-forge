"""Generate application-bound prompts without spending GPU time.

This deliberately uses the same engine bridge, Forge payload builder, payload
serializer, and final dispatch assertion as the running application. It does
not call reForge and it does not write prompt-history or diversity records.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.engine_bridge import generate_compiled
from backend.forge import (
    GenerationSettings, assert_sdxl_dispatch_budget, build_payload,
)
from backend.keywords import BY_ID, CATEGORIES
from backend.prompt_engine import (
    BODY_DETAIL_IDS, BUILD_CORE_IDS, BUILD_CONTRACT_IDS, SKIN_TONE_IDS,
    Decision, ExposureState, FINISH_IDS, MULTIPLE_SUBJECT_NEGATIVE_ID,
    PHOTOGRAPHIC_STYLE_IDS, PRIMARY_MEDIUM_IDS, PROFILE_NEGATIVE,
    SOLO_SUBJECT_TEXT, _environment_affordances, _environment_time,
)

PONY = r"sd\cyberrealisticPony_v180Coreshift.safetensors [9d0e340f5f]"
RAGNAROK = r"sd\juggernautXL_ragnarok.safetensors [dd08fa32f9]"
MODELS = (PONY, RAGNAROK)
ORIENTATIONS = ("front", "side", "back")
AGE_PATTERN = re.compile(r"\b(2[4-9]|3[0-9]|4[0-4])-year-old\b")

ERROR_NAMES = (
    "token_overflow",
    "missing_female_subject",
    "missing_age",
    "missing_body_profile",
    "incompatible_orientation_pose",
    "orientation_camera_contradiction",
    "environment_action_conflict",
    "environment_lighting_contradiction",
    "environment_lighting_affordance_conflict",
    "environment_lighting_time_conflict",
    "composition_affordance_conflict",
    "style_contradiction",
    "positive_negative_contradiction",
    "exposure_wardrobe_contradiction",
    "duplicate_negatives",
    "malformed_material_binding",
    "repair_candidate_dispatched",
)
DISTRIBUTION_NAMES = (
    "identity", "skin", "build_core", "body_detail", "hair",
    "wardrobe", "pose", "environment", "lighting", "camera",
    "primary_medium", "finish",
)


def _normal(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def _negative_fragments(value: str) -> list[str]:
    return [_normal(fragment) for fragment in value.split(",") if fragment.strip()]


def _emitted_ids(compiled) -> set[str]:
    return {concept.id for concept in compiled.positive_fragments}


def _distribution_concepts(compiled) -> dict[str, tuple[Any, ...]]:
    appearance = compiled.ast.appearance

    def tone(concept) -> bool:
        record = BY_ID.get(concept.id)
        if not record:
            return False
        text = concept.text.casefold()
        return record["subcategory"] == "skin_tone" or (
            record["subcategory"] == "skin" and
            "texture" not in text and "freckled" not in text
        )

    return {
        "identity": compiled.ast.identity,
        "skin": tuple(c for c in appearance if tone(c)),
        "build_core": tuple(c for c in appearance if c.id in BUILD_CORE_IDS),
        "body_detail": tuple(c for c in appearance if c.id in BODY_DETAIL_IDS),
        "hair": tuple(c for c in appearance if c.id in BY_ID and
                      str(BY_ID[c.id]["subcategory"]).startswith("hair")),
        "wardrobe": compiled.ast.wardrobe,
        "pose": tuple(c for c in compiled.ast.performance if c.category == "pose_action"),
        "environment": compiled.ast.scene,
        "lighting": compiled.ast.lighting,
        "camera": compiled.ast.composition + compiled.ast.camera,
        # Recipe-backed models express their primary image type through the
        # protected image_type concept. Counting only the optional KB medium
        # falsely reports zero coverage whenever that Tier-3 variation is
        # correctly pruned at Pony's hard token ceiling.
        "primary_medium": compiled.ast.image_type or compiled.ast.style,
        "finish": compiled.ast.finish,
    }


def _pool_size(name: str) -> int:
    if name == "identity":
        return len(CATEGORIES["identity"])
    if name == "skin":
        return sum(record["subcategory"] in {"skin", "skin_tone"}
                   for record in CATEGORIES["appearance"])
    if name == "build_core":
        return len(BUILD_CORE_IDS)
    if name == "body_detail":
        return len(BODY_DETAIL_IDS)
    if name == "hair":
        return sum(str(record["subcategory"]).startswith("hair")
                   for record in CATEGORIES["appearance"])
    category = {
        "pose": "pose_action", "camera": "composition_camera",
        "primary_medium": "style_medium", "finish": "style_medium",
    }.get(name, name)
    if name == "primary_medium":
        return len(PRIMARY_MEDIUM_IDS)
    if name == "finish":
        return len(FINISH_IDS)
    return len(CATEGORIES[category])


def run_preflight(count: int = 10_000) -> dict[str, Any]:
    if count < 1:
        raise ValueError("count must be positive")
    errors = Counter({name: 0 for name in ERROR_NAMES})
    distributions = {name: Counter() for name in DISTRIBUTION_NAMES}
    selected_prompts = Counter()
    emitted_prompts = Counter()
    token_maxima = {"positive": 0, "negative": 0}
    candidate_attempts = Counter()
    candidate_survivors = Counter({value: 0 for value in range(1, 9)})
    candidate_totals = Counter()
    skin_groups = {
        "selected": Counter(),
        "emitted": Counter(),
    }
    reachability = {
        model: {
            "primary_medium": {"selected": Counter(), "emitted": Counter()},
            "finish": {"selected": Counter(), "emitted": Counter()},
        }
        for model in ("cyberrealistic_pony", "juggernaut_ragnarok")
    }

    for index in range(count):
        model = MODELS[index % len(MODELS)]
        orientation = ORIENTATIONS[(index // len(MODELS)) % len(ORIENTATIONS)]
        compiled, used_seed = generate_compiled(
            model, "Safe", orientation, index, style="Photoreal"
        )
        settings = GenerationSettings(
            width=1216,
            height=832,
            model=model,
            rating="Safe",
            style="Photoreal",
            high_res=False,
        )
        payload = build_payload(
            compiled.positive,
            compiled.negative,
            settings,
            seed=used_seed,
            prebuilt=True,
        )
        request_data = payload.as_request(high_res=False)
        try:
            positive_tokens, negative_tokens = assert_sdxl_dispatch_budget(request_data)
        except ValueError:
            errors["token_overflow"] += 1
            positive_tokens = compiled.positive_tokens
            negative_tokens = compiled.negative_tokens
        token_maxima["positive"] = max(token_maxima["positive"], positive_tokens)
        token_maxima["negative"] = max(token_maxima["negative"], negative_tokens)

        positive = str(request_data["prompt"])
        negative = str(request_data["negative_prompt"])
        lowered = positive.casefold()
        emitted = _emitted_ids(compiled)
        candidate_attempts[compiled.candidate_attempts] += 1
        candidate_survivors[compiled.valid_candidates] += 1
        candidate_totals["candidate_attempts"] += compiled.candidate_attempts
        candidate_totals["valid_candidates"] += compiled.valid_candidates
        candidate_totals["rejected_candidates"] += compiled.rejected_candidates
        candidate_totals["repair_candidates"] += compiled.repair_candidates
        if (("woman" not in lowered and "female" not in lowered) or
                _normal(SOLO_SUBJECT_TEXT) not in _normal(positive)):
            errors["missing_female_subject"] += 1
        if AGE_PATTERN.search(lowered) is None:
            errors["missing_age"] += 1
        if not any(value in emitted for value in BUILD_CONTRACT_IDS):
            errors["missing_body_profile"] += 1

        poses = tuple(c for c in compiled.ast.performance if c.category == "pose_action")
        if any(compiled.ast.orientation.value not in c.allowed_views for c in poses):
            errors["incompatible_orientation_pose"] += 1
        view_sensitive = compiled.ast.composition + compiled.ast.camera + tuple(
            c for c in compiled.ast.performance if c.category == "expression_gaze"
        )
        if any(compiled.ast.orientation.value not in c.allowed_views
               for c in view_sensitive):
            errors["orientation_camera_contradiction"] += 1

        # Lighting carries its complete runtime requirements; selected scene
        # compatibility was already solved before this point. Recheck the
        # selected environment using the validator-compatible concept fields.
        scene_record = compiled.ast.scene[0]
        actual_affordances = _environment_affordances(BY_ID[scene_record.id])
        if any(
            not set(action.required_affordances).issubset(actual_affordances) or
            bool(set(action.forbidden_affordances) & actual_affordances) or
            (bool(action.required_any_affordances) and
             not bool(set(action.required_any_affordances) & actual_affordances))
            for action in poses
        ):
            errors["environment_action_conflict"] += 1
        scene_time = _environment_time(BY_ID[scene_record.id])
        light_affordance_conflict = any(
            not set(light.required_affordances).issubset(actual_affordances) or
            bool(set(light.forbidden_affordances) & set(actual_affordances)) or
            (bool(light.required_any_affordances) and
             not bool(set(light.required_any_affordances) & actual_affordances))
            for light in compiled.ast.lighting
        )
        light_time_conflict = any(
            bool(scene_time) and bool(light.allowed_times) and
            scene_time not in light.allowed_times
            for light in compiled.ast.lighting
        )
        if light_affordance_conflict or light_time_conflict:
            errors["environment_lighting_contradiction"] += 1
        if light_affordance_conflict:
            errors["environment_lighting_affordance_conflict"] += 1
        if light_time_conflict:
            errors["environment_lighting_time_conflict"] += 1
        if any(
            not set(concept.required_affordances).issubset(actual_affordances) or
            bool(set(concept.forbidden_affordances) & actual_affordances) or
            (bool(concept.required_any_affordances) and
             not bool(set(concept.required_any_affordances) & actual_affordances))
            for concept in compiled.ast.composition + compiled.ast.camera
        ):
            errors["composition_affordance_conflict"] += 1

        if (compiled.ast.style_parent != "Photoreal" or not compiled.ast.photographic or
                not compiled.ast.style or
                any(c.id not in PHOTOGRAPHIC_STYLE_IDS for c in compiled.ast.style)):
            errors["style_contradiction"] += 1

        negatives = _negative_fragments(negative)
        positive_normal = _normal(positive)
        if any(term and re.search(rf"\b{re.escape(term)}\b", positive_normal)
               for term in negatives):
            errors["positive_negative_contradiction"] += 1
        if len(negatives) != len(set(negatives)):
            errors["duplicate_negatives"] += 1
        if any(_normal(required) not in negatives for required in PROFILE_NEGATIVE):
            errors["missing_body_profile"] += 1
        multiple_subject = BY_ID[MULTIPLE_SUBJECT_NEGATIVE_ID]
        if _normal(str(multiple_subject["canonical_text"])) not in negatives:
            errors["missing_female_subject"] += 1

        if compiled.ast.exposure_state is not ExposureState.CLOTHED:
            errors["exposure_wardrobe_contradiction"] += 1

        bindings = {binding.concept_id: binding for binding in compiled.ast.materials}
        material_concepts = [c for c in compiled.ast.rendering
                             if c.category == "texture_material"]
        if len(bindings) != len(material_concepts) or any(
            c.id not in bindings or not bindings[c.id].target or not bindings[c.id].value
            for c in material_concepts
        ):
            errors["malformed_material_binding"] += 1

        if compiled.validation.decision is not Decision.ACCEPT:
            errors["repair_candidate_dispatched"] += 1

        for category, concepts in _distribution_concepts(compiled).items():
            if concepts:
                selected_prompts[category] += 1
            selected_emitted = [c for c in concepts if c.id in emitted]
            if selected_emitted:
                emitted_prompts[category] += 1
                distributions[category].update(c.id for c in selected_emitted)
        for concept in compiled.ast.appearance:
            if concept.id not in SKIN_TONE_IDS:
                continue
            text = concept.text.casefold()
            if any(term in text for term in (
                "fair", "light", "ivory", "alabaster", "porcelain", "pearl",
                "creamy", "milky", "pale",
            )):
                group = "fair_light_creamy"
            elif "undertone" in text:
                group = "undertone_only"
            elif any(term in text for term in ("brown", "deep", "dark")):
                group = "brown_deep_dark"
            else:
                group = "medium_tan_olive"
            skin_groups["selected"][group] += 1
            if concept.id in emitted:
                skin_groups["emitted"][group] += 1

        model_reachability = reachability[compiled.ast.model_id]
        for name, concepts in (
            ("primary_medium", compiled.ast.style), ("finish", compiled.ast.finish)
        ):
            model_reachability[name]["selected"].update(c.id for c in concepts)
            model_reachability[name]["emitted"].update(
                c.id for c in concepts if c.id in emitted
            )

    report = {
        "generated": count,
        "errors": {name: errors[name] for name in ERROR_NAMES},
        "token_maxima": token_maxima,
        "distribution": {
            name: {
                "pool_size": _pool_size(name),
                "selected_prompts": selected_prompts[name],
                "emitted_prompts": emitted_prompts[name],
                "pruned_prompts": selected_prompts[name] - emitted_prompts[name],
                "selected_percent": round(selected_prompts[name] / count * 100, 2),
                "emitted_percent": round(emitted_prompts[name] / count * 100, 2),
                "pruned_percent_of_selected": round(
                    (selected_prompts[name] - emitted_prompts[name]) /
                    max(selected_prompts[name], 1) * 100, 2
                ),
                "unique_records": len(counter),
                "records": dict(counter.most_common()),
            }
            for name, counter in distributions.items()
        },
        "candidate_survival": {
            "target_valid": 8,
            "max_attempts": 32,
            "candidate_attempts": candidate_totals["candidate_attempts"],
            "valid_candidates": candidate_totals["valid_candidates"],
            "rejected_candidates": candidate_totals["rejected_candidates"],
            "repair_candidates": candidate_totals["repair_candidates"],
            "candidate_attempt_histogram": dict(sorted(candidate_attempts.items())),
            "valid_candidate_histogram": {
                str(value): candidate_survivors[value] for value in range(1, 9)
            },
        },
        "skin_groups": {
            kind: dict(values) for kind, values in skin_groups.items()
        },
        "reachability": {
            model: {
                name: {
                    "pool_size": len(PRIMARY_MEDIUM_IDS if name == "primary_medium" else FINISH_IDS),
                    "eligible": len(PRIMARY_MEDIUM_IDS if name == "primary_medium" else FINISH_IDS),
                    "selected": len(values["selected"]),
                    "emitted": len(values["emitted"]),
                    "selected_records": dict(values["selected"].most_common()),
                    "emitted_records": dict(values["emitted"].most_common()),
                }
                for name, values in families.items()
            }
            for model, families in reachability.items()
        },
    }
    failures = {name: value for name, value in report["errors"].items() if value}
    if failures:
        raise AssertionError(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=10_000)
    args = parser.parse_args()
    report = run_preflight(args.count)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

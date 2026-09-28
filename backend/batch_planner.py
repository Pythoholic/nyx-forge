"""Deterministic, constraint-aware ForgeBAT planning."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import random
from types import MappingProxyType
from typing import Collection, Mapping
from uuid import NAMESPACE_URL, UUID, uuid5

from .engine_bridge import generate_compiled
from .forge import model_profile_for
from .keywords import BY_ID
from .prompt_engine import BUILD_CONTRACT_IDS, CompiledPrompt, PromptEngineError


MAX_DUPLICATE_RETRIES = 5


@dataclass(frozen=True, slots=True)
class BatchPlanItem:
    index: int
    request_id: str
    engine_seed: int
    model: str
    style: str
    content_rating: str
    quality_mode: str
    aspect_ratio: str
    orientation: str
    resolved_facets: tuple[tuple[str, str], ...]
    concept_ids: tuple[str, ...]
    positive_prompt: str
    negative_prompt: str

    def as_dict(self, *, include_prompts: bool = True) -> dict[str, object]:
        value: dict[str, object] = {
            "index": self.index,
            "request_id": self.request_id,
            "engine_seed": self.engine_seed,
            "model": self.model,
            "style": self.style,
            "content_rating": self.content_rating,
            "quality_mode": self.quality_mode,
            "aspect_ratio": self.aspect_ratio,
            "orientation": self.orientation,
            "resolved_facets": dict(self.resolved_facets),
            "concept_ids": list(self.concept_ids),
        }
        if include_prompts:
            value.update({
                "positive_prompt": self.positive_prompt,
                "negative_prompt": self.negative_prompt,
            })
        return value


@dataclass(frozen=True, slots=True)
class BatchPlan:
    request_id: str
    planner_seed: int
    items: tuple[BatchPlanItem, ...]
    coverage: tuple[tuple[str, tuple[tuple[str, int], ...]], ...]
    warnings: tuple[str, ...] = ()

    def as_dict(self, *, preview_limit: int | None = None) -> dict[str, object]:
        visible = self.items if preview_limit is None else self.items[:preview_limit]
        return {
            "request_id": self.request_id,
            "planner_seed": self.planner_seed,
            "count": len(self.items),
            "items": [item.as_dict(include_prompts=False) for item in visible],
            "coverage": {
                facet: [
                    {
                        "id": concept_id,
                        "label": str(BY_ID.get(concept_id, {}).get("canonical_text", concept_id)),
                        "count": count,
                    }
                    for concept_id, count in values
                ]
                for facet, values in self.coverage
            },
            "warnings": list(self.warnings),
        }


def _namespace(request_id: str) -> UUID:
    try:
        return UUID(request_id)
    except ValueError:
        return uuid5(NAMESPACE_URL, request_id)


def _derived_seed(planner_seed: int, index: int, attempt: int) -> int:
    digest = hashlib.blake2b(
        f"{planner_seed}:{index}:{attempt}".encode("ascii"), digest_size=8,
    ).digest()
    return int.from_bytes(digest, "big") % (2**31)


def _scheduled_choices(
    count: int, allowed_ids: Mapping[str, Collection[str]], distribution: str,
    planner_seed: int,
) -> tuple[Mapping[str, str], ...]:
    rng = random.Random(planner_seed)
    columns: dict[str, list[str]] = {}
    for facet in sorted(allowed_ids):
        pool = sorted(set(allowed_ids[facet]))
        if not pool:
            continue
        if distribution == "balanced":
            offset = rng.randrange(len(pool))
            values = [pool[(index + offset) % len(pool)] for index in range(count)]
            rng.shuffle(values)
        elif distribution == "random":
            values = [rng.choice(pool) for _ in range(count)]
        else:
            raise ValueError(f"Unsupported batch distribution: {distribution}")
        columns[facet] = values
    return tuple(MappingProxyType({facet: values[index] for facet, values in columns.items()})
                 for index in range(count))


def _scheduled_pool_values(
    count: int, pool: tuple[str, ...], distribution: str, seed: int,
) -> list[str]:
    rng = random.Random(seed)
    if distribution == "balanced":
        offset = rng.randrange(len(pool))
        values = [pool[(index + offset) % len(pool)] for index in range(count)]
        rng.shuffle(values)
        return values
    return [rng.choice(pool) for _ in range(count)]


def _resolved_facets(compiled: CompiledPrompt, scheduled: Mapping[str, str]) -> dict[str, str]:
    ast = compiled.ast
    resolved = dict(scheduled)
    if ast.subject:
        resolved.setdefault("age", ast.subject[0].id)
    if ast.identity:
        resolved.setdefault("heritage", ast.identity[0].id)
    build = next((concept.id for concept in ast.appearance if concept.id in BUILD_CONTRACT_IDS), None)
    if build:
        resolved.setdefault("body_type", build)
    if ast.wardrobe:
        resolved.setdefault("clothing", ast.wardrobe[0].id)
    if ast.performance:
        resolved.setdefault("pose", ast.performance[0].id)
        if len(ast.performance) > 1:
            resolved.setdefault("expression", ast.performance[1].id)
    if ast.scene:
        resolved.setdefault("environment", ast.scene[0].id)
    if ast.composition:
        resolved.setdefault("camera", ast.composition[0].id)
    if ast.lighting:
        resolved.setdefault("lighting", ast.lighting[0].id)
    if ast.atmosphere:
        resolved.setdefault("mood", ast.atmosphere[0].id)
    return resolved


def build_batch_plan(
    *, count: int, allowed_ids: Mapping[str, Collection[str]], model: str,
    content_rating: str, style: str, orientation: str, distribution: str,
    planner_seed: int, request_id: str, quality_mode: str, aspect_ratio: str,
    orientations: Collection[str] | None = None,
    styles: Collection[str] | None = None,
) -> BatchPlan:
    """Compile exactly ``count`` deterministic, replayable batch recipes."""
    if count < 1:
        raise ValueError("Batch count must be positive.")
    orientation_pool = tuple(dict.fromkeys(orientations or (orientation,)))
    if not orientation_pool or any(value not in {"front", "side", "back"} for value in orientation_pool):
        raise ValueError("Select at least one supported orientation.")
    style_pool = tuple(dict.fromkeys(styles or (style,)))
    supported_styles = model_profile_for(model).styles
    if not style_pool or any(value not in supported_styles for value in style_pool):
        raise ValueError(
            f"Select at least one style supported by this model: {', '.join(supported_styles)}."
        )
    schedule = _scheduled_choices(count, allowed_ids, distribution, planner_seed)
    item_orientations = _scheduled_pool_values(
        count, orientation_pool, distribution, planner_seed ^ 0x4F5249454E54,
    )
    item_styles = _scheduled_pool_values(
        count, style_pool, distribution, planner_seed ^ 0x5354594C45,
    )
    namespace = _namespace(request_id)
    items: list[BatchPlanItem] = []
    prompt_fingerprints: set[str] = set()
    recipe_fingerprints: set[tuple[tuple[str, str], ...]] = set()
    constrained = False

    for index, choices in enumerate(schedule):
        item_orientation = item_orientations[index]
        item_style = item_styles[index]
        chosen: BatchPlanItem | None = None
        last_compile_error: PromptEngineError | None = None
        for attempt in range(MAX_DUPLICATE_RETRIES + 1):
            engine_seed = _derived_seed(planner_seed, index, attempt)
            constraints = {facet: {concept_id} for facet, concept_id in choices.items()}
            try:
                compiled, _ = generate_compiled(
                    model,
                    content_rating,
                    item_orientation,
                    seed=engine_seed,
                    style=item_style,
                    constraints=constraints,
                    enforce_active_pools=True,
                )
            except PromptEngineError as exc:
                # Some otherwise valid recipe seeds cannot fit the selected
                # model's prompt budget. Keep planning with the next stable
                # derived seed instead of failing the entire batch preview.
                last_compile_error = exc
                continue
            concept_ids = tuple(
                concept.id for concept in compiled.positive_fragments
                if concept.category != "control"
            )
            missing = [concept_id for concept_id in choices.values() if concept_id not in {
                concept.id for concept in compiled.ast.positive_concepts()
            }]
            if missing:
                raise PromptEngineError(
                    (), f"the compiler did not preserve explicit selections: {missing}",
                )
            facets = tuple(sorted(_resolved_facets(compiled, choices).items()))
            candidate = BatchPlanItem(
                index=index,
                request_id=str(uuid5(namespace, f"item:{index}")),
                engine_seed=engine_seed,
                model=model,
                style=item_style,
                content_rating=content_rating,
                quality_mode=quality_mode,
                aspect_ratio=aspect_ratio,
                orientation=item_orientation,
                resolved_facets=facets,
                concept_ids=concept_ids,
                positive_prompt=compiled.positive,
                negative_prompt=compiled.negative,
            )
            prompt_key = hashlib.sha256(compiled.positive.encode("utf-8")).hexdigest()
            duplicate = prompt_key in prompt_fingerprints or facets in recipe_fingerprints
            chosen = candidate
            if not duplicate:
                prompt_fingerprints.add(prompt_key)
                recipe_fingerprints.add(facets)
                break
            if attempt == MAX_DUPLICATE_RETRIES:
                constrained = True
                prompt_fingerprints.add(prompt_key)
                recipe_fingerprints.add(facets)
        if chosen is None:
            assert last_compile_error is not None
            raise last_compile_error
        items.append(chosen)

    coverage: dict[str, Counter[str]] = {}
    for item in items:
        coverage.setdefault("orientation", Counter())[item.orientation] += 1
        for facet, concept_id in item.resolved_facets:
            coverage.setdefault(facet, Counter())[concept_id] += 1
    warning = (
        "The selected search space is constrained; some valid recipes may repeat.",
    ) if constrained else ()
    return BatchPlan(
        request_id=request_id,
        planner_seed=planner_seed,
        items=tuple(items),
        coverage=tuple(
            (facet, tuple(sorted(counts.items())))
            for facet, counts in sorted(coverage.items())
        ),
        warnings=warning,
    )

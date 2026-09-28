"""Translate app-level request values into prompt-engine calls.

The engine keys its profiles by a short model id and names the rear view
"rear"; the app passes a checkpoint filename and says "back". This module owns
that translation so neither side has to know about the other.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import re
import secrets
import random
from typing import Collection, Iterable, Mapping

from .forge import MODEL_PROFILES as FORGE_MODEL_PROFILES, model_profile_for
from .diversity import (
    DiversitySnapshot,
    DiversityStore,
    concept_fingerprint,
)
from .batch_profile import profile_for
from .batch_common import RATING_LABELS, normalize_content_rating
from .keywords import BY_ID
from .prompt_engine import (
    MODEL_PROFILES, CompiledPrompt, Concept, Orientation, PromptEngine,
    PromptEngineError, Rating, ReasonCode, STYLE_MEDIUM_PROFILES,
    _safe_wardrobe_record,
)
from .prompt_semantics import (
    CanonicalPromptSpec, PromptFacet, SemanticCatalogItem, SemanticPromptContext,
    SemanticPromptError, SemanticPromptResponse, SemanticSelection,
)
from .novelty import (
    NoveltyContext,
    PromptHistoryItem,
    create_novelty_context,
    maximum_similarity,
)

# Engine profiles exist per checkpoint family, not per checkpoint file.
_FAMILY_TO_ENGINE_MODEL = {
    "illustrious": "wai_illustrious_v17",
    "sdxl_realvis": "realvisxl_v5",
    "flux": "flux1_dev",
    # Base Pony XL shares CyberRealistic's dialect: the same score tags, the
    # same comma-separated tag grammar and the same tokenizer. Without this
    # entry the family resolves to nothing and every curated request for
    # Pony Diffusion V6 XL fails with MODEL_UNSUPPORTED.
    "pony": "cyberrealistic_pony",
    "pony_realism": "cyberrealistic_pony",
    "sdxl": "juggernaut_ragnarok",
}
_STABLE_YOGI_FORGE_PROFILE = FORGE_MODEL_PROFILES["realismbystableyogiponyv65"]

_ORIENTATIONS = {"front": "front", "side": "side", "back": "rear", "rear": "rear"}

_ENGINE = PromptEngine(diversity_store=DiversityStore(), diversity_enabled=True)
_REPLAY_ENGINE = PromptEngine(diversity_enabled=False)

_SEMANTIC_REPAIR_ORDER = (
    PromptFacet.CAMERA,
    PromptFacet.LIGHTING,
    PromptFacet.POSE,
    PromptFacet.CLOTHING,
    PromptFacet.ENVIRONMENT,
    PromptFacet.EXPRESSION,
    PromptFacet.MOOD,
    PromptFacet.FACE_FEATURES,
    PromptFacet.HAIR,
    PromptFacet.HERITAGE,
    PromptFacet.BODY_TYPE,
    PromptFacet.AGE,
)

_CURATED_FACETS = (
    PromptFacet.CAMERA,
    PromptFacet.LIGHTING,
    PromptFacet.POSE,
    PromptFacet.ENVIRONMENT,
    PromptFacet.MOOD,
)
_CURATED_CANDIDATE_COUNT = 6
_CURATED_FACET_LIMIT = 18


@dataclass(frozen=True, slots=True)
class SemanticCompiledPrompt:
    """Validated engine output and its normalized provider proposal."""

    compiled: CompiledPrompt
    semantic: CanonicalPromptSpec
    repair_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CuratedCompiledPrompt:
    """History-aware curated selection compiled entirely by Prompt Engine V1."""

    compiled: CompiledPrompt
    novelty: NoveltyContext
    seed: int
    similarity_score: float
    attempt_count: int
    concept_similarity: float
    canonical_concept_ids: tuple[str, ...]
    repair_notes: tuple[str, ...] = ()
    novelty_signature: str = ""


class _SnapshotDiversityStore:
    """Expose one engine diversity snapshot without recording rejected draws."""

    def __init__(self, snapshot: DiversitySnapshot):
        self.snapshot = snapshot

    def load(self) -> DiversitySnapshot:
        return self.snapshot

    def record(
        self, concept_ids: Iterable[str], model: str, orientation: str,
    ) -> str:
        del model, orientation
        return concept_fingerprint(concept_ids)


def engine_model_for(forge_model: str) -> str | None:
    """Return the engine profile id for a checkpoint, or None if unsupported."""
    forge_profile = model_profile_for(forge_model)
    if forge_profile is _STABLE_YOGI_FORGE_PROFILE:
        return "stable_yogi_pony"
    family = forge_profile.family
    engine_model = _FAMILY_TO_ENGINE_MODEL.get(family)
    return engine_model if engine_model in MODEL_PROFILES else None


def _validate_checkpoint_style(forge_model: str, style: str) -> None:
    """Require both checkpoint permission and an honest existing KB pool."""
    if (style not in model_profile_for(forge_model).styles or
            style not in STYLE_MEDIUM_PROFILES):
        raise PromptEngineError((ReasonCode.STYLE_CONFLICT,), style)


def _semantic_records(
    facet: PromptFacet, *, forge_model: str, rating: str, orientation: str,
) -> tuple[Mapping[str, object], ...]:
    """Return engine-compatible records without exposing the whole KB."""
    # Local import avoids making the ForgeBAT option module and this bridge
    # depend on each other during module initialization.
    from .batch_options import facet_records

    query_rating = RATING_LABELS[normalize_content_rating(rating)]
    records = tuple(facet_records(
        facet.value,
        model=forge_model,
        content_rating=query_rating,
        orientation=orientation,
    ))
    if facet is PromptFacet.CLOTHING:
        records = tuple(record for record in records if _safe_wardrobe_record(record))
    return records


def _catalog_rank(record: Mapping[str, object], relevance: frozenset[str], salt: str) -> tuple[int, str]:
    searchable = " ".join((
        str(record["canonical_text"]),
        *(str(value) for value in record.get("aliases", ())),
        *(str(value) for value in record.get("semantic_tags", ())),
    )).casefold()
    overlap = sum(1 for term in relevance if term in searchable)
    digest = hashlib.sha256(f"{salt}:{record['id']}".encode("utf-8")).hexdigest()
    return -overlap, digest


def build_semantic_context(
    forge_model: str,
    rating: str,
    orientation: str,
    *,
    style: str = "Photoreal",
    novelty_text: str = "",
    seed: int | None = None,
    per_facet_limit: int = 6,
) -> SemanticPromptContext:
    """Build a compact provider catalog from the canonical KB.

    The private compatibility sets may span a facet, but only the small
    ``catalog`` projection is serialized into an LLM request.
    """
    model_id = engine_model_for(forge_model)
    if model_id is None:
        raise PromptEngineError((ReasonCode.MODEL_UNSUPPORTED,), forge_model)
    _validate_checkpoint_style(forge_model, style)
    engine_rating = Rating(normalize_content_rating(rating))
    used_seed = secrets.randbelow(2**31) if seed is None else seed
    resolved_orientation = _ORIENTATIONS.get(orientation) or random.Random(used_seed).choice(
        ("front", "side", "rear")
    )
    engine_orientation = Orientation(resolved_orientation)
    relevance = frozenset(
        word for word in re.findall(r"[a-z0-9]+", novelty_text.casefold())
        if len(word) >= 4
    )
    salt = f"{model_id}:{engine_rating.value}:{engine_orientation.value}:{novelty_text}"
    catalog: dict[PromptFacet, tuple[SemanticCatalogItem, ...]] = {}
    compatible: dict[PromptFacet, frozenset[str]] = {}
    for facet in PromptFacet:
        records = _semantic_records(
            facet,
            forge_model=forge_model,
            rating=rating,
            orientation=engine_orientation.value,
        )
        compatible[facet] = frozenset(str(record["id"]) for record in records)
        selected = sorted(records, key=lambda record: _catalog_rank(record, relevance, salt))[
            :per_facet_limit
        ]
        catalog[facet] = tuple(
            SemanticCatalogItem(
                id=str(record["id"]),
                text=str(record["canonical_text"]),
                tags=tuple(str(value) for value in record.get("semantic_tags", ()))[:4],
            )
            for record in selected
        )
    return SemanticPromptContext(
        checkpoint=forge_model,
        model_id=model_id,
        rating=engine_rating,
        orientation=engine_orientation,
        style_parent=style,
        seed=used_seed,
        catalog=catalog,
        compatible_ids=compatible,
    )


def normalize_semantic_response(
    response: SemanticPromptResponse,
    context: SemanticPromptContext,
) -> CanonicalPromptSpec:
    """Resolve provider JSON to existing Prompt Engine concepts."""
    if response.orientation is not context.orientation:
        raise SemanticPromptError(
            "The provider changed the request-owned orientation."
        )
    selections: list[SemanticSelection] = []
    repairs: list[str] = []
    for facet, concept_id in response.concepts.items():
        if concept_id not in context.compatible_ids.get(facet, frozenset()):
            repairs.append(f"dropped unavailable {facet.value} concept {concept_id}")
            continue
        record = BY_ID.get(concept_id)
        if record is None:
            repairs.append(f"dropped unknown {facet.value} concept {concept_id}")
            continue
        selections.append(SemanticSelection(
            facet,
            Concept.from_record(record, context.model_id),
        ))
    if not selections:
        raise SemanticPromptError("The provider returned no usable canonical concepts.")
    return CanonicalPromptSpec(
        model_id=context.model_id,
        rating=context.rating,
        orientation=context.orientation,
        style_parent=context.style_parent,
        selections=tuple(selections),
        repair_notes=tuple(repairs),
    )


def generate_from_semantic(
    forge_model: str,
    semantic: CanonicalPromptSpec,
    *,
    seed: int,
) -> SemanticCompiledPrompt:
    """Compile a provider proposal with bounded deterministic repair.

    Repairs only relax optional LLM suggestions.  Request-owned model, rating,
    style, and orientation are never changed, and every attempt still passes
    through the existing solver, validator, budget allocator, and compiler.
    """
    expected_model = engine_model_for(forge_model)
    if expected_model != semantic.model_id:
        raise SemanticPromptError("The semantic model does not match the checkpoint.")

    original = {selection.facet: selection for selection in semantic.selections}
    attempts: list[tuple[PromptFacet, ...]] = [()]
    attempts.extend((facet,) for facet in _SEMANTIC_REPAIR_ORDER if facet in original)
    cumulative: list[PromptFacet] = []
    for facet in _SEMANTIC_REPAIR_ORDER:
        if facet in original:
            cumulative.append(facet)
            if len(cumulative) > 1:
                attempts.append(tuple(cumulative))

    seen: set[tuple[PromptFacet, ...]] = set()
    failures: list[ReasonCode] = []
    for dropped in attempts:
        if dropped in seen:
            continue
        seen.add(dropped)
        kept = tuple(
            selection for facet, selection in original.items() if facet not in dropped
        )
        constraints = {
            selection.facet.value: (selection.concept.id,) for selection in kept
        }
        try:
            compiled, _ = generate_compiled(
                forge_model,
                semantic.rating.value,
                semantic.orientation.value,
                seed,
                style=semantic.style_parent,
                constraints=constraints,
            )
        except PromptEngineError as exc:
            failures.extend(exc.reason_codes)
            continue
        repair_notes = semantic.repair_notes + tuple(
            f"replaced incompatible {facet.value} concept {original[facet].concept.id}"
            for facet in dropped
        )
        normalized = replace(
            semantic,
            selections=kept,
            repair_notes=repair_notes,
        )
        return SemanticCompiledPrompt(compiled, normalized, repair_notes)
    raise PromptEngineError(
        tuple(dict.fromkeys(failures)) or (ReasonCode.QUALITY_BELOW_ACCEPT,),
        "semantic proposal could not be deterministically repaired",
    )


def _curated_facet_text(novelty: NoveltyContext, facet: PromptFacet) -> str:
    """Map novelty axes to solver facets; returned prose is never serialized."""
    brief = novelty.brief
    values = {
        PromptFacet.CAMERA: (brief.camera,),
        PromptFacet.LIGHTING: (brief.lighting, brief.palette),
        PromptFacet.POSE: (brief.action,),
        PromptFacet.ENVIRONMENT: (brief.theme, brief.environment),
        PromptFacet.MOOD: (brief.theme, brief.mood, brief.palette),
    }[facet]
    return " ".join((*values, *novelty.quality_guidance))


def _curated_constraints(
    forge_model: str,
    rating: str,
    orientation: str,
    novelty: NoveltyContext,
) -> dict[str, tuple[str, ...]]:
    """Return KB ids whose canonical metadata overlaps the novelty brief."""
    constraints: dict[str, tuple[str, ...]] = {}
    for facet in _CURATED_FACETS:
        relevance = frozenset(
            word for word in re.findall(
                r"[a-z0-9]+", _curated_facet_text(novelty, facet).casefold()
            )
            if len(word) >= 4
        )
        records = _semantic_records(
            facet,
            forge_model=forge_model,
            rating=rating,
            orientation=orientation,
        )
        ranked = sorted(
            (
                (_catalog_rank(record, relevance, facet.value), record)
                for record in records
            ),
            key=lambda item: item[0],
        )
        # No lexical match means this novelty axis cannot safely steer the KB.
        # Leaving the facet unconstrained lets the solver use its normal
        # semantic/model/reliability ranking instead of inventing a mapping.
        relevant = tuple(
            str(record["id"])
            for rank, record in ranked
            if rank[0] < 0
        )[:_CURATED_FACET_LIMIT]
        if relevant:
            constraints[facet.value] = relevant
    return constraints


def _compiled_concept_ids(compiled: CompiledPrompt) -> tuple[str, ...]:
    return tuple(
        concept.id
        for concept in compiled.positive_fragments
        if concept.category != "control"
    )


def _compiled_novelty_signature(compiled: CompiledPrompt) -> str:
    """Describe emitted creative axes rather than the discarded source brief."""
    axes = {
        "environment": {"environment"},
        "action": {"pose_action"},
        "lighting": {"lighting"},
        "camera": {"composition_camera"},
        "mood": {"atmosphere_mood"},
    }
    payload: dict[str, object] = {}
    for axis, categories in axes.items():
        values = tuple(dict.fromkeys(
            concept.text.strip(" ,.")
            for concept in compiled.positive_fragments
            if concept.category in categories and concept.text.strip(" ,.")
        ))
        if values:
            payload[axis] = "; ".join(values)
    concept_ids = _compiled_concept_ids(compiled)
    if concept_ids:
        payload["canonical_concept_ids"] = concept_ids
    return json.dumps(payload, separators=(",", ":"), sort_keys=True)


def _compile_curated_candidate(
    engine: PromptEngine,
    forge_model: str,
    rating: str,
    orientation: str,
    seed: int,
    novelty: NoveltyContext,
    *,
    style: str,
    batch_profile: str | None,
) -> tuple[CompiledPrompt, tuple[str, ...]]:
    """Compile one novelty-steered candidate with deterministic soft repair."""
    model_id = engine_model_for(forge_model)
    if model_id is None:
        raise PromptEngineError((ReasonCode.MODEL_UNSUPPORTED,), forge_model)
    resolved_orientation = _ORIENTATIONS.get(orientation) or random.Random(seed).choice(
        ("front", "side", "rear")
    )
    constraints = _curated_constraints(
        forge_model,
        rating,
        resolved_orientation,
        novelty,
    )
    ordered = tuple(
        facet for facet in _SEMANTIC_REPAIR_ORDER if facet.value in constraints
    )
    attempts = [()]
    attempts.extend(tuple(ordered[:count]) for count in range(1, len(ordered) + 1))
    failures: list[ReasonCode] = []
    for dropped in attempts:
        allowed = {
            facet: ids
            for facet, ids in constraints.items()
            if PromptFacet(facet) not in dropped
        }
        try:
            compiled = engine.generate(
                model_id,
                seed=seed,
                orientation=resolved_orientation,
                rating=normalize_content_rating(rating),
                style_parent=style,
                batch_profile=profile_for(batch_profile),
                allowed_ids=allowed,
            )
        except PromptEngineError as exc:
            failures.extend(exc.reason_codes)
            continue
        notes = tuple(
            f"relaxed novelty constraint for {facet.value}" for facet in dropped
        )
        return compiled, notes
    raise PromptEngineError(
        tuple(dict.fromkeys(failures)) or (ReasonCode.QUALITY_BELOW_ACCEPT,),
        "curated novelty constraints could not be deterministically repaired",
    )


def generate_curated(
    forge_model: str,
    rating: str,
    orientation: str,
    history: Iterable[PromptHistoryItem],
    *,
    creativity_level: str,
    aspect_ratio: str,
    style: str = "Photoreal",
    batch_profile: str | None = None,
    seed: int | None = None,
) -> CuratedCompiledPrompt:
    """Select and compile the least repetitive valid local KB candidate.

    An explicit seed is a replay contract and therefore ignores mutable engine
    diversity state. Ordinary unseeded calls use one concept-diversity snapshot
    for every draw, then record only the selected winner. Prompt history steers
    KB concept pools and scores final prompt similarity; it never formats text.
    """
    _validate_checkpoint_style(forge_model, style)
    recent = tuple(history)
    explicit_seed = seed is not None
    used_seed = secrets.randbelow(2**63) if seed is None else seed
    candidate_rng = random.Random(used_seed)
    novelty_seed = int.from_bytes(
        hashlib.sha256(f"curated-novelty:{used_seed}".encode("utf-8")).digest()[:8],
        "big",
    )
    novelty_rng = random.Random(novelty_seed)

    diversity_store: DiversityStore | None = None
    diversity: DiversitySnapshot | None = None
    candidate_engine = _REPLAY_ENGINE
    if not explicit_seed:
        diversity_store = (
            _ENGINE.diversity_store
            if _ENGINE.diversity_enabled and _ENGINE.diversity_store is not None
            else DiversityStore()
        )
        diversity = diversity_store.load()
        candidate_engine = PromptEngine(
            tokenizer=_ENGINE.tokenizer,
            target_valid=_ENGINE.target_valid,
            max_attempts=_ENGINE.max_attempts,
            diversity_store=_SnapshotDiversityStore(diversity),
            diversity_enabled=True,
        )

    best: tuple[
        tuple[float, float, float, int],
        CompiledPrompt,
        NoveltyContext,
        tuple[str, ...],
    ] | None = None
    attempts = 0
    for index in range(_CURATED_CANDIDATE_COUNT):
        attempts += 1
        novelty = create_novelty_context(
            recent,
            creativity_level=creativity_level,
            aspect_ratio=aspect_ratio,
            content_rating=rating,
            rng=novelty_rng,
        )
        candidate_seed = used_seed if index == 0 else candidate_rng.getrandbits(63)
        compiled, repair_notes = _compile_curated_candidate(
            candidate_engine,
            forge_model,
            rating,
            orientation,
            candidate_seed,
            novelty,
            style=style,
            batch_profile=batch_profile,
        )
        similarity = maximum_similarity(compiled.positive, novelty.recent_prompts)
        concept_ids = _compiled_concept_ids(compiled)
        concept_similarity = (
            diversity.maximum_similarity(concept_ids) if diversity is not None else 0.0
        )
        engine_score = compiled.candidate_scores[compiled.winner_index]
        rank = (similarity, concept_similarity, -engine_score, index)
        candidate = (rank, compiled, novelty, repair_notes)
        if best is None or rank < best[0]:
            best = candidate
        if similarity <= novelty.similarity_threshold:
            break

    assert best is not None
    rank, compiled, novelty, repair_notes = best
    if diversity_store is not None:
        diversity_store.record(
            _compiled_concept_ids(compiled),
            compiled.ast.model_id,
            compiled.ast.orientation.value,
        )
    return CuratedCompiledPrompt(
        compiled=compiled,
        novelty=novelty,
        seed=used_seed,
        similarity_score=rank[0],
        attempt_count=attempts,
        concept_similarity=rank[1],
        canonical_concept_ids=_compiled_concept_ids(compiled),
        repair_notes=repair_notes,
        novelty_signature=_compiled_novelty_signature(compiled),
    )


def generate_compiled(
    forge_model: str,
    rating: str,
    orientation: str,
    seed: int | None = None,
    *,
    style: str = "Photoreal",
    batch_profile: str | None = None,
    constraints: Mapping[str, Collection[str]] | None = None,
    enforce_active_pools: bool = False,
) -> tuple[CompiledPrompt, int]:
    """Return the compiled engine result and replayable seed for app callers.

    Raises PromptEngineError when the checkpoint has no engine profile or the
    engine cannot build a valid prompt, so the caller can fall back rather than
    ship something the validator rejected.
    """
    engine_model = engine_model_for(forge_model)
    if engine_model is None:
        raise PromptEngineError((ReasonCode.MODEL_UNSUPPORTED,), forge_model)
    _validate_checkpoint_style(forge_model, style)
    used_seed = secrets.randbelow(2**31) if seed is None else seed
    # "mixed" is reproducible from the returned generation seed alone.
    resolved = _ORIENTATIONS.get(orientation) or random.Random(used_seed).choice(
        ("front", "side", "rear")
    )
    # User-supplied seeds are an explicit replay contract. Ordinary app calls
    # omit the seed and use durable cross-generation diversity.
    engine = _ENGINE if seed is None else _REPLAY_ENGINE
    compiled = engine.generate(
        engine_model,
        seed=used_seed,
        orientation=resolved,
        rating=normalize_content_rating(rating),
        style_parent=style,
        batch_profile=profile_for(batch_profile),
        allowed_ids=constraints,
        enforce_active_pools=enforce_active_pools,
    )
    return compiled, used_seed


def generate_pair(
    forge_model: str,
    rating: str,
    orientation: str,
    seed: int | None = None,
    *,
    style: str = "Photoreal",
    batch_profile: str | None = None,
    constraints: Mapping[str, Collection[str]] | None = None,
    enforce_active_pools: bool = False,
) -> tuple[str, str, int]:
    """Return (positive, negative, seed) from the application engine bridge."""
    compiled, used_seed = generate_compiled(
        forge_model, rating, orientation, seed, style=style,
        batch_profile=batch_profile,
        constraints=constraints,
        enforce_active_pools=enforce_active_pools,
    )
    return compiled.positive, compiled.negative, used_seed

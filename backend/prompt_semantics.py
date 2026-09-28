"""Shared semantic contract between prompt intelligence and Prompt Engine V1.

Providers select canonical knowledge-base concepts.  They never provide final
checkpoint syntax: the existing constraint solver turns these suggestions into
a validated :class:`PromptAST` and the model adapter serializes it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field

from .prompt_engine import Concept, Orientation, Rating


class PromptFacet(str, Enum):
    """Existing solver facets that an intelligence source may suggest."""

    AGE = "age"
    HERITAGE = "heritage"
    BODY_TYPE = "body_type"
    FACE_FEATURES = "face_features"
    HAIR = "hair"
    CLOTHING = "clothing"
    POSE = "pose"
    EXPRESSION = "expression"
    ENVIRONMENT = "environment"
    CAMERA = "camera"
    LIGHTING = "lighting"
    MOOD = "mood"


class SemanticPromptResponse(BaseModel):
    """Provider-neutral JSON returned by Ollama, DeepSeek, or Anthropic."""

    model_config = ConfigDict(extra="forbid")

    orientation: Orientation
    concepts: dict[PromptFacet, str] = Field(min_length=3, max_length=12)


# Backward-compatible import name for callers that treated the old transport
# schema as the provider's structured "creative scene" response.
CreativeScene = SemanticPromptResponse


@dataclass(frozen=True, slots=True)
class SemanticSelection:
    """One provider suggestion backed by an existing Prompt Engine concept."""

    facet: PromptFacet
    concept: Concept


@dataclass(frozen=True, slots=True)
class CanonicalPromptSpec:
    """Normalized semantic input to the existing Prompt Engine.

    Request-owned controls are strongly typed with the Prompt AST enums.  The
    semantic content is represented by the engine's own ``Concept`` objects,
    not by provider prose or a second keyword model.
    """

    model_id: str
    rating: Rating
    orientation: Orientation
    style_parent: str
    selections: tuple[SemanticSelection, ...]
    repair_notes: tuple[str, ...] = ()

    @property
    def concept_ids(self) -> tuple[str, ...]:
        return tuple(selection.concept.id for selection in self.selections)

    def constraints(self) -> Mapping[str, tuple[str, ...]]:
        return MappingProxyType({
            selection.facet.value: (selection.concept.id,)
            for selection in self.selections
        })


@dataclass(frozen=True, slots=True)
class SemanticCatalogItem:
    """Compact, LLM-visible projection of one canonical KB record."""

    id: str
    text: str
    tags: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {"id": self.id, "text": self.text, "tags": list(self.tags)}


@dataclass(frozen=True, slots=True)
class SemanticPromptContext:
    """Bounded provider context plus private compatibility allow-lists."""

    checkpoint: str
    model_id: str
    rating: Rating
    orientation: Orientation
    style_parent: str
    seed: int
    catalog: Mapping[PromptFacet, tuple[SemanticCatalogItem, ...]]
    compatible_ids: Mapping[PromptFacet, frozenset[str]]

    def catalog_payload(self) -> dict[str, list[dict[str, object]]]:
        return {
            facet.value: [item.as_dict() for item in items]
            for facet, items in self.catalog.items()
            if items
        }


class SemanticPromptError(ValueError):
    """A provider response could not satisfy the shared semantic contract."""

"""Validation and translation of ForgeBAT facet selections."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Collection, Mapping

from .batch_options import FACET_BY_KEY, facet_records
from .keywords import BY_ID


@dataclass(frozen=True, slots=True)
class BatchConstraintError(ValueError):
    facet: str
    concept: str | None
    reason: str
    message: str

    def __str__(self) -> str:
        return self.message

    def detail(self) -> dict[str, str | None]:
        return {
            "facet": self.facet,
            "concept": self.concept,
            "reason": self.reason,
            "message": self.message,
        }


def _belongs_to_facet(facet: str, concept_id: str) -> bool:
    definition = FACET_BY_KEY[facet]
    record = BY_ID[concept_id]
    if record["category"] != definition.category:
        return False
    return not definition.subcategories or record["subcategory"] in definition.subcategories


def validate_facet_selections(
    selections: Mapping[str, Mapping[str, Any]], *, model: str | None,
    content_rating: str, orientation: str = "front",
    orientations: Collection[str] | None = None,
) -> dict[str, frozenset[str]]:
    """Validate UI selections and return solver-ready allowed-id pools."""
    allowed: dict[str, frozenset[str]] = {}
    for facet, selection in selections.items():
        if facet not in FACET_BY_KEY:
            raise BatchConstraintError(
                facet, None, "unknown_facet", f"Unknown batch facet: {facet}",
            )
        mode = str(selection.get("mode", ""))
        if mode not in {"all", "selected"}:
            raise BatchConstraintError(
                facet, None, "invalid_mode", "Facet mode must be 'all' or 'selected'.",
            )
        if mode == "all":
            continue
        raw_ids = selection.get("ids", ())
        if not isinstance(raw_ids, (list, tuple, set, frozenset)):
            raise BatchConstraintError(
                facet, None, "invalid_ids", "Selected facet ids must be a list.",
            )
        ids = tuple(dict.fromkeys(str(value) for value in raw_ids))
        if not ids:
            raise BatchConstraintError(
                facet, None, "empty_selection", "Selected mode requires at least one choice.",
            )
        for concept_id in ids:
            if concept_id not in BY_ID:
                raise BatchConstraintError(
                    facet, concept_id, "unknown_concept",
                    f"Unknown concept id for {FACET_BY_KEY[facet].label}: {concept_id}",
                )
            if not _belongs_to_facet(facet, concept_id):
                raise BatchConstraintError(
                    facet, concept_id, "wrong_category",
                    f"{concept_id} is not a {FACET_BY_KEY[facet].label} choice.",
                )
        selected_orientations = tuple(dict.fromkeys(orientations or (orientation,)))
        compatible = {
            str(record["id"])
            for record in facet_records(
                facet,
                model=model,
                content_rating=content_rating,
                orientations=selected_orientations,
            )
        }
        for concept_id in ids:
            if concept_id not in compatible:
                raise BatchConstraintError(
                    facet, concept_id, "incompatible_concept",
                    f"{concept_id} is not compatible with the selected model, rating, or all selected orientations.",
                )
        allowed[facet] = frozenset(ids)
    return allowed

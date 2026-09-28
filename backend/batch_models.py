"""Validated API request models shared by ForgeBAT preview and creation."""

from __future__ import annotations

from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from .forge import model_profile_for


class FacetSelection(BaseModel):
    mode: Literal["all", "selected"] = "all"
    ids: list[str] = Field(default_factory=list, max_length=500)

    @model_validator(mode="after")
    def selected_has_ids(self) -> "FacetSelection":
        self.ids = list(dict.fromkeys(self.ids))
        if self.mode == "selected" and not self.ids:
            raise ValueError("Selected mode requires at least one concept id.")
        if self.mode == "all":
            self.ids = []
        return self


class BatchCreateRequest(BaseModel):
    count: int = Field(default=16, ge=1, le=1000)
    model: str = Field(min_length=1, max_length=300)
    style: str = Field(default="Photoreal", min_length=1, max_length=80)
    styles: list[str] = Field(default_factory=list)
    content_rating: Literal["Safe"] = "Safe"
    quality_mode: Literal["normal", "high", "super", "4k", "8k", "12k"] = "high"
    aspect_ratio: str = Field(default="Portrait (832x1216)", min_length=1, max_length=80)
    orientation: Literal["front", "side", "back"] = "front"
    orientations: list[Literal["front", "side", "back"]] = Field(default_factory=list, max_length=3)
    distribution: Literal["balanced", "random"] = "balanced"
    batch_seed: int = Field(default_factory=lambda: int.from_bytes(uuid4().bytes[:8], "big"), ge=0, le=2**63 - 1)
    request_id: str = Field(default_factory=lambda: str(uuid4()), min_length=8, max_length=100)
    facets: dict[str, FacetSelection] = Field(default_factory=dict)
    cfg_scale: float | None = Field(default=None, ge=1.0, le=30.0)

    @model_validator(mode="after")
    def normalize_pools(self) -> "BatchCreateRequest":
        """Accept legacy scalar recipes while storing multi-value pools."""
        orientation_values = list(dict.fromkeys(self.orientations or [self.orientation]))
        self.orientations = orientation_values
        self.orientation = orientation_values[0]

        style_values = list(dict.fromkeys(self.styles or [self.style]))
        supported_styles = model_profile_for(self.model).styles
        unsupported = [value for value in style_values if value not in supported_styles]
        if unsupported:
            raise ValueError(
                f"Style {unsupported[0]!r} is not supported by the selected model. "
                f"Choose from: {', '.join(supported_styles)}."
            )
        self.styles = style_values
        self.style = style_values[0]
        return self

"""ForgeBAT create/preview validation produces structured HTTP 422 details."""

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend import main
from backend.batch_models import BatchCreateRequest
from backend.batch_options import facet_records


MODEL = r"sd\juggernautXL_ragnarok.safetensors [dd08fa32f9]"


def _body(facet: str, concept_id: str, *, orientation: str = "front") -> BatchCreateRequest:
    return BatchCreateRequest(
        count=4,
        model=MODEL,
        content_rating="Safe",
        aspect_ratio="Portrait (832x1216)",
        orientation=orientation,
        facets={facet: {"mode": "selected", "ids": [concept_id]}},
    )


@pytest.mark.parametrize(
    ("body", "reason"),
    [
        (_body("heritage", "identity.not-real"), "unknown_concept"),
        (_body("heritage", str(facet_records("pose", content_rating="Safe")[0]["id"])), "wrong_category"),
    ],
)
def test_unknown_and_wrong_category_ids_are_http_422(body, reason) -> None:
    with pytest.raises(HTTPException) as caught:
        main._validated_batch_constraints(body)
    assert caught.value.status_code == 422
    assert caught.value.detail["reason"] == reason
    assert set(caught.value.detail) == {"facet", "concept", "reason", "message"}


def test_invalid_pose_view_is_http_422_before_queueing() -> None:
    front = {str(record["id"]) for record in facet_records("pose", content_rating="Safe", orientation="front")}
    rear = {str(record["id"]) for record in facet_records("pose", content_rating="Safe", orientation="back")}
    body = _body("pose", next(iter(front - rear)), orientation="back")

    with pytest.raises(HTTPException) as caught:
        main._validated_batch_constraints(body)
    assert caught.value.status_code == 422
    assert caught.value.detail["facet"] == "pose"
    assert caught.value.detail["reason"] == "incompatible_concept"


def test_single_orientation_recipes_upgrade_to_a_one_item_pool() -> None:
    body = BatchCreateRequest(model=MODEL, orientation="side")
    assert body.orientations == ["side"]
    assert body.orientation == "side"


def test_multi_orientation_recipe_keeps_legacy_primary_view() -> None:
    body = BatchCreateRequest(model=MODEL, orientations=["side", "front", "side"])
    assert body.orientations == ["side", "front"]
    assert body.orientation == "side"


def test_style_pool_is_normalized_and_keeps_legacy_primary_style() -> None:
    body = BatchCreateRequest(
        model=MODEL,
        styles=["Cinematic", "Photoreal", "Cinematic"],
    )
    assert body.styles == ["Cinematic", "Photoreal"]
    assert body.style == "Cinematic"


def test_checkpoint_rejects_an_unsupported_style_with_supported_choices() -> None:
    with pytest.raises(ValidationError) as caught:
        BatchCreateRequest(model=MODEL, style="Anime")

    message = str(caught.value)
    assert "Style 'Anime' is not supported by the selected model" in message
    assert "Photoreal, Cinematic, Editorial, Fantasy, Concept art" in message

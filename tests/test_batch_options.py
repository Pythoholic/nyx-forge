"""ForgeBAT option browsing is a filtered view of the keyword base."""

import pytest

from backend.batch_common import normalize_content_rating
from backend.batch_options import FACET_BY_KEY, facet_records, get_batch_options
from backend.keywords import BY_ID, CATEGORIES, MANIFEST


PONY_V6 = r"sd\ponyDiffusionV6XL_v6StartWithThisOne.safetensors [67ab2fd8ec]"


def test_option_metadata_is_lazy_and_comes_from_the_kb() -> None:
    metadata = get_batch_options()
    heritage = next(facet for facet in metadata["facets"] if facet["key"] == "heritage")
    loaded = get_batch_options(facet="heritage")
    options = next(facet for facet in loaded["facets"] if facet["key"] == "heritage")["options"]

    assert metadata["kb_version"] == MANIFEST["kb_version"]
    assert heritage["options"] == []
    assert heritage["option_count"] == len(options)
    assert options
    assert {option["id"] for option in options} <= {record["id"] for record in CATEGORIES["identity"]}
    assert {BY_ID[option["id"]]["subcategory"] for option in options} == {"heritage", "nationality"}
    assert {option["group"] for option in options} == {"Region / Heritage", "Country / Nationality"}
    assert all("canonical_text" not in option for option in options)


def test_style_capabilities_come_from_the_selected_model_profile() -> None:
    metadata = get_batch_options(model=PONY_V6, style="Anime")

    assert metadata["capabilities"]["styles"] == ["Anime", "Illustration", "Cartoon"]


def test_every_facet_option_has_the_declared_category() -> None:
    for key, definition in FACET_BY_KEY.items():
        records = facet_records(key, content_rating="Safe")
        assert records
        assert all(record["category"] == definition.category for record in records)


def test_option_search_and_orientation_only_narrow_the_kb_pool() -> None:
    all_pose_ids = {record["id"] for record in facet_records("pose", content_rating="Safe")}
    searched = facet_records("pose", content_rating="Safe", search="sitting")
    rear = facet_records("pose", content_rating="Safe", orientation="back")

    assert searched
    assert {record["id"] for record in searched} <= all_pose_ids
    assert {record["id"] for record in rear} <= {record["id"] for record in CATEGORIES["pose_action"]}


def test_multi_orientation_options_are_safe_for_every_selected_view() -> None:
    records = facet_records(
        "pose", content_rating="Safe", orientations=("front", "side", "back"),
    )
    front = {record["id"] for record in facet_records("pose", content_rating="Safe", orientation="front")}
    side = {record["id"] for record in facet_records("pose", content_rating="Safe", orientation="side")}
    back = {record["id"] for record in facet_records("pose", content_rating="Safe", orientation="back")}
    assert records
    assert {record["id"] for record in records} <= front & side & back


def test_body_type_includes_an_adult_pregnancy_option() -> None:
    options = get_batch_options(facet="body_type")
    body_types = next(facet for facet in options["facets"] if facet["key"] == "body_type")
    assert any(option["id"] == "appearance.build_pregnant_maternity_silhouette" for option in body_types["options"])


def test_expression_and_mood_explain_their_distinct_roles() -> None:
    facets = {facet["key"]: facet for facet in get_batch_options()["facets"]}
    assert facets["expression"]["label"] == "Facial Expression"
    assert "subject's face" in facets["expression"]["description"]
    assert facets["mood"]["label"] == "Scene Mood"
    assert "setting" in facets["mood"]["description"]


def test_heritage_facet_exposes_regions_and_specific_nationalities() -> None:
    response = get_batch_options(facet="heritage")
    facet = next(value for value in response["facets"] if value["key"] == "heritage")
    labels = {option["label"] for option in facet["options"]}
    assert facet["label"] == "Country / Region Heritage"
    assert {"East Asian", "French", "Indian", "Lebanese"} <= labels


@pytest.mark.parametrize(
    ("label", "engine_value"),
    [("Safe", "SAFE")],
)
def test_rating_normalization(label: str, engine_value: str) -> None:
    assert normalize_content_rating(label) == engine_value


@pytest.mark.parametrize("label", ["Unsupported", "Mature"])
def test_non_safe_rating_is_rejected(label: str) -> None:
    with pytest.raises(ValueError):
        normalize_content_rating(label)

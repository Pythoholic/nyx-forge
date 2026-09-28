"""Run installed ADetailer YOLO weights and emit PNG mask proposals as JSON."""

from __future__ import annotations

import argparse
import base64
from io import BytesIO
import json
from pathlib import Path

import numpy as np
from PIL import Image


def load_yolo(weights: Path):
    """Import Ultralytics only inside the Forge detector runtime."""
    from ultralytics import YOLO

    return YOLO(weights)


def encode_mask(mask: np.ndarray, size: tuple[int, int]) -> str:
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    output = BytesIO()
    image.save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def model_path(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if not matches:
        raise FileNotFoundError(name)
    return matches[0]


# Medium beats nano on outline accuracy, which is what decides whether an
# outfit change stays inside the person or bleeds into the background. Fall
# back through the smaller weights if the better ones are not installed.
PERSON_WEIGHTS = ("person_yolov8m-seg.pt", "person_yolov8s-seg.pt", "person_yolov8n-seg.pt")


def person_model(models: Path) -> Path:
    for name in PERSON_WEIGHTS:
        try:
            return model_path(models, name)
        except FileNotFoundError:
            continue
    raise FileNotFoundError("no person segmentation weights installed")


def source_sized_mask(mask: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Move an inference-space binary mask into source-image coordinates."""
    image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
    if image.size != size:
        image = image.resize(size, Image.Resampling.NEAREST)
    return np.asarray(image) > 0


def protect_face_regions(person_mask: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Subtract the head, hair, and neck from a source-space person mask.

    The head is protected in proportion to the detected face, never to a
    target coverage number. On a close-up the head legitimately occupies most
    of the frame, and a retention floor there trades away the face - the one
    thing this function exists to keep - to hit an arbitrary ratio.
    """
    protected = person_mask.copy()
    height, width = protected.shape
    for coordinates in boxes:
        x1, y1, x2, y2 = (float(value) for value in coordinates)
        box_width, box_height = x2 - x1, y2 - y1
        # Hair sits above and beside the face box; the neck/collar bridge
        # below it stops high-denoise inpainting leaving a seam at the jaw.
        left = max(0, int(np.floor(x1 - box_width * 0.45)))
        right = min(width, int(np.ceil(x2 + box_width * 0.45)))
        top = max(0, int(np.floor(y1 - box_height * 0.9)))
        bottom = min(height, int(np.ceil(y2 + box_height * 0.55)))
        protected[top:bottom, left:right] = False
        # Belt and braces: rounding above must never leave the detected face
        # itself editable.
        protected[
            max(0, int(np.floor(y1))) : min(height, int(np.ceil(y2))),
            max(0, int(np.floor(x1))) : min(width, int(np.ceil(x2))),
        ] = False
    return protected


def person_mask(source: Path, models: Path) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Return one union person mask in source coordinates."""
    result = load_yolo(person_model(models))(str(source), verbose=False, device="cpu")[0]
    if result.masks is None:
        return None
    with Image.open(source) as image:
        size = image.size
    inference_masks = result.masks.data.cpu().numpy()
    return source_sized_mask(np.any(inference_masks > 0.5, axis=0), size), size


def segmented_people(source: Path, models: Path) -> list[dict[str, str]]:
    detected = person_mask(source, models)
    if detected is None:
        return []
    mask, size = detected
    return [{"id": "person", "label": "Detected person", "mask": encode_mask(mask, size)}]


def segmented_body(source: Path, models: Path) -> list[dict[str, str]]:
    """Return the segmented person with face, hair, and jaw protected."""
    detected = person_mask(source, models)
    if detected is None:
        return []
    mask, size = detected
    face_result = load_yolo(model_path(models, "face_yolov8n.pt"))(str(source), verbose=False, device="cpu")[0]
    boxes = face_result.boxes.xyxy.cpu().numpy() if face_result.boxes is not None else np.empty((0, 4))
    body = protect_face_regions(mask, boxes)
    return [{"id": "body", "label": "Body (face protected)", "mask": encode_mask(body, size)}]


# SegFormer human-parsing labels. Semantic classes beat rectangle geometry:
# these follow a collar, an open jacket front, and a sleeve cuff exactly,
# where a box cannot.
CLOTHES_MODEL = "mattmdjaga/segformer_b2_clothes"
GARMENT_LABELS = (4, 5, 6, 7, 8, 17)   # upper-clothes, skirt, pants, dress, belt, scarf
SKIN_LABELS = (11, 12, 13, 14, 15)     # face, legs, arms
HEAD_LABELS = (1, 2, 3, 11)            # hat, hair, sunglasses, face


def parse_human(source: Path):
    """Return a per-pixel human-parsing map, or None if the model is absent."""
    try:
        import torch
        from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    except ImportError:
        return None
    try:
        processor = SegformerImageProcessor.from_pretrained(CLOTHES_MODEL)
        model = AutoModelForSemanticSegmentation.from_pretrained(CLOTHES_MODEL)
    except Exception:
        return None
    with Image.open(source) as opened:
        image = opened.convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    upsampled = torch.nn.functional.interpolate(
        logits, size=image.size[::-1], mode="bilinear", align_corners=False
    )
    return upsampled.argmax(dim=1)[0].numpy(), image.size


def segmented_garments(source: Path, models: Path) -> list[dict[str, str]]:
    """Mask the clothing itself, leaving skin, face, and hair untouched."""
    parsed = parse_human(source)
    if parsed is None:
        # Fall back to the geometric body mask when the parser is unavailable.
        return segmented_body(source, models)
    segmentation, size = parsed
    garment = np.isin(segmentation, GARMENT_LABELS)
    if not garment.any():
        return segmented_body(source, models)
    return [{"id": "garment", "label": "Clothing", "mask": encode_mask(garment, size)}]


def segmented_skin_region(source: Path, models: Path) -> list[dict[str, str]]:
    """Mask clothing plus the torso it covers, while protecting the head."""
    parsed = parse_human(source)
    if parsed is None:
        return segmented_body(source, models)
    segmentation, size = parsed
    region = np.isin(segmentation, GARMENT_LABELS + SKIN_LABELS)
    region &= ~np.isin(segmentation, HEAD_LABELS)
    if not region.any():
        return segmented_body(source, models)
    return [{"id": "body", "label": "Body (head protected)", "mask": encode_mask(region, size)}]


def detected_face_box(source: Path, models: Path) -> list[int] | None:
    """Return the largest detected face with 30% head-context padding."""
    with Image.open(source) as image:
        width, height = image.size
    result = load_yolo(model_path(models, "face_yolov8n.pt"))(
        str(source), verbose=False, device="cpu"
    )[0]
    if result.boxes is None:
        return None
    boxes = result.boxes.xyxy.cpu().numpy()
    if not len(boxes):
        return None
    x1, y1, x2, y2 = max(
        boxes,
        key=lambda coordinates: max(0.0, float(coordinates[2] - coordinates[0]))
        * max(0.0, float(coordinates[3] - coordinates[1])),
    )
    box_width, box_height = float(x2 - x1), float(y2 - y1)
    return [
        max(0, int(np.floor(x1 - box_width * 0.3))),
        max(0, int(np.floor(y1 - box_height * 0.3))),
        min(width, int(np.ceil(x2 + box_width * 0.3))),
        min(height, int(np.ceil(y2 + box_height * 0.3))),
    ]


def detected_features(source: Path, models: Path) -> list[dict[str, str]]:
    with Image.open(source) as image:
        width, height = image.size
    proposals: list[dict[str, str]] = []
    for label, filename in (("Face", "face_yolov8n.pt"), ("Hand", "hand_yolov8n.pt")):
        result = load_yolo(model_path(models, filename))(str(source), verbose=False, device="cpu")[0]
        if result.boxes is None:
            continue
        for index, coordinates in enumerate(result.boxes.xyxy.cpu().numpy()):
            x1, y1, x2, y2 = coordinates.tolist()
            padding = max(x2 - x1, y2 - y1) * 0.12
            x1, y1 = max(0, int(x1 - padding)), max(0, int(y1 - padding))
            x2, y2 = min(width, int(x2 + padding)), min(height, int(y2 + padding))
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[y1:y2, x1:x2] = 1
            proposals.append({"id": f"{label.lower()}-{index + 1}", "label": f"{label} {index + 1}", "mask": encode_mask(mask, (width, height))})
    return proposals


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("models", type=Path)
    parser.add_argument("detector", choices=("person", "body", "garment", "features", "face-box"))
    args = parser.parse_args()
    if args.detector == "face-box":
        print(json.dumps({"face_box": detected_face_box(args.source, args.models)}, separators=(",", ":")))
        return
    detectors = {
        "person": segmented_people,
        "body": segmented_skin_region,
        "garment": segmented_garments,
        "features": detected_features,
    }
    proposals = detectors[args.detector](args.source, args.models)
    print(json.dumps({"proposals": proposals}, separators=(",", ":")))


if __name__ == "__main__":
    main()

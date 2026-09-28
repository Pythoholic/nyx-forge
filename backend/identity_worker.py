"""Small InsightFace subprocess used by ForgeIMG's local identity providers.

This module is intentionally executed with reForge's Python environment, where
InsightFace and ONNX Runtime are already installed. Keeping it out of the API
process avoids coupling ForgeAI startup to reForge's native ML dependencies.
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


def _analysis(root: Path) -> FaceAnalysis:
    analyzer = FaceAnalysis(
        name="buffalo_l",
        root=str(root),
        providers=["CPUExecutionProvider"],
    )
    analyzer.prepare(ctx_id=-1, det_size=(640, 640))
    return analyzer


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def _largest_face(analyzer: FaceAnalysis, image: np.ndarray, label: str):
    faces = analyzer.get(image)
    if not faces:
        raise ValueError(f"No face detected in the {label} image.")
    return max(
        faces,
        key=lambda face: float(
            (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
        ),
    )


def _similarity(source_face, target_face) -> float:
    source = np.asarray(source_face.normed_embedding, dtype=np.float32)
    target = np.asarray(target_face.normed_embedding, dtype=np.float32)
    return float(np.clip(np.dot(source, target), -1.0, 1.0))


def analyze_identity(reference_path: Path, *, root: Path) -> dict[str, object]:
    """Return detector, pose, exposure, sharpness, and embedding evidence."""
    analyzer = _analysis(root)
    image = _read_image(reference_path)
    faces = analyzer.get(image)
    if not faces:
        raise ValueError("No face detected in the reference crop.")
    face = max(
        faces,
        key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ),
    )
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    embedding = np.asarray(face.normed_embedding, dtype=np.float32)
    pose = np.asarray(getattr(face, "pose", np.zeros(3)), dtype=np.float32)
    bbox = np.asarray(face.bbox, dtype=np.float32)
    return {
        "provider": "insightface-buffalo-l",
        "face_count": len(faces),
        "bbox": [float(value) for value in bbox.tolist()],
        "det_score": float(getattr(face, "det_score", 0.0)),
        "pose": [float(value) for value in pose.tolist()],
        "image_size": [int(width), int(height)],
        "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness": float(gray.mean()),
        "dark_clip_ratio": float(np.mean(gray <= 8)),
        "light_clip_ratio": float(np.mean(gray >= 247)),
        "embedding": base64.b64encode(embedding.tobytes()).decode("ascii"),
    }


def lock_identity(
    source_path: Path,
    target_path: Path,
    output_path: Path,
    *,
    root: Path,
    model_path: Path,
) -> dict[str, object]:
    analyzer = _analysis(root)
    source_image = _read_image(source_path)
    target_image = _read_image(target_path)
    source_face = _largest_face(analyzer, source_image, "reference")
    target_face = _largest_face(analyzer, target_image, "candidate")
    before = _similarity(source_face, target_face)
    swapper = get_model(
        str(model_path),
        download=False,
        providers=["CPUExecutionProvider"],
    )
    locked = swapper.get(target_image, target_face, source_face, paste_back=True)
    if locked is None:
        raise ValueError("The identity-lock model returned no image.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), locked):
        raise OSError(f"Could not write identity-locked image: {output_path}")
    locked_face = _largest_face(analyzer, locked, "identity-locked")
    return {
        "provider": "insightface-inswapper-128",
        "similarity_before": before,
        "similarity_after": _similarity(source_face, locked_face),
        "output": str(output_path),
    }


def verify_identity(
    source_path: Path,
    target_path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    analyzer = _analysis(root)
    source_face = _largest_face(analyzer, _read_image(source_path), "reference")
    target_face = _largest_face(analyzer, _read_image(target_path), "result")
    return {
        "provider": "insightface-buffalo-l",
        "similarity": _similarity(source_face, target_face),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("analyze", "lock", "verify"))
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "analyze":
            result = analyze_identity(args.source, root=args.root)
        elif args.command == "lock":
            if args.target is None or args.output is None or args.model is None:
                raise ValueError("Identity lock requires --target, --output, and --model.")
            result = lock_identity(
                args.source,
                args.target,
                args.output,
                root=args.root,
                model_path=args.model,
            )
        else:
            if args.target is None:
                raise ValueError("Identity verification requires --target.")
            result = verify_identity(args.source, args.target, root=args.root)
        print(json.dumps(result, separators=(",", ":")))
        return 0
    except Exception as exc:
        print(json.dumps({"error": str(exc)}, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

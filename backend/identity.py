"""Local identity-lock and verification provider abstractions."""

from __future__ import annotations

from dataclasses import dataclass
import base64
import json
import os
from pathlib import Path
import subprocess
from typing import Protocol

from .database import backend_config


ROOT_DIR = Path(__file__).resolve().parents[1]


def _configured_reforge_dir() -> Path | None:
    """Return the administrator-selected reForge folder when available."""
    try:
        value = backend_config("reforge").package_dir
    except KeyError:
        return None
    return Path(value) if value else None


class IdentityProviderError(RuntimeError):
    """A local identity stage could not produce a trustworthy result."""


@dataclass(frozen=True, slots=True)
class IdentityLockResult:
    output_path: Path
    provider: str
    similarity_before: float
    similarity_after: float


@dataclass(frozen=True, slots=True)
class IdentityVerification:
    score: float
    threshold: float
    provider: str

    @property
    def passed(self) -> bool:
        return self.score >= self.threshold


@dataclass(frozen=True, slots=True)
class IdentityAnalysis:
    provider: str
    face_count: int
    bbox: tuple[float, float, float, float]
    det_score: float
    pose: tuple[float, float, float]
    image_size: tuple[int, int]
    sharpness: float
    brightness: float
    dark_clip_ratio: float
    light_clip_ratio: float
    embedding: bytes


class IdentityLockProvider(Protocol):
    def lock(self, source: Path, target: Path, output: Path) -> IdentityLockResult:
        ...


class IdentityVerifierProvider(Protocol):
    def verify(self, source: Path, target: Path) -> IdentityVerification:
        ...


class InsightFaceLocalIdentityProvider:
    """Run CPU InsightFace stages in reForge's already-provisioned venv."""

    def __init__(self, *, threshold: float | None = None) -> None:
        reforge_dir = _configured_reforge_dir() or ROOT_DIR / ".unconfigured-forge"
        identity_python = reforge_dir / "venv" / (
            "Scripts/python.exe" if os.name == "nt" else "bin/python"
        )
        insightface_root = reforge_dir / "models" / "insightface"
        lock_model = insightface_root / "inswapper_128.onnx"
        self.python = Path(os.getenv("FORGE_IDENTITY_PYTHON", str(identity_python)))
        self.root = Path(os.getenv("FORGE_INSIGHTFACE_ROOT", str(insightface_root)))
        self.model = Path(os.getenv("FORGE_IDENTITY_LOCK_MODEL", str(lock_model)))
        configured = os.getenv("FORGE_IDENTITY_THRESHOLD")
        self.threshold = float(configured) if configured is not None else (
            0.60 if threshold is None else float(threshold)
        )

    def readiness_error(self) -> str | None:
        if not self.python.is_file():
            return "The reForge identity Python environment is missing."
        if not (self.root / "models" / "buffalo_l" / "w600k_r50.onnx").is_file():
            return "The buffalo_l identity verifier is not installed."
        if not self.model.is_file():
            return "The local Identity Lock model is not installed."
        return None

    def analysis_readiness_error(self) -> str | None:
        if not self.python.is_file():
            return "The reForge identity Python environment is missing."
        if not (self.root / "models" / "buffalo_l" / "w600k_r50.onnx").is_file():
            return "The buffalo_l identity analyzer is not installed."
        return None

    def _run(self, *arguments: str, timeout: float = 300.0) -> dict[str, object]:
        readiness = self.readiness_error()
        if readiness:
            raise IdentityProviderError(readiness)
        command = [
            str(self.python),
            "-m",
            "backend.identity_worker",
            *arguments,
            "--root",
            str(self.root),
        ]
        try:
            result = subprocess.run(
                command,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise IdentityProviderError(f"Identity provider could not run: {exc}") from exc
        output_lines = [line for line in result.stdout.splitlines() if line.strip()]
        error_lines = [line for line in result.stderr.splitlines() if line.strip()]
        raw = output_lines[-1] if output_lines else error_lines[-1] if error_lines else "{}"
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IdentityProviderError("Identity provider returned invalid output.") from exc
        if result.returncode != 0 or not isinstance(payload, dict) or payload.get("error"):
            message = payload.get("error") if isinstance(payload, dict) else None
            raise IdentityProviderError(str(message or "Identity provider failed."))
        return payload

    def lock(self, source: Path, target: Path, output: Path) -> IdentityLockResult:
        payload = self._run(
            "lock",
            "--source",
            str(source),
            "--target",
            str(target),
            "--output",
            str(output),
            "--model",
            str(self.model),
        )
        if not output.is_file():
            raise IdentityProviderError("Identity Lock completed without an output image.")
        return IdentityLockResult(
            output,
            str(payload.get("provider") or "insightface-inswapper-128"),
            float(payload.get("similarity_before", 0.0)),
            float(payload.get("similarity_after", 0.0)),
        )

    def analyze(self, source: Path) -> IdentityAnalysis:
        readiness = self.analysis_readiness_error()
        if readiness:
            raise IdentityProviderError(readiness)
        payload = self._run("analyze", "--source", str(source))
        try:
            bbox = tuple(float(value) for value in payload["bbox"])
            pose = tuple(float(value) for value in payload.get("pose", (0, 0, 0)))
            image_size = tuple(int(value) for value in payload["image_size"])
            embedding = base64.b64decode(str(payload["embedding"]), validate=True)
            if len(bbox) != 4 or len(pose) != 3 or len(image_size) != 2 or not embedding:
                raise ValueError("Incomplete identity analysis")
        except (KeyError, TypeError, ValueError) as exc:
            raise IdentityProviderError("Identity analyzer returned invalid output.") from exc
        return IdentityAnalysis(
            provider=str(payload.get("provider") or "insightface-buffalo-l"),
            face_count=int(payload.get("face_count", 0)),
            bbox=bbox,  # type: ignore[arg-type]
            det_score=float(payload.get("det_score", 0.0)),
            pose=pose,  # type: ignore[arg-type]
            image_size=image_size,  # type: ignore[arg-type]
            sharpness=float(payload.get("sharpness", 0.0)),
            brightness=float(payload.get("brightness", 0.0)),
            dark_clip_ratio=float(payload.get("dark_clip_ratio", 0.0)),
            light_clip_ratio=float(payload.get("light_clip_ratio", 0.0)),
            embedding=embedding,
        )

    def verify(self, source: Path, target: Path) -> IdentityVerification:
        payload = self._run(
            "verify",
            "--source",
            str(source),
            "--target",
            str(target),
        )
        return IdentityVerification(
            score=float(payload.get("similarity", 0.0)),
            threshold=self.threshold,
            provider=str(payload.get("provider") or "insightface-buffalo-l"),
        )

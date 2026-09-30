"""FastAPI application for the ForgeAI suite."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import asdict, replace
import base64
import binascii
import hashlib
from io import BytesIO
import json
import logging
import os
import ipaddress
from pathlib import Path
import re
import secrets
import shutil
import socket
import sqlite3
import subprocess
import sys
from threading import Lock, Thread
from time import monotonic
from typing import Literal
from urllib.parse import quote, urlencode, urlsplit
from uuid import uuid4

import qrcode
import requests
from fastapi import FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, SecretStr, ValidationError

from .analytics import summarize_analytics
from .auth import (
    SESSION_DAYS,
    TOTP_ISSUER,
    AuthenticationError,
    AuthenticationRateLimitError,
    account_can_access_job_generation,
    account_for_session,
    authenticate_authenticator,
    authenticate_account,
    authenticator_account,
    begin_authenticator_setup,
    confirm_authenticator_setup,
    create_account,
    create_job_access,
    create_session,
    initialize_auth_database,
    registration_is_open,
    revoke_session,
    update_max_active_jobs,
    set_session_workspace,
    workspace_for_session,
    verify_generation_access,
    verify_job_access,
    verify_job_generation_access,
)
from .database import (
    BackendConfig,
    DATA_DIR,
    configured_backends,
    configured_data_dir,
    write_storage_config,
    set_generation_favorite,
    delete_generation,
    delete_jobs,
    GENERATED_DIR,
    THUMBNAILS_DIR,
    IDENTITY_PROFILES_DIR,
    UPLOADS_DIR,
    VIDEOS_DIR,
    JobRecord,
    JobQueueLimitError,
    DuplicateJobSubmission,
    analytics_records,
    average_generation_duration,
    create_job,
    create_batch_with_items,
    cancel_pending_batch_items,
    create_identity_profile,
    delete_identity_profile,
    finish_job,
    finish_prompt_run,
    generation_filename,
    get_generation,
    get_batch,
    get_batch_items,
    get_job,
    get_identity_profile,
    identity_profile_has_active_job,
    initialize_database,
    job_kind_for_generation,
    rate_generation,
    recent_generations,
    recent_batches,
    recent_jobs,
    list_identity_profiles,
    recent_prompt_history,
    resumable_job_ids,
    reconcile_batch_jobs,
    retry_failed_batch_items,
    save_generation,
    save_prompt_run,
    touch_identity_profile,
    rename_identity_profile,
    update_job_request_payload,
    update_job_progress,
    delete_video_generation,
    get_video_generation,
    recent_video_generations,
    save_video_generation,
    set_video_favorite,
)
from .comfy import CAMERA_MOTIONS, INTERPOLATION_MULTIPLIERS, OUTPUT_RESOLUTIONS, T2V_SIZES, ComfyApiClient, build_workflow, derive_i2v_size, derive_upscale_size
from .forge import (
    RATING_RULES,
    ForgeApiClient,
    GenerationSettings,
    Img2ImgPayload,
    Txt2ImgPayload,
    build_img2img_payload,
    build_instantid_payload,
    build_payload,
    aspect_ratios_for,
    generate_random_prompt_pair,  # legacy import/test seam; removal belongs to phase 5
    model_profile_for,
    nearest_native_resolution,
    release_other_backends,
)
from .identity import IdentityProviderError, InsightFaceLocalIdentityProvider
from .ollama_prompt import OllamaPromptClient
from .cloud_prompt import CloudPromptClient, CloudPromptError
from .video_prompt import analyse_source_frame, cloud_video_prompt, compose_i2v_prompt, curated_video_prompt, ollama_video_prompt, strip_sdxl_tags, validate_video_prompt
from .engine_bridge import (
    engine_model_for,
    generate_curated,
    generate_pair as engine_generate_pair,
)
from .prompt_engine import PromptEngineError, ReasonCode
from .batch_options import get_batch_options
from .batch_constraints import BatchConstraintError, validate_facet_selections
from .batch_models import BatchCreateRequest
from .batch_planner import build_batch_plan
from .novelty import (
    CREATIVITY_LEVELS,
    NoveltyContext,
    create_novelty_context,
    maximum_similarity,
)
from .jobs import JobExecutionError, JobRunner
from .batch_coordinator import BatchCoordinator
from . import backend_manager


LOGGER = logging.getLogger(__name__)
APP_BOOT_ID = secrets.token_urlsafe(24)
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
MAX_UPLOAD_BODY_BYTES = MAX_UPLOAD_BYTES + 64 * 1024
MAX_IMG2IMG_BODY_BYTES = 32 * 1024 * 1024
UPLOAD_CHUNK_BYTES = 1024 * 1024
UPLOAD_CONTENT_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})
UPLOAD_FORMATS = frozenset({"PNG", "JPEG", "WEBP"})
UPLOAD_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
CHARACTER_REFINEMENT_SCALE = 1.5
CHARACTER_REFINEMENT_UPSCALER = "4x_NMKD-Siax_200k"
CHARACTER_CODEFORMER_VISIBILITY = 0.6
CHARACTER_CODEFORMER_WEIGHT = 0.6
QUALITY_MODES = ("normal", "high", "super", "4k", "8k", "12k")
QualityMode = Literal["normal", "high", "super", "4k", "8k", "12k"]


def _resolve_quality_mode(
    quality_mode: QualityMode | None, high_res: bool, super_res: bool
) -> QualityMode:
    """Resolve the enum while preserving legacy boolean request semantics."""
    if quality_mode is not None:
        return quality_mode
    if super_res:
        return "super"
    if high_res:
        return "high"
    return "normal"


def _quality_flags(mode: QualityMode) -> tuple[bool, bool]:
    """Map a mode onto the byte-compatible legacy Forge detail recipe."""
    return mode != "normal", mode in {"super", "8k", "12k"}


def _quality_target_size(width: int, height: int, mode: QualityMode) -> tuple[int, int]:
    scale = {
        "normal": 1.0,
        "high": 1.5,
        "super": 2.0,
        "4k": 3.0,
        "8k": 8.0,
        "12k": 12.0,
    }[mode]
    return round(width * scale), round(height * scale)


class GenerateRequest(BaseModel):
    """Validated generation request accepted from the React client."""

    prompt: str = Field(min_length=1, max_length=8000)
    negative_prompt: str = Field(default="", max_length=8000)
    model: str = "Forge default"
    style: str = "Photoreal"
    aspect_ratio: str = "Portrait (832x1216)"
    content_rating: str = "Safe"
    high_res: bool = False
    super_res: bool = False
    quality_mode: QualityMode | None = None
    prompt_run_id: int | None = Field(default=None, ge=1)
    prompt_engine: str = "manual"
    creativity_level: str = "balanced"
    orientation: Literal["mixed", "front", "side", "back"] = "mixed"
    # None keeps the checkpoint profile's tuned CFG. Bounded because an
    # out-of-range value reaches Forge and wastes a full render.
    cfg_scale: float | None = Field(default=None, ge=1.0, le=12.0)
    request_id: str | None = Field(default=None, min_length=8, max_length=100)
    seed: int = Field(default=-1, ge=-1, le=2**63 - 1)


class UpscaleRequest(BaseModel):
    """Request to re-render an existing generation with the detail pass enabled."""

    super_res: bool = False
    access_token: SecretStr | None = None
    request_id: str | None = Field(default=None, min_length=8, max_length=100)


class Img2ImgRequest(BaseModel):
    """Transform an authorized gallery image through Forge img2img."""

    source_generation_id: int | None = Field(default=None, ge=1)
    upload_id: str | None = Field(default=None, min_length=32, max_length=32)
    identity_profile_id: int | None = Field(default=None, ge=1)
    prompt: str = Field(default="", max_length=8000)
    negative_prompt: str = Field(default="", max_length=8000)
    model: str = "Forge default"
    style: str = "Photoreal"
    content_rating: str = "Safe"
    preset: str | None = None
    denoising_strength: float = Field(default=0.4, ge=0.05, le=0.95)
    cfg_scale: float | None = Field(default=None, ge=1.0, le=12.0)
    prompt_engine: str = "manual"
    request_id: str | None = Field(default=None, min_length=8, max_length=100)
    access_token: SecretStr | None = None
    mask: str | None = None
    auto_mask: Literal["person", "body", "features"] | None = None
    mask_blur: int = Field(default=8, ge=0, le=64)
    inpaint_full_res: bool = True
    inpaint_full_res_padding: int = Field(default=32, ge=0, le=256)
    invert_mask: bool = False
    soft_inpainting: bool = False
    face_identity: bool = False
    # Background replacement generates only the scene: the subject is masked
    # out and preserved, so subject and rating scaffolding in the prompt fights
    # the very region being generated.
    region_only: Literal["", "scene", "region"] = ""


class UploadResponse(BaseModel):
    """Opaque handle and sanitized dimensions for one uploaded source."""

    upload_id: str
    width: int
    height: int


class UploadFromUrlRequest(BaseModel):
    """Remote HTTPS image selected by an authenticated user."""

    url: str = Field(min_length=1, max_length=2048)


class MaskProposalRequest(BaseModel):
    """Authorized source and detector family for a Simple-mode mask."""

    source_generation_id: int | None = Field(default=None, ge=1)
    upload_id: str | None = Field(default=None, min_length=32, max_length=32)
    access_token: SecretStr | None = None
    detector: Literal["person", "body", "garment", "features"]


class MaskProposal(BaseModel):
    id: str
    label: str
    mask: str


class MaskProposalResponse(BaseModel):
    proposals: list[MaskProposal]


class TransformSuggestionRequest(BaseModel):
    source_generation_id: int | None = Field(default=None, ge=1)
    upload_id: str | None = Field(default=None, min_length=32, max_length=32)
    identity_profile_id: int | None = Field(default=None, ge=1)
    access_token: SecretStr | None = None
    kind: Literal["background", "outfit", "scene"]
    prompt_engine: Literal["curated", "engine", "ollama", "cloud"] = "curated"
    ollama_model: str | None = None
    cloud_provider: str | None = None
    cloud_model: str | None = None
    cloud_api_key: SecretStr | None = None


class TransformSuggestionResponse(BaseModel):
    suggestion: str
    engine: str
    notice: str | None = None


class IdentityCrop(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    width: float = Field(gt=0.0, le=1.0)
    height: float = Field(gt=0.0, le=1.0)


class IdentitySourceRequest(BaseModel):
    source_generation_id: int | None = Field(default=None, ge=1)
    upload_id: str | None = Field(default=None, min_length=32, max_length=32)
    access_token: SecretStr | None = None


class IdentityCropRequest(IdentitySourceRequest):
    crop: IdentityCrop


class IdentityProfileCreateRequest(IdentityCropRequest):
    name: str = Field(min_length=1, max_length=60)


class IdentityProfileRenameRequest(BaseModel):
    name: str = Field(min_length=1, max_length=60)


class IdentityCropSuggestionResponse(BaseModel):
    crop: IdentityCrop


class IdentityValidationResponse(BaseModel):
    usable: bool
    status: Literal["ready", "usable", "invalid"]
    reasons: list[str]
    metrics: dict[str, object]


class IdentityProfileResponse(BaseModel):
    id: int
    name: str
    thumbnail_url: str
    quality: dict[str, object]
    created_at: str
    updated_at: str
    last_used_at: str | None = None


class PixelUpscaleRequest(BaseModel):
    """Request to resize an existing image with no re-render, at any stage."""

    multiplier: float = Field(ge=1.0, le=4.0)
    access_token: SecretStr | None = None


class GenerationResponse(BaseModel):
    """Serializable generation metadata and public image URLs."""

    id: int
    thumbnail_url: str
    full_url: str
    timestamp: str
    positive_prompt: str
    negative_prompt: str
    width: int
    height: int
    rating: str
    style: str
    style_variant: str | None = None
    quality_mode: str = "normal"
    high_res: bool
    super_res: bool
    model: str
    user_rating: int | None = None
    favorite: bool = False
    feedback_reasons: list[str] = Field(default_factory=list)
    sampler_name: str
    scheduler: str
    steps: int
    cfg_scale: float
    seed: int
    duration_seconds: float | None = None
    diffusion_duration_seconds: float | None = None
    upscale_duration_seconds: float | None = None
    source_generation_id: int | None = None
    job_id: int | None = None
    access_token: str | None = None


class HistoryPageResponse(BaseModel):
    """One cursor-addressable page of persistent generations."""

    items: list[GenerationResponse]
    next_cursor: int | None = None


class SurpriseRequest(BaseModel):
    """Complete active generation context for randomized prompt construction."""

    model: str = "Forge default"
    style: str = "Photoreal"
    content_rating: str = "Safe"
    aspect_ratio: str = "Portrait (832x1216)"
    high_res: bool = False
    super_res: bool = False
    quality_mode: QualityMode | None = None
    prompt_engine: str = "curated"
    creativity_level: str = "balanced"
    orientation: Literal["mixed", "front", "side", "back"] = "mixed"
    cfg_scale: float | None = Field(default=None, ge=1.0, le=12.0)
    ollama_model: str | None = None
    cloud_provider: str | None = None
    cloud_model: str | None = None
    cloud_api_key: SecretStr | None = None
    request_id: str | None = Field(default=None, min_length=8, max_length=100)
    batch_profile: None = None
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1)


class CloudCredentialRequest(BaseModel):
    """One-shot cloud credential validation; the secret is never persisted."""

    provider: str
    api_key: SecretStr


class CloudCredentialResponse(BaseModel):
    """Safe provider metadata returned after successful authentication."""

    provider: str
    models: list[str]
    recommended: str


class AuthRegisterRequest(BaseModel):
    """Validated fields for creating a local ForgeAI account."""

    display_name: str = Field(min_length=2, max_length=50)
    email: str = Field(min_length=5, max_length=254)
    password: SecretStr


class AuthLoginRequest(BaseModel):
    """Credentials used for a local account session."""

    email: str = Field(min_length=5, max_length=254)
    password: SecretStr


class AuthenticatorSetupRequest(BaseModel):
    """Password confirmation required before displaying a new TOTP secret."""

    password: SecretStr = Field(min_length=10, max_length=256)


class AuthenticatorCodeRequest(BaseModel):
    """A six-digit TOTP or one saved recovery code."""

    code: SecretStr = Field(min_length=6, max_length=32)


class AuthUserResponse(BaseModel):
    """Public account identity returned to the React client."""

    id: int
    email: str
    display_name: str
    is_admin: bool
    created_at: str
    authenticator_enabled: bool = False
    max_active_jobs: int = 3


class StorageSettingsRequest(BaseModel):
    """A new artifact storage directory chosen from Settings."""

    data_dir: str = Field(min_length=2, max_length=4096)


class StorageSettingsResponse(BaseModel):
    """Where artifacts are stored now, and where they will be after a restart."""

    active_dir: str
    configured_dir: str | None
    pending_restart: bool
    env_override: bool
    free_gb: float | None = None


class BackendSettingsRequest(BaseModel):
    """Editable fields for a local Forge installation."""

    port: int = Field(ge=1, le=65535)
    package_dir: str = Field(min_length=2, max_length=4096)


class AccountSettingsRequest(BaseModel):
    """Durable account-level execution preferences."""

    max_active_jobs: int = Field(ge=1, le=50)


class WorkspaceSelectionRequest(BaseModel):
    """Creator product selected after credentials have been validated."""

    workspace: Literal["forgeai", "forgeimg", "forgevid", "forgebat"]


class VideoRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)
    negative_prompt: str = Field(default="low quality, blurry, distorted, morphing, flickering, jitter, text, watermark", max_length=4000)
    seed: int = Field(default=-1, ge=-1, le=1125899906842624)
    source_generation_id: int | None = Field(default=None, ge=1)
    upload_id: str | None = Field(default=None, max_length=64)
    camera_motion: str = Field(default="Zoom In", max_length=40)
    interpolation: Literal[1, 2, 4] = 2
    output_resolution: Literal["native", "720p", "1080p"] = "native"
    steps: int = Field(default=20, ge=20, le=30)
    prompt_engine: Literal["curated", "ollama", "cloud", "reuse", "manual"] = "manual"
    prompt_run_id: int | None = Field(default=None, ge=1)
    orientation: Literal["landscape", "portrait", "square"] = "landscape"


class VideoPromptRequest(BaseModel):
    engine: Literal["curated", "ollama", "cloud", "reuse"]
    mode: Literal["t2v", "i2v"] = "t2v"
    source_prompt: str = Field(default="", max_length=4000)
    source_generation_id: int | None = Field(default=None, ge=1)
    upload_id: str | None = Field(default=None, max_length=64)
    ollama_model: str | None = None
    cloud_provider: str | None = None
    cloud_model: str | None = None
    cloud_api_key: SecretStr | None = None


class VideoPromptResponse(BaseModel):
    prompt: str
    negative_prompt: str
    prompt_run_id: int
    engine: str
    # Set when a frame was analysed, so the UI can align the camera embedding
    # with the motion the prompt actually describes.
    camera_motion: str | None = None
    analysis_note: str | None = None


class VideoResponse(BaseModel):
    id: int
    prompt: str
    negative_prompt: str
    mode: str
    source_generation_id: int | None
    width: int
    height: int
    frame_count: int
    fps: int
    seed: int
    duration_seconds: float
    camera_motion: str | None
    favorite: bool
    created_at: str
    file_url: str
    poster_url: str | None = None


class VideoHistoryResponse(BaseModel):
    items: list[VideoResponse]
    next_cursor: int | None = None


class VideoStatusResponse(BaseModel):
    connected: bool
    backend: str = "ComfyUI"


class VideoFavoriteRequest(BaseModel):
    favorite: bool


class AuthSessionResponse(BaseModel):
    """Current browser authentication state."""

    authenticated: bool
    user: AuthUserResponse | None = None
    registration_open: bool = False
    authenticator_available: bool = False
    authenticator_display_name: str | None = None
    active_workspace: Literal["forgeai", "forgeimg", "forgevid", "forgebat"] | None = None
    boot_id: str = APP_BOOT_ID


class AuthenticatorSetupResponse(BaseModel):
    """One-time enrollment material for a compatible authenticator app."""

    manual_secret: str
    otpauth_uri: str
    qr_data_url: str


class AuthenticatorConfirmResponse(BaseModel):
    """The activated account and recovery codes shown exactly once."""

    user: AuthUserResponse
    recovery_codes: list[str]


class ProgressResponse(BaseModel):
    """Normalized live or history-estimated generation progress."""

    active: bool
    source: str
    percent: float
    current_step: int | None = None
    total_steps: int | None = None
    eta_seconds: float | None = None
    current_image: str | None = None
    stage: str


class PromptPairResponse(BaseModel):
    """Randomized positive and negative prompt pair."""

    prompt: str
    negative_prompt: str
    engine: str = "curated"
    notice: str | None = None
    prompt_run_id: int


class RatingRequest(BaseModel):
    """Validated user quality feedback for one generated image."""

    score: int = Field(ge=1, le=5)
    reasons: list[
        Literal[
            "prompt_match",
            "eyes",
            "anatomy",
            "composition",
            "style",
            "detail",
            "content_rating",
        ]
    ] = Field(default_factory=list, max_length=7)
    access_token: SecretStr | None = None


class RatingResponse(BaseModel):
    """Persisted image-quality feedback."""

    generation_id: int
    score: int
    reasons: list[str]


class JobStartResponse(BaseModel):
    """Capability-bearing handle returned immediately after enqueueing."""

    job_id: int
    access_token: str | None = None


class JobResponse(BaseModel):
    """Observable state and optional final artifact for one durable job."""

    id: int
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    request_payload: dict[str, object]
    progress_percent: float
    progress_stage: str
    current_step: int | None = None
    total_steps: int | None = None
    eta_seconds: float | None = None
    current_image: str | None = None
    generation_id: int | None = None
    generation: GenerationResponse | None = None
    positive_prompt: str | None = None
    negative_prompt: str | None = None
    error_message: str | None = None


class JobListGenerationResponse(BaseModel):
    """Generation fields used by job-history rows and their lightbox."""

    id: int
    thumbnail_url: str
    full_url: str
    width: int
    height: int
    rating: str
    style_variant: str | None = None
    seed: int
    source_generation_id: int | None = None
    favorite: bool = False


class JobListResponse(BaseModel):
    """Compact observable state for one job-history row."""

    id: int
    kind: str
    status: str
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    request_payload: dict[str, object]
    progress_percent: float
    progress_stage: str
    current_step: int | None = None
    total_steps: int | None = None
    eta_seconds: float | None = None
    current_image: str | None = None
    generation_id: int | None = None
    generation: JobListGenerationResponse | None = None
    error_message: str | None = None


class JobsPageResponse(BaseModel):
    """One cursor-addressable page of jobs."""

    items: list[JobListResponse]
    next_cursor: int | None = None


class GenerationLineageNodeResponse(BaseModel):
    """One real persisted generation in an ancestor chain."""

    id: int
    source_generation_id: int | None = None
    pipeline_kind: str | None = None
    timestamp: str
    model: str
    style: str
    rating: str
    width: int
    height: int
    high_res: bool
    super_res: bool
    quality_mode: QualityMode | None = None
    thumbnail_url: str | None = None


class GenerationLineageResponse(BaseModel):
    """Oldest-to-newest generation ancestry returned in one request."""

    generation_id: int
    items: list[GenerationLineageNodeResponse]


def serialize_generation(record: object) -> GenerationResponse:
    """Attach same-origin static URLs to a database generation record."""
    values = asdict(record)  # type: ignore[arg-type]
    values.pop("filename")
    # A relative URL works through Vite's development proxy and through a
    # same-origin production reverse proxy, including reliable downloads.
    image_url = f"/api/images/{values['id']}"
    thumbnail_url = f"/api/images/{values['id']}/thumbnail"
    return GenerationResponse(
        **values,
        thumbnail_url=thumbnail_url,
        full_url=image_url,
    )


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistence and supervise the sole Forge worker."""
    global _job_runner, _batch_coordinator
    initialize_database()
    initialize_auth_database()
    _job_runner = JobRunner(_execute_job)
    _job_runner.start()
    for job_id in resumable_job_ids():
        _job_runner.enqueue(job_id)
    _batch_coordinator = BatchCoordinator(_job_runner)
    _batch_coordinator.start()
    try:
        yield
    finally:
        _batch_coordinator.stop()
        _batch_coordinator = None
        _job_runner.stop()
        _job_runner = None


app = FastAPI(title="NyxForge API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def limit_image_request_bodies(request: Request, call_next):
    """Bound upload and base64-mask requests before application parsing."""
    limit = (
        MAX_UPLOAD_BODY_BYTES
        if request.url.path == "/api/uploads"
        else MAX_IMG2IMG_BODY_BYTES
        if request.url.path == "/api/img2img"
        else None
    )
    if limit is None or request.method != "POST":
        return await call_next(request)
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            if int(content_length) > limit:
                return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
        except ValueError:
            return JSONResponse(status_code=422, content={"detail": "Invalid request body."})
    body = bytearray()
    async for chunk in request.stream():
        body.extend(chunk)
        if len(body) > limit:
            return JSONResponse(status_code=413, content={"detail": "Request body is too large."})
    request._body = bytes(body)  # Starlette replays the bounded cache downstream.
    return await call_next(request)
_progress_lock = Lock()
_progress_sessions: dict[str, dict[str, object]] = {}
AUTH_COOKIE = "aiqg_session"
# Set once a browser has signed in here. Carries no identity and grants no
# access; it only marks a browser as one that may be offered the
# authenticator shortcut, so a stranger is not told an admin exists.
KNOWN_BROWSER_COOKIE = "aiqg_known"
_job_runner: JobRunner | None = None
_batch_coordinator: BatchCoordinator | None = None


def _raise_if_job_cancelled(job_id: int | None) -> None:
    """Stop a worker at safe boundaries after a cancellation request."""
    runner = _job_runner
    if job_id is not None and runner is not None and runner.cancellation_requested(job_id):
        raise JobExecutionError("Cancelled by user.")


def _create_prompt_progress_session(request_id: str | None) -> None:
    """Register the prompt-intelligence phase for Simple mode."""
    if request_id:
        with _progress_lock:
            _progress_sessions[request_id] = {
                "started_at": monotonic(),
                "stage_started_at": monotonic(),
                "stage": "prompt_generation",
                "milestone_percent": 4.0,
            }


def _set_progress_stage(
    request_id: str | None, stage: str, milestone_percent: float
) -> None:
    """Advance an in-flight workflow to an observable local milestone."""
    if not request_id:
        return
    with _progress_lock:
        session = _progress_sessions.get(request_id)
        if session is not None:
            session.update(
                stage=stage,
                milestone_percent=max(0.0, min(100.0, milestone_percent)),
                stage_started_at=monotonic(),
            )


def _start_progress_session(
    request_id: str | None,
    expected: float,
    total_steps: int,
    *,
    high_res: bool,
    super_res: bool,
) -> None:
    """Register one in-flight request for honest progress or fallback timing."""
    if request_id:
        with _progress_lock:
            now = monotonic()
            session = _progress_sessions.setdefault(
                request_id, {"started_at": now}
            )
            session.update(
                forge_started_at=now,
                stage_started_at=now,
                expected=expected,
                total_steps=total_steps,
                high_res=high_res,
                super_res=super_res,
                stage="model_loading",
                milestone_percent=14.0,
            )


def _finish_progress_session(request_id: str | None) -> None:
    if request_id:
        with _progress_lock:
            _progress_sessions.pop(request_id, None)


@app.get("/api/status")
def status() -> dict[str, object]:
    """Expose Forge connectivity for the sidebar status chip."""
    client = ForgeApiClient()
    connected = client.is_reachable()
    # Forge answering /docs is not the same as Forge being able to work. A
    # phantom job leaves it reachable while every request queues behind it,
    # which otherwise looks like the app silently doing nothing.
    stalled = client.stalled_job_name() if connected else None
    return {"connected": connected, "stalled_job": stalled}


@app.get("/api/instance")
def instance_identity() -> dict[str, str]:
    """Identify the local server without performing an external health check."""
    return {"app": "nyx-forge", "version": app.version}


def _set_auth_cookie(response: Response, request: Request, token: str) -> None:
    """Attach a private, same-site session cookie suitable for local or HTTPS use."""
    response.set_cookie(
        key=AUTH_COOKIE,
        value=token,
        max_age=SESSION_DAYS * 24 * 60 * 60,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )
    response.set_cookie(
        key=KNOWN_BROWSER_COOKIE,
        value="1",
        max_age=365 * 24 * 60 * 60,
        httponly=True,
        secure=request.url.scheme == "https",
        samesite="lax",
        path="/",
    )


def _auth_response(
    account: object, active_workspace: str | None = None
) -> AuthSessionResponse:
    values = asdict(account)  # type: ignore[arg-type]
    return AuthSessionResponse(
        authenticated=True,
        user=AuthUserResponse(**values),
        registration_open=False,
        authenticator_available=bool(values.get("authenticator_enabled")),
        authenticator_display_name=str(values["display_name"]),
        active_workspace=active_workspace,
    )


@app.get("/api/auth/session", response_model=AuthSessionResponse)
def auth_session(request: Request) -> AuthSessionResponse:
    """Return the signed-in account, or an explicit guest state."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    if account is None:
        known = request.cookies.get(KNOWN_BROWSER_COOKIE) is not None
        authenticator_user = authenticator_account() if known else None
        return AuthSessionResponse(
            authenticated=False,
            registration_open=registration_is_open(),
            authenticator_available=authenticator_user is not None,
            authenticator_display_name=(
                authenticator_user.display_name if authenticator_user else None
            ),
        )
    return _auth_response(
        account, workspace_for_session(request.cookies.get(AUTH_COOKIE))
    )


@app.post("/api/auth/workspace", response_model=AuthSessionResponse)
def auth_workspace(
    body: WorkspaceSelectionRequest, request: Request
) -> AuthSessionResponse:
    """Set or switch the creator workspace for this authenticated session."""
    token = request.cookies.get(AUTH_COOKIE)
    account = _require_authenticated_account(request)
    try:
        selected = set_session_workspace(token, body.workspace)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not selected:
        raise HTTPException(status_code=401, detail="Your session has expired. Sign in again.")
    return _auth_response(account, body.workspace)


@app.patch("/api/auth/settings", response_model=AuthUserResponse)
def auth_settings(body: AccountSettingsRequest, request: Request) -> AuthUserResponse:
    """Persist queue preferences for the signed-in account."""
    account = _require_authenticated_account(request)
    try:
        updated = update_max_active_jobs(account.id, body.max_active_jobs)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return AuthUserResponse(**asdict(updated))


def _storage_state() -> StorageSettingsResponse:
    """Describe the active and configured storage directories."""
    configured = configured_data_dir()
    override = bool(os.environ.get("NYX_FORGE_DATA_DIR"))
    free_gb = None
    try:
        free_gb = round(shutil.disk_usage(DATA_DIR).free / 1e9, 1)
    except OSError:
        pass
    return StorageSettingsResponse(
        active_dir=str(DATA_DIR),
        configured_dir=str(configured) if configured else None,
        # DATA_DIR is bound at import, so a newly saved path needs a restart.
        pending_restart=bool(configured and configured != DATA_DIR),
        env_override=override,
        free_gb=free_gb,
    )


@app.get("/api/settings/storage", response_model=StorageSettingsResponse)
def get_storage_settings(request: Request) -> StorageSettingsResponse:
    """Report where artifacts are stored."""
    _require_authenticated_account(request)
    return _storage_state()


@app.put("/api/settings/storage", response_model=StorageSettingsResponse)
def set_storage_settings(
    body: StorageSettingsRequest, request: Request
) -> StorageSettingsResponse:
    """Persist a new artifact storage directory for the next restart."""
    _require_authenticated_account(request)
    candidate = Path(body.data_dir.strip()).expanduser()
    if not candidate.is_absolute():
        raise HTTPException(status_code=422, detail="Enter an absolute path.")
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise HTTPException(
            status_code=422, detail=f"That folder cannot be created: {exc.strerror or exc}"
        ) from exc
    if not os.access(candidate, os.W_OK):
        raise HTTPException(status_code=422, detail="That folder is not writable.")
    probe = candidate / ".aiqg-write-test"
    try:
        probe.write_bytes(b"ok")
        probe.unlink()
    except OSError as exc:
        raise HTTPException(
            status_code=422, detail=f"That folder is not writable: {exc.strerror or exc}"
        ) from exc
    try:
        write_storage_config({"data_dir": str(candidate)})
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save the setting: {exc.strerror or exc}"
        ) from exc
    return _storage_state()


@app.post("/api/auth/register", response_model=AuthSessionResponse)
def auth_register(
    body: AuthRegisterRequest, request: Request, response: Response
) -> AuthSessionResponse:
    """Create a local account and immediately establish its browser session."""
    password = body.password.get_secret_value()
    try:
        account = create_account(body.email, body.display_name, password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    _set_auth_cookie(response, request, create_session(account.id))
    return _auth_response(account)


@app.post("/api/auth/login", response_model=AuthSessionResponse)
def auth_login(
    body: AuthLoginRequest, request: Request, response: Response
) -> AuthSessionResponse:
    """Authenticate a local account and rotate its HTTP-only browser session."""
    try:
        account = authenticate_account(body.email, body.password.get_secret_value())
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    revoke_session(request.cookies.get(AUTH_COOKIE))
    _set_auth_cookie(response, request, create_session(account.id))
    return _auth_response(account)


@app.post("/api/auth/logout", response_model=AuthSessionResponse)
def auth_logout(request: Request, response: Response) -> AuthSessionResponse:
    """Revoke the current session and clear its browser cookie."""
    revoke_session(request.cookies.get(AUTH_COOKIE))
    response.delete_cookie(AUTH_COOKIE, path="/", samesite="lax")
    known = request.cookies.get(KNOWN_BROWSER_COOKIE) is not None
    authenticator_user = authenticator_account() if known else None
    return AuthSessionResponse(
        authenticated=False,
        registration_open=registration_is_open(),
        authenticator_available=authenticator_user is not None,
        authenticator_display_name=(
            authenticator_user.display_name if authenticator_user else None
        ),
    )


def _admin_account(request: Request):
    """Return the local administrator or reject the request."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    if account is None or not account.is_admin:
        raise HTTPException(status_code=403, detail="Admin sign-in required.")
    return account


@app.get("/api/backends")
def backend_statuses() -> list[dict[str, object]]:
    """Report process and API health for every configured local backend."""
    return [item.as_dict() for item in backend_manager.statuses()]


@app.get("/api/settings/backends")
def backend_settings(request: Request) -> list[dict[str, object]]:
    """Return process-control configuration to the local administrator."""
    _admin_account(request)
    return [
        {
            "id": item.id,
            "label": item.label,
            "port": item.port,
            "package_dir": item.package_dir,
        }
        for item in configured_backends()
    ]


@app.post("/api/settings/backends/browse")
def browse_backend_package(request: Request) -> dict[str, object]:
    """Open the local Windows picker and return a validated Forge package."""
    _admin_account(request)
    if sys.platform != "win32":
        raise HTTPException(
            status_code=501,
            detail="The native Forge file picker is currently available on Windows only.",
        )
    script = r"""
Add-Type -AssemblyName System.Windows.Forms
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$dialog = New-Object System.Windows.Forms.OpenFileDialog
$dialog.Title = 'Select the Forge launch.py file'
$dialog.Filter = 'Forge launcher (launch.py)|launch.py'
$dialog.CheckFileExists = $true
$dialog.Multiselect = $false
try {
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        [Console]::Out.Write($dialog.FileName)
    }
} finally {
    $dialog.Dispose()
}
"""
    try:
        completed = subprocess.run(
            [
                "powershell.exe",
                "-NoProfile",
                "-STA",
                "-Command",
                script,
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            check=True,
        )
    except subprocess.TimeoutExpired as exc:
        raise HTTPException(
            status_code=504,
            detail="The Forge file picker timed out. Try Browse again or paste the path manually.",
        ) from exc
    except (OSError, subprocess.SubprocessError) as exc:
        raise HTTPException(
            status_code=500,
            detail="The Windows file picker could not be opened. Paste the package path manually.",
        ) from exc

    selected = completed.stdout.strip().lstrip("\ufeff")
    if not selected:
        return {"cancelled": True, "package_dir": None}
    launch_script = Path(selected)
    if launch_script.name.casefold() != "launch.py" or not launch_script.is_file():
        raise HTTPException(
            status_code=422,
            detail="Select the launch.py file inside the Forge package folder.",
        )
    return {"cancelled": False, "package_dir": str(launch_script.parent)}


@app.put("/api/backends/{backend_id}")
def update_backend_settings(
    backend_id: str, body: BackendSettingsRequest, request: Request
) -> dict[str, object]:
    """Persist editable backend settings without allowing remote hosts."""
    _admin_account(request)
    package_dir = Path(body.package_dir.strip()).expanduser()
    if not package_dir.is_absolute():
        raise HTTPException(status_code=422, detail="Enter an absolute package path.")
    if not package_dir.is_dir():
        raise HTTPException(
            status_code=422,
            detail="That package folder does not exist or is not a directory.",
        )
    if not (package_dir / "launch.py").is_file():
        raise HTTPException(
            status_code=422,
            detail="Select the Forge package folder that contains launch.py.",
        )
    backends = list(configured_backends())
    for index, current in enumerate(backends):
        if current.id == backend_id:
            backends[index] = BackendConfig(
                id=current.id,
                label=current.label,
                port=body.port,
                package_dir=str(package_dir),
                launch_args=current.launch_args,
            )
            break
    else:
        raise HTTPException(status_code=404, detail="Backend not found.")
    try:
        write_storage_config({"backends": [asdict(item) for item in backends]})
    except OSError as exc:
        raise HTTPException(
            status_code=500, detail=f"Could not save the setting: {exc.strerror or exc}"
        ) from exc
    return backend_manager.status(backend_id).as_dict()


def _control_backend(backend_id: str, operation: str) -> dict[str, object]:
    try:
        action = getattr(backend_manager, operation)
        return action(backend_id).as_dict()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Backend not found.") from exc
    except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@app.post("/api/backends/{backend_id}/start")
def start_backend(backend_id: str, request: Request) -> dict[str, object]:
    _admin_account(request)
    return _control_backend(backend_id, "start")


@app.post("/api/backends/{backend_id}/stop")
def stop_backend(backend_id: str, request: Request) -> dict[str, object]:
    _admin_account(request)
    return _control_backend(backend_id, "stop")


@app.post("/api/backends/{backend_id}/restart")
def restart_backend(backend_id: str, request: Request) -> dict[str, object]:
    _admin_account(request)
    return _control_backend(backend_id, "restart")


def _authenticator_uri(secret: str, account_email: str) -> str:
    """Build the interoperable Google Authenticator key URI."""
    label = quote(f"{TOTP_ISSUER}:{account_email}", safe="")
    query = urlencode(
        {
            "secret": secret,
            "issuer": TOTP_ISSUER,
            "algorithm": "SHA1",
            "digits": "6",
            "period": "30",
        }
    )
    return f"otpauth://totp/{label}?{query}"


def _qr_data_url(value: str) -> str:
    """Render enrollment material in memory so no secret-bearing file is written."""
    qr = qrcode.QRCode(
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=7,
        border=4,
    )
    qr.add_data(value)
    qr.make(fit=True)
    image = qr.make_image(fill_color="black", back_color="white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@app.post("/api/auth/authenticator/setup", response_model=AuthenticatorSetupResponse)
def authenticator_setup(
    body: AuthenticatorSetupRequest, request: Request, response: Response
) -> AuthenticatorSetupResponse:
    """Start or replace authenticator enrollment after password confirmation."""
    signed_in = _admin_account(request)
    try:
        secret, account = begin_authenticator_setup(
            signed_in.id, body.password.get_secret_value()
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    uri = _authenticator_uri(secret, account.email)
    response.headers["Cache-Control"] = "no-store"
    return AuthenticatorSetupResponse(
        manual_secret=secret,
        otpauth_uri=uri,
        qr_data_url=_qr_data_url(uri),
    )


@app.post(
    "/api/auth/authenticator/confirm", response_model=AuthenticatorConfirmResponse
)
def authenticator_confirm(
    body: AuthenticatorCodeRequest, request: Request, response: Response
) -> AuthenticatorConfirmResponse:
    """Verify the scanned secret before making authenticator login active."""
    signed_in = _admin_account(request)
    try:
        account, recovery_codes = confirm_authenticator_setup(
            signed_in.id, body.code.get_secret_value()
        )
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    response.headers["Cache-Control"] = "no-store"
    return AuthenticatorConfirmResponse(
        user=AuthUserResponse(**asdict(account)), recovery_codes=recovery_codes
    )


@app.post("/api/auth/authenticator/login", response_model=AuthSessionResponse)
def authenticator_login(
    body: AuthenticatorCodeRequest, request: Request, response: Response
) -> AuthSessionResponse:
    """Create a browser session from one TOTP or unused recovery code."""
    try:
        account = authenticate_authenticator(body.code.get_secret_value())
    except AuthenticationRateLimitError as exc:
        raise HTTPException(status_code=429, detail=str(exc)) from exc
    except AuthenticationError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc
    revoke_session(request.cookies.get(AUTH_COOKIE))
    _set_auth_cookie(response, request, create_session(account.id))
    return _auth_response(account)


def _require_safe_rating(request: Request, content_rating: str) -> None:
    """Reject any request outside the public Safe-only generation boundary."""
    del request
    if content_rating != "Safe":
        raise HTTPException(
            status_code=422,
            detail="NyxForge public edition supports Safe content only.",
        )


def _require_authenticated_account(request: Request):
    """Require a signed-in local account without broadening guest capabilities."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    if account is None:
        raise HTTPException(status_code=403, detail="Sign in is required.")
    return account


def _require_workspace(request: Request, required: Literal["forgeai", "forgeimg", "forgevid", "forgebat"]):
    """Require the authenticated session's currently selected workspace."""
    account = _require_authenticated_account(request)
    selected = workspace_for_session(request.cookies.get(AUTH_COOKIE))
    if selected is None:
        raise HTTPException(
            status_code=409,
            detail="Choose a creator workspace before starting creator work.",
        )
    if selected != required:
        label = {
            "forgeai": "ForgeAI", "forgeimg": "ForgeIMG",
            "forgevid": "ForgeVID", "forgebat": "ForgeBAT",
        }[required]
        raise HTTPException(
            status_code=403,
            detail=f"Switch to {label} to start this operation.",
        )
    return account


def _upload_path(upload_id: str) -> Path | None:
    """Resolve only canonical opaque upload handles within UPLOADS_DIR."""
    if UPLOAD_ID_PATTERN.fullmatch(upload_id) is None:
        return None
    return UPLOADS_DIR / f"{upload_id}.png"


def _sanitize_and_store_upload(buffered: BytesIO) -> UploadResponse:
    """Verify, normalize, and atomically store one untrusted image body."""
    try:
        buffered.seek(0)
        with Image.open(buffered) as source:
            if source.format not in UPLOAD_FORMATS:
                raise ValueError("Unsupported image format.")
            if source.width > 8192 or source.height > 8192:
                raise ValueError("Image dimensions exceed the limit.")
            source.load()
            sanitized = ImageOps.exif_transpose(source).convert("RGB")
        if sanitized.width > 2048 or sanitized.height > 2048:
            sanitized.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded image is invalid.") from exc
    upload_id = secrets.token_hex(16)
    destination = UPLOADS_DIR / f"{upload_id}.png"
    temporary = UPLOADS_DIR / f".{upload_id}.tmp"
    try:
        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
        sanitized.save(temporary, format="PNG", optimize=True)
        temporary.replace(destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise HTTPException(status_code=507, detail="The upload could not be stored.") from exc
    return UploadResponse(upload_id=upload_id, width=sanitized.width, height=sanitized.height)


def _validate_public_https_url(value: str) -> str:
    """Reject non-HTTPS and any hostname resolving outside public IP space."""
    try:
        parsed = urlsplit(value)
        port = parsed.port or 443
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="Enter a valid HTTPS image URL.") from exc
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise HTTPException(status_code=422, detail="Image links must use HTTPS.")
    try:
        addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise HTTPException(status_code=422, detail="The image link host could not be resolved.") from exc
    if not addresses:
        raise HTTPException(status_code=422, detail="The image link host could not be resolved.")
    for address in addresses:
        try:
            resolved = ipaddress.ip_address(str(address[4][0]).split("%", 1)[0])
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="The image link resolved to an invalid address.") from exc
        if not resolved.is_global:
            raise HTTPException(status_code=422, detail="Private or local image links are not allowed.")
    return value


@app.post("/api/uploads", response_model=UploadResponse)
async def upload_image(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Validate and sanitize one untrusted image into a bounded PNG input."""
    _require_authenticated_account(request)
    if file.content_type not in UPLOAD_CONTENT_TYPES:
        raise HTTPException(status_code=422, detail="The uploaded image is invalid.")
    buffered = BytesIO()
    total = 0
    try:
        while chunk := await file.read(UPLOAD_CHUNK_BYTES):
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="The uploaded image exceeds 25 MB.")
            buffered.write(chunk)
    finally:
        await file.close()
    return _sanitize_and_store_upload(buffered)


@app.post("/api/uploads/from-url", response_model=UploadResponse)
def upload_image_from_url(body: UploadFromUrlRequest, request: Request) -> UploadResponse:
    """Fetch one public HTTPS image with SSRF and streaming-size defenses."""
    _require_authenticated_account(request)
    url = _validate_public_https_url(body.url.strip())
    response: requests.Response | None = None
    try:
        response = requests.get(
            url,
            stream=True,
            allow_redirects=False,
            timeout=(5, 15),
            headers={"Accept": "image/png,image/jpeg,image/webp"},
        )
        if 300 <= response.status_code < 400:
            raise HTTPException(status_code=422, detail="Redirecting image links are not allowed.")
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if content_type not in UPLOAD_CONTENT_TYPES:
            raise HTTPException(status_code=422, detail="The linked file is not a supported image.")
        buffered = BytesIO()
        total = 0
        for chunk in response.iter_content(chunk_size=UPLOAD_CHUNK_BYTES):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="The uploaded image exceeds 25 MB.")
            buffered.write(chunk)
    except HTTPException:
        raise
    except requests.RequestException as exc:
        raise HTTPException(status_code=422, detail="The image link could not be fetched.") from exc
    finally:
        if response is not None:
            response.close()
    return _sanitize_and_store_upload(buffered)


@app.get("/api/uploads/{upload_id}")
def uploaded_image(upload_id: str, request: Request) -> FileResponse:
    """Preview one opaque sanitized upload for an authenticated picker."""
    _require_authenticated_account(request)
    path = _upload_path(upload_id)
    if path is None or not path.is_file():
        raise HTTPException(status_code=404, detail="Upload not found.")
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, no-store"})


def _identity_profile_response(record) -> IdentityProfileResponse:
    return IdentityProfileResponse(
        id=record.id,
        name=record.name,
        thumbnail_url=f"/api/identity-profiles/{record.id}/thumbnail",
        quality=record.quality,
        created_at=record.created_at,
        updated_at=record.updated_at,
        last_used_at=record.last_used_at,
    )


def _authorized_identity_source(body: IdentitySourceRequest, request: Request) -> Path:
    if (body.source_generation_id is None) == (body.upload_id is None):
        raise HTTPException(
            status_code=422,
            detail="Supply exactly one source_generation_id or upload_id.",
        )
    if body.source_generation_id is not None:
        source = get_generation(body.source_generation_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        access = body.access_token.get_secret_value() if body.access_token else None
        if not _can_access_generation(request, source.id, access):
            raise HTTPException(status_code=403, detail="Image access denied.")
        path = GENERATED_DIR / source.filename
    else:
        _require_authenticated_account(request)
        path = _upload_path(body.upload_id or "")
        if path is None:
            raise HTTPException(status_code=422, detail="Invalid upload identifier.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Source image file is missing.")
    return path


def _identity_crop_image(source_path: Path, crop: IdentityCrop) -> Image.Image:
    try:
        with Image.open(source_path) as opened:
            source = ImageOps.exif_transpose(opened).convert("RGB")
            x1 = int(round(crop.x * source.width))
            y1 = int(round(crop.y * source.height))
            x2 = int(round((crop.x + crop.width) * source.width))
            y2 = int(round((crop.y + crop.height) * source.height))
            if x2 > source.width or y2 > source.height or x2 <= x1 or y2 <= y1:
                raise HTTPException(status_code=422, detail="The face crop is outside the source image.")
            if min(x2 - x1, y2 - y1) < 192:
                raise HTTPException(status_code=422, detail="Make the face crop at least 192 pixels wide and high.")
            return source.crop((x1, y1, x2, y2))
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="The source image is invalid.") from exc


def _analyze_identity_crop(crop: Image.Image) -> tuple[IdentityValidationResponse, bytes, str]:
    token = uuid4().hex
    temporary = IDENTITY_PROFILES_DIR / f".validate-{token}.png"
    IDENTITY_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    try:
        crop.save(temporary, format="PNG", optimize=True)
        analysis = InsightFaceLocalIdentityProvider().analyze(temporary)
    except IdentityProviderError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        temporary.unlink(missing_ok=True)

    width, height = analysis.image_size
    x1, y1, x2, y2 = analysis.bbox
    face_width = max(0.0, x2 - x1)
    face_height = max(0.0, y2 - y1)
    face_area_ratio = (face_width * face_height) / max(1.0, width * height)
    fatal: list[str] = []
    warnings: list[str] = []
    if analysis.face_count != 1:
        fatal.append("Crop the image so it contains exactly one face.")
    if min(face_width, face_height) < 128:
        fatal.append("The detected face is too small; zoom in before saving.")
    if analysis.det_score < 0.55:
        fatal.append("The face detector is not confident enough in this crop.")
    margin_x, margin_y = width * 0.015, height * 0.015
    if x1 <= margin_x or y1 <= margin_y or x2 >= width - margin_x or y2 >= height - margin_y:
        fatal.append("Leave a little space around the entire face.")
    if face_area_ratio < 0.12:
        warnings.append("The face is relatively small; a tighter crop may preserve identity better.")
    if face_area_ratio > 0.72:
        warnings.append("The crop is very tight; include a little more head context.")
    if analysis.sharpness < 45:
        warnings.append("The reference is soft or blurred.")
    if not 45 <= analysis.brightness <= 215:
        warnings.append("The face is unusually dark or bright.")
    if max(abs(value) for value in analysis.pose) > 35:
        warnings.append("A more front-facing reference will usually preserve identity better.")
    if analysis.dark_clip_ratio > 0.25 or analysis.light_clip_ratio > 0.25:
        warnings.append("Large parts of the crop have clipped shadows or highlights.")
    status: Literal["ready", "usable", "invalid"] = (
        "invalid" if fatal else "usable" if warnings else "ready"
    )
    metrics: dict[str, object] = {
        "face_count": analysis.face_count,
        "detector_confidence": round(analysis.det_score, 4),
        "face_pixel_size": [round(face_width), round(face_height)],
        "face_area_ratio": round(face_area_ratio, 4),
        "sharpness": round(analysis.sharpness, 2),
        "brightness": round(analysis.brightness, 2),
        "pose": [round(value, 2) for value in analysis.pose],
        "crop_size": [width, height],
    }
    return (
        IdentityValidationResponse(
            usable=not fatal,
            status=status,
            reasons=[*fatal, *warnings],
            metrics=metrics,
        ),
        analysis.embedding,
        analysis.provider,
    )


@app.post("/api/identity-profiles/suggest-crop", response_model=IdentityCropSuggestionResponse)
def suggest_identity_crop(body: IdentitySourceRequest, request: Request) -> IdentityCropSuggestionResponse:
    _require_authenticated_account(request)
    source_path = _authorized_identity_source(body, request)
    try:
        with Image.open(source_path) as opened:
            width, height = ImageOps.exif_transpose(opened).size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=422, detail="The source image is invalid.") from exc
    box = _detected_face_box(source_path)
    if box is None:
        size = min(width, height) * 0.7
        x, y = (width - size) / 2, (height - size) / 2
    else:
        x1, y1, x2, y2 = box
        face_width, face_height = x2 - x1, y2 - y1
        size = min(float(min(width, height)), max(face_width, face_height) * 1.35)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        x = min(max(0.0, center_x - size / 2), width - size)
        y = min(max(0.0, center_y - size / 2), height - size)
    return IdentityCropSuggestionResponse(
        crop=IdentityCrop(x=x / width, y=y / height, width=size / width, height=size / height)
    )


@app.post("/api/identity-profiles/validate", response_model=IdentityValidationResponse)
def validate_identity_crop(body: IdentityCropRequest, request: Request) -> IdentityValidationResponse:
    _require_authenticated_account(request)
    crop = _identity_crop_image(_authorized_identity_source(body, request), body.crop)
    validation, _, _ = _analyze_identity_crop(crop)
    return validation


@app.post("/api/identity-profiles", response_model=IdentityProfileResponse, status_code=201)
def create_identity_profile_endpoint(
    body: IdentityProfileCreateRequest, request: Request
) -> IdentityProfileResponse:
    account = _require_authenticated_account(request)
    name = " ".join(body.name.strip().split())
    if not name:
        raise HTTPException(status_code=422, detail="Give this Face ID a name.")
    crop = _identity_crop_image(_authorized_identity_source(body, request), body.crop)
    validation, embedding, provider = _analyze_identity_crop(crop)
    if not validation.usable:
        raise HTTPException(status_code=422, detail=" ".join(validation.reasons))
    token = uuid4().hex
    filename = f"identity-{token}.png"
    thumbnail_filename = f"identity-{token}.jpg"
    image_path = IDENTITY_PROFILES_DIR / filename
    thumbnail_path = IDENTITY_PROFILES_DIR / thumbnail_filename
    try:
        crop.save(image_path, format="PNG", optimize=True)
        thumbnail = ImageOps.fit(crop, (320, 320), method=Image.Resampling.LANCZOS)
        thumbnail.save(thumbnail_path, format="JPEG", quality=86, optimize=True)
        record = create_identity_profile(
            account_id=account.id,
            name=name,
            filename=filename,
            thumbnail_filename=thumbnail_filename,
            embedding=embedding,
            detector=provider,
            embedding_model=provider,
            quality={"status": validation.status, "reasons": validation.reasons, **validation.metrics},
        )
    except (OSError, sqlite3.Error) as exc:
        image_path.unlink(missing_ok=True)
        thumbnail_path.unlink(missing_ok=True)
        raise HTTPException(status_code=507, detail="The Face ID could not be saved.") from exc
    return _identity_profile_response(record)


@app.get("/api/identity-profiles", response_model=list[IdentityProfileResponse])
def identity_profiles(request: Request) -> list[IdentityProfileResponse]:
    account = _require_authenticated_account(request)
    return [_identity_profile_response(record) for record in list_identity_profiles(account.id)]


@app.get("/api/identity-profiles/{profile_id}/thumbnail")
def identity_profile_thumbnail(profile_id: int, request: Request) -> FileResponse:
    account = _require_authenticated_account(request)
    record = get_identity_profile(profile_id, account_id=account.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Face ID not found.")
    path = IDENTITY_PROFILES_DIR / record.thumbnail_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Face ID preview is missing.")
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, no-store"})


@app.patch("/api/identity-profiles/{profile_id}", response_model=IdentityProfileResponse)
def update_identity_profile(
    profile_id: int, body: IdentityProfileRenameRequest, request: Request
) -> IdentityProfileResponse:
    account = _require_authenticated_account(request)
    name = " ".join(body.name.strip().split())
    if not name or not rename_identity_profile(profile_id, account.id, name):
        raise HTTPException(status_code=404, detail="Face ID not found.")
    record = get_identity_profile(profile_id, account_id=account.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Face ID not found.")
    return _identity_profile_response(record)


@app.delete("/api/identity-profiles/{profile_id}")
def remove_identity_profile(profile_id: int, request: Request) -> dict[str, bool]:
    account = _require_authenticated_account(request)
    if identity_profile_has_active_job(profile_id, account.id):
        raise HTTPException(
            status_code=409,
            detail="This Face ID is being used by an active character job.",
        )
    record = delete_identity_profile(profile_id, account.id)
    if record is None:
        raise HTTPException(status_code=404, detail="Face ID not found.")
    (IDENTITY_PROFILES_DIR / record.filename).unlink(missing_ok=True)
    (IDENTITY_PROFILES_DIR / record.thumbnail_filename).unlink(missing_ok=True)
    return {"deleted": True}


# The checkpoints this app is actually built and tuned for. Forge reports
# every file in its models directory, including stray copies and one-off
# downloads; offering those invites generations nothing here is calibrated for.
def _model_filename(title: str) -> str:
    """The checkpoint's bare filename, ignoring which folder an instance uses."""
    return title.rsplit("\\", 1)[-1].split(" [", 1)[0].casefold()


def _forge_error_summary(exc: requests.HTTPError) -> str:
    """Return Forge's own error text from a failed response, if it sent any."""
    if exc.response is None:
        return ""
    try:
        body = exc.response.json()
    except ValueError:
        return exc.response.text[:200].strip()
    if not isinstance(body, dict):
        return ""
    parts = [str(body.get(key, "")).strip() for key in ("error", "message", "detail")]
    return " - ".join(part for part in parts if part)[:300]


SUPPORTED_MODEL_MARKERS = (
    "realvisxlv50_v50bakedvae",
    "waiillustrioussdxl_v170",
    "cyberrealisticpony",
    "juggernautxl_ragnarok",
    "ponydiffusionv6xl",
    "realismbystableyogi",
    # The GGUF build only. flux1-dev-bnb-nf4-v2 fails to load in this Forge
    # build with flattened state_dict shapes, so it is not offered.
    "flux1-dev-q5_k_s",
)


def _is_supported_model(title: str) -> bool:
    """Return whether a Forge checkpoint title is one this app is tuned for."""
    filename = title.rsplit("\\", 1)[-1].casefold().replace(" ", "")
    # A duplicate such as "name (1).safetensors" is the same checkpoint twice.
    if "(1)" in filename:
        return False
    return any(marker in filename for marker in SUPPORTED_MODEL_MARKERS)


@app.get("/api/models")
def models() -> dict[str, object]:
    """Return available Forge checkpoints and connectivity state."""
    registry = configured_backends()
    titles: list[str] = []
    known: set[str] = set()
    default_connected = False
    default_client = ForgeApiClient()
    for index, backend in enumerate(registry):
        client = ForgeApiClient(backend.id)
        try:
            backend_titles = client.available_models()
            if index == 0:
                default_connected = True
            for title in backend_titles:
                filename = _model_filename(title)
                if filename not in known:
                    known.add(filename)
                    titles.append(title)
        except (requests.RequestException, ValueError):
            LOGGER.info("Forge backend %s is not reachable; skipping its models", backend.id)
    if not titles:
        default = "Forge default"
        profile = model_profile_for(default)
        return {
            "models": [default],
            "profiles": {
                default: {
                    **asdict(profile),
                    "display_name": "Forge Default",
                    "aspect_ratios": list(aspect_ratios_for(default)),
                }
            },
            "connected": False,
        }
    try:
        available = [title for title in titles if _is_supported_model(title)]
        profiles: dict[str, dict[str, object]] = {}
        for title in available:
            profile_values: dict[str, object] = {
                **asdict(model_profile_for(title)),
                "aspect_ratios": list(aspect_ratios_for(title)),
            }
            if profile_values["display_name"] == "General SDXL":
                filename = title.rsplit("\\", 1)[-1]
                profile_values["display_name"] = filename.split(
                    ".safetensors", 1
                )[0].replace("_", " ")
            # Forge can report duplicate files with the same checkpoint hash.
            # Keep both selectable while making the copy unambiguous to users.
            if "(1).safetensors" in title.lower():
                profile_values["display_name"] = (
                    f"{profile_values['display_name']} (copy)"
                )
            profiles[title] = profile_values
        return {
            "models": available,
            "profiles": profiles,
            "connected": default_connected,
            "stalled_job": default_client.stalled_job_name() if default_connected else None,
        }
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.get("/api/batch/options")
def batch_options(
    request: Request,
    facet: str | None = None,
    model: str | None = None,
    content_rating: str = "Safe",
    style: str = "Photoreal",
    orientation: str = "front",
    orientations: str | None = None,
    search: str | None = None,
) -> dict[str, object]:
    """Return lazily-loaded ForgeBAT choices derived from the keyword base."""
    _require_workspace(request, "forgebat")
    _require_safe_rating(request, content_rating)
    try:
        selected_orientations = tuple(
            value.strip() for value in (orientations or orientation).split(",") if value.strip()
        )
        return get_batch_options(
            facet=facet,
            model=model,
            content_rating=content_rating,
            style=style,
            orientations=selected_orientations,
            search=search,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _validated_batch_constraints(body: BatchCreateRequest) -> dict[str, frozenset[str]]:
    if engine_model_for(body.model) is None:
        raise HTTPException(
            status_code=422,
            detail={
                "facet": "model", "concept": body.model,
                "reason": "model_unsupported",
                "message": "The selected model is not supported by the structured prompt engine.",
            },
        )
    if body.aspect_ratio not in aspect_ratios_for(body.model):
        raise HTTPException(
            status_code=422,
            detail={
                "facet": "aspect_ratio", "concept": body.aspect_ratio,
                "reason": "unsupported_aspect_ratio",
                "message": "The selected aspect ratio is not supported by this model.",
            },
        )
    try:
        return validate_facet_selections(
            {key: value.model_dump(mode="python") for key, value in body.facets.items()},
            model=body.model,
            content_rating=body.content_rating,
            orientations=body.orientations,
        )
    except BatchConstraintError as exc:
        raise HTTPException(status_code=422, detail=exc.detail()) from exc


@app.post("/api/batch/preview")
def preview_batch(body: BatchCreateRequest, request: Request) -> dict[str, object]:
    """Compile a deterministic batch plan without starting any GPU work."""
    _require_workspace(request, "forgebat")
    _require_safe_rating(request, body.content_rating)
    allowed = _validated_batch_constraints(body)
    try:
        plan = build_batch_plan(
            count=body.count,
            allowed_ids=allowed,
            model=body.model,
            content_rating=body.content_rating,
            style=body.style,
            styles=body.styles,
            orientation=body.orientation,
            orientations=body.orientations,
            distribution=body.distribution,
            planner_seed=body.batch_seed,
            request_id=body.request_id,
            quality_mode=body.quality_mode,
            aspect_ratio=body.aspect_ratio,
        )
    except (PromptEngineError, ValueError) as exc:
        reason_codes = getattr(exc, "reason_codes", ())
        LOGGER.warning(
            "ForgeBAT preview rejected request_id=%s planner_seed=%s model=%r "
            "rating=%s orientation=%s distribution=%s facets=%s reason=%s",
            body.request_id,
            body.batch_seed,
            body.model,
            body.content_rating,
            body.orientation,
            body.distribution,
            {
                key: selection.model_dump(mode="json")
                for key, selection in body.facets.items()
            },
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail={
                "facet": "recipe",
                "concept": None,
                "reason": reason_codes[0].value if reason_codes else "compiler_rejected",
                "message": str(exc),
            },
        ) from exc
    return plan.as_dict(preview_limit=12)


@app.post("/api/batches", status_code=202)
def create_batch(body: BatchCreateRequest, request: Request) -> dict[str, object]:
    """Validate, plan and durably persist a batch before scheduling children."""
    account = _require_workspace(request, "forgebat")
    _require_safe_rating(request, body.content_rating)
    allowed = _validated_batch_constraints(body)
    try:
        plan = build_batch_plan(
            count=body.count,
            allowed_ids=allowed,
            model=body.model,
            content_rating=body.content_rating,
            style=body.style,
            styles=body.styles,
            orientation=body.orientation,
            orientations=body.orientations,
            distribution=body.distribution,
            planner_seed=body.batch_seed,
            request_id=body.request_id,
            quality_mode=body.quality_mode,
            aspect_ratio=body.aspect_ratio,
        )
        record = create_batch_with_items(
            account_id=account.id,
            request_id=body.request_id,
            requested_count=body.count,
            planner_seed=body.batch_seed,
            config=body.model_dump(mode="json"),
            plan_items=[item.as_dict(include_prompts=True) for item in plan.items],
        )
    except (PromptEngineError, ValueError, sqlite3.Error) as exc:
        if isinstance(exc, sqlite3.Error):
            LOGGER.exception("Could not persist ForgeBAT plan")
            raise HTTPException(
                status_code=503,
                detail="The batch could not be persisted; no generation was started.",
            ) from exc
        reason_codes = getattr(exc, "reason_codes", ())
        raise HTTPException(
            status_code=422,
            detail={
                "facet": "recipe", "concept": None,
                "reason": reason_codes[0].value if reason_codes else "compiler_rejected",
                "message": str(exc),
            },
        ) from exc
    if _batch_coordinator is not None:
        _batch_coordinator.wake()
    return {
        "batch_id": record.id,
        "request_id": record.request_id,
        "status": record.status,
        "requested_count": record.requested_count,
    }


def _serialize_batch(batch_id: int, account_id: int, *, include_items: bool) -> dict[str, object]:
    reconcile_batch_jobs(batch_id)
    record = get_batch(batch_id)
    if record is None or record.account_id != account_id:
        raise HTTPException(status_code=404, detail="Batch not found.")
    items = get_batch_items(batch_id)
    counts = {
        status: sum(item.status == status for item in items)
        for status in ("pending", "queued", "running", "succeeded", "failed", "cancelled")
    }
    terminal_count = counts["succeeded"] + counts["failed"] + counts["cancelled"]
    value: dict[str, object] = {
        "id": record.id,
        "request_id": record.request_id,
        "status": record.status,
        "requested_count": record.requested_count,
        "planner_seed": record.planner_seed,
        "config": record.config,
        "counts": counts,
        "completed_count": counts["succeeded"],
        "terminal_count": terminal_count,
        "progress_percent": round(terminal_count / max(record.requested_count, 1) * 100, 2),
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "started_at": record.started_at,
        "finished_at": record.finished_at,
    }
    if include_items:
        generations = {
            item.generation_id: get_generation(item.generation_id)
            for item in items
            if item.generation_id is not None
        }
        value["items"] = [
            {
                "id": item.id,
                "index": item.item_index,
                "request_id": item.request_id,
                "status": item.status,
                "resolved_facets": item.plan.get("resolved_facets", {}),
                "job_id": item.job_id,
                "generation_id": item.generation_id,
                "error": item.error,
                "thumbnail_url": (
                    f"/api/images/{item.generation_id}/thumbnail"
                    if item.generation_id is not None else None
                ),
                "full_url": (
                    f"/api/images/{item.generation_id}"
                    if item.generation_id is not None else None
                ),
                "favorite": bool(generations[item.generation_id].favorite)
                if item.generation_id is not None and generations.get(item.generation_id) is not None
                else False,
            }
            for item in items
        ]
    return value


@app.get("/api/batches")
def list_batches(request: Request, limit: int = Query(default=50, ge=1, le=100)) -> dict[str, object]:
    # Batch status is shared shell context: it remains readable while the user
    # works elsewhere. Creating or mutating a batch still requires ForgeBAT.
    account = _require_authenticated_account(request)
    return {
        "batches": [
            _serialize_batch(record.id, account.id, include_items=False)
            for record in recent_batches(account.id, limit=limit)
        ]
    }


@app.get("/api/batches/{batch_id}")
def batch_detail(batch_id: int, request: Request) -> dict[str, object]:
    account = _require_workspace(request, "forgebat")
    return _serialize_batch(batch_id, account.id, include_items=True)


@app.post("/api/batches/{batch_id}/cancel")
def cancel_batch(batch_id: int, request: Request) -> dict[str, object]:
    """Cancel only the remaining work linked to one batch."""
    account = _require_workspace(request, "forgebat")
    record = get_batch(batch_id)
    if record is None or record.account_id != account.id:
        raise HTTPException(status_code=404, detail="Batch not found.")
    if record.status in {"completed", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="This batch has already finished.")
    items = get_batch_items(batch_id)
    cancel_pending_batch_items(batch_id)
    runner = _job_runner
    if runner is None:
        raise HTTPException(status_code=503, detail="The local job worker is unavailable.")
    for item in items:
        if item.status not in {"queued", "running"} or item.job_id is None:
            continue
        job = get_job(item.job_id)
        if (
            job is None
            or job.kind != "batch_generate"
            or int(job.request_payload.get("batch_id", -1)) != batch_id
        ):
            continue
        non_interruptible = job.progress_stage in {"decoding", "upscaling", "saving"}
        try:
            outcome = runner.request_cancel(
                job.id,
                interrupt_active=(None if non_interruptible else ForgeApiClient().interrupt_generation),
                allow_active=not non_interruptible,
            )
        except requests.RequestException:
            LOGGER.warning("Forge did not accept batch %s child interrupt", batch_id, exc_info=True)
            continue
        if outcome == "queued":
            finish_job(job.id, status="cancelled", error_message="Cancelled before execution.")
    reconcile_batch_jobs(batch_id)
    if _batch_coordinator is not None:
        _batch_coordinator.wake()
    return _serialize_batch(batch_id, account.id, include_items=True)


@app.post("/api/batches/{batch_id}/retry-failed")
def retry_batch_failures(batch_id: int, request: Request) -> dict[str, object]:
    """Retry failed items from their existing plans without replanning."""
    account = _require_workspace(request, "forgebat")
    record = get_batch(batch_id)
    if record is None or record.account_id != account.id:
        raise HTTPException(status_code=404, detail="Batch not found.")
    reconcile_batch_jobs(batch_id)
    retried = retry_failed_batch_items(batch_id)
    if not retried:
        raise HTTPException(status_code=409, detail="This batch has no failed items to retry.")
    if _batch_coordinator is not None:
        _batch_coordinator.wake()
    value = _serialize_batch(batch_id, account.id, include_items=True)
    value["retried_count"] = retried
    return value


@app.get("/api/prompt-models")
def prompt_models() -> dict[str, object]:
    """Discover local Ollama text models without making Ollama mandatory."""
    try:
        available = list(OllamaPromptClient().available_models())
    except (requests.RequestException, ValueError, TypeError):
        return {"connected": False, "models": [], "recommended": None}
    preference = ("gemma4:12b", "qwen3:8b", "llama3.1:8b")
    recommended = next((name for name in preference if name in available), None)
    return {
        "connected": True,
        "models": available,
        "recommended": recommended or (available[0] if available else None),
    }


def validate_cloud_credentials(body: CloudCredentialRequest) -> CloudCredentialResponse:
    """Validate one API key without caching, logging, or returning it."""
    api_key = body.api_key.get_secret_value()
    if not 8 <= len(api_key) <= 512:
        raise HTTPException(status_code=422, detail="The API key appears incomplete.")
    try:
        result = CloudPromptClient().validate_credentials(
            body.provider, api_key
        )
    except CloudPromptError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
    return CloudCredentialResponse(
        provider=result.provider,
        models=list(result.models),
        recommended=result.recommended,
    )


@app.post("/api/cloud/test", response_model=CloudCredentialResponse)
def test_cloud_credentials_endpoint(
    body: CloudCredentialRequest, request: Request
) -> CloudCredentialResponse:
    """Restrict provider configuration to the local administrator."""
    _admin_account(request)
    return validate_cloud_credentials(body)


@app.get("/api/history", response_model=HistoryPageResponse)
def history(
    request: Request,
    limit: int = Query(60, ge=1, le=100),
    before_id: int | None = Query(default=None, ge=1),
) -> HistoryPageResponse:
    """Return one stable cursor page of the administrator's local history."""
    _admin_account(request)
    records = recent_generations(limit + 1, before_id=before_id)
    has_more = len(records) > limit
    page = records[:limit]
    return HistoryPageResponse(
        items=[serialize_generation(item) for item in page],
        next_cursor=page[-1].id if has_more and page else None,
    )


@app.get("/api/analytics")
def analytics(
    request: Request, days: int = Query(30, ge=0, le=3650)
) -> dict[str, object]:
    """Return local, aggregate generation telemetry for the dashboard."""
    _admin_account(request)
    runs, attempts = analytics_records(days)
    return summarize_analytics(runs, attempts, days=days)


def _authorize_generation_file(
    request: Request, generation_id: int, access: str | None
) -> str:
    """Resolve an image filename for an admin or a valid guest capability."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    account_allowed = bool(
        account
        and (
            account.is_admin
            or account_can_access_job_generation(account.id, generation_id)
        )
    )
    if not account_allowed and not (
        verify_generation_access(generation_id, access)
        or verify_job_generation_access(generation_id, access)
    ):
        raise HTTPException(status_code=403, detail="Image access denied.")
    filename = generation_filename(generation_id)
    if filename is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    return filename


@app.get("/api/images/{generation_id}", response_class=FileResponse)
def generation_image(
    generation_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> FileResponse:
    """Serve a full image only to the admin or its originating guest."""
    filename = _authorize_generation_file(request, generation_id, access)
    path = GENERATED_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    return FileResponse(
        path,
        media_type="image/png",
        filename=f"generation-{generation_id}.png",
        content_disposition_type="inline",
    )


@app.get("/api/images/{generation_id}/thumbnail", response_class=FileResponse)
def generation_thumbnail(
    generation_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> FileResponse:
    """Serve the protected lightweight preview for one generation."""
    filename = _authorize_generation_file(request, generation_id, access)
    path = THUMBNAILS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Thumbnail not found.")
    return FileResponse(path, media_type="image/jpeg")


@app.delete("/api/images/{generation_id}", status_code=204)
def delete_generation_image(
    generation_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> Response:
    """Delete a generation available to the same principals that can view it."""
    _authorize_generation_file(request, generation_id, access)
    if delete_generation(generation_id) is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    return Response(status_code=204)


@app.get("/api/generate/progress", response_model=ProgressResponse)
def generation_progress(request_id: str = Query(min_length=8, max_length=100)) -> ProgressResponse:
    """Proxy Forge's live snapshot, falling back to observed generation time."""
    with _progress_lock:
        session = _progress_sessions.get(request_id)
    if session is None:
        return ProgressResponse(active=False, source="none", percent=0, stage="queued")

    session = dict(session)
    stage = str(session.get("stage", "queued"))
    milestone = float(session.get("milestone_percent", 0.0))
    if stage in {"prompt_generation", "prompt_ready", "decoding", "upscaling", "saving"}:
        return ProgressResponse(
            active=True,
            source="workflow",
            percent=round(milestone, 1),
            eta_seconds=None,
            stage=stage,
        )

    elapsed = max(
        0.0,
        monotonic() - float(session.get("forge_started_at", session.get("started_at", monotonic()))),
    )
    expected = max(1.0, float(session.get("expected", 30.0)))
    estimated_percent = min(92.0, elapsed / expected * 100.0)
    high_res = bool(session.get("high_res", False))
    try:
        snapshot = ForgeApiClient().generation_progress()
        progress = max(0.0, min(1.0, float(snapshot.get("progress", 0) or 0)))
        state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
        current_step = int(state.get("sampling_step", 0) or 0)
        total_steps = int(state.get("sampling_steps", session["total_steps"]) or session["total_steps"])
        preview = snapshot.get("current_image")
        eta = max(0.0, float(snapshot.get("eta_relative", 0) or 0))
        if isinstance(preview, str) and preview and not preview.startswith("data:"):
            preview = f"data:image/png;base64,{preview}"
        stage = (
            "model_loading"
            if progress <= 0 and current_step <= 0
            else "finalizing"
            if progress >= 0.98 or (total_steps > 0 and current_step >= total_steps)
            else "refining"
            if high_res and progress >= 0.68
            else "sampling"
        )
        forge_span = 70.0 if high_res else 78.0
        overall_percent = milestone if stage == "model_loading" else 14.0 + progress * forge_span
        return ProgressResponse(
            active=True,
            source="forge",
            percent=round(min(92.0, overall_percent), 1),
            current_step=current_step or None,
            total_steps=total_steps or None,
            eta_seconds=round(eta, 1) if eta > 0 else None,
            current_image=preview if isinstance(preview, str) and preview else None,
            stage=stage,
        )
    except (requests.RequestException, ValueError, TypeError):
        overall_percent = min(88.0, 14.0 + estimated_percent * 0.78)
        stage = (
            "model_loading"
            if estimated_percent < 8
            else "refining"
            if high_res and estimated_percent >= 70
            else "sampling"
        )
        return ProgressResponse(
            active=True,
            source="estimate",
            percent=round(overall_percent, 1),
            eta_seconds=round(max(0.0, expected - elapsed), 1),
            stage=stage,
        )


def _safe_finish_prompt_run(
    prompt_run_id: int,
    *,
    status: str,
    generation_id: int | None = None,
    error_message: str | None = None,
) -> bool:
    """Finalize telemetry without masking the generation's primary outcome."""
    try:
        finish_prompt_run(
            prompt_run_id,
            status=status,
            generation_id=generation_id,
            error_message=error_message,
        )
    except sqlite3.Error:
        LOGGER.exception(
            "Could not finalize prompt run %s as %s", prompt_run_id, status
        )
        return False
    return True


def _save_prompt_run_or_http(**values: object) -> int:
    """Persist a prompt attempt or stop before consuming GPU time."""
    try:
        return save_prompt_run(**values)  # type: ignore[arg-type]
    except sqlite3.Error as exc:
        LOGGER.exception("Could not create the prompt-run record")
        raise HTTPException(
            status_code=503,
            detail="Local history is temporarily unavailable. No generation was started.",
        ) from exc


def _quality_artifact(job_id: int, name: str) -> Path:
    directory = GENERATED_DIR / "quality-jobs" / f"job-{job_id}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _ensure_model_backend(model: str | None) -> ForgeApiClient:
    """Make exactly the selected model runner available.

    The local Forge Neo instance is a FLUX appliance with instance-wide model
    components. It must not sit beside reForge consuming VRAM for ordinary
    SD/SDXL/Pony work. Selecting a FLUX profile is the sole automatic start
    signal; selecting any other profile stops Neo before generation.
    """
    profile = model_profile_for(model or "Forge default")
    backend_id = profile.forge_backend_id
    forge = ForgeApiClient.for_model(model)

    if backend_id == "forge_neo":
        release_other_backends(forge.base_url)
        state = backend_manager.status("forge_neo")
        if not state.reachable:
            backend_manager.start("forge_neo")
            state = backend_manager.wait_until_reachable("forge_neo", timeout=300.0)
        if not state.reachable:
            raise RuntimeError(
                "Forge Neo could not start for the selected FLUX model."
            )
        return forge

    neo_state = backend_manager.status("forge_neo")
    if neo_state.running:
        backend_manager.stop("forge_neo")
    default_id = configured_backends()[0].id
    state = backend_manager.status(default_id)
    if not state.reachable:
        backend_manager.start(default_id)
        state = backend_manager.wait_until_reachable(default_id, timeout=300.0)
    if not state.reachable:
        raise RuntimeError("reForge could not start for the selected model.")
    return forge


def _run_quality_pipeline(
    payload: Txt2ImgPayload,
    mode: QualityMode,
    *,
    job_id: int | None,
    on_stage,
) -> tuple[bytes, int, float, float, Path | None]:
    """Execute a fixed quality ladder, resuming persisted 12K artifacts."""
    high_res, super_res = _quality_flags(mode)
    forge = _ensure_model_backend(payload.model_checkpoint)
    if mode != "12k" or job_id is None:
        generated = forge.generate_image(
            payload,
            high_res=high_res,
            super_res=super_res,
            on_stage=on_stage,
        )
        image_bytes = generated.image_bytes
        upscale_seconds = generated.upscale_duration_seconds
        if mode in {"4k", "8k", "12k"}:
            targets = (
                [(payload.width * 8, payload.height * 8),
                 (payload.width * 12, payload.height * 12)]
                if mode == "12k"
                else [_quality_target_size(payload.width, payload.height, mode)]
            )
            for index, (target_width, target_height) in enumerate(targets, start=1):
                on_stage(
                    f"upscale_{index}_12k"
                    if mode == "12k"
                    else f"final_upscale_{mode}"
                )
                upscale_started = monotonic()
                image_bytes = forge.upscale_image_bytes(
                    image_bytes,
                    target_width=target_width,
                    target_height=target_height,
                    upscaler=payload.output_upscaler,
                )
                upscale_seconds += monotonic() - upscale_started
        return (
            image_bytes,
            generated.seed,
            generated.diffusion_duration_seconds,
            upscale_seconds,
            None,
        )

    artifact_dir = _quality_artifact(job_id, "detail-passed.png").parent
    detail_path = artifact_dir / "detail-passed.png"
    upscale_1_path = artifact_dir / "upscale-1.png"
    upscale_2_path = artifact_dir / "upscale-2.png"
    job = get_job(job_id)
    metadata = job.request_payload if job is not None else {}
    seed = int(metadata.get("quality_seed", payload.seed))
    diffusion_seconds = float(metadata.get("quality_diffusion_seconds", 0.0))
    upscale_seconds = float(metadata.get("quality_upscale_seconds", 0.0))

    if not detail_path.is_file():
        generated = forge.generate_image(
            payload, high_res=True, super_res=True, on_stage=on_stage
        )
        detail_path.write_bytes(generated.image_bytes)
        seed = generated.seed
        diffusion_seconds = generated.diffusion_duration_seconds
        upscale_seconds = generated.upscale_duration_seconds
        update_job_request_payload(
            job_id,
            quality_stage="BASE_RENDERED",
            quality_base_artifact=str(detail_path.relative_to(GENERATED_DIR)),
            quality_seed=seed,
            quality_diffusion_seconds=diffusion_seconds,
            quality_upscale_seconds=upscale_seconds,
        )
        update_job_request_payload(job_id, quality_stage="DETAIL_PASSED")

    if not upscale_1_path.is_file():
        _raise_if_job_cancelled(job_id)
        on_stage("upscale_1_8k")
        started = monotonic()
        upscale_1_path.write_bytes(
            forge.upscale_image_bytes(
                detail_path.read_bytes(),
                target_width=payload.width * 8,
                target_height=payload.height * 8,
                upscaler=payload.output_upscaler,
            )
        )
        upscale_seconds += monotonic() - started
        update_job_request_payload(
            job_id,
            quality_stage="UPSCALE_1_DONE",
            quality_upscale_1_artifact=str(upscale_1_path.relative_to(GENERATED_DIR)),
            quality_upscale_seconds=upscale_seconds,
        )

    if not upscale_2_path.is_file():
        _raise_if_job_cancelled(job_id)
        on_stage("upscale_2_12k")
        started = monotonic()
        upscale_2_path.write_bytes(
            forge.upscale_image_bytes(
                upscale_1_path.read_bytes(),
                target_width=payload.width * 12,
                target_height=payload.height * 12,
                upscaler=payload.output_upscaler,
            )
        )
        upscale_seconds += monotonic() - started
        update_job_request_payload(
            job_id,
            quality_stage="UPSCALE_2_DONE",
            quality_upscale_2_artifact=str(upscale_2_path.relative_to(GENERATED_DIR)),
            quality_upscale_seconds=upscale_seconds,
        )
    return (
        upscale_2_path.read_bytes(), seed, diffusion_seconds, upscale_seconds, artifact_dir
    )


def _run_generation(
    payload: Txt2ImgPayload,
    settings: GenerationSettings,
    *,
    prompt_run_id: int,
    request_id: str | None,
    high_res: bool,
    super_res: bool,
    source_generation_id: int | None = None,
    job_id: int | None = None,
):
    """Render via Forge, persist the result, and translate failures to HTTP errors."""
    if job_id is not None:
        update_job_request_payload(
            job_id,
            resolved_positive_prompt=payload.prompt,
            resolved_negative_prompt=payload.negative_prompt,
        )
    try:
        expected_duration = average_generation_duration(
            model=settings.model,
            style=settings.style,
            high_res=high_res,
            super_res=super_res,
        )
    except sqlite3.Error:
        LOGGER.warning("Could not read generation timing history", exc_info=True)
        expected_duration = 120.0 if super_res else 60.0 if high_res else 30.0
    _start_progress_session(
        request_id,
        expected_duration,
        payload.steps,
        high_res=high_res,
        super_res=super_res,
    )
    if job_id is not None:
        update_job_progress(
            job_id,
            status="running",
            percent=14,
            stage="model_loading",
        )
    started_at = monotonic()
    try:
        def on_forge_stage(stage: str) -> None:
            _raise_if_job_cancelled(job_id)
            milestone = (
                96.0 if stage == "upscale_2_12k"
                else 92.0 if stage in {
                    "upscaling", "final_upscale_4k", "final_upscale_8k",
                    "upscale_1_8k", "upscale_1_12k",
                }
                else 88.0
            )
            _set_progress_stage(request_id, stage, milestone)
            if job_id is not None:
                update_job_progress(job_id, percent=milestone, stage=stage)

        _raise_if_job_cancelled(job_id)
        quality_mode = settings.quality_mode
        image_bytes, actual_seed, diffusion_seconds, upscale_seconds, artifact_dir = (
            _run_quality_pipeline(
                payload,
                quality_mode,  # type: ignore[arg-type]
                job_id=job_id,
                on_stage=on_forge_stage,
            )
        )
        _raise_if_job_cancelled(job_id)
        _set_progress_stage(request_id, "saving", 98.0)
        if job_id is not None:
            update_job_progress(job_id, percent=98, stage="saving")
        record = save_generation(
            image_bytes,
            payload,
            settings,
            actual_seed=actual_seed,
            duration_seconds=monotonic() - started_at,
            diffusion_duration_seconds=diffusion_seconds,
            upscale_duration_seconds=upscale_seconds,
            prompt_run_id=prompt_run_id,
            source_generation_id=source_generation_id,
        )
        if job_id is not None and quality_mode == "12k":
            update_job_request_payload(
                job_id, quality_stage="COMPLETED", quality_generation_id=record.id
            )
            if artifact_dir is not None:
                shutil.rmtree(artifact_dir, ignore_errors=True)
    except requests.Timeout as exc:
        _safe_finish_prompt_run(
            prompt_run_id, status="failed", error_message="Forge timed out"
        )
        raise HTTPException(status_code=504, detail="Forge timed out while generating.") from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = (
            "Forge returned an internal server error. Check its console for model or memory errors."
            if status_code >= 500
            else f"Forge rejected the request (HTTP {status_code})."
        )
        # Forge puts the actual cause in the response body. Without it every
        # failure reads the same and the console is the only way to tell a
        # missing upscaler from an OOM.
        detail = f"{detail} {_forge_error_summary(exc)}".strip()
        _safe_finish_prompt_run(prompt_run_id, status="failed", error_message=detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    except (requests.RequestException, ValueError) as exc:
        _safe_finish_prompt_run(prompt_run_id, status="failed", error_message=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        _safe_finish_prompt_run(
            prompt_run_id,
            status="failed",
            error_message="Local history database was busy; image queued for recovery",
        )
        LOGGER.exception("Database failure while saving generation")
        raise HTTPException(
            status_code=503,
            detail="The image was generated and preserved for automatic history recovery. Please restart ForgeAI if it does not appear shortly.",
        ) from exc
    except OSError as exc:
        _safe_finish_prompt_run(
            prompt_run_id,
            status="failed",
            error_message="Local image storage failed",
        )
        LOGGER.exception("Filesystem failure while saving generation")
        raise HTTPException(
            status_code=507,
            detail="The image was generated, but local storage could not save it. Check available disk space.",
        ) from exc
    finally:
        _finish_progress_session(request_id)
    _safe_finish_prompt_run(
        prompt_run_id, status="succeeded", generation_id=record.id
    )
    return serialize_generation(record)


def generate(body: GenerateRequest, *, job_id: int | None = None):
    """Build the existing Forge payload, generate, persist, and return an image."""
    aspect_ratios = aspect_ratios_for(body.model)
    if body.aspect_ratio not in aspect_ratios:
        raise HTTPException(status_code=422, detail="Unsupported aspect ratio.")
    if body.style not in model_profile_for(body.model).styles:
        raise HTTPException(
            status_code=422,
            detail=f"Style '{body.style}' is not supported by the selected model.",
        )
    if body.content_rating not in RATING_RULES:
        raise HTTPException(status_code=422, detail="Unsupported content rating.")

    width, height = aspect_ratios[body.aspect_ratio]
    quality_mode = _resolve_quality_mode(
        body.quality_mode, body.high_res, body.super_res
    )
    uses_high_res, uses_super_res = _quality_flags(quality_mode)
    settings = GenerationSettings(
        width=width,
        height=height,
        model=body.model,
        rating=body.content_rating,
        style=body.style,
        high_res=uses_high_res,
        super_res=uses_super_res,
        cfg_scale=body.cfg_scale,
        quality_mode=quality_mode,
    )
    try:
        # The prompt engine compiles a finished pair against a measured token
        # budget. Letting Forge rebuild it appends scaffolding the engine
        # already accounted for and pushes the result past that budget.
        payload = build_payload(
            body.prompt,
            body.negative_prompt,
            settings,
            seed=body.seed,
            prebuilt=body.prompt_engine in {"curated", "engine", "ollama", "cloud"},
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if job_id is not None:
        update_job_request_payload(
            job_id,
            resolved_positive_prompt=payload.prompt,
            resolved_negative_prompt=payload.negative_prompt,
            style_variant=payload.style_variant,
            quality_mode=quality_mode,
            base_resolution=[width, height],
            final_resolution=list(_quality_target_size(width, height, quality_mode)),
        )
    prompt_run_id = body.prompt_run_id or _save_prompt_run_or_http(
        positive_prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        source="submitted",
        engine=body.prompt_engine,
        model=body.model,
        style=body.style,
        aspect_ratio=body.aspect_ratio,
        content_rating=body.content_rating,
        high_res=uses_high_res,
        super_res=uses_super_res,
        creativity_level=body.creativity_level,
    )
    return _run_generation(
        payload,
        settings,
        prompt_run_id=prompt_run_id,
        request_id=body.request_id,
        high_res=uses_high_res,
        super_res=uses_super_res,
        job_id=job_id,
    )


def _request_payload(body: BaseModel, **extra: object) -> dict[str, object]:
    """Return observable job inputs while excluding transient credentials."""
    payload = body.model_dump(
        mode="json",
        exclude={"cloud_api_key", "access_token"},
    )
    payload.update(extra)
    return payload


def _enqueue_forge_job(
    kind: str,
    body: BaseModel,
    request: Request,
    *,
    runtime_payload: object | None = None,
    **extra: object,
) -> JobStartResponse:
    """Persist and enqueue one job, returning a guest capability when needed."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    raw_access: str | None = None
    access_hash: str | None = None
    if account is None:
        raw_access, access_hash = create_job_access()
    try:
        job_id = create_job(
            kind,
            _request_payload(body, **extra),
            account.id if account is not None else None,
            access_hash,
            max_outstanding=account.max_active_jobs if account is not None else None,
        )
        runner = _job_runner
        if runner is None:
            finish_job(
                job_id,
                status="failed",
                error_message="The local job worker is unavailable.",
            )
            raise HTTPException(status_code=503, detail="The local job worker is unavailable.")
        try:
            runner.enqueue(job_id, runtime_payload if runtime_payload is not None else body)
        except RuntimeError as exc:
            finish_job(
                job_id,
                status="failed",
                error_message="The local job worker is unavailable.",
            )
            raise HTTPException(
                status_code=503,
                detail="The local job worker is unavailable.",
            ) from exc
    except DuplicateJobSubmission as exc:
        # A retried HTTP request is the same user action, not another GPU job.
        return JobStartResponse(job_id=exc.job_id, access_token=None)
    except JobQueueLimitError as exc:
        raise HTTPException(
            status_code=429,
            detail=(
                f"Queue limit reached ({exc.limit} outstanding jobs). "
                "Wait for a job to finish, cancel one, or raise the limit in Queue settings."
            ),
        ) from exc
    except sqlite3.Error as exc:
        LOGGER.exception("Could not enqueue Forge job")
        raise HTTPException(
            status_code=503,
            detail="The local job queue is temporarily unavailable. No generation was started.",
        ) from exc
    return JobStartResponse(job_id=job_id, access_token=raw_access)


def _validate_generate_request(body: GenerateRequest) -> None:
    """Run inexpensive generation validation before accepting a job."""
    ratios = aspect_ratios_for(body.model)
    if body.aspect_ratio not in ratios:
        raise HTTPException(status_code=422, detail="Unsupported aspect ratio.")
    if body.style not in model_profile_for(body.model).styles:
        raise HTTPException(
            status_code=422,
            detail=f"Style '{body.style}' is not supported by the selected model.",
        )
    if body.content_rating not in RATING_RULES:
        raise HTTPException(status_code=422, detail="Unsupported content rating.")
    width, height = ratios[body.aspect_ratio]
    settings = GenerationSettings(
        width=width,
        height=height,
        model=body.model,
        rating=body.content_rating,
        style=body.style,
        high_res=body.high_res or body.super_res,
        super_res=body.super_res,
        cfg_scale=body.cfg_scale,
    )
    try:
        build_payload(body.prompt, body.negative_prompt, settings)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/generate", response_model=JobStartResponse, status_code=202)
def generate_endpoint(body: GenerateRequest, request: Request) -> JobStartResponse:
    """Validate and enqueue generation without holding the client connection."""
    _require_workspace(request, "forgeai")
    _require_safe_rating(request, body.content_rating)
    _validate_generate_request(body)
    return _enqueue_forge_job("generate", body, request)


def _can_access_generation(
    request: Request, generation_id: int, access_token: str | None
) -> bool:
    """Share one authorization rule across files, feedback, and derived jobs."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    if account and (
        account.is_admin or account_can_access_job_generation(account.id, generation_id)
    ):
        return True
    return verify_generation_access(
        generation_id, access_token
    ) or verify_job_generation_access(generation_id, access_token)


def _mask_detector_runtime() -> tuple[Path, Path]:
    """Locate the Forge Python runtime and its installed ADetailer weights."""
    configured_python = os.getenv("FORGE_PYTHON")
    configured_models = os.getenv("FORGE_ADETAILER_MODELS_DIR")
    package_dirs = [
        Path(item.package_dir)
        for item in configured_backends()
        if item.package_dir
    ]
    python_candidates = [
        Path(configured_python) if configured_python else None,
    ]
    for package_dir in package_dirs:
        python_candidates.extend(
            (
                package_dir / "venv" / "Scripts" / "python.exe",
                package_dir / "venv" / "bin" / "python",
            )
        )
    model_candidates = [
        Path(configured_models) if configured_models else None,
    ]
    for package_dir in package_dirs:
        model_candidates.extend(
            (
                package_dir / "models" / "diffusers" / "models--Bingsu--adetailer",
                package_dir / "models",
            )
        )
    python = next((candidate for candidate in python_candidates if candidate and candidate.is_file()), None)
    models = next((candidate for candidate in model_candidates if candidate and candidate.is_dir()), None)
    if python is None or models is None:
        raise HTTPException(
            status_code=503,
            detail="ADetailer mask detection is unavailable. Configure FORGE_PYTHON and FORGE_ADETAILER_MODELS_DIR.",
        )
    return python, models


def _run_mask_detector(source_path: Path, detector: str) -> MaskProposalResponse:
    """Run one installed detector and validate its machine-readable response."""
    python, models = _mask_detector_runtime()
    detector_script = Path(__file__).with_name("detect_masks.py")
    try:
        completed = subprocess.run(
            [str(python), str(detector_script), str(source_path), str(models), detector],
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        output = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), "")
        return MaskProposalResponse.model_validate(json.loads(output))
    except (subprocess.SubprocessError, json.JSONDecodeError, ValidationError) as exc:
        LOGGER.exception("ADetailer mask proposal failed")
        raise HTTPException(status_code=503, detail="Forge could not detect an editable region in this image.") from exc


def _detected_face_box(source_path: Path) -> list[int] | None:
    """Return the detector's padded source-space face box when available."""
    python, models = _mask_detector_runtime()
    detector_script = Path(__file__).with_name("detect_masks.py")
    try:
        completed = subprocess.run(
            [str(python), str(detector_script), str(source_path), str(models), "face-box"],
            capture_output=True,
            text=True,
            timeout=90,
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        output = next((line for line in reversed(completed.stdout.splitlines()) if line.startswith("{")), "")
        face_box = json.loads(output).get("face_box")
        if not isinstance(face_box, list) or len(face_box) != 4:
            return None
        return [int(value) for value in face_box]
    except (subprocess.SubprocessError, json.JSONDecodeError, ValueError, TypeError) as exc:
        LOGGER.warning("FaceID face detection failed: %s", exc)
        return None


def _face_reference(source_path: Path, source_image: Image.Image) -> str:
    """Crop a padded face reference in Forge's detector subprocess."""
    face_box = _detected_face_box(source_path)
    reference = source_image.crop(tuple(face_box)) if face_box else source_image
    return _png_base64(reference)


def _automatic_mask(source_path: Path, detector: str, size: tuple[int, int]) -> str:
    """Resolve detector proposals into one exact-size mask for direct API use."""
    proposals = _run_mask_detector(source_path, detector).proposals
    if not proposals:
        raise HTTPException(status_code=422, detail="No suitable region was detected in the source image.")
    if len(proposals) == 1:
        return proposals[0].mask
    union = Image.new("L", size, 0)
    for proposal in proposals:
        with _decode_mask(proposal.mask) as proposal_mask:
            if proposal_mask.size != size:
                raise HTTPException(status_code=503, detail="A detector returned a mask with invalid dimensions.")
            union = ImageOps.lighter(union, proposal_mask)
    return _png_base64(union)


@app.post("/api/img2img/mask-proposals", response_model=MaskProposalResponse)
def img2img_mask_proposals(body: MaskProposalRequest, request: Request) -> MaskProposalResponse:
    """Return exact-source-size masks proposed by Forge's installed YOLO detectors."""
    if (body.source_generation_id is None) == (body.upload_id is None):
        raise HTTPException(status_code=422, detail="Supply exactly one source_generation_id or upload_id.")
    if body.source_generation_id is not None:
        source = get_generation(body.source_generation_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        access = body.access_token.get_secret_value() if body.access_token else None
        if not _can_access_generation(request, source.id, access):
            raise HTTPException(status_code=403, detail="Image access denied.")
        source_path = GENERATED_DIR / source.filename
    else:
        _require_authenticated_account(request)
        source_path = _upload_path(body.upload_id or "")
        if source_path is None:
            raise HTTPException(status_code=422, detail="Invalid upload identifier.")
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source image file is missing.")
    return _run_mask_detector(source_path, body.detector)


_TRANSFORM_SUGGESTIONS = {
    "background": (
        "a rain-slicked Tokyo alley glowing with violet neon",
        "a quiet alpine lake beneath soft morning mist",
        "an elegant hotel lobby with warm amber lighting",
        "a windswept beach beneath a dramatic overcast sky",
        "a lush glasshouse filled with tropical plants",
        "a minimalist concrete studio with diffused window light",
        "a moonlit rooftop overlooking a futuristic city",
        "an autumn forest path covered in copper leaves",
    ),
    "outfit": (
        "a tailored charcoal suit with a silk ivory blouse",
        "a flowing emerald evening dress with subtle gold accents",
        "a cropped leather jacket with high-waisted black trousers",
        "a soft cream sweater with a pleated midi skirt",
        "a structured navy jumpsuit with a slim metallic belt",
        "a red satin cocktail dress with understated jewelry",
        "a modern white streetwear set with lavender accents",
        "a vintage denim jacket over a floral summer dress",
    ),
    "scene": (
        "walking through a quiet Tokyo street beneath warm evening lanterns",
        "standing beside an alpine lake in soft misty morning light",
        "reading near tall windows in an elegant amber-lit hotel lounge",
        "walking along a windswept beach beneath a dramatic overcast sky",
        "exploring a lush glasshouse filled with tropical plants",
        "posing in a minimalist concrete studio with diffused window light",
        "standing on a moonlit rooftop overlooking a futuristic city",
        "walking along an autumn forest path covered in copper leaves",
    ),
}


@app.post("/api/img2img/suggestion", response_model=TransformSuggestionResponse)
def img2img_suggestion(body: TransformSuggestionRequest, request: Request) -> TransformSuggestionResponse:
    """Generate one concise contextual value for a ForgeIMG preset."""
    if sum(value is not None for value in (body.source_generation_id, body.upload_id, body.identity_profile_id)) != 1:
        raise HTTPException(status_code=422, detail="Supply exactly one source image, upload, or Face ID.")
    if body.source_generation_id is not None:
        source = get_generation(body.source_generation_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        access = body.access_token.get_secret_value() if body.access_token else None
        if not _can_access_generation(request, source.id, access):
            raise HTTPException(status_code=403, detail="Image access denied.")
        source_path = GENERATED_DIR / source.filename
    elif body.upload_id is not None:
        _require_authenticated_account(request)
        source_path = _upload_path(body.upload_id or "")
        if source_path is None:
            raise HTTPException(status_code=422, detail="Invalid upload identifier.")
    else:
        account = _require_authenticated_account(request)
        profile = get_identity_profile(body.identity_profile_id or 0, account_id=account.id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Face ID not found.")
        source_path = IDENTITY_PROFILES_DIR / profile.filename
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source image file is missing.")
    fallback = secrets.choice(_TRANSFORM_SUGGESTIONS[body.kind])
    if body.prompt_engine == "curated":
        return TransformSuggestionResponse(suggestion=fallback, engine="curated")
    try:
        if body.prompt_engine == "ollama":
            if not body.ollama_model:
                raise HTTPException(status_code=422, detail="Choose an Ollama prompt model.")
            suggestion = OllamaPromptClient().suggest_transform(
                ollama_model=body.ollama_model,
                kind=body.kind,
                image_base64=base64.b64encode(source_path.read_bytes()).decode("ascii"),
            )
            engine = f"ollama:{body.ollama_model}"
        else:
            _admin_account(request)
            if not body.cloud_provider or not body.cloud_model or not body.cloud_api_key:
                raise HTTPException(status_code=422, detail="Configure a Cloud API provider and key first.")
            suggestion = CloudPromptClient().suggest_transform(
                provider=body.cloud_provider,
                api_key=body.cloud_api_key.get_secret_value(),
                cloud_model=body.cloud_model,
                kind=body.kind,
            )
            engine = f"cloud:{body.cloud_provider}:{body.cloud_model}"
        if re.search(r"\b(nude|naked|topless|lingerie|underwear)\b", suggestion, re.I):
            raise ValueError("Unsafe transform suggestion")
        return TransformSuggestionResponse(suggestion=suggestion, engine=engine)
    except HTTPException:
        raise
    except (requests.RequestException, CloudPromptError, ValueError, KeyError) as exc:
        LOGGER.warning("Transform suggestion fell back to curated: %s", exc)
        return TransformSuggestionResponse(
            suggestion=fallback,
            engine="curated",
            notice="The selected prompt engine was unavailable; a local suggestion was used.",
        )


def _validate_img2img_request(body: Img2ImgRequest, request: Request) -> None:
    """Validate the source authorization and model-bound transform settings."""
    supplied_assets = sum(
        value is not None
        for value in (body.source_generation_id, body.upload_id, body.identity_profile_id)
    )
    if supplied_assets != 1:
        raise HTTPException(
            status_code=422,
            detail="Supply exactly one source image, upload, or Face ID.",
        )
    if body.identity_profile_id is not None and not body.face_identity:
        raise HTTPException(status_code=422, detail="Face IDs can only be used for character scenes.")
    if body.mask is not None and body.auto_mask is not None:
        raise HTTPException(status_code=422, detail="Supply either mask or auto_mask, not both.")
    if body.face_identity and not model_profile_for(body.model).supports_face_identity:
        profile = model_profile_for(body.model)
        raise HTTPException(
            status_code=422,
            detail=(
                "Character identity needs an SDXL checkpoint. "
                f"{profile.display_name} is a different architecture, so the "
                "FaceID adapter cannot condition it."
            ),
        )
    source_size: tuple[int, int]
    if body.source_generation_id is not None:
        source = get_generation(body.source_generation_id)
        if source is None:
            raise HTTPException(status_code=404, detail="Generation not found.")
        access_token = (
            body.access_token.get_secret_value() if body.access_token is not None else None
        )
        if not _can_access_generation(request, body.source_generation_id, access_token):
            raise HTTPException(status_code=403, detail="Image access denied.")
        source_size = (source.width, source.height)
    elif body.upload_id is not None:
        _require_authenticated_account(request)
        upload_path = _upload_path(body.upload_id or "")
        if upload_path is None:
            raise HTTPException(status_code=422, detail="Invalid upload identifier.")
        if not upload_path.is_file():
            raise HTTPException(status_code=404, detail="Upload not found.")
        try:
            with Image.open(upload_path) as uploaded:
                source_size = uploaded.size
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=422, detail="The uploaded image is invalid.") from exc
    else:
        account = _require_authenticated_account(request)
        profile = get_identity_profile(body.identity_profile_id or 0, account_id=account.id)
        if profile is None:
            raise HTTPException(status_code=404, detail="Face ID not found.")
        profile_path = IDENTITY_PROFILES_DIR / profile.filename
        if not profile_path.is_file():
            raise HTTPException(status_code=404, detail="Face ID image is missing.")
        try:
            with Image.open(profile_path) as identity_image:
                source_size = identity_image.size
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=422, detail="The Face ID image is invalid.") from exc
    if body.mask is not None:
        try:
            with _decode_mask(body.mask) as mask:
                if mask.size != source_size:
                    raise HTTPException(
                        status_code=422,
                        detail="Mask dimensions must match the source image.",
                    )
        except HTTPException:
            raise
        except (UnidentifiedImageError, OSError, ValueError, binascii.Error, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=422, detail="The mask image is invalid.") from exc
    if body.style not in model_profile_for(body.model).styles:
        raise HTTPException(
            status_code=422,
            detail=f"Style '{body.style}' is not supported by the selected model.",
        )
    if body.content_rating not in RATING_RULES:
        raise HTTPException(status_code=422, detail="Unsupported content rating.")


def _decode_mask(encoded: str) -> Image.Image:
    """Decode a bounded base64 mask into a fully-loaded greyscale image."""
    raw = base64.b64decode(encoded, validate=True)
    with Image.open(BytesIO(raw)) as source:
        source.load()
        return source.convert("L")


def _png_base64(image: Image.Image) -> str:
    """Encode an application-owned image as base64 PNG without a data prefix."""
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


RATING_SEVERITY = {"Safe": 0}


def _rating_rank(rating: str) -> int:
    """Order ratings so a derived image can never soften its source."""
    return RATING_SEVERITY.get(rating, 0)


def _run_img2img_job(body: Img2ImgRequest, *, job_id: int) -> GenerationResponse:
    """Load and downscale one source, transform it, and persist lineage."""
    source = (
        get_generation(body.source_generation_id)
        if body.source_generation_id is not None
        else None
    )
    if body.source_generation_id is not None and source is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    source_path = (
        GENERATED_DIR / source.filename
        if source is not None
        else _upload_path(body.upload_id or "")
    )
    if source_path is None:
        raise HTTPException(status_code=422, detail="Invalid upload identifier.")
    try:
        with Image.open(source_path) as opened_source:
            opened_source.load()
            source_image = ImageOps.exif_transpose(opened_source).convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(status_code=404, detail="Source image file is missing.") from exc
    original_size = source_image.size
    face_reference = _face_reference(source_path, source_image) if body.face_identity else ""
    target_width, target_height = nearest_native_resolution(
        body.model, source_image.width, source_image.height
    )
    if source_image.size != (target_width, target_height):
        source_image = ImageOps.fit(
            source_image,
            (target_width, target_height),
            method=Image.Resampling.LANCZOS,
        )
    encoded_source = _png_base64(source_image)
    encoded_mask: str | None = None
    requested_mask = None if body.face_identity else body.mask or (
        _automatic_mask(source_path, body.auto_mask, original_size)
        if body.auto_mask is not None else None
    )
    if requested_mask is not None:
        try:
            mask_image = _decode_mask(requested_mask)
        except (UnidentifiedImageError, OSError, ValueError, binascii.Error, Image.DecompressionBombError) as exc:
            raise HTTPException(status_code=422, detail="The mask image is invalid.") from exc
        if mask_image.size != original_size:
            raise HTTPException(status_code=422, detail="Mask dimensions must match the source image.")
        if mask_image.size != (target_width, target_height):
            mask_image = ImageOps.fit(
                mask_image,
                (target_width, target_height),
                method=Image.Resampling.LANCZOS,
            )
        encoded_mask = _png_base64(mask_image)
    # Imported and generated sources must remain inside the Safe-only boundary.
    effective_rating = body.content_rating
    if source is not None and _rating_rank(source.rating) > _rating_rank(effective_rating):
        effective_rating = source.rating
    settings = GenerationSettings(
        width=target_width,
        height=target_height,
        model=body.model,
        rating=effective_rating,
        style=body.style,
        high_res=False,
        super_res=False,
        cfg_scale=body.cfg_scale,
    )
    try:
        payload = build_img2img_payload(
            body.prompt,
            body.negative_prompt,
            settings,
            encoded_source,
            body.denoising_strength,
            mask=encoded_mask,
            mask_blur=body.mask_blur,
            inpaint_full_res=body.inpaint_full_res,
            inpaint_full_res_padding=body.inpaint_full_res_padding,
            invert_mask=body.invert_mask,
            soft_inpainting=body.soft_inpainting,
            region_only=body.region_only,
            face_reference=face_reference,
            preserve_unmasked_regions=(
                encoded_mask is not None
                and body.preset in {"background", "outfit"}
            ),
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    update_job_request_payload(
        job_id,
        resolved_positive_prompt=payload.prompt,
        resolved_negative_prompt=payload.negative_prompt,
        style_variant=payload.style_variant,
    )
    prompt_run_id = _save_prompt_run_or_http(
        positive_prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        source="submitted",
        engine=body.prompt_engine,
        model=body.model,
        style=body.style,
        aspect_ratio=f"{target_width}x{target_height}",
        content_rating=body.content_rating,
        high_res=False,
        super_res=False,
        creativity_level=body.preset or "advanced",
    )
    update_job_progress(job_id, status="running", percent=14, stage="model_loading")
    started_at = monotonic()
    try:
        def on_forge_stage(stage: str) -> None:
            _raise_if_job_cancelled(job_id)
            update_job_progress(job_id, percent=88, stage=stage)

        _raise_if_job_cancelled(job_id)
        forge = _ensure_model_backend(payload.model_checkpoint)
        forge_result = forge.transform_image(payload, on_stage=on_forge_stage)
        _raise_if_job_cancelled(job_id)
        update_job_progress(job_id, percent=98, stage="saving")
        record = save_generation(
            forge_result.image_bytes,
            payload,
            settings,
            actual_seed=forge_result.seed,
            duration_seconds=monotonic() - started_at,
            diffusion_duration_seconds=forge_result.diffusion_duration_seconds,
            upscale_duration_seconds=0.0,
            prompt_run_id=prompt_run_id,
            source_generation_id=body.source_generation_id,
        )
    except requests.Timeout as exc:
        _safe_finish_prompt_run(prompt_run_id, status="failed", error_message="Forge timed out")
        raise HTTPException(status_code=504, detail="Forge timed out while generating.") from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = (
            "Forge returned an internal server error. Check its console for model or memory errors."
            if status_code >= 500
            else f"Forge rejected the request (HTTP {status_code})."
        )
        # Forge puts the actual cause in the response body. Without it every
        # failure reads the same and the console is the only way to tell a
        # missing upscaler from an OOM.
        detail = f"{detail} {_forge_error_summary(exc)}".strip()
        _safe_finish_prompt_run(prompt_run_id, status="failed", error_message=detail)
        raise HTTPException(status_code=502, detail=detail) from exc
    except (requests.RequestException, ValueError) as exc:
        _safe_finish_prompt_run(prompt_run_id, status="failed", error_message=str(exc))
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except sqlite3.Error as exc:
        _safe_finish_prompt_run(
            prompt_run_id,
            status="failed",
            error_message="Local history database was busy; image queued for recovery",
        )
        LOGGER.exception("Database failure while saving img2img generation")
        raise HTTPException(
            status_code=503,
            detail="The image was generated and preserved for automatic history recovery. Please restart ForgeAI if it does not appear shortly.",
        ) from exc
    except OSError as exc:
        _safe_finish_prompt_run(prompt_run_id, status="failed", error_message="Local image storage failed")
        raise HTTPException(
            status_code=507,
            detail="The image was generated, but local storage could not save it. Check available disk space.",
        ) from exc
    _safe_finish_prompt_run(prompt_run_id, status="succeeded", generation_id=record.id)
    return serialize_generation(record)


def _character_artifact(job_id: int, name: str) -> Path:
    directory = GENERATED_DIR / "character-jobs" / f"job-{job_id}"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / name


def _character_metadata(job_id: int) -> dict[str, object]:
    record = get_job(job_id)
    if record is None:
        raise JobExecutionError("The character job no longer exists.")
    return record.request_payload


def _run_character_job(body: Img2ImgRequest, *, job_id: int) -> GenerationResponse:
    """Generate a new scene, lock the reference identity, and verify it.

    Every expensive output is written before its stage marker is persisted, so
    an interrupted job can resume from the last valid artifact rather than
    repeating diffusion.
    """
    source = (
        get_generation(body.source_generation_id)
        if body.source_generation_id is not None
        else None
    )
    if body.source_generation_id is not None and source is None:
        raise JobExecutionError("The source generation is no longer available.")
    if source is not None:
        source_path = GENERATED_DIR / source.filename
    elif body.upload_id is not None:
        source_path = _upload_path(body.upload_id)
    else:
        job = get_job(job_id)
        profile = get_identity_profile(
            body.identity_profile_id or 0,
            account_id=job.account_id if job is not None else None,
        )
        if profile is None:
            raise JobExecutionError("The selected Face ID is no longer available.")
        source_path = IDENTITY_PROFILES_DIR / profile.filename
        if job is not None and job.account_id is not None:
            touch_identity_profile(profile.id, job.account_id)
    if source_path is None or not source_path.is_file():
        raise JobExecutionError("The reference image is no longer available.")

    completed_id = _character_metadata(job_id).get("character_generation_id")
    if completed_id is not None:
        completed = get_generation(int(completed_id))
        if completed is not None:
            return serialize_generation(completed)

    reference_path = _character_artifact(job_id, "reference.png")
    if not reference_path.is_file():
        try:
            with Image.open(source_path) as opened:
                reference = ImageOps.exif_transpose(opened).convert("RGB")
                original_size = reference.size
                reference.save(reference_path, format="PNG")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise JobExecutionError("The reference image could not be prepared.") from exc
        update_job_request_payload(
            job_id,
            character_stage="REFERENCE_READY",
            character_reference=str(reference_path.relative_to(GENERATED_DIR)),
            character_source_hash=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            character_source_size=list(original_size),
        )
    try:
        with Image.open(reference_path) as prepared:
            reference_size = prepared.size
    except OSError as exc:
        raise JobExecutionError("The persisted reference artifact is invalid.") from exc

    identity = InsightFaceLocalIdentityProvider()
    readiness = identity.readiness_error()
    if readiness:
        raise JobExecutionError(readiness)
    metadata = _character_metadata(job_id)
    if "character_reference_score" not in metadata:
        update_job_progress(job_id, percent=8, stage="identity_analysis")
        try:
            reference_check = identity.verify(reference_path, reference_path)
        except IdentityProviderError as exc:
            raise JobExecutionError(str(exc)) from exc
        update_job_request_payload(
            job_id,
            character_stage="REFERENCE_READY",
            character_reference_score=reference_check.score,
            identity_verifier=reference_check.provider,
            identity_threshold=reference_check.threshold,
        )

    target_width, target_height = nearest_native_resolution(
        body.model, reference_size[0], reference_size[1]
    )
    settings = GenerationSettings(
        width=target_width,
        height=target_height,
        model=body.model,
        rating=body.content_rating,
        style=body.style,
        high_res=False,
        super_res=False,
        cfg_scale=body.cfg_scale,
    )
    forge = ForgeApiClient()
    face_reference_path = _character_artifact(job_id, "face-reference.png")
    try:
        with Image.open(reference_path) as prepared:
            face_reference = _face_reference(
                reference_path,
                ImageOps.exif_transpose(prepared).convert("RGB"),
            )
        if not face_reference_path.is_file():
            face_reference_path.write_bytes(base64.b64decode(face_reference))
        adapter_model = forge.resolve_controlnet_model("ip-adapter_instant_id_sdxl")
        control_model = forge.resolve_controlnet_model("control_instant_id_sdxl")
        instantid_payload = build_instantid_payload(
            body.prompt,
            body.negative_prompt,
            settings,
            face_reference,
            adapter_model=adapter_model,
            controlnet_model=control_model,
        )
        # Identity Lock is the final authority for the delivered face. Keep a
        # generator-only SDXL fallback for reForge builds where the paired
        # InstantID units load successfully but return an invalid latent. The
        # fallback is accepted only after the same lock + verifier gate.
        fallback_payload = replace(
            build_payload(
                body.prompt,
                body.negative_prompt,
                settings,
            ),
            prompt=instantid_payload.prompt,
            negative_prompt=instantid_payload.negative_prompt,
            detail_pass_models=(),
            controlnet_units=(),
        )
    except (requests.RequestException, ValueError, OSError, binascii.Error) as exc:
        raise JobExecutionError(str(exc)) from exc
    update_job_request_payload(
        job_id,
        resolved_positive_prompt=instantid_payload.prompt,
        resolved_negative_prompt=instantid_payload.negative_prompt,
        character_generator="instantid-sdxl-with-verified-sdxl-fallback",
        character_adapter_model=adapter_model,
        character_controlnet_model=control_model,
        character_face_reference=str(face_reference_path.relative_to(GENERATED_DIR)),
        character_target_size=[target_width, target_height],
    )

    metadata = _character_metadata(job_id)
    prompt_run_value = metadata.get("character_prompt_run_id")
    if prompt_run_value is None:
        prompt_run_id = _save_prompt_run_or_http(
            positive_prompt=instantid_payload.prompt,
            negative_prompt=instantid_payload.negative_prompt,
            source="submitted",
            engine=body.prompt_engine,
            model=body.model,
            style=body.style,
            aspect_ratio=f"{target_width}x{target_height}",
            content_rating=body.content_rating,
            high_res=False,
            super_res=False,
            creativity_level="same-character",
        )
        update_job_request_payload(job_id, character_prompt_run_id=prompt_run_id)
    else:
        prompt_run_id = int(prompt_run_value)

    started_at = monotonic()

    def attempt(
        label: str,
        base_percent: float,
        candidate_payload: Txt2ImgPayload,
        generator: str,
    ) -> tuple[Path, float, int, float, float, Txt2ImgPayload, str] | None:
        suffix = label.lower()
        candidate_path = _character_artifact(job_id, f"candidate-{suffix}-{generator}.png")
        locked_path = _character_artifact(job_id, f"locked-{suffix}-{generator}.png")
        current = _character_metadata(job_id)
        seed_key = f"character_candidate_{suffix}_seed"
        duration_key = f"character_candidate_{suffix}_diffusion_seconds"
        if not candidate_path.is_file():
            _raise_if_job_cancelled(job_id)
            update_job_progress(job_id, percent=base_percent, stage="identity_generation")
            try:
                generated = forge.generate_image(candidate_payload, high_res=False)
            except requests.Timeout as exc:
                raise JobExecutionError("Forge timed out during identity scene generation.") from exc
            except requests.HTTPError as exc:
                detail = "Forge ran out of memory or rejected the identity scene request."
                raise JobExecutionError(detail) from exc
            except (requests.RequestException, ValueError) as exc:
                raise JobExecutionError(str(exc)) from exc
            candidate_path.write_bytes(generated.image_bytes)
            update_job_request_payload(
                job_id,
                character_stage=f"CANDIDATE_{label}_GENERATED",
                **{
                    seed_key: generated.seed,
                    duration_key: generated.diffusion_duration_seconds,
                    f"character_candidate_{suffix}": str(candidate_path.relative_to(GENERATED_DIR)),
                    f"character_candidate_{suffix}_generator": generator,
                },
            )
            seed = generated.seed
            diffusion_seconds = generated.diffusion_duration_seconds
        else:
            seed = int(current.get(seed_key, -1))
            diffusion_seconds = float(current.get(duration_key, 0.0))

        # A lock must refine a real generated face, never manufacture one on a
        # headless crop. Verify the untouched candidate first and reject both
        # missing faces and anti-correlated identities before inswapper runs.
        current = _character_metadata(job_id)
        prelock_key = f"character_candidate_{suffix}_prelock_score"
        if prelock_key in current:
            prelock_score = float(current[prelock_key])
        else:
            _raise_if_job_cancelled(job_id)
            update_job_progress(job_id, percent=base_percent + 20, stage="identity_verification")
            try:
                prelock = identity.verify(reference_path, candidate_path)
            except IdentityProviderError as exc:
                if "No face detected" in str(exc):
                    update_job_request_payload(
                        job_id,
                        **{f"character_candidate_{suffix}_failure": str(exc)},
                    )
                    return None
                raise JobExecutionError(str(exc)) from exc
            prelock_score = prelock.score
            update_job_request_payload(
                job_id,
                **{prelock_key: prelock_score},
            )
        if prelock_score <= 0.0:
            update_job_request_payload(
                job_id,
                **{
                    f"character_candidate_{suffix}_failure": (
                        f"Pre-lock identity score {prelock_score:.3f} is not positive."
                    ),
                    f"character_candidate_{suffix}_identity_passed": False,
                },
            )
            return None

        if not locked_path.is_file():
            _raise_if_job_cancelled(job_id)
            update_job_progress(job_id, percent=base_percent + 27, stage="identity_lock")
            try:
                lock_result = identity.lock(reference_path, candidate_path, locked_path)
            except IdentityProviderError as exc:
                if "No face detected in the candidate" in str(exc):
                    update_job_request_payload(
                        job_id,
                        **{f"character_candidate_{suffix}_failure": str(exc)},
                    )
                    return None
                raise JobExecutionError(str(exc)) from exc
            update_job_request_payload(
                job_id,
                character_stage=f"CANDIDATE_{label}_LOCKED",
                identity_lock_provider=lock_result.provider,
                **{
                    f"character_locked_{suffix}": str(locked_path.relative_to(GENERATED_DIR)),
                    f"character_candidate_{suffix}_lock_input_score": lock_result.similarity_before,
                    f"character_candidate_{suffix}_lock_score": lock_result.similarity_after,
                },
            )

        _raise_if_job_cancelled(job_id)
        update_job_progress(job_id, percent=base_percent + 38, stage="identity_verification")
        try:
            verification = identity.verify(reference_path, locked_path)
        except IdentityProviderError as exc:
            raise JobExecutionError(str(exc)) from exc
        update_job_request_payload(
            job_id,
            character_stage=f"CANDIDATE_{label}_VERIFIED",
            **{
                f"character_candidate_{suffix}_identity_score": verification.score,
                f"character_candidate_{suffix}_identity_passed": verification.passed,
            },
        )
        if not verification.passed:
            return None

        # Inswapper's internal face representation is only 128px. Restore its
        # texture before a conservative pixel upscale, then verify identity a
        # second time. The already-verified raw lock remains the safe fallback
        # if Extras is unavailable or restoration weakens identity.
        refined_path = _character_artifact(job_id, f"refined-{suffix}-{generator}.png")
        refinement_duration_key = f"character_candidate_{suffix}_refinement_seconds"
        refinement_seconds = float(
            _character_metadata(job_id).get(refinement_duration_key, 0.0)
        )
        if not refined_path.is_file():
            _raise_if_job_cancelled(job_id)
            update_job_progress(
                job_id, percent=base_percent + 41, stage="identity_refinement"
            )
            refinement_started_at = monotonic()
            try:
                with Image.open(locked_path) as locked_image:
                    locked_width, locked_height = locked_image.size
                refined_bytes = forge.restore_and_upscale_image_bytes(
                    locked_path.read_bytes(),
                    target_width=round(locked_width * CHARACTER_REFINEMENT_SCALE),
                    target_height=round(locked_height * CHARACTER_REFINEMENT_SCALE),
                    upscaler=CHARACTER_REFINEMENT_UPSCALER,
                    codeformer_visibility=CHARACTER_CODEFORMER_VISIBILITY,
                    codeformer_weight=CHARACTER_CODEFORMER_WEIGHT,
                )
                with Image.open(BytesIO(refined_bytes)) as refined_image:
                    refined_image.verify()
                refined_path.write_bytes(refined_bytes)
                refinement_seconds = monotonic() - refinement_started_at
                update_job_request_payload(
                    job_id,
                    character_stage=f"CANDIDATE_{label}_REFINED",
                    **{
                        f"character_refined_{suffix}": str(
                            refined_path.relative_to(GENERATED_DIR)
                        ),
                        refinement_duration_key: refinement_seconds,
                        f"character_candidate_{suffix}_refinement_upscaler": CHARACTER_REFINEMENT_UPSCALER,
                        f"character_candidate_{suffix}_codeformer_visibility": CHARACTER_CODEFORMER_VISIBILITY,
                        f"character_candidate_{suffix}_codeformer_weight": CHARACTER_CODEFORMER_WEIGHT,
                    },
                )
            except (
                OSError,
                requests.RequestException,
                ValueError,
            ) as exc:
                update_job_request_payload(
                    job_id,
                    **{
                        f"character_candidate_{suffix}_refinement_selected": False,
                        f"character_candidate_{suffix}_refinement_error": str(exc),
                    },
                )
                return (
                    locked_path,
                    verification.score,
                    seed,
                    diffusion_seconds,
                    0.0,
                    candidate_payload,
                    generator,
                )

        _raise_if_job_cancelled(job_id)
        update_job_progress(
            job_id, percent=base_percent + 43, stage="identity_verification"
        )
        try:
            refined_verification = identity.verify(reference_path, refined_path)
        except IdentityProviderError as exc:
            update_job_request_payload(
                job_id,
                **{
                    f"character_candidate_{suffix}_refinement_selected": False,
                    f"character_candidate_{suffix}_refinement_error": str(exc),
                },
            )
            return (
                locked_path,
                verification.score,
                seed,
                diffusion_seconds,
                0.0,
                candidate_payload,
                generator,
            )
        update_job_request_payload(
            job_id,
            character_stage=f"CANDIDATE_{label}_REFINEMENT_VERIFIED",
            **{
                f"character_candidate_{suffix}_refined_identity_score": refined_verification.score,
                f"character_candidate_{suffix}_refined_identity_passed": refined_verification.passed,
                f"character_candidate_{suffix}_refinement_selected": refined_verification.passed,
            },
        )
        if refined_verification.passed:
            return (
                refined_path,
                refined_verification.score,
                seed,
                diffusion_seconds,
                refinement_seconds,
                candidate_payload,
                generator,
            )
        return (
            locked_path,
            verification.score,
            seed,
            diffusion_seconds,
            0.0,
            candidate_payload,
            generator,
        )

    selected = attempt("A", 15, instantid_payload, "instantid-sdxl")
    selected_label = "A"
    if selected is None:
        selected_label = "B"
        selected = attempt("B", 51, fallback_payload, "sdxl-identity-lock")
    if selected is None:
        _safe_finish_prompt_run(
            prompt_run_id,
            status="failed",
            error_message="Identity verification failed for both candidates.",
        )
        raise JobExecutionError(
            "Identity verification failed for both candidates; no mismatched face was saved."
        )

    (
        output_path,
        identity_score,
        actual_seed,
        diffusion_seconds,
        refinement_seconds,
        selected_payload,
        selected_generator,
    ) = selected
    _raise_if_job_cancelled(job_id)
    update_job_progress(job_id, percent=96, stage="saving")
    try:
        record = save_generation(
            output_path.read_bytes(),
            selected_payload,
            settings,
            actual_seed=actual_seed,
            duration_seconds=monotonic() - started_at,
            diffusion_duration_seconds=diffusion_seconds,
            upscale_duration_seconds=refinement_seconds,
            prompt_run_id=prompt_run_id,
            source_generation_id=body.source_generation_id,
        )
    except (OSError, sqlite3.Error) as exc:
        raise JobExecutionError("The verified character image could not be saved.") from exc
    update_job_request_payload(
        job_id,
        character_stage="COMPLETED",
        character_selected_candidate=selected_label,
        character_selected_generator=selected_generator,
        character_identity_score=identity_score,
        character_refinement_selected=refinement_seconds > 0,
        character_output=str(output_path.relative_to(GENERATED_DIR)),
        character_generation_id=record.id,
    )
    _safe_finish_prompt_run(prompt_run_id, status="succeeded", generation_id=record.id)
    return serialize_generation(record)


@app.post("/api/img2img", response_model=JobStartResponse, status_code=202)
def img2img_endpoint(body: Img2ImgRequest, request: Request) -> JobStartResponse:
    """Validate and enqueue a gallery-sourced image transform."""
    _require_workspace(request, "forgeimg")
    _require_safe_rating(request, body.content_rating)
    _validate_img2img_request(body, request)
    return _enqueue_forge_job("character" if body.face_identity else "img2img", body, request)


def _run_upscale_job(
    generation_id: int,
    body: UpscaleRequest,
    *,
    job_id: int,
) -> GenerationResponse:
    """Execute a validated detail-pass job outside request context."""
    source = get_generation(generation_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    # Anchor the re-render to this model's native resolution rather than the
    # source's current (possibly already-upscaled) size, so a detail pass on
    # an already-large image adds detail without compounding resolution or
    # forcing Forge into a VRAM-starved tiled-VAE fallback.
    native_width, native_height = nearest_native_resolution(
        source.model, source.width, source.height
    )
    settings = GenerationSettings(
        width=native_width,
        height=native_height,
        model=source.model,
        rating=source.rating,
        style=source.style,
        high_res=True,
        super_res=body.super_res,
        quality_mode="super" if body.super_res else "high",
        # Keep the CFG the source image was rendered at, so a detail pass
        # refines that image instead of re-rendering it at the profile default.
        cfg_scale=source.cfg_scale,
    )
    try:
        payload = build_payload(
            source.positive_prompt, source.negative_prompt, settings, seed=source.seed
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    prompt_run_id = _save_prompt_run_or_http(
        positive_prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        source="submitted",
        engine="upscale",
        model=source.model,
        style=source.style,
        aspect_ratio=f"{native_width}x{native_height}",
        content_rating=source.rating,
        high_res=True,
        super_res=body.super_res,
        creativity_level="balanced",
    )
    return _run_generation(
        payload,
        settings,
        prompt_run_id=prompt_run_id,
        request_id=None,
        high_res=True,
        super_res=body.super_res,
        source_generation_id=generation_id,
        job_id=job_id,
    )


@app.post(
    "/api/generations/{generation_id}/upscale",
    response_model=JobStartResponse,
    status_code=202,
)
def upscale_generation(
    generation_id: int, body: UpscaleRequest, request: Request
) -> JobStartResponse:
    """Validate access and enqueue a detail-pass re-render."""
    _require_workspace(request, "forgeai")
    source = get_generation(generation_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    access_token = (
        body.access_token.get_secret_value() if body.access_token is not None else None
    )
    if not _can_access_generation(request, generation_id, access_token):
        raise HTTPException(status_code=403, detail="Image access denied.")
    _require_safe_rating(request, source.rating)
    return _enqueue_forge_job(
        "upscale",
        body,
        request,
        runtime_payload=(generation_id, body),
        generation_id=generation_id,
    )


def _run_pixel_upscale_job(
    generation_id: int,
    body: PixelUpscaleRequest,
    *,
    job_id: int,
) -> GenerationResponse:
    """Execute a validated pixel-upscale job outside request context."""
    source = get_generation(generation_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    source_path = GENERATED_DIR / (generation_filename(generation_id) or "")
    try:
        source_bytes = source_path.read_bytes()
    except OSError as exc:
        raise HTTPException(
            status_code=404, detail="Source image file is missing."
        ) from exc

    target_width = round(source.width * body.multiplier)
    target_height = round(source.height * body.multiplier)
    update_job_progress(job_id, percent=18, stage="refining")
    started_at = monotonic()
    try:
        # Refine before enlarging: a plain resize spreads existing detail over
        # a bigger canvas, which is what made upscaled images look softer.
        upscaled_bytes = ForgeApiClient().refine_and_upscale_image_bytes(
            source_bytes,
            prompt=source.positive_prompt,
            negative_prompt=source.negative_prompt,
            target_width=target_width,
            target_height=target_height,
            sampler_name=source.sampler_name,
            scheduler=source.scheduler,
            steps=source.steps,
            cfg_scale=source.cfg_scale,
            seed=source.seed,
            multiplier=body.multiplier,
        )
    except requests.Timeout as exc:
        raise HTTPException(
            status_code=504, detail="Forge timed out while upscaling."
        ) from exc
    except requests.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else 502
        detail = (
            "Forge returned an internal server error while upscaling."
            if status_code >= 500
            else f"Forge rejected the upscale request (HTTP {status_code})."
        )
        raise HTTPException(status_code=502, detail=detail) from exc
    except (requests.RequestException, ValueError) as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    upscale_duration_seconds = monotonic() - started_at
    update_job_progress(job_id, percent=94, stage="saving")

    payload = Txt2ImgPayload(
        prompt=source.positive_prompt,
        negative_prompt=source.negative_prompt,
        width=target_width,
        height=target_height,
        sampler_name=source.sampler_name,
        scheduler=source.scheduler,
        steps=source.steps,
        cfg_scale=source.cfg_scale,
        seed=source.seed,
    )
    settings = GenerationSettings(
        width=target_width,
        height=target_height,
        model=source.model,
        rating=source.rating,
        style=source.style,
        high_res=source.high_res,
        super_res=source.super_res,
        quality_mode=source.quality_mode,
        cfg_scale=source.cfg_scale,
    )
    try:
        record = save_generation(
            upscaled_bytes,
            payload,
            settings,
            actual_seed=source.seed,
            upscale_duration_seconds=upscale_duration_seconds,
            source_generation_id=generation_id,
        )
    except sqlite3.Error as exc:
        LOGGER.exception("Database failure while saving pixel upscale")
        raise HTTPException(
            status_code=503,
            detail="The upscaled image could not be saved. Please try again.",
        ) from exc
    except OSError as exc:
        LOGGER.exception("Filesystem failure while saving pixel upscale")
        raise HTTPException(
            status_code=507,
            detail="The upscaled image was created, but local storage could not save it.",
        ) from exc
    return serialize_generation(record)


@app.post(
    "/api/generations/{generation_id}/pixel-upscale",
    response_model=JobStartResponse,
    status_code=202,
)
def pixel_upscale_generation(
    generation_id: int, body: PixelUpscaleRequest, request: Request
) -> JobStartResponse:
    """Validate access and enqueue a pure pixel upscale."""
    _require_workspace(request, "forgeai")
    source = get_generation(generation_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    access_token = (
        body.access_token.get_secret_value() if body.access_token is not None else None
    )
    if not _can_access_generation(request, generation_id, access_token):
        raise HTTPException(status_code=403, detail="Image access denied.")
    source_path = GENERATED_DIR / (generation_filename(generation_id) or "")
    if not source_path.is_file():
        raise HTTPException(status_code=404, detail="Source image file is missing.")
    return _enqueue_forge_job(
        "pixel_upscale",
        body,
        request,
        runtime_payload=(generation_id, body),
        generation_id=generation_id,
    )


def _engine_prompt_response(
    body: SurpriseRequest,
    *,
    prompt_started_at: float,
    novelty: NoveltyContext,
    response_engine: str = "engine",
    notice: str | None = None,
    novelty_signature: str | None = None,
) -> PromptPairResponse:
    """Compile an engine-owned prompt, including provider fallbacks."""
    try:
        prompt, negative_prompt, engine_seed = engine_generate_pair(
            body.model,
            body.content_rating,
            body.orientation,
            style=body.style,
            batch_profile=body.batch_profile,
        )
    except PromptEngineError as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt engine could not build a prompt: {exc}",
        ) from exc
    stored_engine = (
        f"engine:{engine_seed}"
        if response_engine == "engine"
        else f"curated:engine:{engine_seed}"
    )
    quality_mode = _resolve_quality_mode(
        body.quality_mode, body.high_res, body.super_res
    )
    uses_high_res, uses_super_res = _quality_flags(quality_mode)
    prompt_run_id = _save_prompt_run_or_http(
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        source="auto_generated",
        engine=stored_engine,
        model=body.model,
        style=body.style,
        aspect_ratio=body.aspect_ratio,
        content_rating=body.content_rating,
        high_res=uses_high_res,
        super_res=uses_super_res,
        creativity_level=body.creativity_level,
        novelty_signature=novelty_signature or novelty.brief.as_json(),
        prompt_duration_seconds=monotonic() - prompt_started_at,
    )
    return PromptPairResponse(
        prompt=prompt,
        negative_prompt=negative_prompt,
        engine=response_engine,
        notice=notice,
        prompt_run_id=prompt_run_id,
    )


_CURATED_COMPATIBILITY_REASONS = frozenset({
    ReasonCode.MODEL_UNSUPPORTED,
    ReasonCode.STYLE_CONFLICT,
})


def _legacy_curated_compatibility_response(
    body: SurpriseRequest,
    *,
    prompt_started_at: float,
    history: list[object],
    novelty: NoveltyContext,
    uses_high_res: bool,
    uses_super_res: bool,
    reason_code: ReasonCode,
) -> PromptPairResponse:
    """Temporarily route unsupported Phase 3 combinations to legacy curated.

    Phase 4 model/style compilation removes this compatibility path.
    """
    LOGGER.warning(
        "Curated Prompt Engine compatibility fallback: model=%s style=%s "
        "reason_code=%s",
        body.model,
        body.style,
        reason_code.value,
    )
    best: tuple[float, str, str, NoveltyContext] | None = None
    curated_attempts = 0
    for _ in range(6):
        curated_attempts += 1
        brief_parts = (
            novelty.brief.action,
            novelty.brief.environment,
            novelty.brief.lighting,
            f"color palette of {novelty.brief.palette}",
            novelty.brief.mood,
            *novelty.quality_guidance,
        )
        prompt, negative_prompt = generate_random_prompt_pair(
            body.model,
            body.style,
            body.content_rating,
            body.aspect_ratio,
            uses_high_res,
            brief_parts,
            orientation=body.orientation,
        )
        similarity = maximum_similarity(prompt, novelty.recent_prompts)
        candidate = (similarity, prompt, negative_prompt, novelty)
        if best is None or similarity < best[0]:
            best = candidate
        if similarity <= novelty.similarity_threshold:
            break
        novelty = create_novelty_context(
            history,
            creativity_level=body.creativity_level,
            aspect_ratio=body.aspect_ratio,
            content_rating=body.content_rating,
        )
    assert best is not None
    similarity, prompt, negative_prompt, selected_novelty = best
    novelty_metadata = json.loads(selected_novelty.brief.as_json())
    novelty_metadata["fallback_occurrence"] = {
        "from": "prompt_engine_v1",
        "to": "legacy_curated",
        "reason_code": reason_code.value,
    }
    prompt_run_id = _save_prompt_run_or_http(
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        source="auto_generated",
        engine=f"curated:legacy-fallback:{reason_code.value}",
        model=body.model,
        style=body.style,
        aspect_ratio=body.aspect_ratio,
        content_rating=body.content_rating,
        high_res=uses_high_res,
        super_res=uses_super_res,
        creativity_level=body.creativity_level,
        novelty_signature=json.dumps(
            novelty_metadata, separators=(",", ":"), sort_keys=True
        ),
        similarity_score=similarity,
        novelty_retry_count=curated_attempts - 1,
        prompt_duration_seconds=monotonic() - prompt_started_at,
    )
    return PromptPairResponse(
        prompt=prompt,
        negative_prompt=negative_prompt,
        engine="curated",
        notice=(
            "Prompt Engine V1 does not support this model/style combination; "
            f"the legacy curated compatibility path was used ({reason_code.value})."
        ),
        prompt_run_id=prompt_run_id,
    )


def surprise(body: SurpriseRequest) -> PromptPairResponse:
    """Return a randomized Pony/SDXL prompt pair for the active preset."""
    prompt_started_at = monotonic()
    if body.style not in model_profile_for(body.model).styles:
        raise HTTPException(
            status_code=422,
            detail=f"Style '{body.style}' is not supported by the selected model.",
        )
    if body.content_rating not in RATING_RULES:
        raise HTTPException(status_code=422, detail="Unsupported content rating.")
    if body.aspect_ratio not in aspect_ratios_for(body.model):
        raise HTTPException(status_code=422, detail="Unsupported aspect ratio.")
    if body.prompt_engine not in {"curated", "engine", "ollama", "cloud"}:
        raise HTTPException(status_code=422, detail="Unsupported prompt engine.")
    if body.creativity_level not in CREATIVITY_LEVELS:
        raise HTTPException(status_code=422, detail="Unsupported creativity level.")
    quality_mode = _resolve_quality_mode(
        body.quality_mode, body.high_res, body.super_res
    )
    uses_high_res, uses_super_res = _quality_flags(quality_mode)
    try:
        history = recent_prompt_history(
            model=body.model,
            style=body.style,
            content_rating=body.content_rating,
            limit=150,
        )
    except sqlite3.Error:
        LOGGER.warning("Could not read prompt history; continuing without it", exc_info=True)
        history = []
    novelty = create_novelty_context(
        history,
        creativity_level=body.creativity_level,
        aspect_ratio=body.aspect_ratio,
        content_rating=body.content_rating,
    )
    notice: str | None = None
    if body.prompt_engine == "cloud":
        if not body.cloud_provider or not body.cloud_model or not body.cloud_api_key:
            raise HTTPException(
                status_code=422,
                detail="Configure a Cloud API provider and key in Settings first.",
            )
        api_key = body.cloud_api_key.get_secret_value()
        if not 8 <= len(api_key) <= 512:
            raise HTTPException(status_code=422, detail="The cloud API key is invalid.")
        try:
            result = CloudPromptClient().generate(
                provider=body.cloud_provider,
                api_key=api_key,
                cloud_model=body.cloud_model,
                forge_model=body.model,
                style=body.style,
                rating=body.content_rating,
                aspect_ratio=body.aspect_ratio,
                high_res=uses_high_res,
                orientation=body.orientation,
                novelty=novelty,
            )
        except CloudPromptError as exc:
            fallback_metadata = json.loads(novelty.brief.as_json())
            fallback_metadata.update({
                "prompt_source": "cloud",
                "intelligence_provider": body.cloud_provider,
                "intelligence_model": body.cloud_model,
                "fallback_occurrence": {
                    "from": "cloud",
                    "to": "curated",
                    "status_code": exc.status_code,
                },
            })
            notice = (
                f"{body.cloud_provider.title()} could not create a usable scene; "
                "Curated selection through Prompt Engine was used instead."
            )
            LOGGER.warning(
                "CloudAI prompt fell back to Prompt Engine: provider=%s model=%s error=%s",
                body.cloud_provider,
                body.cloud_model,
                exc,
            )
            return _engine_prompt_response(
                body,
                prompt_started_at=prompt_started_at,
                novelty=novelty,
                response_engine="curated",
                notice=notice,
                novelty_signature=json.dumps(
                    fallback_metadata, separators=(",", ":"), sort_keys=True
                ),
            )
        else:
            engine = f"cloud:{result.provider}:{result.model}"
            cloud_metadata = json.loads(result.novelty_signature or "{}")
            cloud_metadata.update({
                "prompt_source": "cloud",
                "intelligence_provider": result.provider,
                "intelligence_model": result.model,
                "model_profile": (
                    result.semantic.model_id if result.semantic is not None else None
                ),
                "canonical_concept_ids": (
                    list(result.semantic.concept_ids) if result.semantic is not None else []
                ),
                "validation": (
                    {
                        "decision": result.validation.decision.value,
                        "reason_codes": [
                            reason.value for reason in result.validation.reason_codes
                        ],
                        "score": result.validation.score,
                    }
                    if result.validation is not None else None
                ),
                "repair_notes": list(result.repair_notes),
            })
            prompt_run_id = _save_prompt_run_or_http(
                positive_prompt=result.prompt,
                negative_prompt=result.negative_prompt,
                source="auto_generated",
                engine=engine,
                model=body.model,
                style=body.style,
                aspect_ratio=body.aspect_ratio,
                content_rating=body.content_rating,
                high_res=uses_high_res,
                super_res=uses_super_res,
                creativity_level=body.creativity_level,
                novelty_signature=json.dumps(
                    cloud_metadata, separators=(",", ":"), sort_keys=True
                ),
                similarity_score=result.similarity_score,
                novelty_retry_count=result.retry_count,
                prompt_duration_seconds=monotonic() - prompt_started_at,
            )
            return PromptPairResponse(
                prompt=result.prompt,
                negative_prompt=result.negative_prompt,
                engine=engine,
                prompt_run_id=prompt_run_id,
            )
    if body.prompt_engine == "ollama" and body.ollama_model:
        try:
            result = OllamaPromptClient().generate(
                ollama_model=body.ollama_model,
                forge_model=body.model,
                style=body.style,
                rating=body.content_rating,
                aspect_ratio=body.aspect_ratio,
                high_res=uses_high_res,
                orientation=body.orientation,
                novelty=novelty,
            )
            prompt_run_id = _save_prompt_run_or_http(
                positive_prompt=result.prompt,
                negative_prompt=result.negative_prompt,
                source="auto_generated",
                engine=f"ollama:{result.model}",
                model=body.model,
                style=body.style,
                aspect_ratio=body.aspect_ratio,
                content_rating=body.content_rating,
                high_res=uses_high_res,
                super_res=uses_super_res,
                creativity_level=body.creativity_level,
                novelty_signature=result.novelty_signature,
                similarity_score=result.similarity_score,
                novelty_retry_count=result.retry_count,
                prompt_duration_seconds=monotonic() - prompt_started_at,
            )
            return PromptPairResponse(
                prompt=result.prompt,
                negative_prompt=result.negative_prompt,
                engine=f"ollama:{result.model}",
                prompt_run_id=prompt_run_id,
            )
        except (requests.RequestException, PromptEngineError, ValueError, TypeError, KeyError) as exc:
            notice = (
                "Ollama could not create a valid prompt; Curated selection through "
                f"Prompt Engine was used instead ({exc})."
            )
            LOGGER.warning("Ollama prompt fell back to Prompt Engine: %s", exc)
            return _engine_prompt_response(
                body,
                prompt_started_at=prompt_started_at,
                novelty=novelty,
                response_engine="curated",
                notice=notice,
            )
    elif body.prompt_engine == "ollama":
        notice = (
            "No Ollama model is selected; Curated selection through Prompt Engine was used."
        )
        return _engine_prompt_response(
            body,
            prompt_started_at=prompt_started_at,
            novelty=novelty,
            response_engine="curated",
            notice=notice,
        )

    if body.prompt_engine == "engine":
        return _engine_prompt_response(
            body,
            prompt_started_at=prompt_started_at,
            novelty=novelty,
        )

    try:
        curated = generate_curated(
            body.model,
            body.content_rating,
            body.orientation,
            history,
            creativity_level=body.creativity_level,
            aspect_ratio=body.aspect_ratio,
            style=body.style,
            batch_profile=body.batch_profile,
            seed=body.seed,
        )
    except PromptEngineError as exc:
        reason_codes = frozenset(exc.reason_codes)
        if reason_codes and reason_codes <= _CURATED_COMPATIBILITY_REASONS:
            return _legacy_curated_compatibility_response(
                body,
                prompt_started_at=prompt_started_at,
                history=history,
                novelty=novelty,
                uses_high_res=uses_high_res,
                uses_super_res=uses_super_res,
                reason_code=exc.reason_codes[0],
            )
        raise HTTPException(
            status_code=422,
            detail=f"Curated prompt engine could not build a prompt: {exc}",
        ) from exc
    prompt = curated.compiled.positive
    negative_prompt = curated.compiled.negative
    prompt_run_id = _save_prompt_run_or_http(
        positive_prompt=prompt,
        negative_prompt=negative_prompt,
        source="auto_generated",
        engine=f"curated:engine:{curated.seed}",
        model=body.model,
        style=body.style,
        aspect_ratio=body.aspect_ratio,
        content_rating=body.content_rating,
        high_res=uses_high_res,
        super_res=uses_super_res,
        creativity_level=body.creativity_level,
        novelty_signature=curated.novelty_signature,
        similarity_score=curated.similarity_score,
        novelty_retry_count=curated.attempt_count - 1,
        prompt_duration_seconds=monotonic() - prompt_started_at,
    )
    return PromptPairResponse(
        prompt=prompt,
        negative_prompt=negative_prompt,
        engine="curated",
        notice=notice,
        prompt_run_id=prompt_run_id,
    )


@app.post("/api/surprise", response_model=PromptPairResponse)
def surprise_endpoint(body: SurpriseRequest, request: Request) -> PromptPairResponse:
    """Authorize the requested rating before constructing an automatic prompt."""
    _require_safe_rating(request, body.content_rating)
    if body.prompt_engine == "cloud":
        _admin_account(request)
    return surprise(body)


def _validate_surprise_request(
    body: SurpriseRequest, *, prompt_already_resolved: bool = False
) -> None:
    """Reject unsupported automatic-prompt settings before enqueueing."""
    if body.style not in model_profile_for(body.model).styles:
        raise HTTPException(
            status_code=422,
            detail=f"Style '{body.style}' is not supported by the selected model.",
        )
    if body.content_rating not in RATING_RULES:
        raise HTTPException(status_code=422, detail="Unsupported content rating.")
    if body.aspect_ratio not in aspect_ratios_for(body.model):
        raise HTTPException(status_code=422, detail="Unsupported aspect ratio.")
    if body.prompt_engine not in {"curated", "engine", "ollama", "cloud"}:
        raise HTTPException(status_code=422, detail="Unsupported prompt engine.")
    if body.creativity_level not in CREATIVITY_LEVELS:
        raise HTTPException(status_code=422, detail="Unsupported creativity level.")
    # A key is only needed to produce a prompt. Re-running a job whose prompt
    # already exists never calls the provider, so requiring one there blocks a
    # replay that would work fine.
    if (
        body.prompt_engine == "cloud"
        and not prompt_already_resolved
        and (not body.cloud_provider or not body.cloud_model or not body.cloud_api_key)
    ):
        raise HTTPException(
            status_code=422,
            detail="Configure a Cloud API provider and key in Settings first.",
        )


def _prepared_surprise_request(
    body: SurpriseRequest,
    prompt: str,
    negative_prompt: str,
    prompt_run_id: int | None,
) -> GenerateRequest:
    """Convert a resolved automatic prompt into its Forge-bound request."""
    quality_mode = _resolve_quality_mode(
        body.quality_mode, body.high_res, body.super_res
    )
    high_res, super_res = _quality_flags(quality_mode)
    return GenerateRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=body.model,
        style=body.style,
        aspect_ratio=body.aspect_ratio,
        content_rating=body.content_rating,
        high_res=high_res,
        super_res=super_res,
        quality_mode=quality_mode,
        prompt_run_id=prompt_run_id,
        prompt_engine=body.prompt_engine,
        creativity_level=body.creativity_level,
        cfg_scale=body.cfg_scale,
        request_id=body.request_id,
        seed=body.seed if body.seed is not None else -1,
    )


@app.post(
    "/api/surprise-generate",
    response_model=JobStartResponse,
    status_code=202,
)
def surprise_generate(
    body: SurpriseRequest, request: Request
) -> JobStartResponse:
    """Validate and enqueue the complete prompt-plus-image workflow."""
    _require_workspace(request, "forgeai")
    _require_safe_rating(request, body.content_rating)
    if body.prompt_engine == "cloud":
        _admin_account(request)
    _validate_surprise_request(body)
    started = _enqueue_forge_job("surprise_generate", body, request)
    if body.prompt_engine in {"cloud", "ollama"}:
        # Prompt preparation consumes network or local-LLM capacity, not
        # Forge's GPU, so it runs off the request thread and off the single GPU
        # FIFO. Resolving it here rather than in the worker means a queued job
        # already has its prompt by the time the GPU frees up, and the click
        # returns immediately instead of waiting on the provider.
        #
        # The worker resolves the prompt itself when this has not finished in
        # time, so a slow or failed provider costs nothing but the old
        # behaviour. Curated prompts build in-process in under a millisecond
        # and gain nothing from any of this.
        _resolve_prompt_in_background(started.job_id, body)
    return started


def _resolve_prompt_in_background(job_id: int, body: SurpriseRequest) -> None:
    """Store a job's prompt ahead of its turn on the GPU queue."""

    def resolve() -> None:
        try:
            prompts = surprise(body)
        except Exception:  # noqa: BLE001 - the worker retries and reports
            LOGGER.warning("Could not pre-resolve job %s prompt", job_id, exc_info=True)
            return
        try:
            update_job_request_payload(
                job_id,
                resolved_positive_prompt=prompts.prompt,
                resolved_negative_prompt=prompts.negative_prompt,
                resolved_prompt_run_id=prompts.prompt_run_id,
                resolved_prompt_engine=prompts.engine,
            )
        except sqlite3.Error:
            LOGGER.warning("Could not store job %s prompt", job_id, exc_info=True)

    Thread(target=resolve, name=f"aiqg-prompt-{job_id}", daemon=True).start()



def _run_video_job(body: VideoRequest, *, mode: Literal["t2v", "i2v"], job_id: int) -> None:
    """Run one allowlisted Wan workflow and persist only its file metadata."""
    seed = body.seed if body.seed >= 0 else secrets.randbelow(1125899906842624)
    source_path: Path | None = None
    generation_size = T2V_SIZES[body.orientation]
    if mode == "i2v":
        if (body.source_generation_id is None) == (body.upload_id is None):
            raise ValueError("Animate image requires exactly one gallery image or upload.")
        if body.source_generation_id is not None:
            source = get_generation(body.source_generation_id)
            if source is None: raise ValueError("Source generation was not found.")
            source_path = GENERATED_DIR / source.filename
        else:
            source_path = _upload_path(body.upload_id or "")
        if source_path is None or not source_path.is_file(): raise ValueError("Source image file is missing.")
        try:
            with Image.open(source_path) as source_image:
                generation_size = derive_i2v_size(*ImageOps.exif_transpose(source_image).size)
        except (OSError, UnidentifiedImageError) as exc:
            raise ValueError("Source image file is invalid.") from exc
        if body.camera_motion not in CAMERA_MOTIONS: raise ValueError("Unsupported camera motion.")
    client = ComfyApiClient()
    uploaded = client.upload_image(source_path) if source_path else None
    graph = build_workflow(mode, prompt=body.prompt, negative_prompt=body.negative_prompt,
        seed=seed, source_filename=uploaded, camera_motion=body.camera_motion,
        interpolation=body.interpolation, output_resolution=body.output_resolution, steps=body.steps,
        generation_size=generation_size, orientation=body.orientation)
    update_job_progress(job_id, percent=4, stage="submitted")
    def progress(value: int, maximum: int, stage: str = "sampling") -> None:
        try:
            _raise_if_job_cancelled(job_id)
        except JobExecutionError:
            client.cancel(str(get_job(job_id).request_payload.get("comfy_prompt_id") or ""))
            raise
        if stage == "sampling":
            percent = 8 + (max(0, value) / max(1, maximum)) * 67
            update_job_progress(job_id, percent=percent, stage=stage, current_step=value, total_steps=maximum)
        else:
            percent = {"interpolating": 78, "upscaling": 84, "encoding": 94}.get(stage, 76)
            update_job_progress(job_id, percent=percent, stage=stage)
    known_prompt_id = str(get_job(job_id).request_payload.get("comfy_prompt_id") or "") or None
    result = client.generate_video(graph, progress=progress,
        on_submitted=lambda prompt_id: update_job_request_payload(job_id, comfy_prompt_id=prompt_id),
        known_prompt_id=known_prompt_id)
    update_job_request_payload(job_id, comfy_prompt_id=result.prompt_id)
    update_job_progress(job_id, percent=93, stage="encoding")
    record = save_video_generation(content=result.content, prompt=body.prompt,
        negative_prompt=body.negative_prompt, mode=mode,
        source_generation_id=body.source_generation_id,
        width=derive_upscale_size(*generation_size, body.output_resolution)[0],
        height=derive_upscale_size(*generation_size, body.output_resolution)[1],
        frame_count=(49 - 1) * body.interpolation + 1, fps=16 * body.interpolation, seed=seed,
        duration_seconds=((49 - 1) * body.interpolation + 1) / (16 * body.interpolation),
        camera_motion=body.camera_motion if mode == "i2v" else None,
        comfy_prompt_id=result.prompt_id, poster_content=result.poster_content)
    update_job_request_payload(job_id, video_id=record.id, resolved_seed=seed)
    update_job_progress(job_id, percent=98, stage="saving")
    return None


def _execute_job(job_id: int, runtime_payload: object | None) -> int | None:
    """Dispatch one queued job through the existing Forge logic."""
    record = get_job(job_id)
    if record is None:
        raise JobExecutionError("The queued job no longer exists.")
    _raise_if_job_cancelled(job_id)
    try:
        if record.kind == "generate":
            body = (
                runtime_payload
                if isinstance(runtime_payload, GenerateRequest)
                else GenerateRequest.model_validate(record.request_payload)
            )
            _raise_if_job_cancelled(job_id)
            return generate(body, job_id=job_id).id
        if record.kind == "batch_generate":
            # A batch plan already contains a validated, token-budgeted engine
            # prompt. Execution rejoins the standard generation path so
            # persistence, Gallery, progress and Forge serialization match an
            # interactive image.
            body = (
                runtime_payload
                if isinstance(runtime_payload, GenerateRequest)
                else GenerateRequest.model_validate(record.request_payload)
            )
            _raise_if_job_cancelled(job_id)
            return generate(body, job_id=job_id).id
        if record.kind == "surprise_generate":
            body = (
                runtime_payload
                if isinstance(runtime_payload, SurpriseRequest)
                else SurpriseRequest.model_validate(record.request_payload)
            )
            prepared = runtime_payload if isinstance(runtime_payload, GenerateRequest) else None
            if prepared is None and record.request_payload.get("resolved_positive_prompt"):
                prepared = _prepared_surprise_request(
                    body,
                    str(record.request_payload["resolved_positive_prompt"]),
                    str(record.request_payload.get("resolved_negative_prompt") or ""),
                    (
                        int(record.request_payload["resolved_prompt_run_id"])
                        if record.request_payload.get("resolved_prompt_run_id") is not None
                        else None
                    ),
                )
            if prepared is None:
                update_job_progress(job_id, percent=4, stage="prompt_generation")
                prompts = surprise(body)
                prepared = _prepared_surprise_request(
                    body,
                    prompts.prompt,
                    prompts.negative_prompt,
                    prompts.prompt_run_id,
                )
                update_job_request_payload(
                    job_id,
                    resolved_positive_prompt=prompts.prompt,
                    resolved_negative_prompt=prompts.negative_prompt,
                    resolved_prompt_run_id=prompts.prompt_run_id,
                    resolved_prompt_engine=prompts.engine,
                )
            update_job_progress(job_id, percent=11, stage="prompt_ready")
            _raise_if_job_cancelled(job_id)
            generation = generate(prepared, job_id=job_id)
            return generation.id
        if record.kind in {"img2img", "character"}:
            body = (
                runtime_payload
                if isinstance(runtime_payload, Img2ImgRequest)
                else Img2ImgRequest.model_validate(record.request_payload)
            )
            _raise_if_job_cancelled(job_id)
            if record.kind == "character":
                return _run_character_job(body, job_id=job_id).id
            return _run_img2img_job(body, job_id=job_id).id
        if record.kind in {"upscale", "pixel_upscale"}:
            if isinstance(runtime_payload, tuple) and len(runtime_payload) == 2:
                generation_id, body = runtime_payload
            else:
                values = dict(record.request_payload)
                generation_id = int(values.pop("generation_id"))
                body = (
                    UpscaleRequest.model_validate(values)
                    if record.kind == "upscale"
                    else PixelUpscaleRequest.model_validate(values)
                )
            if record.kind == "upscale":
                if not isinstance(body, UpscaleRequest):
                    raise ValueError("Invalid upscale job payload.")
                _raise_if_job_cancelled(job_id)
                return _run_upscale_job(int(generation_id), body, job_id=job_id).id
            if not isinstance(body, PixelUpscaleRequest):
                raise ValueError("Invalid pixel-upscale job payload.")
            _raise_if_job_cancelled(job_id)
            return _run_pixel_upscale_job(int(generation_id), body, job_id=job_id).id
        if record.kind in {"video_t2v", "video_i2v"}:
            body = runtime_payload if isinstance(runtime_payload, VideoRequest) else VideoRequest.model_validate(record.request_payload)
            return _run_video_job(body, mode="t2v" if record.kind == "video_t2v" else "i2v", job_id=job_id)
        raise JobExecutionError(f"Unsupported job kind: {record.kind}")
    except HTTPException as exc:
        detail = exc.detail if isinstance(exc.detail, str) else "The Forge job failed."
        raise JobExecutionError(detail) from exc
    except (ValueError, TypeError) as exc:
        raise JobExecutionError(str(exc)) from exc


def _refresh_running_job(record: JobRecord) -> JobRecord:
    """Sample Forge's real progress when the active job is being observed."""
    if record.status != "running" or record.progress_stage in {
        "prompt_generation",
        "prompt_ready",
        "upscaling",
        "final_upscale_4k",
        "final_upscale_8k",
        "upscale_1_8k",
        "upscale_2_12k",
        "upscale_1_12k",
        "saving",
    }:
        return record
    try:
        snapshot = ForgeApiClient().generation_progress()
        progress = max(0.0, min(1.0, float(snapshot.get("progress", 0) or 0)))
        state = snapshot.get("state") if isinstance(snapshot.get("state"), dict) else {}
        current_step = int(state.get("sampling_step", 0) or 0)
        total_steps = int(state.get("sampling_steps", 0) or 0)
        eta = max(0.0, float(snapshot.get("eta_relative", 0) or 0))
        preview = snapshot.get("current_image")
        if isinstance(preview, str) and preview and not preview.startswith("data:"):
            preview = f"data:image/png;base64,{preview}"
        high_res = bool(record.request_payload.get("high_res")) or bool(
            record.request_payload.get("super_res")
        )
        stage = (
            "model_loading"
            if progress <= 0 and current_step <= 0
            else "finalizing"
            if progress >= 0.98 or (total_steps > 0 and current_step >= total_steps)
            else "refining"
            if high_res and progress >= 0.68
            else "sampling"
        )
        percent = min(92.0, 14.0 + progress * (70.0 if high_res else 78.0))
        update_job_progress(
            record.id,
            percent=percent,
            stage=stage,
            current_step=current_step or None,
            total_steps=total_steps or None,
            eta_seconds=eta if eta > 0 else None,
            current_image=preview if isinstance(preview, str) and preview else None,
        )
        return get_job(record.id) or record
    except (requests.RequestException, sqlite3.Error, ValueError, TypeError):
        return record


def _authorize_job(request: Request, record: JobRecord, access: str | None) -> None:
    """Allow an admin, submitting account, or matching guest capability."""
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    if account and (account.is_admin or record.account_id == account.id):
        return
    if verify_job_access(record.id, access):
        return
    raise HTTPException(status_code=403, detail="Job access denied.")


def _serialize_job(record: JobRecord, access: str | None = None) -> JobResponse:
    generation: GenerationResponse | None = None
    if record.generation_id is not None:
        generation_record = get_generation(record.generation_id)
        if generation_record is not None:
            generation = serialize_generation(generation_record)
            if access:
                generation = generation.model_copy(
                    update={
                        "access_token": access,
                        "full_url": f"{generation.full_url}?access={access}",
                        "thumbnail_url": f"{generation.thumbnail_url}?access={access}",
                    }
                )
    prompt_source = generation_record if record.generation_id is not None else None
    if prompt_source is None:
        source_value = record.request_payload.get("source_generation_id")
        if source_value is None:
            source_value = record.request_payload.get("generation_id")
        try:
            prompt_source = get_generation(int(source_value)) if source_value is not None else None
        except (TypeError, ValueError):
            prompt_source = None
    positive_prompt = (
        generation.positive_prompt
        if generation is not None
        else str(record.request_payload.get("resolved_positive_prompt") or record.request_payload.get("prompt") or "")
        or (prompt_source.positive_prompt if prompt_source is not None else None)
    )
    negative_prompt = (
        generation.negative_prompt
        if generation is not None
        else str(record.request_payload.get("resolved_negative_prompt") or record.request_payload.get("negative_prompt") or "")
        or (prompt_source.negative_prompt if prompt_source is not None else None)
    )
    return JobResponse(
        id=record.id,
        kind=record.kind,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        request_payload=record.request_payload,
        progress_percent=record.progress_percent,
        progress_stage=record.progress_stage,
        current_step=record.current_step,
        total_steps=record.total_steps,
        eta_seconds=record.eta_seconds,
        current_image=record.current_image,
        generation_id=record.generation_id,
        generation=generation,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt,
        error_message=record.error_message,
    )


JOB_LIST_REQUEST_KEYS = frozenset(
    {
        "aspect_ratio",
        "batch_id",
        "camera_motion",
        "character_generator",
        "denoising_strength",
        "high_res",
        "model",
        "multiplier",
        "preset",
        "quality_mode",
        "source_generation_id",
        "style",
        "style_variant",
        "super_res",
    }
)


def _serialize_job_list_item(record: JobRecord) -> JobListResponse:
    """Serialize only fields rendered before a job's detail view is opened."""
    request_payload = {
        key: value
        for key, value in record.request_payload.items()
        if key in JOB_LIST_REQUEST_KEYS
    }
    if record.request_payload.get("character_selected_generator") is not None:
        request_payload["character_generator"] = record.request_payload[
            "character_selected_generator"
        ]
    # Older derived-image requests store their input under generation_id. The
    # list presents either spelling as the source generation, while the detail
    # endpoint preserves the original full request exactly.
    if (
        "source_generation_id" not in request_payload
        and record.request_payload.get("generation_id") is not None
    ):
        request_payload["source_generation_id"] = record.request_payload["generation_id"]

    generation: JobListGenerationResponse | None = None
    if record.generation_id is not None:
        generation_record = get_generation(record.generation_id)
        if generation_record is not None:
            generation = JobListGenerationResponse(
                id=generation_record.id,
                thumbnail_url=f"/api/images/{generation_record.id}/thumbnail",
                full_url=f"/api/images/{generation_record.id}",
                width=generation_record.width,
                height=generation_record.height,
                rating=generation_record.rating,
                style_variant=generation_record.style_variant,
                seed=generation_record.seed,
                source_generation_id=generation_record.source_generation_id,
                favorite=generation_record.favorite,
            )

    return JobListResponse(
        id=record.id,
        kind=record.kind,
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        request_payload=request_payload,
        progress_percent=record.progress_percent,
        progress_stage=record.progress_stage,
        current_step=record.current_step,
        total_steps=record.total_steps,
        eta_seconds=record.eta_seconds,
        current_image=record.current_image,
        generation_id=record.generation_id,
        generation=generation,
        error_message=record.error_message,
    )


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_detail(
    job_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> JobResponse:
    """Return live state and the final artifact for one authorized job."""
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _authorize_job(request, record, access)
    record = _refresh_running_job(record)
    return _serialize_job(record, access)


def _rerun_job(record: JobRecord, request: Request) -> JobStartResponse:
    """Reconstruct and enqueue one persisted job without reusing transient IDs."""
    required_workspace = "forgevid" if record.kind in {"video_t2v", "video_i2v"} else "forgeimg" if record.kind in {"img2img", "character"} else "forgeai"
    _require_workspace(request, required_workspace)
    values = dict(record.request_payload)
    values.pop("request_id", None)
    values.pop("prompt_run_id", None)
    values.pop("comfy_prompt_id", None)
    values.pop("video_id", None)
    values.pop("resolved_seed", None)
    try:
        if record.kind == "generate":
            body = GenerateRequest.model_validate(values)
            _require_safe_rating(request, body.content_rating)
            _validate_generate_request(body)
            return _enqueue_forge_job("generate", body, request)
        if record.kind == "surprise_generate":
            # Only block a re-run that would still need the provider. Once the
            # prompt has been generated it is stored on the job, and the worker
            # reuses it without calling out, so the missing key is irrelevant.
            # Older jobs pre-date the resolved_* fields but still produced a
            # generation, and that record holds the prompt the UI displays.
            # Recover it so a completed job is always re-runnable.
            if not values.get("resolved_positive_prompt") and record.generation_id:
                completed = get_generation(record.generation_id)
                if completed is not None and completed.positive_prompt:
                    values["resolved_positive_prompt"] = completed.positive_prompt
                    values["resolved_negative_prompt"] = completed.negative_prompt or ""
                    values["resolved_prompt_engine"] = values.get("prompt_engine")
            if values.get("prompt_engine") == "cloud" and not values.get(
                "resolved_positive_prompt"
            ):
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "This cloud prompt job never produced a prompt, and API "
                        "keys are intentionally not stored. Start a new job."
                    ),
                )
            body = SurpriseRequest.model_validate(values)
            _require_safe_rating(request, body.content_rating)
            _validate_surprise_request(
                body, prompt_already_resolved=bool(values.get("resolved_positive_prompt"))
            )
            # SurpriseRequest does not carry the resolved prompt, so forward it
            # explicitly. Without this the re-run would call the provider again
            # and fail on the key that was deliberately never stored.
            resolved = {
                key: values[key]
                for key in (
                    "resolved_positive_prompt",
                    "resolved_negative_prompt",
                    "resolved_prompt_run_id",
                    "resolved_prompt_engine",
                )
                if values.get(key) is not None
            }
            return _enqueue_forge_job(
                "surprise_generate", body, request, **resolved
            )
        if record.kind in {"img2img", "character"}:
            body = Img2ImgRequest.model_validate(values)
            _require_safe_rating(request, body.content_rating)
            _validate_img2img_request(body, request)
            return _enqueue_forge_job(record.kind, body, request)
        if record.kind in {"video_t2v", "video_i2v"}:
            body = VideoRequest.model_validate(values)
            return _enqueue_forge_job(record.kind, body, request)
        generation_id = int(values.pop("generation_id"))
        source = get_generation(generation_id)
        if source is None:
            raise HTTPException(status_code=409, detail="The source generation is no longer available.")
        if record.kind == "upscale":
            body = UpscaleRequest.model_validate(values)
            _require_safe_rating(request, source.rating)
            return _enqueue_forge_job(
                "upscale",
                body,
                request,
                runtime_payload=(generation_id, body),
                generation_id=generation_id,
            )
        if record.kind == "pixel_upscale":
            body = PixelUpscaleRequest.model_validate(values)
            return _enqueue_forge_job(
                "pixel_upscale",
                body,
                request,
                runtime_payload=(generation_id, body),
                generation_id=generation_id,
            )
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=409,
            detail="This job's stored configuration is no longer valid for re-run.",
        ) from exc
    raise HTTPException(status_code=409, detail="This job type cannot be re-run.")


@app.post("/api/jobs/{job_id}/rerun", response_model=JobStartResponse, status_code=202)
def rerun_job(
    job_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> JobStartResponse:
    """Enqueue a fresh execution from an authorized job's stored settings."""
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _authorize_job(request, record, access)
    return _rerun_job(record, request)


class DeleteJobsRequest(BaseModel):
    """Job ids to remove from the execution history."""

    job_ids: list[int] = Field(min_length=1, max_length=500)


class DeleteJobsResponse(BaseModel):
    deleted: int


class FavoriteRequest(BaseModel):
    favorite: bool


@app.post("/api/generations/{generation_id}/favorite", response_model=GenerationResponse)
def set_favorite(
    generation_id: int,
    body: FavoriteRequest,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> GenerationResponse:
    """Mark or unmark one generation as a favourite."""
    if not _can_access_generation(request, generation_id, access):
        raise HTTPException(status_code=403, detail="Image access denied.")
    if not set_generation_favorite(generation_id, body.favorite):
        raise HTTPException(status_code=404, detail="Generation not found.")
    record = get_generation(generation_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    return serialize_generation(record)


@app.post("/api/jobs/delete", response_model=DeleteJobsResponse)
def delete_job_records(body: DeleteJobsRequest, request: Request) -> DeleteJobsResponse:
    """Remove finished job records without touching the images they produced."""
    _require_authenticated_account(request)
    # delete_jobs only removes terminal rows, so a queued or running job the
    # runner still references cannot be deleted out from under it.
    return DeleteJobsResponse(deleted=delete_jobs(body.job_ids))


@app.post("/api/jobs/{job_id}/cancel", response_model=JobResponse)
def cancel_job(
    job_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> JobResponse:
    """Cancel queued work or safely interrupt the active diffusion request."""
    record = get_job(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    _authorize_job(request, record, access)
    if record.status in {"succeeded", "failed", "cancelled"}:
        raise HTTPException(status_code=409, detail="This job has already finished.")
    runner = _job_runner
    if runner is None:
        raise HTTPException(status_code=503, detail="The local job worker is unavailable.")

    non_interruptible = record.kind == "pixel_upscale" or record.progress_stage in {
        "decoding",
        "upscaling",
        "saving",
    }
    interrupt = None
    if not non_interruptible and record.progress_stage not in {
        "prompt_generation",
        "prompt_ready",
    }:
        interrupt = (
            lambda: ComfyApiClient().cancel(str(record.request_payload.get("comfy_prompt_id") or ""))
            if record.kind in {"video_t2v", "video_i2v"}
            else ForgeApiClient().interrupt_generation()
        )
    try:
        outcome = runner.request_cancel(
            job_id,
            interrupt_active=interrupt,
            allow_active=not non_interruptible,
        )
    except requests.RequestException as exc:
        raise HTTPException(
            status_code=502,
            detail="Forge did not accept the interrupt request; the job is still running.",
        ) from exc

    if outcome == "not_interruptible":
        raise HTTPException(
            status_code=409,
            detail="This job is already encoding or upscaling and cannot be safely interrupted.",
        )
    if outcome == "not_active":
        refreshed = get_job(job_id)
        if refreshed is not None and refreshed.status in {"succeeded", "failed", "cancelled"}:
            return _serialize_job(refreshed, access)
        raise HTTPException(status_code=409, detail="The job is finishing and can no longer be cancelled.")
    if outcome == "queued":
        finish_job(
            job_id,
            status="cancelled",
            error_message="Cancelled before execution.",
        )
    refreshed = get_job(job_id)
    if refreshed is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return _serialize_job(refreshed, access)


@app.get(
    "/api/generations/{generation_id}/lineage",
    response_model=GenerationLineageResponse,
)
def generation_lineage(
    generation_id: int,
    request: Request,
    access: str | None = Query(default=None, max_length=256),
) -> GenerationLineageResponse:
    """Return a complete, bounded ancestor chain in one authorized request."""
    root = get_generation(generation_id)
    if root is None:
        raise HTTPException(status_code=404, detail="Generation not found.")
    if not _can_access_generation(request, generation_id, access):
        raise HTTPException(status_code=403, detail="Generation access denied.")

    records = []
    seen: set[int] = set()
    current = root
    while current is not None and current.id not in seen and len(records) < 32:
        records.append(current)
        seen.add(current.id)
        current = (
            get_generation(current.source_generation_id)
            if current.source_generation_id is not None
            else None
        )
    records.reverse()

    nodes: list[GenerationLineageNodeResponse] = []
    for record in records:
        can_view_thumbnail = _can_access_generation(request, record.id, access)
        thumbnail_url = f"/api/images/{record.id}/thumbnail" if can_view_thumbnail else None
        if thumbnail_url and access:
            thumbnail_url = f"{thumbnail_url}?access={access}"
        nodes.append(
            GenerationLineageNodeResponse(
                id=record.id,
                source_generation_id=record.source_generation_id,
                pipeline_kind=job_kind_for_generation(record.id),
                timestamp=record.timestamp,
                model=record.model,
                style=record.style,
                rating=record.rating,
                width=record.width,
                height=record.height,
                high_res=record.high_res,
                super_res=record.super_res,
                thumbnail_url=thumbnail_url,
            )
        )
    return GenerationLineageResponse(generation_id=generation_id, items=nodes)


def _serialize_video(record) -> VideoResponse:
    return VideoResponse(id=record.id, prompt=record.prompt, negative_prompt=record.negative_prompt,
        mode=record.mode, source_generation_id=record.source_generation_id, width=record.width,
        height=record.height, frame_count=record.frame_count, fps=record.fps, seed=record.seed,
        duration_seconds=record.duration_seconds, camera_motion=record.camera_motion,
        favorite=record.favorite, created_at=record.created_at, file_url=f"/api/video/{record.id}/file",
        poster_url=f"/api/video/{record.id}/poster" if record.poster_filename else None)


@app.get("/api/video/status", response_model=VideoStatusResponse)
def video_status() -> VideoStatusResponse:
    try:
        ComfyApiClient().health()
        return VideoStatusResponse(connected=True)
    except requests.RequestException:
        return VideoStatusResponse(connected=False)


@app.post("/api/video/prompt", response_model=VideoPromptResponse)
def create_video_prompt(body: VideoPromptRequest, request: Request) -> VideoPromptResponse:
    """Generate Wan motion prose using the shared prompt-run ledger."""
    _require_workspace(request, "forgevid")
    started = monotonic()
    image_base64: str | None = None
    camera_motion: str | None = None
    analysis_note: str | None = None
    if body.mode == "i2v" and body.engine == "ollama":
        if (body.source_generation_id is None) == (body.upload_id is None):
            raise HTTPException(status_code=422, detail="Choose exactly one source image before generating an I2V prompt.")
        if body.source_generation_id is not None:
            source = get_generation(body.source_generation_id)
            if source is None:
                raise HTTPException(status_code=404, detail="Source generation not found.")
            if not _can_access_generation(request, source.id, None):
                raise HTTPException(status_code=403, detail="Image access denied.")
            source_path = GENERATED_DIR / source.filename
        else:
            source_path = _upload_path(body.upload_id or "")
        if source_path is None or not source_path.is_file():
            raise HTTPException(status_code=422, detail="Source image is unavailable.")
        image_base64 = base64.b64encode(source_path.read_bytes()).decode("ascii")
    try:
        if body.engine == "curated":
            prompt = curated_video_prompt(mode=body.mode, source_prompt=body.source_prompt)
        elif body.engine == "reuse":
            prompt = strip_sdxl_tags(body.source_prompt)
            if not prompt:
                raise ValueError("Choose a prompt to reuse.")
        elif body.engine == "ollama":
            if not body.ollama_model:
                raise ValueError("Choose an Ollama model.")
            if image_base64:
                # Structured analysis first: the frame decides what can move and
                # which camera move the geometry supports, instead of the prompt
                # and the camera embedding being chosen independently.
                try:
                    analysis = analyse_source_frame(body.ollama_model, image_base64)
                    prompt = compose_i2v_prompt(analysis)
                    camera_motion = analysis["camera"]
                    analysis_note = f"{analysis['pose']} subject; camera {analysis['camera']}"
                except (ValueError, KeyError):
                    # Vision failed or returned unusable JSON; fall back to the
                    # prose path rather than blocking the user.
                    prompt = ollama_video_prompt(body.ollama_model, context=body.source_prompt, image_base64=image_base64)
            else:
                prompt = ollama_video_prompt(body.ollama_model, context=body.source_prompt)
        else:
            if not body.cloud_provider or not body.cloud_model or body.cloud_api_key is None:
                raise ValueError("Complete the Cloud API settings first.")
            prompt = cloud_video_prompt(body.cloud_provider, body.cloud_api_key.get_secret_value(), body.cloud_model, context=body.source_prompt)
    except (ValueError, requests.RequestException) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    negative = "low quality, blurry, distorted, morphing, flickering, jitter, unstable motion, text, watermark"
    prompt_run_id = _save_prompt_run_or_http(
        positive_prompt=prompt, negative_prompt=negative, source="forgevid", engine=body.engine,
        model="Wan 2.1", style="Video motion", aspect_ratio="Video", content_rating="Safe",
        high_res=False, super_res=False, prompt_duration_seconds=monotonic() - started,
    )
    return VideoPromptResponse(
        prompt=prompt, negative_prompt=negative, prompt_run_id=prompt_run_id,
        engine=body.engine, camera_motion=camera_motion, analysis_note=analysis_note,
    )


@app.post("/api/video/t2v", response_model=JobStartResponse)
def create_t2v(body: VideoRequest, request: Request) -> JobStartResponse:
    _require_workspace(request, "forgevid")
    if body.source_generation_id is not None or body.upload_id is not None:
        raise HTTPException(status_code=422, detail="Text-to-video does not accept a source image.")
    if body.prompt_engine != "manual":
        try:
            validate_video_prompt(body.prompt)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _enqueue_forge_job("video_t2v", body, request)


@app.post("/api/video/i2v", response_model=JobStartResponse)
def create_i2v(body: VideoRequest, request: Request) -> JobStartResponse:
    _require_workspace(request, "forgevid")
    if (body.source_generation_id is None) == (body.upload_id is None):
        raise HTTPException(status_code=422, detail="Animate image requires exactly one gallery image or upload.")
    if body.source_generation_id is not None:
        source = get_generation(body.source_generation_id)
        if source is None: raise HTTPException(status_code=404, detail="Source generation not found.")
        if not _can_access_generation(request, source.id, None): raise HTTPException(status_code=403, detail="Image access denied.")
    elif _upload_path(body.upload_id or "") is None:
        raise HTTPException(status_code=422, detail="Invalid upload identifier.")
    if body.camera_motion not in CAMERA_MOTIONS:
        raise HTTPException(status_code=422, detail="Unsupported camera motion.")
    if body.prompt_engine != "manual":
        try:
            validate_video_prompt(body.prompt)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
    return _enqueue_forge_job("video_i2v", body, request)


@app.get("/api/video/history", response_model=VideoHistoryResponse)
def video_history(request: Request, limit: int = Query(60, ge=1, le=100), before_id: int | None = Query(default=None, ge=1)) -> VideoHistoryResponse:
    _require_authenticated_account(request)
    records = recent_video_generations(before_id=before_id, limit=limit + 1)
    page = records[:limit]
    return VideoHistoryResponse(items=[_serialize_video(item) for item in page], next_cursor=page[-1].id if len(records) > limit and page else None)


@app.get("/api/video/{video_id}/file")
def video_file(video_id: int, request: Request):
    _require_authenticated_account(request)
    record = get_video_generation(video_id)
    if record is None: raise HTTPException(status_code=404, detail="Video not found.")
    path = VIDEOS_DIR / record.filename
    if not path.is_file(): raise HTTPException(status_code=404, detail="Video file is missing.")
    size = path.stat().st_size
    range_header = request.headers.get("range")
    if not range_header:
        return FileResponse(path, media_type="video/mp4", headers={"Accept-Ranges": "bytes"})
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", range_header.strip())
    if not match: raise HTTPException(status_code=416, detail="Invalid byte range.", headers={"Content-Range": f"bytes */{size}"})
    start_text, end_text = match.groups()
    if not start_text and not end_text: raise HTTPException(status_code=416, detail="Invalid byte range.")
    if start_text:
        start = int(start_text); end = min(int(end_text), size - 1) if end_text else size - 1
    else:
        length = min(int(end_text), size); start, end = size - length, size - 1
    if start >= size or start > end: raise HTTPException(status_code=416, detail="Range is outside the file.", headers={"Content-Range": f"bytes */{size}"})
    def chunks():
        with path.open("rb") as stream:
            stream.seek(start); remaining = end - start + 1
            while remaining:
                chunk = stream.read(min(1024 * 1024, remaining))
                if not chunk: break
                remaining -= len(chunk); yield chunk
    return StreamingResponse(chunks(), status_code=206, media_type="video/mp4", headers={"Accept-Ranges":"bytes","Content-Range":f"bytes {start}-{end}/{size}","Content-Length":str(end-start+1)})


@app.get("/api/video/{video_id}/poster")
def video_poster(video_id: int, request: Request) -> FileResponse:
    """Serve the native final frame retained for a future continuation."""
    _require_authenticated_account(request)
    record = get_video_generation(video_id)
    if record is None or not record.poster_filename:
        raise HTTPException(status_code=404, detail="Video final frame not found.")
    path = VIDEOS_DIR / record.poster_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Video final frame is missing.")
    return FileResponse(path, media_type="image/png")


@app.post("/api/video/{video_id}/favorite", response_model=VideoResponse)
def favorite_video(video_id: int, body: VideoFavoriteRequest, request: Request) -> VideoResponse:
    _require_authenticated_account(request)
    if not set_video_favorite(video_id, body.favorite): raise HTTPException(status_code=404, detail="Video not found.")
    return _serialize_video(get_video_generation(video_id))


@app.delete("/api/video/{video_id}", status_code=204)
def delete_video(video_id: int, request: Request) -> Response:
    _require_authenticated_account(request)
    if delete_video_generation(video_id) is None: raise HTTPException(status_code=404, detail="Video not found.")
    return Response(status_code=204)


@app.get("/api/jobs", response_model=JobsPageResponse)
def jobs_history(
    request: Request,
    limit: int = Query(50, ge=1, le=100),
    before_id: int | None = Query(default=None, ge=1),
    status: str | None = Query(default=None),
    kind: list[str] | None = Query(default=None),
    mine: bool = Query(default=False),
) -> JobsPageResponse:
    """Return administrator-wide or account-scoped durable job history."""
    if status is not None and status not in {
        "queued", "running", "succeeded", "failed", "cancelled"
    }:
        raise HTTPException(status_code=422, detail="Unsupported job status filter.")
    allowed_kinds = {
        "generate", "surprise_generate", "batch_generate", "img2img", "character", "upscale", "pixel_upscale", "video_t2v", "video_i2v"
    }
    requested_kinds = tuple(dict.fromkeys(kind or ()))
    if any(value not in allowed_kinds for value in requested_kinds):
        raise HTTPException(status_code=422, detail="Unsupported job kind filter.")
    account = account_for_session(request.cookies.get(AUTH_COOKIE))
    if account is None:
        return JobsPageResponse(items=[])
    records = recent_jobs(
        limit + 1,
        before_id=before_id,
        status=status,
        account_id=account.id if mine or not account.is_admin else None,
        kinds=requested_kinds or None,
    )
    records = [
        _refresh_running_job(record) if record.status == "running" else record
        for record in records
    ]
    has_more = len(records) > limit
    page = records[:limit]
    return JobsPageResponse(
        items=[_serialize_job_list_item(record) for record in page],
        next_cursor=page[-1].id if has_more and page else None,
    )


@app.put("/api/generations/{generation_id}/rating", response_model=RatingResponse)
def update_generation_rating(
    generation_id: int, body: RatingRequest, request: Request
) -> RatingResponse:
    """Idempotently create or update quality feedback for a generation."""
    access_token = (
        body.access_token.get_secret_value() if body.access_token is not None else None
    )
    if not _can_access_generation(request, generation_id, access_token):
        raise HTTPException(status_code=403, detail="Generation access denied.")
    try:
        score, reasons = rate_generation(
            generation_id, body.score, tuple(body.reasons)
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return RatingResponse(
        generation_id=generation_id, score=score, reasons=list(reasons)
    )


# Registered last so protected /api/* image and data routes always take precedence.
# The launcher builds this directory before importing the ASGI app. The explicit
# history fallback keeps product routes bookmarkable without hash fragments.
FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"
if FRONTEND_DIST.exists():
    frontend_assets = FRONTEND_DIST / "assets"
    if frontend_assets.exists():
        app.mount(
            "/assets",
            StaticFiles(directory=frontend_assets),
            name="frontend-assets",
        )

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend_app(frontend_path: str) -> FileResponse:
        if frontend_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found.")
        candidate = (FRONTEND_DIST / frontend_path).resolve()
        try:
            candidate.relative_to(FRONTEND_DIST.resolve())
        except ValueError:
            candidate = FRONTEND_DIST / "index.html"
        if candidate.is_file():
            headers = {"Cache-Control": "no-store"} if candidate.name == "index.html" else None
            return FileResponse(candidate, headers=headers)
        return FileResponse(
            FRONTEND_DIST / "index.html",
            headers={"Cache-Control": "no-store"},
        )

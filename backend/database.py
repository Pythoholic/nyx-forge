"""SQLite persistence and static image materialization."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import sleep, time
from typing import Iterator, Sequence
from uuid import uuid4

from PIL import Image, ImageOps

# Pillow refuses images over ~89 MP as possible decompression bombs. Every
# image here was rendered by this application - a 12k upscale is 145 MP - so
# the guard only produced warnings and silently skipped the dimension check.
Image.MAX_IMAGE_PIXELS = 300_000_000

from .analytics import AnalyticsRun, AttemptCounts
from .forge import GenerationSettings, Img2ImgPayload, Txt2ImgPayload
from .versioning import PROMPT_COMPILER_VERSION

ROOT_DIR = Path(__file__).resolve().parents[1]
# Artifact storage can live off the repo drive: generated images, uploads and
# the database (which stores image bytes inline) all grow without bound.
# Resolution order: NYX_FORGE_DATA_DIR wins, then the path saved from Settings, then
# the repo directory, so an unconfigured install behaves exactly as before.
STORAGE_CONFIG_PATH = ROOT_DIR / "storage-config.json"
STORAGE_CONFIG_LOCK = Lock()


@dataclass(frozen=True, slots=True)
class BackendConfig:
    """One locally managed Forge installation."""

    id: str
    label: str
    port: int
    package_dir: str
    launch_args: tuple[str, ...] = ()

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# Keep the public defaults portable. Package folders are intentionally blank
# until an administrator selects their own installation in Forge backends.
# Neo keeps an empty package model folder under Stability Matrix, so its model
# arguments use a token that backend_manager expands relative to the selected
# package folder at launch time.
DEFAULT_BACKENDS = (
    BackendConfig(
        id="reforge",
        label="reForge",
        port=7860,
        package_dir="",
        launch_args=("--skip-install", "--cuda-malloc"),
    ),
    BackendConfig(
        id="forge_neo",
        label="Forge Neo",
        port=7861,
        package_dir="",
        launch_args=(
            "--skip-install",
            "--cuda-malloc",
            "--cuda-stream",
            "--ckpt-dirs", "{model_store}/StableDiffusion",
            "--vae-dirs", "{model_store}/VAE",
            "--text-encoder-dirs", "{model_store}/TextEncoders",
        ),
    ),
)


def _read_storage_config() -> dict[str, object]:
    try:
        raw = json.loads(STORAGE_CONFIG_PATH.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except (OSError, ValueError):
        return {}


def write_storage_config(updates: dict[str, object]) -> None:
    """Merge settings into storage-config.json without losing other sections."""
    with STORAGE_CONFIG_LOCK:
        raw = _read_storage_config()
        raw.update(updates)
        STORAGE_CONFIG_PATH.write_text(json.dumps(raw, indent=2), encoding="utf-8")


def configured_backends() -> tuple[BackendConfig, ...]:
    """Return the local backend registry, using defaults for older installs."""
    entries = _read_storage_config().get("backends")
    if not isinstance(entries, list):
        return DEFAULT_BACKENDS
    backends: list[BackendConfig] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        try:
            raw_launch_args = entry.get("launch_args", ())
            if not isinstance(raw_launch_args, (list, tuple)):
                raise TypeError("launch_args must be a list")
            launch_args = tuple(str(arg) for arg in raw_launch_args)
            forbidden_flags = ("--port", "--listen", "--host", "--server-name")
            if any(
                arg.casefold() == flag or arg.casefold().startswith(f"{flag}=")
                for arg in launch_args
                for flag in forbidden_flags
            ):
                raise ValueError("launch_args cannot override the local host or port")
            backend = BackendConfig(
                id=str(entry["id"]).strip(),
                label=str(entry["label"]).strip(),
                port=int(entry["port"]),
                package_dir=str(entry["package_dir"]).strip(),
                launch_args=launch_args,
            )
        except (KeyError, TypeError, ValueError):
            return DEFAULT_BACKENDS
        if not (backend.id and backend.label and 1 <= backend.port <= 65535):
            return DEFAULT_BACKENDS
        backends.append(backend)
    if not backends or len({backend.id for backend in backends}) != len(backends):
        return DEFAULT_BACKENDS
    return tuple(backends)


def backend_config(backend_id: str | None = None) -> BackendConfig:
    """Resolve a registry id; None selects the first/default backend."""
    backends = configured_backends()
    if backend_id is None:
        return backends[0]
    for backend in backends:
        if backend.id == backend_id:
            return backend
    raise KeyError(backend_id)


def configured_data_dir() -> Path | None:
    """Return the data directory saved from Settings, if one is valid."""
    try:
        raw = _read_storage_config()
        value = str(raw.get("data_dir") or "").strip()
        return Path(value).expanduser() if value else None
    except (OSError, ValueError, AttributeError):
        return None


DATA_DIR = (
    Path(os.environ["NYX_FORGE_DATA_DIR"]).expanduser()
    if os.environ.get("NYX_FORGE_DATA_DIR")
    else configured_data_dir() or ROOT_DIR
)
DATABASE_PATH = DATA_DIR / "prompts.db"
GENERATED_DIR = DATA_DIR / "generated"
UPLOADS_DIR = DATA_DIR / "uploads"
THUMBNAILS_DIR = GENERATED_DIR / "thumbnails"
RECOVERY_DIR = GENERATED_DIR / "recovery"
IDENTITY_PROFILES_DIR = GENERATED_DIR / "identity-profiles"
VIDEOS_DIR = DATA_DIR / "videos"
HISTORY_QUERY_LIMIT = 101
INTERRUPTED_RUN_GRACE_MINUTES = 0
SQLITE_WRITE_RETRIES = 3
SQLITE_RETRY_DELAYS = (0.2, 0.5)
LOGGER = logging.getLogger(__name__)
UPLOAD_RETENTION_SECONDS = 7 * 24 * 60 * 60
FEEDBACK_REASON_CODES = frozenset(
    {
        "prompt_match",
        "eyes",
        "anatomy",
        "composition",
        "style",
        "detail",
        "content_rating",
    }
)


@dataclass(frozen=True, slots=True)
class GenerationRecord:
    """Generation metadata returned to the web client."""

    id: int
    filename: str
    positive_prompt: str
    negative_prompt: str
    timestamp: str
    width: int
    height: int
    rating: str
    style: str
    high_res: bool
    super_res: bool
    model: str
    user_rating: int | None = None
    sampler_name: str = "DPM++ 2M SDE"
    scheduler: str = "Karras"
    steps: int = 28
    cfg_scale: float = 6.0
    seed: int = -1
    duration_seconds: float | None = None
    diffusion_duration_seconds: float | None = None
    upscale_duration_seconds: float | None = None
    source_generation_id: int | None = None
    job_id: int | None = None
    feedback_reasons: tuple[str, ...] = ()
    style_variant: str | None = None
    quality_mode: str = "normal"
    favorite: bool = False


@dataclass(frozen=True, slots=True)
class PromptHistoryRecord:
    """Local prompt history used to steer novelty without exposing raw history."""

    positive_prompt: str
    novelty_signature: str | None
    user_rating: int | None
    feedback_reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PendingGeneration:
    """A completed Forge image durably staged until SQLite can index it."""

    token: str
    manifest_path: Path
    image_path: Path
    thumbnail_path: Path
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class JobRecord:
    """One durable Forge-bound execution tracked independently of HTTP."""

    id: int
    kind: str
    status: str
    created_at: str
    started_at: str | None
    finished_at: str | None
    request_payload: dict[str, object]
    progress_percent: float
    progress_stage: str
    current_step: int | None
    total_steps: int | None
    eta_seconds: float | None
    current_image: str | None
    generation_id: int | None
    error_message: str | None
    account_id: int | None
    access_token_hash: str | None


@dataclass(frozen=True, slots=True)
class BatchRecord:
    id: int
    account_id: int
    request_id: str
    status: str
    requested_count: int
    planner_seed: int
    config: dict[str, object]
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None


@dataclass(frozen=True, slots=True)
class BatchItemRecord:
    id: int
    batch_id: int
    item_index: int
    request_id: str
    status: str
    plan: dict[str, object]
    job_id: int | None
    generation_id: int | None
    error: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True, slots=True)
class VideoGenerationRecord:
    id: int
    filename: str
    poster_filename: str | None
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
    comfy_prompt_id: str
    favorite: bool
    created_at: str


class JobQueueLimitError(ValueError):
    """An account already has its configured number of outstanding jobs."""

    def __init__(self, limit: int) -> None:
        self.limit = limit
        super().__init__(f"Queue limit reached ({limit} outstanding jobs).")


class DuplicateJobSubmission(ValueError):
    """The same authenticated request id already created a durable job."""

    def __init__(self, job_id: int) -> None:
        self.job_id = job_id
        super().__init__(f"Request already submitted as job {job_id}.")


@dataclass(frozen=True, slots=True)
class IdentityProfileRecord:
    """One account-owned, validated facial identity reference."""

    id: int
    account_id: int
    name: str
    filename: str
    thumbnail_filename: str
    detector: str
    embedding_model: str
    quality: dict[str, object]
    created_at: str
    updated_at: str
    last_used_at: str | None


@contextmanager
def _connection() -> Iterator[sqlite3.Connection]:
    """Yield a configured connection and always release its file handles."""
    connection = sqlite3.connect(DATABASE_PATH, timeout=5.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        with connection:
            yield connection
    finally:
        connection.close()


def ensure_diversity_tables(connection: sqlite3.Connection) -> None:
    """Create the additive diversity schema for lazy engine-only callers."""
    connection.executescript(
        """
        CREATE TABLE IF NOT EXISTS diversity_fingerprints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fingerprint TEXT NOT NULL,
            concept_ids TEXT NOT NULL,
            model TEXT NOT NULL,
            orientation TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS diversity_concept_usage (
            concept_id TEXT PRIMARY KEY,
            use_count INTEGER NOT NULL,
            last_sequence INTEGER NOT NULL
        );
        CREATE TABLE IF NOT EXISTS diversity_cooccurrence (
            concept_key TEXT NOT NULL,
            arity INTEGER NOT NULL CHECK(arity IN (2, 3)),
            use_count INTEGER NOT NULL,
            last_sequence INTEGER NOT NULL,
            PRIMARY KEY (concept_key, arity)
        );
        CREATE INDEX IF NOT EXISTS idx_diversity_fingerprints_id
            ON diversity_fingerprints(id DESC);
        CREATE INDEX IF NOT EXISTS idx_diversity_usage_sequence
            ON diversity_concept_usage(last_sequence);
        CREATE INDEX IF NOT EXISTS idx_diversity_cooccurrence_sequence
            ON diversity_cooccurrence(last_sequence);
        """
    )


def _thumbnail_bytes(image_bytes: bytes) -> bytes:
    """Encode a lightweight square JPEG without holding a database lock."""
    with Image.open(BytesIO(image_bytes)) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
        thumbnail = ImageOps.fit(
            image,
            (480, 480),
            method=Image.Resampling.LANCZOS,
            centering=(0.5, 0.5),
        )
        output = BytesIO()
        thumbnail.save(output, "JPEG", quality=82, optimize=True)
    return output.getvalue()


def _metadata_free_png_bytes(image_bytes: bytes) -> tuple[bytes, int, int]:
    """Losslessly re-encode a render without carrying PNG metadata forward."""
    with Image.open(BytesIO(image_bytes)) as source:
        source.load()
        actual_width, actual_height = source.size
        image = (
            source.convert("RGBA")
            if source.mode == "P" and "transparency" in source.info
            else source.copy()
        )
    image.info.clear()
    output = BytesIO()
    image.save(output, "PNG")
    return output.getvalue(), actual_width, actual_height


def write_thumbnail(image_bytes: bytes, generation_id: int) -> None:
    """Write a lightweight square JPEG preview for gallery rendering."""
    target = THUMBNAILS_DIR / f"thumb-{generation_id}.jpg"
    if not target.exists():
        target.write_bytes(_thumbnail_bytes(image_bytes))


def _generations_table_requires_rebuild(connection: sqlite3.Connection) -> bool:
    """Detect the legacy schema where image_bytes still forbids NULL."""
    row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='generations'"
    ).fetchone()
    if row is None or row[0] is None:
        return False
    return "image_bytes BLOB NOT NULL" in row[0]


def _rebuild_generations_table_nullable_image_bytes() -> None:
    """Relax image_bytes to nullable via SQLite's copy/rename table rebuild.

    ALTER TABLE cannot drop a column constraint in SQLite, and images now
    live in generated/*.png rather than this BLOB, so new rows insert NULL
    here. Older databases created before that change still enforce NOT NULL
    and must be rebuilt once to accept them.

    This runs on its own connection, with foreign_keys turned off before any
    statement opens a transaction. SQLite only honors that pragma outside an
    active transaction, and dropping a table that prompt_runs/generation_feedback/
    generation_access_tokens reference fires their ON DELETE actions (nulling
    or cascading those rows) the instant foreign key enforcement is on -
    sharing the caller's already-open connection would trigger that data loss.
    """
    connection = sqlite3.connect(DATABASE_PATH, timeout=5.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=OFF")
        if not _generations_table_requires_rebuild(connection):
            return
        with connection:
            columns = [
                row["name"]
                for row in connection.execute("PRAGMA table_info(generations)")
            ]
            column_list = ", ".join(columns)
            connection.execute(
                """
                CREATE TABLE generations_rebuild (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_bytes BLOB,
                    positive_prompt TEXT NOT NULL,
                    negative_prompt TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    width INTEGER NOT NULL,
                    height INTEGER NOT NULL,
                    rating TEXT NOT NULL,
                    style TEXT NOT NULL,
                    high_res INTEGER NOT NULL,
                    super_res INTEGER NOT NULL DEFAULT 0,
                    model TEXT NOT NULL,
                    filename TEXT
                )
                """
            )
            for column, definition in {
                "sampler_name": "TEXT NOT NULL DEFAULT 'DPM++ 2M SDE'",
                "scheduler": "TEXT NOT NULL DEFAULT 'Karras'",
                "steps": "INTEGER NOT NULL DEFAULT 28",
                "cfg_scale": "REAL NOT NULL DEFAULT 6.0",
                "seed": "INTEGER NOT NULL DEFAULT -1",
                "duration_seconds": "REAL",
                "diffusion_duration_seconds": "REAL",
                "upscale_duration_seconds": "REAL",
                "style_variant": "TEXT",
                "quality_mode": "TEXT NOT NULL DEFAULT 'normal'",
                "favorite": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if column in columns:
                    connection.execute(
                        f"ALTER TABLE generations_rebuild ADD COLUMN {column} {definition}"
                    )
            connection.execute(
                f"INSERT INTO generations_rebuild ({column_list}) "
                f"SELECT {column_list} FROM generations"
            )
            connection.execute("DROP TABLE generations")
            connection.execute(
                "ALTER TABLE generations_rebuild RENAME TO generations"
            )
        violations = connection.execute("PRAGMA foreign_key_check").fetchall()
        if violations:
            raise sqlite3.IntegrityError(
                f"generations rebuild left {len(violations)} dangling foreign key(s)."
            )
        LOGGER.info("Rebuilt generations table with image_bytes made nullable.")
    finally:
        connection.close()


def initialize_database() -> None:
    """Create/migrate tables and export legacy BLOBs to static PNG files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if os.name == "nt":
        try:
            import ctypes

            ctypes.windll.kernel32.SetFileAttributesW(str(DATA_DIR), 2)
        except (AttributeError, OSError):
            LOGGER.warning("Could not mark the data directory as hidden.")
    GENERATED_DIR.mkdir(parents=True, exist_ok=True)
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    THUMBNAILS_DIR.mkdir(parents=True, exist_ok=True)
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    IDENTITY_PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    if DATABASE_PATH.exists():
        _rebuild_generations_table_nullable_image_bytes()
    with _connection() as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prompts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                positive_prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL,
                timestamp TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS prompt_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                positive_prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                source TEXT NOT NULL,
                engine TEXT NOT NULL,
                model TEXT NOT NULL,
                style TEXT NOT NULL,
                aspect_ratio TEXT NOT NULL,
                content_rating TEXT NOT NULL,
                high_res INTEGER NOT NULL DEFAULT 0,
                super_res INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'created',
                generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL,
                error_message TEXT,
                prompt_duration_seconds REAL,
                compiler_version TEXT NOT NULL DEFAULT 'legacy'
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_feedback (
                generation_id INTEGER PRIMARY KEY REFERENCES generations(id) ON DELETE CASCADE,
                score INTEGER NOT NULL CHECK(score BETWEEN 1 AND 5),
                reason_codes TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
            """
        )
        feedback_columns = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(generation_feedback)")
        }
        if "reason_codes" not in feedback_columns:
            connection.execute(
                "ALTER TABLE generation_feedback ADD COLUMN reason_codes TEXT NOT NULL DEFAULT '[]'"
            )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_runs_generation ON prompt_runs(generation_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_runs_created ON prompt_runs(created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_prompt_runs_status_created ON prompt_runs(status, created_at)"
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_prompt_runs_novelty_context
            ON prompt_runs(model, style, content_rating, id DESC)
            """
        )
        # Prompt-engine diversity memory is intentionally separate from image
        # history: prompt compilation happens before a generation row exists.
        # Sequence IDs provide a clock which is deterministic in tests and does
        # not depend on wall-clock precision.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diversity_fingerprints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL,
                concept_ids TEXT NOT NULL,
                model TEXT NOT NULL,
                orientation TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diversity_concept_usage (
                concept_id TEXT PRIMARY KEY,
                use_count INTEGER NOT NULL,
                last_sequence INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS diversity_cooccurrence (
                concept_key TEXT NOT NULL,
                arity INTEGER NOT NULL CHECK(arity IN (2, 3)),
                use_count INTEGER NOT NULL,
                last_sequence INTEGER NOT NULL,
                PRIMARY KEY (concept_key, arity)
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_diversity_fingerprints_id "
            "ON diversity_fingerprints(id DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_diversity_usage_sequence "
            "ON diversity_concept_usage(last_sequence)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_diversity_cooccurrence_sequence "
            "ON diversity_cooccurrence(last_sequence)"
        )
        prompt_run_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(prompt_runs)")
        }
        prompt_run_migrations = {
            "creativity_level": "TEXT NOT NULL DEFAULT 'balanced'",
            "novelty_signature": "TEXT",
            "similarity_score": "REAL",
            "novelty_retry_count": "INTEGER NOT NULL DEFAULT 0",
            "prompt_duration_seconds": "REAL",
            "compiler_version": "TEXT NOT NULL DEFAULT 'legacy'",
        }
        for column, definition in prompt_run_migrations.items():
            if column not in prompt_run_columns:
                connection.execute(
                    f"ALTER TABLE prompt_runs ADD COLUMN {column} {definition}"
                )
        _recover_interrupted_prompt_runs(
            connection, grace_minutes=INTERRUPTED_RUN_GRACE_MINUTES
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_bytes BLOB,
                positive_prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                rating TEXT NOT NULL,
                style TEXT NOT NULL,
                high_res INTEGER NOT NULL,
                super_res INTEGER NOT NULL DEFAULT 0,
                model TEXT NOT NULL,
                filename TEXT
            )
            """
        )
        columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(generations)")
        }
        if "filename" not in columns:
            connection.execute("ALTER TABLE generations ADD COLUMN filename TEXT")
        if "super_res" not in columns:
            connection.execute(
                "ALTER TABLE generations ADD COLUMN super_res INTEGER NOT NULL DEFAULT 0"
            )
        generation_migrations = {
            "sampler_name": "TEXT NOT NULL DEFAULT 'DPM++ 2M SDE'",
            "scheduler": "TEXT NOT NULL DEFAULT 'Karras'",
            "steps": "INTEGER NOT NULL DEFAULT 28",
            "cfg_scale": "REAL NOT NULL DEFAULT 6.0",
            "seed": "INTEGER NOT NULL DEFAULT -1",
            "duration_seconds": "REAL",
            "diffusion_duration_seconds": "REAL",
            "upscale_duration_seconds": "REAL",
            "source_generation_id": "INTEGER REFERENCES generations(id) ON DELETE SET NULL",
            "style_variant": "TEXT",
            "quality_mode": "TEXT NOT NULL DEFAULT 'normal'",
            "favorite": "INTEGER NOT NULL DEFAULT 0",
        }
        for column, definition in generation_migrations.items():
            if column not in columns:
                connection.execute(
                    f"ALTER TABLE generations ADD COLUMN {column} {definition}"
                )
        connection.execute(
            """
            UPDATE generations
            SET quality_mode = CASE
                WHEN super_res = 1 THEN 'super'
                WHEN high_res = 1 THEN 'high'
                ELSE 'normal'
            END
            WHERE quality_mode IS NULL
               OR quality_mode = ''
               OR (quality_mode = 'normal' AND (high_res = 1 OR super_res = 1))
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                kind TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'queued',
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                started_at TEXT,
                finished_at TEXT,
                request_payload TEXT NOT NULL,
                progress_percent REAL NOT NULL DEFAULT 0,
                progress_stage TEXT NOT NULL DEFAULT 'queued',
                current_step INTEGER,
                total_steps INTEGER,
                eta_seconds REAL,
                current_image TEXT,
                generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL,
                error_message TEXT,
                account_id INTEGER,
                access_token_hash TEXT
            )
            """
        )
        job_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(jobs)")
        }
        job_migrations = {
            "started_at": "TEXT",
            "finished_at": "TEXT",
            "progress_percent": "REAL NOT NULL DEFAULT 0",
            "progress_stage": "TEXT NOT NULL DEFAULT 'queued'",
            "current_step": "INTEGER",
            "total_steps": "INTEGER",
            "eta_seconds": "REAL",
            "current_image": "TEXT",
            "generation_id": "INTEGER REFERENCES generations(id) ON DELETE SET NULL",
            "error_message": "TEXT",
            "account_id": "INTEGER",
            "access_token_hash": "TEXT",
        }
        for column, definition in job_migrations.items():
            if column not in job_columns:
                connection.execute(f"ALTER TABLE jobs ADD COLUMN {column} {definition}")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_jobs_generation ON jobs(generation_id)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                request_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                requested_count INTEGER NOT NULL,
                planner_seed INTEGER NOT NULL,
                config_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                started_at TEXT,
                finished_at TEXT
            )
            """
        )
        batch_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(batches)")
        }
        batch_migrations = {
            "updated_at": "TEXT NOT NULL DEFAULT ''",
            "started_at": "TEXT",
            "finished_at": "TEXT",
        }
        for column, definition in batch_migrations.items():
            if column not in batch_columns:
                connection.execute(f"ALTER TABLE batches ADD COLUMN {column} {definition}")
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS batch_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                item_index INTEGER NOT NULL,
                request_id TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL DEFAULT 'pending',
                plan_json TEXT NOT NULL,
                job_id INTEGER REFERENCES jobs(id) ON DELETE SET NULL,
                generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL,
                error TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                UNIQUE(batch_id, item_index)
            )
            """
        )
        item_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(batch_items)")
        }
        item_migrations = {
            "job_id": "INTEGER REFERENCES jobs(id) ON DELETE SET NULL",
            "generation_id": "INTEGER REFERENCES generations(id) ON DELETE SET NULL",
            "error": "TEXT",
            "updated_at": "TEXT NOT NULL DEFAULT ''",
        }
        for column, definition in item_migrations.items():
            if column not in item_columns:
                connection.execute(f"ALTER TABLE batch_items ADD COLUMN {column} {definition}")
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_batches_account_created "
            "ON batches(account_id, created_at DESC)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_batches_status ON batches(status)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_items_batch_status "
            "ON batch_items(batch_id, status, item_index)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_items_job ON batch_items(job_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_batch_items_generation ON batch_items(generation_id)"
        )
        # ponytail: poster_filename has no ALTER TABLE migration, unlike the
        # other added columns here. Fine while no deployed database predates it
        # (IF NOT EXISTS creates the table complete); add a migration before
        # shipping to any install that already has video_generations.
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS video_generations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                poster_filename TEXT,
                prompt TEXT NOT NULL,
                negative_prompt TEXT NOT NULL,
                mode TEXT NOT NULL CHECK(mode IN ('t2v','i2v')),
                source_generation_id INTEGER REFERENCES generations(id) ON DELETE SET NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                frame_count INTEGER NOT NULL,
                fps INTEGER NOT NULL,
                seed INTEGER NOT NULL,
                duration_seconds REAL NOT NULL,
                camera_motion TEXT,
                comfy_prompt_id TEXT NOT NULL,
                favorite INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime'))
            )
            """
        )
        connection.execute("CREATE INDEX IF NOT EXISTS idx_video_created ON video_generations(id DESC)")
        VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS identity_profiles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                account_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name TEXT NOT NULL,
                filename TEXT NOT NULL UNIQUE,
                thumbnail_filename TEXT NOT NULL UNIQUE,
                embedding BLOB NOT NULL,
                detector TEXT NOT NULL,
                embedding_model TEXT NOT NULL,
                quality_json TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now','localtime')),
                last_used_at TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_identity_profiles_account ON identity_profiles(account_id, id DESC)"
        )

        # Release the schema write lock before any potentially expensive image
        # decoding or thumbnail work. A second startup can then never starve a
        # generation that is finishing in the active server.
        connection.commit()
        rows = connection.execute(
            "SELECT id, image_bytes, filename, width, height FROM generations"
        ).fetchall()
        for row in rows:
            filename = row["filename"] or f"generation-{row['id']}.png"
            path = GENERATED_DIR / filename
            if not path.exists() and row["image_bytes"]:
                path.write_bytes(bytes(row["image_bytes"]))
            if row["image_bytes"]:
                write_thumbnail(bytes(row["image_bytes"]), int(row["id"]))
            if row["filename"] != filename:
                connection.execute(
                    "UPDATE generations SET filename = ? WHERE id = ?",
                    (filename, row["id"]),
                )
                connection.commit()
            if path.exists():
                try:
                    with Image.open(path) as source:
                        actual_width, actual_height = source.size
                    if (actual_width, actual_height) != (row["width"], row["height"]):
                        connection.execute(
                            "UPDATE generations SET width = ?, height = ? WHERE id = ?",
                            (actual_width, actual_height, row["id"]),
                        )
                        connection.commit()
                except OSError:
                    pass
    migrated_generations = migrate_legacy_generation_filenames()
    if migrated_generations:
        LOGGER.info(
            "Migrated %s generation file(s) to opaque names.", migrated_generations
        )
    recover_pending_generations()
    # Generated artifacts are durable user data. Startup may happen while
    # another app process is committing a generation, and a read taken in
    # that window must never be allowed to classify and delete its files.
    # Keep the audit for diagnostics, but leave cleanup to an explicit,
    # user-authorized maintenance action.
    sweep_orphan_generation_files()
    removed_uploads = sweep_stale_uploads()
    if removed_uploads:
        LOGGER.info("Removed %s expired ForgeIMG upload(s).", removed_uploads)
    recover_stalled_jobs()
    recover_nonterminal_batches()


def _copy_for_crash_safe_rename(source: Path, destination: Path) -> None:
    """Create the new name while retaining the old name until SQLite commits."""
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _remove_empty_legacy_generation_directories() -> None:
    """Remove the retired hidden vault directories, but never their contents."""
    legacy_directory = GENERATED_DIR / ".nyx-private"
    for directory in (legacy_directory / "thumbnails", legacy_directory):
        try:
            directory.rmdir()
        except FileNotFoundError:
            pass
        except OSError:
            # A non-empty legacy directory is intentionally left untouched.
            pass


def migrate_legacy_generation_filenames() -> int:
    """Replace legacy extensioned image names with one opaque name per row.

    Keeping the old name until after the row commits makes every crash state
    recoverable: before commit the old name is authoritative; after commit the
    new name is authoritative and any old name is an orphan for startup sweep.
    """
    with _connection() as connection:
        rows = connection.execute(
            "SELECT id, filename FROM generations WHERE filename IS NOT NULL"
        ).fetchall()

    migrated = 0
    for row in rows:
        generation_id = int(row["id"])
        old_filename = str(row["filename"])
        if not old_filename.endswith(".png"):
            continue
        old_image = GENERATED_DIR / old_filename
        if not old_image.is_file():
            LOGGER.warning(
                "Generation %s still uses legacy filename %s, but its image is missing.",
                generation_id,
                old_filename,
            )
            continue

        while True:
            filename = uuid4().hex
            image_destination = GENERATED_DIR / filename
            thumbnail_destination = THUMBNAILS_DIR / filename
            if not image_destination.exists() and not thumbnail_destination.exists():
                break

        old_thumbnail = (
            old_image.parent / "thumbnails" / f"thumb-{generation_id}.jpg"
        )
        flat_thumbnail = THUMBNAILS_DIR / f"thumb-{generation_id}.jpg"
        if not old_thumbnail.is_file() and flat_thumbnail.is_file():
            old_thumbnail = flat_thumbnail
        created: list[Path] = []
        try:
            _copy_for_crash_safe_rename(old_image, image_destination)
            created.append(image_destination)
            if old_thumbnail.is_file():
                _copy_for_crash_safe_rename(old_thumbnail, thumbnail_destination)
                created.append(thumbnail_destination)
            else:
                LOGGER.warning(
                    "Generation %s has no legacy thumbnail to migrate.", generation_id
                )
            with _connection() as connection:
                updated = connection.execute(
                    "UPDATE generations SET filename = ? WHERE id = ? AND filename = ?",
                    (filename, generation_id, old_filename),
                )
                if updated.rowcount != 1:
                    raise sqlite3.IntegrityError(
                        f"Generation {generation_id} changed during filename migration."
                    )
        except Exception:
            current_filename = generation_filename(generation_id)
            if current_filename != filename:
                for path in reversed(created):
                    try:
                        path.unlink(missing_ok=True)
                    except OSError:
                        LOGGER.warning("Could not clean up migration target %s", path)
            LOGGER.exception(
                "Could not migrate generation %s from %s.",
                generation_id,
                old_filename,
            )
            continue

        for path in (old_thumbnail, old_image):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Could not remove migrated legacy file %s", path)
        migrated += 1
    _remove_empty_legacy_generation_directories()
    return migrated


def sweep_orphan_generation_files() -> int:
    """Audit unindexed generation artifacts without deleting user data.

    This used to unlink files automatically during startup. Besides making a
    restart destructive, that creates a race with another process finishing a
    generation: its files can become visible just before its SQLite row. An
    orphan is therefore only a diagnostic observation, never proof that an
    image is disposable.
    """
    with _connection() as connection:
        filenames = [
            str(row["filename"])
            for row in connection.execute(
                "SELECT filename FROM generations WHERE filename IS NOT NULL"
            )
        ]

    protected_paths = {
        (root / filename).resolve()
        for filename in filenames
        for root in (GENERATED_DIR, THUMBNAILS_DIR)
    }

    protected_directories = {
        ".nyx-private",
        "recovery",
        "quality-jobs",
        "character-jobs",
        "identity-profiles",
        "thumbnails",
    }
    orphan_count = 0
    for path in GENERATED_DIR.iterdir():
        if path.name in protected_directories or not path.is_file():
            continue
        if path.resolve() not in protected_paths:
            orphan_count += 1
    for path in THUMBNAILS_DIR.iterdir():
        if not path.is_file() or path.resolve() in protected_paths:
            continue
        orphan_count += 1
    if orphan_count:
        LOGGER.warning(
            "Found %s unindexed generation artifact(s); preserving them because "
            "automatic cleanup of gallery data is disabled.",
            orphan_count,
        )
    return 0


def sweep_stale_uploads(*, now: float | None = None) -> int:
    """Remove re-encoded upload inputs older than the seven-day retention window."""
    if not UPLOADS_DIR.exists():
        return 0
    cutoff = (time() if now is None else now) - UPLOAD_RETENTION_SECONDS
    removed = 0
    for path in UPLOADS_DIR.glob("*.png"):
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
                removed += 1
        except OSError:
            LOGGER.warning("Could not inspect or remove stale upload %s", path)
    return removed


def _prepare_pending_generation(
    image_bytes: bytes,
    payload: Txt2ImgPayload | Img2ImgPayload,
    settings: GenerationSettings,
    *,
    actual_seed: int | None,
    duration_seconds: float | None,
    diffusion_duration_seconds: float | None,
    upscale_duration_seconds: float | None,
    prompt_run_id: int | None,
    source_generation_id: int | None = None,
) -> PendingGeneration:
    """Durably stage a completed render before attempting a database write."""
    sanitized_image_bytes, actual_width, actual_height = _metadata_free_png_bytes(
        image_bytes
    )
    thumbnail_bytes = _thumbnail_bytes(sanitized_image_bytes)
    RECOVERY_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    manifest_path = RECOVERY_DIR / f"{token}.json"
    temporary_manifest = RECOVERY_DIR / f"{token}.json.tmp"
    image_path = RECOVERY_DIR / f"{token}.png"
    thumbnail_path = RECOVERY_DIR / f"{token}.jpg"
    payload_metadata = asdict(payload)
    if isinstance(payload, Img2ImgPayload):
        # Recovery only needs the common generation metadata. Never persist
        # the source image's base64 data a second time in the manifest.
        payload_metadata = {
            key: value
            for key, value in payload_metadata.items()
            if key
            not in {
                "init_images",
                "denoising_strength",
                "mask",
                "mask_blur",
                "inpaint_full_res",
                "inpaint_full_res_padding",
                "invert_mask",
                "soft_inpainting",
                "resize_mode",
                "face_reference",
            }
        }
    metadata: dict[str, object] = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "actual_width": actual_width,
        "actual_height": actual_height,
        "actual_seed": payload.seed if actual_seed is None else actual_seed,
        "duration_seconds": duration_seconds,
        "diffusion_duration_seconds": diffusion_duration_seconds,
        "upscale_duration_seconds": upscale_duration_seconds,
        "prompt_run_id": prompt_run_id,
        "source_generation_id": source_generation_id,
        "payload": payload_metadata,
        "settings": asdict(settings),
    }
    try:
        image_path.write_bytes(sanitized_image_bytes)
        thumbnail_path.write_bytes(thumbnail_bytes)
        temporary_manifest.write_text(
            json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary_manifest.replace(manifest_path)
    except Exception:
        for path in (temporary_manifest, manifest_path, thumbnail_path, image_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    return PendingGeneration(
        token, manifest_path, image_path, thumbnail_path, metadata
    )


def _pending_from_manifest(manifest_path: Path) -> PendingGeneration:
    """Load and minimally validate a durable recovery manifest."""
    metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("Recovery manifest must contain an object.")
    token = manifest_path.stem
    pending = PendingGeneration(
        token,
        manifest_path,
        RECOVERY_DIR / f"{token}.png",
        RECOVERY_DIR / f"{token}.jpg",
        metadata,
    )
    if not pending.image_path.is_file() or not pending.thumbnail_path.is_file():
        raise FileNotFoundError("Recovery image or thumbnail is missing.")
    return pending


def _restore_pending_file(source: Path, destination: Path) -> None:
    """Best-effort restore of a staged file after transaction rollback."""
    if destination.exists() and not source.exists():
        try:
            destination.replace(source)
        except OSError:
            LOGGER.exception("Could not restore staged generation file %s", source)


def _commit_pending_generation(pending: PendingGeneration) -> GenerationRecord:
    """Index a staged render using a deliberately short write transaction."""
    metadata = pending.metadata
    payload_values = metadata.get("payload")
    settings_values = metadata.get("settings")
    if not isinstance(payload_values, dict) or not isinstance(settings_values, dict):
        raise ValueError("Recovery manifest is missing payload settings.")
    payload = Txt2ImgPayload(**payload_values)
    settings = GenerationSettings(**settings_values)
    timestamp = str(metadata["timestamp"])
    actual_width = int(metadata["actual_width"])
    actual_height = int(metadata["actual_height"])
    actual_seed = int(metadata["actual_seed"])
    prompt_run_value = metadata.get("prompt_run_id")
    prompt_run_id = int(prompt_run_value) if prompt_run_value is not None else None
    duration_seconds = metadata.get("duration_seconds")
    diffusion_duration_seconds = metadata.get("diffusion_duration_seconds")
    upscale_duration_seconds = metadata.get("upscale_duration_seconds")
    source_generation_value = metadata.get("source_generation_id")
    source_generation_id = (
        int(source_generation_value) if source_generation_value is not None else None
    )
    image_destination: Path | None = None
    thumbnail_destination: Path | None = None
    try:
        with _connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO generations (
                    image_bytes, positive_prompt, negative_prompt, timestamp,
                    width, height, rating, style, high_res, super_res, model,
                    sampler_name, scheduler, steps, cfg_scale, seed, duration_seconds,
                    diffusion_duration_seconds, upscale_duration_seconds,
                    source_generation_id, style_variant, quality_mode, filename
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL)
                """,
                (
                    None, payload.prompt, payload.negative_prompt, timestamp,
                    actual_width, actual_height, settings.rating, settings.style,
                    int(settings.high_res), int(settings.super_res), settings.model,
                    payload.sampler_name, payload.scheduler, payload.steps,
                    payload.cfg_scale, actual_seed, duration_seconds,
                    diffusion_duration_seconds, upscale_duration_seconds,
                    source_generation_id, payload.style_variant, settings.quality_mode,
                ),
            )
            generation_id = int(cursor.lastrowid)
            filename = uuid4().hex
            image_destination = GENERATED_DIR / filename
            thumbnail_destination = THUMBNAILS_DIR / filename
            if image_destination.exists() or thumbnail_destination.exists():
                raise FileExistsError(
                    f"Static files already exist for generation {generation_id}."
                )
            pending.image_path.replace(image_destination)
            pending.thumbnail_path.replace(thumbnail_destination)
            connection.execute(
                "UPDATE generations SET filename = ? WHERE id = ?",
                (filename, generation_id),
            )
            if prompt_run_id is not None:
                connection.execute(
                    """
                    UPDATE prompt_runs
                    SET status = 'succeeded', generation_id = ?, error_message = NULL
                    WHERE id = ?
                    """,
                    (generation_id, prompt_run_id),
                )
    except Exception:
        if thumbnail_destination is not None:
            _restore_pending_file(pending.thumbnail_path, thumbnail_destination)
        if image_destination is not None:
            _restore_pending_file(pending.image_path, image_destination)
        raise
    try:
        pending.manifest_path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove recovered manifest %s", pending.manifest_path)
    return GenerationRecord(
        generation_id, filename, payload.prompt, payload.negative_prompt,
        timestamp, actual_width, actual_height, settings.rating, settings.style,
        settings.high_res, settings.super_res, settings.model, None,
        payload.sampler_name, payload.scheduler, payload.steps, payload.cfg_scale,
        actual_seed,
        float(duration_seconds) if duration_seconds is not None else None,
        float(diffusion_duration_seconds) if diffusion_duration_seconds is not None else None,
        float(upscale_duration_seconds) if upscale_duration_seconds is not None else None,
        source_generation_id,
        style_variant=payload.style_variant,
        quality_mode=settings.quality_mode,
    )


def _is_busy_error(error: sqlite3.OperationalError) -> bool:
    message = str(error).lower()
    return "locked" in message or "busy" in message


def _commit_with_retry(pending: PendingGeneration) -> GenerationRecord:
    """Retry only transient SQLite contention, never arbitrary failures."""
    for attempt in range(SQLITE_WRITE_RETRIES):
        try:
            return _commit_pending_generation(pending)
        except sqlite3.OperationalError as exc:
            if not _is_busy_error(exc) or attempt + 1 >= SQLITE_WRITE_RETRIES:
                raise
            sleep(SQLITE_RETRY_DELAYS[min(attempt, len(SQLITE_RETRY_DELAYS) - 1)])
    raise RuntimeError("SQLite retry loop exited unexpectedly.")


def recover_pending_generations() -> int:
    """Re-index completed images retained after a transient persistence failure."""
    if not RECOVERY_DIR.exists():
        return 0
    recovered = 0
    for manifest_path in sorted(RECOVERY_DIR.glob("*.json")):
        try:
            pending = _pending_from_manifest(manifest_path)
            _commit_with_retry(pending)
            recovered += 1
        except sqlite3.OperationalError as exc:
            if _is_busy_error(exc):
                LOGGER.warning("SQLite remains busy; pending generations stay recoverable.")
                break
            LOGGER.exception("Could not recover pending generation %s", manifest_path)
        except (OSError, TypeError, ValueError, KeyError):
            LOGGER.exception("Invalid pending generation manifest %s", manifest_path)
            try:
                manifest_path.replace(manifest_path.with_suffix(".invalid"))
            except OSError:
                pass
    return recovered


def save_generation(
    image_bytes: bytes,
    payload: Txt2ImgPayload | Img2ImgPayload,
    settings: GenerationSettings,
    *,
    actual_seed: int | None = None,
    duration_seconds: float | None = None,
    diffusion_duration_seconds: float | None = None,
    upscale_duration_seconds: float | None = None,
    prompt_run_id: int | None = None,
    source_generation_id: int | None = None,
) -> GenerationRecord:
    """Stage, retry, and atomically index a completed Forge generation."""
    pending = _prepare_pending_generation(
        image_bytes,
        payload,
        settings,
        actual_seed=actual_seed,
        duration_seconds=duration_seconds,
        diffusion_duration_seconds=diffusion_duration_seconds,
        upscale_duration_seconds=upscale_duration_seconds,
        prompt_run_id=prompt_run_id,
        source_generation_id=source_generation_id,
    )
    return _commit_with_retry(pending)


def _job_from_row(row: sqlite3.Row) -> JobRecord:
    """Convert one SQLite row into a typed job record."""
    payload = json.loads(str(row["request_payload"]))
    if not isinstance(payload, dict):
        raise ValueError("Job request payload must be a JSON object.")
    return JobRecord(
        id=int(row["id"]),
        kind=str(row["kind"]),
        status=str(row["status"]),
        created_at=str(row["created_at"]),
        started_at=str(row["started_at"]) if row["started_at"] is not None else None,
        finished_at=str(row["finished_at"]) if row["finished_at"] is not None else None,
        request_payload=payload,
        progress_percent=float(row["progress_percent"]),
        progress_stage=str(row["progress_stage"]),
        current_step=int(row["current_step"]) if row["current_step"] is not None else None,
        total_steps=int(row["total_steps"]) if row["total_steps"] is not None else None,
        eta_seconds=float(row["eta_seconds"]) if row["eta_seconds"] is not None else None,
        current_image=str(row["current_image"]) if row["current_image"] else None,
        generation_id=int(row["generation_id"]) if row["generation_id"] is not None else None,
        error_message=str(row["error_message"]) if row["error_message"] else None,
        account_id=int(row["account_id"]) if row["account_id"] is not None else None,
        access_token_hash=str(row["access_token_hash"]) if row["access_token_hash"] else None,
    )


def create_job(
    kind: str,
    request_payload: dict[str, object],
    account_id: int | None,
    access_token_hash: str | None,
    *,
    max_outstanding: int | None = None,
) -> int:
    """Persist a queued job before handing it to the in-process worker."""
    if kind not in {
        "generate", "surprise_generate", "batch_generate", "img2img", "character", "upscale", "pixel_upscale", "video_t2v", "video_i2v"
    }:
        raise ValueError(f"Unsupported job kind: {kind}")
    with _connection() as connection:
        if account_id is not None and max_outstanding is not None:
            # Reserve the slot and insert under one write transaction so two
            # simultaneous browser requests cannot both pass the same count.
            connection.execute("BEGIN IMMEDIATE")
            request_id = str(request_payload.get("request_id") or "").strip()
            if request_id and kind != "batch_generate":
                duplicate = connection.execute(
                    """
                    SELECT id FROM jobs
                    WHERE account_id = ? AND kind = ?
                      AND json_extract(request_payload, '$.request_id') = ?
                    ORDER BY id DESC LIMIT 1
                    """,
                    (account_id, kind, request_id),
                ).fetchone()
                if duplicate is not None:
                    raise DuplicateJobSubmission(int(duplicate["id"]))
            row = connection.execute(
                """
                SELECT COUNT(*) AS count FROM jobs
                WHERE account_id = ? AND status IN ('queued', 'running')
                """,
                (account_id,),
            ).fetchone()
            if int(row["count"] if row is not None else 0) >= max_outstanding:
                raise JobQueueLimitError(max_outstanding)
        cursor = connection.execute(
            """
            INSERT INTO jobs (kind, request_payload, account_id, access_token_hash, created_at)
            VALUES (?, ?, ?, ?, datetime('now','localtime'))
            """,
            (
                kind,
                json.dumps(request_payload, ensure_ascii=False, separators=(",", ":")),
                account_id,
                access_token_hash,
            ),
        )
        return int(cursor.lastrowid)


def update_job_progress(
    job_id: int,
    *,
    status: str | None = None,
    percent: float | None = None,
    stage: str | None = None,
    current_step: int | None = None,
    total_steps: int | None = None,
    eta_seconds: float | None = None,
    current_image: str | None = None,
) -> None:
    """Apply only the supplied progress fields to one durable job."""
    assignments: list[str] = []
    values: list[object] = []
    if status is not None:
        if status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
            raise ValueError(f"Unsupported job status: {status}")
        assignments.append("status = ?")
        values.append(status)
        if status == "running":
            assignments.append("started_at = COALESCE(started_at, datetime('now','localtime'))")
    if percent is not None:
        assignments.append("progress_percent = ?")
        values.append(max(0.0, min(100.0, float(percent))))
    if stage is not None:
        assignments.append("progress_stage = ?")
        values.append(stage)
    if current_step is not None:
        assignments.append("current_step = ?")
        values.append(max(0, int(current_step)))
    if total_steps is not None:
        assignments.append("total_steps = ?")
        values.append(max(0, int(total_steps)))
    if eta_seconds is not None:
        assignments.append("eta_seconds = ?")
        values.append(max(0.0, float(eta_seconds)))
    if current_image is not None:
        assignments.append("current_image = ?")
        values.append(current_image)
    if not assignments:
        return
    values.append(job_id)
    with _connection() as connection:
        connection.execute(
            f"UPDATE jobs SET {', '.join(assignments)} WHERE id = ?",
            values,
        )


def update_job_request_payload(job_id: int, **updates: object) -> None:
    """Merge durable execution metadata into a job's stored request payload."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT request_payload FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
        if row is None:
            raise ValueError("Job not found.")
        payload = json.loads(str(row["request_payload"]))
        if not isinstance(payload, dict):
            raise ValueError("Job request payload must be a JSON object.")
        payload.update(updates)
        connection.execute(
            "UPDATE jobs SET request_payload = ? WHERE id = ?",
            (json.dumps(payload, ensure_ascii=False, separators=(",", ":")), job_id),
        )


def finish_job(
    job_id: int,
    *,
    status: str,
    generation_id: int | None = None,
    error_message: str | None = None,
) -> None:
    """Set one job's terminal state and optional generated artifact."""
    if status not in {"succeeded", "failed", "cancelled"}:
        raise ValueError("finish_job requires a terminal status.")
    with _connection() as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = ?, finished_at = datetime('now','localtime'),
                progress_percent = CASE WHEN ? = 'succeeded' THEN 100 ELSE progress_percent END,
                progress_stage = CASE WHEN ? = 'succeeded' THEN 'completed' ELSE progress_stage END,
                generation_id = ?, error_message = ?, current_image = NULL
            WHERE id = ?
            """,
            (
                status,
                status,
                status,
                generation_id,
                error_message,
                job_id,
            ),
        )


def get_job(job_id: int) -> JobRecord | None:
    """Return one durable job, or None when it does not exist."""
    with _connection() as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return _job_from_row(row) if row is not None else None


def _batch_from_row(row: sqlite3.Row) -> BatchRecord:
    config = json.loads(str(row["config_json"]))
    if not isinstance(config, dict):
        raise ValueError("Batch config must be a JSON object.")
    return BatchRecord(
        id=int(row["id"]),
        account_id=int(row["account_id"]),
        request_id=str(row["request_id"]),
        status=str(row["status"]),
        requested_count=int(row["requested_count"]),
        planner_seed=int(row["planner_seed"]),
        config=config,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        started_at=str(row["started_at"]) if row["started_at"] is not None else None,
        finished_at=str(row["finished_at"]) if row["finished_at"] is not None else None,
    )


def _batch_item_from_row(row: sqlite3.Row) -> BatchItemRecord:
    plan = json.loads(str(row["plan_json"]))
    if not isinstance(plan, dict):
        raise ValueError("Batch item plan must be a JSON object.")
    return BatchItemRecord(
        id=int(row["id"]),
        batch_id=int(row["batch_id"]),
        item_index=int(row["item_index"]),
        request_id=str(row["request_id"]),
        status=str(row["status"]),
        plan=plan,
        job_id=int(row["job_id"]) if row["job_id"] is not None else None,
        generation_id=int(row["generation_id"]) if row["generation_id"] is not None else None,
        error=str(row["error"]) if row["error"] else None,
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def create_batch_with_items(
    *, account_id: int, request_id: str, requested_count: int,
    planner_seed: int, config: dict[str, object],
    plan_items: Sequence[dict[str, object]],
) -> BatchRecord:
    """Atomically persist a parent and its exact immutable child plan."""
    if requested_count < 1 or len(plan_items) != requested_count:
        raise ValueError("A batch must persist exactly its requested item count.")
    expected_indexes = list(range(requested_count))
    indexes = [int(item.get("index", -1)) for item in plan_items]
    if indexes != expected_indexes:
        raise ValueError("Batch item indexes must be contiguous and ordered from zero.")
    if len({str(item.get("request_id", "")) for item in plan_items}) != requested_count:
        raise ValueError("Batch item request ids must be unique.")
    config_json = json.dumps(config, ensure_ascii=False, separators=(",", ":"))
    if any(secret in config_json.casefold() for secret in ("cloud_api_key", "api_key")):
        raise ValueError("Batch config cannot contain cloud credentials.")

    with _connection() as connection:
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT * FROM batches WHERE request_id = ?", (request_id,)
        ).fetchone()
        if existing is not None:
            record = _batch_from_row(existing)
            if record.account_id != account_id:
                raise ValueError("Batch request id is already in use.")
            return record
        cursor = connection.execute(
            """
            INSERT INTO batches (
                account_id, request_id, status, requested_count, planner_seed,
                config_json, created_at, updated_at
            ) VALUES (?, ?, 'pending', ?, ?, ?, datetime('now','localtime'), datetime('now','localtime'))
            """,
            (account_id, request_id, requested_count, planner_seed, config_json),
        )
        batch_id = int(cursor.lastrowid)
        for item in plan_items:
            plan_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            if any(secret in plan_json.casefold() for secret in ("cloud_api_key", "api_key")):
                raise ValueError("Batch plans cannot contain cloud credentials.")
            connection.execute(
                """
                INSERT INTO batch_items (
                    batch_id, item_index, request_id, status, plan_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 'pending', ?, datetime('now','localtime'), datetime('now','localtime'))
                """,
                (batch_id, int(item["index"]), str(item["request_id"]), plan_json),
            )
        row = connection.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    assert row is not None
    return _batch_from_row(row)


def get_batch(batch_id: int) -> BatchRecord | None:
    with _connection() as connection:
        row = connection.execute("SELECT * FROM batches WHERE id = ?", (batch_id,)).fetchone()
    return _batch_from_row(row) if row is not None else None


def get_batch_by_request_id(request_id: str) -> BatchRecord | None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM batches WHERE request_id = ?", (request_id,)
        ).fetchone()
    return _batch_from_row(row) if row is not None else None


def get_batch_items(batch_id: int) -> list[BatchItemRecord]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM batch_items WHERE batch_id = ? ORDER BY item_index",
            (batch_id,),
        ).fetchall()
    return [_batch_item_from_row(row) for row in rows]


def recent_batches(account_id: int, *, limit: int = 50) -> list[BatchRecord]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM batches WHERE account_id = ? ORDER BY id DESC LIMIT ?",
            (account_id, max(1, min(100, limit))),
        ).fetchall()
    return [_batch_from_row(row) for row in rows]


def update_batch_status(batch_id: int, status: str) -> None:
    if status not in {"pending", "running", "completed", "partial", "failed", "cancelled"}:
        raise ValueError(f"Unsupported batch status: {status}")
    with _connection() as connection:
        connection.execute(
            """
            UPDATE batches SET status = ?, updated_at = datetime('now','localtime'),
                started_at = CASE WHEN ? = 'running' THEN COALESCE(started_at, datetime('now','localtime')) ELSE started_at END,
                finished_at = CASE WHEN ? IN ('completed','partial','failed','cancelled') THEN datetime('now','localtime') ELSE NULL END
            WHERE id = ?
            """,
            (status, status, status, batch_id),
        )


def update_batch_item(
    item_id: int, *, status: str | None = None, job_id: int | None = None,
    generation_id: int | None = None, error: str | None = None,
) -> None:
    if status is not None and status not in {
        "pending", "queued", "running", "succeeded", "failed", "cancelled",
    }:
        raise ValueError(f"Unsupported batch item status: {status}")
    assignments = ["updated_at = datetime('now','localtime')"]
    values: list[object] = []
    if status is not None:
        assignments.append("status = ?")
        values.append(status)
    if job_id is not None:
        assignments.append("job_id = ?")
        values.append(job_id)
    if generation_id is not None:
        assignments.append("generation_id = ?")
        values.append(generation_id)
    if error is not None:
        assignments.append("error = ?")
        values.append(error)
    values.append(item_id)
    with _connection() as connection:
        connection.execute(
            f"UPDATE batch_items SET {', '.join(assignments)} WHERE id = ?", values,
        )


def next_pending_batch_item(*, after_batch_id: int | None = None) -> BatchItemRecord | None:
    """Choose one pending item, rotating across non-terminal parents."""
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT batch_items.* FROM batch_items
            JOIN batches ON batches.id = batch_items.batch_id
            WHERE batch_items.status = 'pending'
              AND batches.status IN ('pending', 'running')
            ORDER BY CASE WHEN batches.id > ? THEN 0 ELSE 1 END,
                     batches.id, batch_items.item_index
            LIMIT 1
            """,
            (after_batch_id or 0,),
        ).fetchone()
    return _batch_item_from_row(row) if row is not None else None


def active_batch_child_counts() -> tuple[int, int]:
    """Return (running, queued) batch children after reconciliation."""
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'running' THEN 1 ELSE 0 END) AS running,
                SUM(CASE WHEN status = 'queued' THEN 1 ELSE 0 END) AS queued
            FROM batch_items
            """
        ).fetchone()
    return int(row["running"] or 0), int(row["queued"] or 0)


def account_job_limit(account_id: int) -> int | None:
    with _connection() as connection:
        try:
            row = connection.execute(
                "SELECT max_active_jobs FROM users WHERE id = ?", (account_id,)
            ).fetchone()
        except sqlite3.OperationalError:
            return None
    return int(row["max_active_jobs"]) if row and row["max_active_jobs"] is not None else None


def reconcile_batch_jobs(batch_id: int | None = None) -> int:
    """Copy durable child job outcomes into items and derive parent states."""
    updated = 0
    with _connection() as connection:
        clause = "AND batch_items.batch_id = ?" if batch_id is not None else ""
        values: tuple[object, ...] = (batch_id,) if batch_id is not None else ()
        rows = connection.execute(
            f"""
            SELECT batch_items.id AS item_id, batch_items.status AS item_status,
                   batch_items.batch_id, jobs.status AS job_status,
                   jobs.generation_id, jobs.error_message
            FROM batch_items
            JOIN jobs ON jobs.id = batch_items.job_id
            WHERE batch_items.status IN ('queued','running') {clause}
            """,
            values,
        ).fetchall()
        affected: set[int] = set()
        for row in rows:
            job_status = str(row["job_status"])
            if job_status not in {"queued", "running", "succeeded", "failed", "cancelled"}:
                continue
            generation_id = row["generation_id"]
            error = row["error_message"]
            if (
                job_status != row["item_status"]
                or generation_id is not None
                or error is not None
            ):
                connection.execute(
                    """
                    UPDATE batch_items SET status = ?, generation_id = ?, error = ?,
                        updated_at = datetime('now','localtime')
                    WHERE id = ?
                    """,
                    (job_status, generation_id, error, row["item_id"]),
                )
                updated += 1
            affected.add(int(row["batch_id"]))
        if batch_id is not None:
            affected.add(batch_id)
        if batch_id is None:
            affected.update(int(row[0]) for row in connection.execute(
                "SELECT id FROM batches WHERE status IN ('pending','running')"
            ))

        terminal = {"succeeded", "failed", "cancelled"}
        for parent_id in affected:
            statuses = [str(row[0]) for row in connection.execute(
                "SELECT status FROM batch_items WHERE batch_id = ? ORDER BY item_index",
                (parent_id,),
            )]
            if not statuses:
                continue
            if all(status == "succeeded" for status in statuses):
                parent_status = "completed"
            elif all(status in terminal for status in statuses):
                if all(status == "cancelled" for status in statuses):
                    parent_status = "cancelled"
                elif any(status == "succeeded" for status in statuses):
                    parent_status = "partial"
                else:
                    parent_status = "failed"
            elif any(status != "pending" for status in statuses):
                parent_status = "running"
            else:
                parent_status = "pending"
            connection.execute(
                """
                UPDATE batches SET status = ?, updated_at = datetime('now','localtime'),
                    started_at = CASE WHEN ? = 'running' THEN COALESCE(started_at, datetime('now','localtime')) ELSE started_at END,
                    finished_at = CASE WHEN ? IN ('completed','partial','failed','cancelled') THEN datetime('now','localtime') ELSE NULL END
                WHERE id = ?
                """,
                (parent_status, parent_status, parent_status, parent_id),
            )
    return updated


def cancel_pending_batch_items(batch_id: int) -> int:
    """Cancel only work that has not yet been promoted for one parent."""
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE batch_items SET status = 'cancelled',
                error = 'Cancelled before submission.',
                updated_at = datetime('now','localtime')
            WHERE batch_id = ? AND status = 'pending'
            """,
            (batch_id,),
        )
        connection.execute(
            "UPDATE batches SET updated_at = datetime('now','localtime') WHERE id = ?",
            (batch_id,),
        )
        return max(0, int(cursor.rowcount))


def retry_failed_batch_items(batch_id: int) -> int:
    """Reset failed items in place while preserving their immutable plan."""
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE batch_items SET status = 'pending', error = NULL,
                generation_id = NULL, updated_at = datetime('now','localtime')
            WHERE batch_id = ? AND status = 'failed'
            """,
            (batch_id,),
        )
        retried = max(0, int(cursor.rowcount))
        if retried:
            connection.execute(
                """
                UPDATE batches SET status = 'running', finished_at = NULL,
                    updated_at = datetime('now','localtime') WHERE id = ?
                """,
                (batch_id,),
            )
        return retried


def recover_nonterminal_batches() -> int:
    """Repair item/job linkage and state after an interrupted server process."""
    repaired = 0
    with _connection() as connection:
        active_batch_ids = {
            int(row[0]) for row in connection.execute(
                "SELECT id FROM batches WHERE status IN ('pending','running')"
            )
        }
        if not active_batch_ids:
            return 0
        placeholders = ",".join("?" for _ in active_batch_ids)
        item_rows = connection.execute(
            f"SELECT id, batch_id, status, job_id FROM batch_items "
            f"WHERE batch_id IN ({placeholders})",
            tuple(sorted(active_batch_ids)),
        ).fetchall()
        items_by_id = {int(row["id"]): row for row in item_rows}
        latest_jobs: dict[int, sqlite3.Row] = {}
        for job in connection.execute(
            "SELECT * FROM jobs WHERE kind = 'batch_generate' ORDER BY id DESC"
        ):
            try:
                payload = json.loads(str(job["request_payload"]))
                item_id = int(payload.get("batch_item_id"))
            except (TypeError, ValueError, AttributeError):
                continue
            if item_id in items_by_id and item_id not in latest_jobs:
                latest_jobs[item_id] = job

        for item_id, item in items_by_id.items():
            linked = None
            if item["job_id"] is not None:
                linked = connection.execute(
                    "SELECT * FROM jobs WHERE id = ?", (item["job_id"],)
                ).fetchone()
            crash_window_job = latest_jobs.get(item_id)
            if linked is None and crash_window_job is not None and str(crash_window_job["status"]) in {
                "queued", "running", "succeeded",
            }:
                connection.execute(
                    """
                    UPDATE batch_items SET job_id = ?, status = ?,
                        generation_id = ?, error = ?,
                        updated_at = datetime('now','localtime') WHERE id = ?
                    """,
                    (
                        crash_window_job["id"], crash_window_job["status"],
                        crash_window_job["generation_id"], crash_window_job["error_message"],
                        item_id,
                    ),
                )
                repaired += 1
            elif linked is None and str(item["status"]) in {"queued", "running"}:
                connection.execute(
                    """
                    UPDATE batch_items SET status = 'pending', job_id = NULL,
                        error = NULL, updated_at = datetime('now','localtime')
                    WHERE id = ?
                    """,
                    (item_id,),
                )
                repaired += 1
    repaired += reconcile_batch_jobs()
    return repaired


def save_video_generation(*, content: bytes, prompt: str, negative_prompt: str, mode: str,
    source_generation_id: int | None, width: int, height: int, frame_count: int,
    fps: int, seed: int, duration_seconds: float, camera_motion: str | None,
    comfy_prompt_id: str, poster_content: bytes | None = None) -> VideoGenerationRecord:
    """Write MP4 bytes to DATA_DIR and persist metadata only in SQLite."""
    if not content:
        raise ValueError("Video output is empty.")
    VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    token = uuid4().hex
    filename = f"video-{token}.mp4"
    temporary = VIDEOS_DIR / f".{filename}.tmp"
    temporary.write_bytes(content)
    temporary.replace(VIDEOS_DIR / filename)
    poster_filename: str | None = None
    if poster_content:
        poster_filename = f"video-{token}-last-frame.png"
        poster_temporary = VIDEOS_DIR / f".{poster_filename}.tmp"
        poster_temporary.write_bytes(poster_content)
        poster_temporary.replace(VIDEOS_DIR / poster_filename)
    with _connection() as connection:
        cursor = connection.execute("""
            INSERT INTO video_generations
            (filename, poster_filename, prompt, negative_prompt, mode, source_generation_id, width, height,
             frame_count, fps, seed, duration_seconds, camera_motion, comfy_prompt_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (filename, poster_filename, prompt, negative_prompt, mode, source_generation_id, width, height,
              frame_count, fps, seed, duration_seconds, camera_motion, comfy_prompt_id))
        video_id = int(cursor.lastrowid)
    return get_video_generation(video_id)  # type: ignore[return-value]


def _video_from_row(row: sqlite3.Row) -> VideoGenerationRecord:
    return VideoGenerationRecord(id=int(row["id"]), filename=str(row["filename"]),
        poster_filename=str(row["poster_filename"]) if row["poster_filename"] else None,
        prompt=str(row["prompt"]), negative_prompt=str(row["negative_prompt"]), mode=str(row["mode"]),
        source_generation_id=int(row["source_generation_id"]) if row["source_generation_id"] is not None else None,
        width=int(row["width"]), height=int(row["height"]), frame_count=int(row["frame_count"]),
        fps=int(row["fps"]), seed=int(row["seed"]), duration_seconds=float(row["duration_seconds"]),
        camera_motion=str(row["camera_motion"]) if row["camera_motion"] else None,
        comfy_prompt_id=str(row["comfy_prompt_id"]), favorite=bool(row["favorite"]), created_at=str(row["created_at"]))


def get_video_generation(video_id: int) -> VideoGenerationRecord | None:
    with _connection() as connection:
        row = connection.execute("SELECT * FROM video_generations WHERE id = ?", (video_id,)).fetchone()
    return _video_from_row(row) if row else None


def recent_video_generations(*, before_id: int | None = None, limit: int = 60) -> list[VideoGenerationRecord]:
    clause = "WHERE id < ?" if before_id is not None else ""
    values = (before_id, limit) if before_id is not None else (limit,)
    with _connection() as connection:
        rows = connection.execute(f"SELECT * FROM video_generations {clause} ORDER BY id DESC LIMIT ?", values).fetchall()
    return [_video_from_row(row) for row in rows]


def set_video_favorite(video_id: int, favorite: bool) -> bool:
    with _connection() as connection:
        return connection.execute("UPDATE video_generations SET favorite = ? WHERE id = ?", (int(favorite), video_id)).rowcount == 1


def delete_video_generation(video_id: int) -> VideoGenerationRecord | None:
    record = get_video_generation(video_id)
    if record is None: return None
    with _connection() as connection:
        connection.execute("DELETE FROM video_generations WHERE id = ?", (video_id,))
    (VIDEOS_DIR / record.filename).unlink(missing_ok=True)
    if record.poster_filename: (VIDEOS_DIR / record.poster_filename).unlink(missing_ok=True)
    return record


def delete_generation(generation_id: int) -> GenerationRecord | None:
    """Delete an image row before unlinking its now-orphaned artifacts."""
    record = get_generation(generation_id)
    if record is None:
        return None
    with _connection() as connection:
        connection.execute("DELETE FROM generations WHERE id = ?", (generation_id,))
    (GENERATED_DIR / record.filename).unlink(missing_ok=True)
    (THUMBNAILS_DIR / record.filename).unlink(missing_ok=True)
    return record


def _identity_profile_from_row(row: sqlite3.Row) -> IdentityProfileRecord:
    try:
        quality = json.loads(str(row["quality_json"]))
    except (TypeError, ValueError):
        quality = {}
    return IdentityProfileRecord(
        id=int(row["id"]),
        account_id=int(row["account_id"]),
        name=str(row["name"]),
        filename=str(row["filename"]),
        thumbnail_filename=str(row["thumbnail_filename"]),
        detector=str(row["detector"]),
        embedding_model=str(row["embedding_model"]),
        quality=quality if isinstance(quality, dict) else {},
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
        last_used_at=str(row["last_used_at"]) if row["last_used_at"] else None,
    )


def create_identity_profile(
    *,
    account_id: int,
    name: str,
    filename: str,
    thumbnail_filename: str,
    embedding: bytes,
    detector: str,
    embedding_model: str,
    quality: dict[str, object],
) -> IdentityProfileRecord:
    """Persist metadata for an already-written validated identity crop."""
    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO identity_profiles (
                account_id, name, filename, thumbnail_filename, embedding,
                detector, embedding_model, quality_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                account_id,
                name,
                filename,
                thumbnail_filename,
                embedding,
                detector,
                embedding_model,
                json.dumps(quality, ensure_ascii=False, separators=(",", ":")),
            ),
        )
        row = connection.execute(
            "SELECT * FROM identity_profiles WHERE id = ?", (cursor.lastrowid,)
        ).fetchone()
    if row is None:
        raise sqlite3.IntegrityError("Identity profile insert returned no record.")
    return _identity_profile_from_row(row)


def get_identity_profile(
    profile_id: int, *, account_id: int | None = None
) -> IdentityProfileRecord | None:
    clauses = ["id = ?"]
    values: list[object] = [profile_id]
    if account_id is not None:
        clauses.append("account_id = ?")
        values.append(account_id)
    with _connection() as connection:
        row = connection.execute(
            f"SELECT * FROM identity_profiles WHERE {' AND '.join(clauses)}", values
        ).fetchone()
    return _identity_profile_from_row(row) if row is not None else None


def list_identity_profiles(account_id: int) -> list[IdentityProfileRecord]:
    with _connection() as connection:
        rows = connection.execute(
            "SELECT * FROM identity_profiles WHERE account_id = ? ORDER BY id DESC",
            (account_id,),
        ).fetchall()
    return [_identity_profile_from_row(row) for row in rows]


def touch_identity_profile(profile_id: int, account_id: int) -> None:
    with _connection() as connection:
        connection.execute(
            "UPDATE identity_profiles SET last_used_at = datetime('now','localtime') WHERE id = ? AND account_id = ?",
            (profile_id, account_id),
        )


def rename_identity_profile(profile_id: int, account_id: int, name: str) -> bool:
    with _connection() as connection:
        cursor = connection.execute(
            """
            UPDATE identity_profiles
            SET name = ?, updated_at = datetime('now','localtime')
            WHERE id = ? AND account_id = ?
            """,
            (name, profile_id, account_id),
        )
    return cursor.rowcount == 1


def delete_identity_profile(profile_id: int, account_id: int) -> IdentityProfileRecord | None:
    with _connection() as connection:
        row = connection.execute(
            "SELECT * FROM identity_profiles WHERE id = ? AND account_id = ?",
            (profile_id, account_id),
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            "DELETE FROM identity_profiles WHERE id = ? AND account_id = ?",
            (profile_id, account_id),
        )
    return _identity_profile_from_row(row)


def identity_profile_has_active_job(profile_id: int, account_id: int) -> bool:
    """Protect a reference until queued/running character work has copied it."""
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT request_payload FROM jobs
            WHERE account_id = ? AND kind = 'character' AND status IN ('queued', 'running')
            """,
            (account_id,),
        ).fetchall()
    for row in rows:
        try:
            payload = json.loads(str(row["request_payload"]))
        except (TypeError, ValueError):
            continue
        if isinstance(payload, dict) and payload.get("identity_profile_id") == profile_id:
            return True
    return False


def job_kind_for_generation(generation_id: int) -> str | None:
    """Return the pipeline kind that produced one generation, when tracked."""
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT kind FROM jobs
            WHERE generation_id = ? AND status = 'succeeded'
            ORDER BY id DESC LIMIT 1
            """,
            (generation_id,),
        ).fetchone()
    return str(row["kind"]) if row is not None else None


def recent_jobs(
    limit: int = 50,
    *,
    before_id: int | None = None,
    status: str | None = None,
    account_id: int | None = None,
    kinds: tuple[str, ...] | None = None,
) -> list[JobRecord]:
    """Return a cursor-addressable job page, optionally scoped and filtered."""
    safe_limit = max(1, min(int(limit), HISTORY_QUERY_LIMIT))
    clauses = ["(? IS NULL OR id < ?)"]
    values: list[object] = [before_id, before_id]
    if status is not None:
        clauses.append("status = ?")
        values.append(status)
    if account_id is not None:
        clauses.append("account_id = ?")
        values.append(account_id)
    if kinds:
        placeholders = ",".join("?" for _ in kinds)
        clauses.append(f"kind IN ({placeholders})")
        values.extend(kinds)
    values.append(safe_limit)
    with _connection() as connection:
        rows = connection.execute(
            f"SELECT * FROM jobs WHERE {' AND '.join(clauses)} ORDER BY id DESC LIMIT ?",
            values,
        ).fetchall()
    return [_job_from_row(row) for row in rows]


# A job holding the single worker forever blocks every queued job behind it.
# Nine hours is far past any legitimate render, so treat it as a dead worker.
STALLED_JOB_SECONDS = 3600.0


def set_generation_favorite(generation_id: int, favorite: bool) -> bool:
    """Mark or unmark one generation, returning whether the row existed."""
    with _connection() as connection:
        cursor = connection.execute(
            "UPDATE generations SET favorite = ? WHERE id = ?",
            (1 if favorite else 0, int(generation_id)),
        )
        return cursor.rowcount > 0


def delete_jobs(job_ids: list[int]) -> int:
    """Remove finished job records, leaving their generations untouched.

    Only terminal jobs are deletable: removing a queued or running record
    would orphan work the runner still holds a reference to.
    """
    if not job_ids:
        return 0
    placeholders = ",".join("?" for _ in job_ids)
    with _connection() as connection:
        cursor = connection.execute(
            f"""
            DELETE FROM jobs
            WHERE id IN ({placeholders})
              AND status IN ('succeeded', 'failed', 'cancelled')
            """,
            [int(job_id) for job_id in job_ids],
        )
        return max(0, int(cursor.rowcount))


def fail_stalled_running_jobs(max_age_seconds: float = STALLED_JOB_SECONDS) -> list[int]:
    """Fail running jobs whose worker stopped reporting, freeing the queue."""
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT id FROM jobs
            WHERE status = 'running'
              AND started_at IS NOT NULL
              AND (julianday('now','localtime') - julianday(started_at)) * 86400.0 > ?
            """,
            (float(max_age_seconds),),
        ).fetchall()
        stalled = [int(row["id"]) for row in rows]
        if stalled:
            connection.executemany(
                """
                UPDATE jobs
                SET status = 'failed', finished_at = datetime('now','localtime'),
                    current_image = NULL,
                    error_message = 'The job stopped responding and was released.'
                WHERE id = ?
                """,
                [(job_id,) for job_id in stalled],
            )
        return stalled


def recover_stalled_jobs() -> int:
    """Fail work interrupted mid-render and requeue anything that never began.

    A queued job has not started, so failing it is wrong twice over: the work
    was never attempted, and the runner may still hold it in memory and execute
    it anyway - which showed a job as failed and then quietly succeeded.
    """
    with _connection() as connection:
        resumable = connection.execute(
            """
            UPDATE jobs
            SET status = 'queued', finished_at = NULL, current_image = NULL,
                error_message = NULL
            WHERE status = 'queued'
               OR (kind IN ('character', 'video_t2v', 'video_i2v', 'batch_generate') AND status = 'running')
               OR (
                    kind IN ('generate', 'surprise_generate')
                    AND status = 'running'
                    AND json_extract(request_payload, '$.quality_mode') = '12k'
               )
            """
        )
        failed = connection.execute(
            """
            UPDATE jobs
            SET status = 'failed', finished_at = datetime('now','localtime'),
                current_image = NULL,
                error_message = 'Interrupted by server restart'
            WHERE kind NOT IN ('character', 'video_t2v', 'video_i2v', 'batch_generate') AND status = 'running'
              AND NOT (
                    kind IN ('generate', 'surprise_generate')
                    AND json_extract(request_payload, '$.quality_mode') = '12k'
              )
            """
        )
        return max(0, int(resumable.rowcount)) + max(0, int(failed.rowcount))


def resumable_job_ids() -> list[int]:
    """Return every persisted job awaiting the single GPU worker."""
    with _connection() as connection:
        rows = connection.execute(
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY id"
        ).fetchall()
    return [int(row["id"]) for row in rows]


def resumable_character_job_ids() -> list[int]:
    """Deprecated alias retained for callers expecting character-only work."""
    return resumable_job_ids()


def recent_generations(
    limit: int = 24, *, before_id: int | None = None
) -> list[GenerationRecord]:
    """Return a cursor-addressable page of generations, newest first."""
    safe_limit = max(1, min(limit, HISTORY_QUERY_LIMIT))
    safe_before_id = max(1, int(before_id)) if before_id is not None else None
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT generations.id, filename, positive_prompt, negative_prompt, timestamp,
                   width, height, rating, style, style_variant, quality_mode, favorite, high_res, super_res, model,
                   generation_feedback.score AS user_rating,
                   generation_feedback.reason_codes AS feedback_reasons,
                   sampler_name, scheduler, steps, cfg_scale, seed, duration_seconds,
                   diffusion_duration_seconds, upscale_duration_seconds,
                   source_generation_id,
                   (SELECT jobs.id FROM jobs WHERE jobs.generation_id = generations.id
                    ORDER BY jobs.id DESC LIMIT 1) AS job_id
            FROM generations
            LEFT JOIN generation_feedback ON generation_feedback.generation_id = generations.id
            WHERE filename IS NOT NULL
              AND (? IS NULL OR generations.id < ?)
            ORDER BY generations.id DESC LIMIT ?
            """,
            (safe_before_id, safe_before_id, safe_limit),
        ).fetchall()
    return [
        GenerationRecord(
            id=int(row["id"]),
            filename=str(row["filename"]),
            positive_prompt=str(row["positive_prompt"]),
            negative_prompt=str(row["negative_prompt"]),
            timestamp=str(row["timestamp"]),
            width=int(row["width"]),
            height=int(row["height"]),
            rating=str(row["rating"]),
            style=str(row["style"]),
            high_res=bool(row["high_res"]),
            super_res=bool(row["super_res"]),
            model=str(row["model"]),
            user_rating=int(row["user_rating"]) if row["user_rating"] is not None else None,
            sampler_name=str(row["sampler_name"]),
            scheduler=str(row["scheduler"]),
            steps=int(row["steps"]),
            cfg_scale=float(row["cfg_scale"]),
            seed=int(row["seed"]),
            duration_seconds=float(row["duration_seconds"]) if row["duration_seconds"] is not None else None,
            diffusion_duration_seconds=(
                float(row["diffusion_duration_seconds"])
                if row["diffusion_duration_seconds"] is not None
                else None
            ),
            upscale_duration_seconds=(
                float(row["upscale_duration_seconds"])
                if row["upscale_duration_seconds"] is not None
                else None
            ),
            source_generation_id=(
                int(row["source_generation_id"])
                if row["source_generation_id"] is not None
                else None
            ),
            job_id=(int(row["job_id"]) if row["job_id"] is not None else None),
            feedback_reasons=_decode_feedback_reasons(row["feedback_reasons"]),
            favorite=bool(row["favorite"]),
            style_variant=(
                str(row["style_variant"]) if row["style_variant"] is not None else None
            ),
            quality_mode=str(row["quality_mode"]),
        )
        for row in rows
    ]


def generation_filename(generation_id: int) -> str | None:
    """Return the materialized filename for one persisted generation."""
    with _connection() as connection:
        row = connection.execute(
            "SELECT filename FROM generations WHERE id = ? AND filename IS NOT NULL",
            (generation_id,),
        ).fetchone()
    return str(row["filename"]) if row is not None else None


def get_generation(generation_id: int) -> GenerationRecord | None:
    """Return one persisted generation's full metadata, or None if missing."""
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT generations.id, filename, positive_prompt, negative_prompt, timestamp,
                   width, height, rating, style, style_variant, quality_mode, favorite, high_res, super_res, model,
                   generation_feedback.score AS user_rating,
                   generation_feedback.reason_codes AS feedback_reasons,
                   sampler_name, scheduler, steps, cfg_scale, seed, duration_seconds,
                   diffusion_duration_seconds, upscale_duration_seconds,
                   source_generation_id,
                   (SELECT jobs.id FROM jobs WHERE jobs.generation_id = generations.id
                    ORDER BY jobs.id DESC LIMIT 1) AS job_id
            FROM generations
            LEFT JOIN generation_feedback ON generation_feedback.generation_id = generations.id
            WHERE generations.id = ? AND filename IS NOT NULL
            """,
            (generation_id,),
        ).fetchone()
    if row is None:
        return None
    return GenerationRecord(
        id=int(row["id"]),
        filename=str(row["filename"]),
        positive_prompt=str(row["positive_prompt"]),
        negative_prompt=str(row["negative_prompt"]),
        timestamp=str(row["timestamp"]),
        width=int(row["width"]),
        height=int(row["height"]),
        rating=str(row["rating"]),
        style=str(row["style"]),
        high_res=bool(row["high_res"]),
        super_res=bool(row["super_res"]),
        model=str(row["model"]),
        user_rating=int(row["user_rating"]) if row["user_rating"] is not None else None,
        sampler_name=str(row["sampler_name"]),
        scheduler=str(row["scheduler"]),
        steps=int(row["steps"]),
        cfg_scale=float(row["cfg_scale"]),
        seed=int(row["seed"]),
        duration_seconds=float(row["duration_seconds"]) if row["duration_seconds"] is not None else None,
        diffusion_duration_seconds=(
            float(row["diffusion_duration_seconds"])
            if row["diffusion_duration_seconds"] is not None
            else None
        ),
        upscale_duration_seconds=(
            float(row["upscale_duration_seconds"])
            if row["upscale_duration_seconds"] is not None
            else None
        ),
        source_generation_id=(
            int(row["source_generation_id"])
            if row["source_generation_id"] is not None
            else None
        ),
        feedback_reasons=_decode_feedback_reasons(row["feedback_reasons"]),
        favorite=bool(row["favorite"]),
        style_variant=(
            str(row["style_variant"]) if row["style_variant"] is not None else None
        ),
        quality_mode=str(row["quality_mode"]),
    )


def average_generation_duration(
    *, model: str, style: str, high_res: bool, super_res: bool
) -> float:
    """Return a context-specific observed duration or a conservative fallback."""
    with _connection() as connection:
        row = connection.execute(
            """
            SELECT AVG(duration_seconds) AS average_seconds
            FROM (
                SELECT duration_seconds
                FROM generations
                WHERE model = ? AND style = ? AND high_res = ? AND super_res = ?
                  AND duration_seconds IS NOT NULL AND duration_seconds > 0
                ORDER BY id DESC LIMIT 20
            )
            """,
            (model, style, int(high_res), int(super_res)),
        ).fetchone()
    if row and row["average_seconds"] is not None:
        return max(1.0, float(row["average_seconds"]))
    return 120.0 if super_res else 60.0 if high_res else 30.0


def save_prompt_run(
    *,
    positive_prompt: str,
    negative_prompt: str,
    source: str,
    engine: str,
    model: str,
    style: str,
    aspect_ratio: str,
    content_rating: str,
    high_res: bool,
    super_res: bool,
    creativity_level: str = "balanced",
    novelty_signature: str | None = None,
    similarity_score: float | None = None,
    novelty_retry_count: int = 0,
    prompt_duration_seconds: float | None = None,
    compiler_version: str = PROMPT_COMPILER_VERSION,
) -> int:
    """Persist an immutable prompt attempt before image generation begins."""
    with _connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO prompt_runs (
                positive_prompt, negative_prompt, source, engine, model, style,
                aspect_ratio, content_rating, high_res, super_res, creativity_level,
                novelty_signature, similarity_score, novelty_retry_count,
                prompt_duration_seconds, compiler_version, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now','localtime'))
            """,
            (
                positive_prompt,
                negative_prompt,
                source,
                engine,
                model,
                style,
                aspect_ratio,
                content_rating,
                int(high_res),
                int(super_res),
                creativity_level,
                novelty_signature,
                similarity_score,
                novelty_retry_count,
                prompt_duration_seconds,
                compiler_version,
            ),
        )
        return int(cursor.lastrowid)


def _recover_interrupted_prompt_runs(
    connection: sqlite3.Connection, *, grace_minutes: int
) -> int:
    """Mark abandoned in-flight prompt attempts as failed after a restart."""
    safe_grace = max(0, min(int(grace_minutes), 24 * 60))
    cursor = connection.execute(
        """
        UPDATE prompt_runs
        SET status = 'failed',
            error_message = 'Interrupted by server restart'
        WHERE status = 'created'
          AND datetime(created_at) <= datetime('now','localtime', ?)
        """,
        (f"-{safe_grace} minutes",),
    )
    return max(0, int(cursor.rowcount))


def recover_interrupted_prompt_runs(
    grace_minutes: int = INTERRUPTED_RUN_GRACE_MINUTES,
) -> int:
    """Recover stale attempts without touching recently-started generations."""
    with _connection() as connection:
        return _recover_interrupted_prompt_runs(
            connection, grace_minutes=grace_minutes
        )


def analytics_records(days: int = 30) -> tuple[list[AnalyticsRun], AttemptCounts]:
    """Return completed run telemetry and terminal workflow counts."""
    safe_days = max(0, min(days, 3650))
    generation_filter = ""
    attempt_filter = ""
    parameters: tuple[object, ...] = ()
    attempt_parameters: tuple[object, ...] = (PROMPT_COMPILER_VERSION,)
    if safe_days:
        modifier = f"-{safe_days} days"
        generation_filter = "WHERE datetime(g.timestamp) >= datetime('now', 'localtime', ?)"
        attempt_filter = "AND datetime(created_at) >= datetime('now', 'localtime', ?)"
        parameters = (modifier,)
        attempt_parameters = (PROMPT_COMPILER_VERSION, modifier)
    with _connection() as connection:
        rows = connection.execute(
            f"""
            SELECT g.id, g.timestamp, g.model,
                   CASE WHEN g.style_variant IS NOT NULL
                        THEN g.style || ' · ' || g.style_variant
                        ELSE g.style END AS style,
                   g.rating,
                   COALESCE(pr.aspect_ratio, printf('%dx%d', g.width, g.height)) AS aspect_ratio,
                   COALESCE(NULLIF(g.quality_mode, ''),
                       CASE WHEN g.super_res = 1 THEN 'super'
                            WHEN g.high_res = 1 THEN 'high'
                            ELSE 'normal' END) AS quality_mode,
                   g.width, g.height, g.sampler_name, g.scheduler, g.steps,
                   g.cfg_scale, g.duration_seconds, g.diffusion_duration_seconds,
                   g.upscale_duration_seconds,
                   pr.prompt_duration_seconds, COALESCE(pr.engine, 'unknown') AS prompt_engine,
                   COALESCE(pr.creativity_level, 'unknown') AS creativity_level,
                   pr.similarity_score, COALESCE(pr.novelty_retry_count, 0) AS novelty_retry_count,
                   COALESCE(pr.compiler_version, 'legacy') AS compiler_version,
                   generation_feedback.score AS user_rating,
                   generation_feedback.reason_codes AS feedback_reasons
            FROM generations AS g
            LEFT JOIN prompt_runs AS pr ON pr.id = (
                SELECT MAX(candidate.id) FROM prompt_runs AS candidate
                WHERE candidate.generation_id = g.id
            )
            LEFT JOIN generation_feedback
              ON generation_feedback.generation_id = g.id
            {generation_filter}
            ORDER BY g.id DESC
            """,
            parameters,
        ).fetchall()
        attempt_rows = connection.execute(
            f"""
            SELECT status, COUNT(*) AS count
            FROM prompt_runs
            WHERE status IN ('succeeded', 'failed')
              AND compiler_version = ? {attempt_filter}
            GROUP BY status
            """,
            attempt_parameters,
        ).fetchall()
        failure_rows = connection.execute(
            f"""
            SELECT CASE
                     WHEN lower(COALESCE(error_message, '')) LIKE '%timeout%' THEN 'Forge timeout'
                     WHEN lower(COALESCE(error_message, '')) LIKE '%storage%'
                       OR lower(COALESCE(error_message, '')) LIKE '%disk%' THEN 'Local storage'
                     WHEN lower(COALESCE(error_message, '')) LIKE '%database%'
                       OR lower(COALESCE(error_message, '')) LIKE '%history%' THEN 'Local database'
                     WHEN lower(COALESCE(error_message, '')) LIKE '%memory%'
                       OR lower(COALESCE(error_message, '')) LIKE '%500%' THEN 'Forge or VRAM'
                     WHEN trim(COALESCE(error_message, '')) = '' THEN 'Unknown interruption'
                     ELSE 'Prompt or provider'
                   END AS reason,
                   COUNT(*) AS count
            FROM prompt_runs
            WHERE status = 'failed'
              AND compiler_version = ? {attempt_filter}
            GROUP BY reason
            ORDER BY count DESC, reason
            """,
            attempt_parameters,
        ).fetchall()
    records = [
        AnalyticsRun(
            id=int(row["id"]),
            timestamp=str(row["timestamp"]),
            model=str(row["model"]),
            style=str(row["style"]),
            content_rating=str(row["rating"]),
            aspect_ratio=str(row["aspect_ratio"]),
            quality_mode=str(row["quality_mode"]),
            width=int(row["width"]),
            height=int(row["height"]),
            sampler_name=str(row["sampler_name"]),
            scheduler=str(row["scheduler"]),
            steps=int(row["steps"]),
            cfg_scale=float(row["cfg_scale"]),
            duration_seconds=(
                float(row["duration_seconds"])
                if row["duration_seconds"] is not None
                else None
            ),
            diffusion_duration_seconds=(
                float(row["diffusion_duration_seconds"])
                if row["diffusion_duration_seconds"] is not None
                else None
            ),
            upscale_duration_seconds=(
                float(row["upscale_duration_seconds"])
                if row["upscale_duration_seconds"] is not None
                else None
            ),
            prompt_duration_seconds=(
                float(row["prompt_duration_seconds"])
                if row["prompt_duration_seconds"] is not None
                else None
            ),
            prompt_engine=str(row["prompt_engine"]),
            creativity_level=str(row["creativity_level"]),
            similarity_score=(
                float(row["similarity_score"])
                if row["similarity_score"] is not None
                else None
            ),
            novelty_retry_count=int(row["novelty_retry_count"]),
            user_rating=(
                int(row["user_rating"]) if row["user_rating"] is not None else None
            ),
            feedback_reasons=_decode_feedback_reasons(row["feedback_reasons"]),
            compiler_version=str(row["compiler_version"]),
        )
        for row in rows
    ]
    counts = {str(row["status"]): int(row["count"]) for row in attempt_rows}
    return records, AttemptCounts(
        succeeded=counts.get("succeeded", 0),
        failed=counts.get("failed", 0),
        failure_reasons=tuple(
            (str(row["reason"]), int(row["count"])) for row in failure_rows
        ),
    )


def recent_prompt_history(
    *,
    model: str,
    style: str,
    content_rating: str,
    limit: int = 20,
) -> list[PromptHistoryRecord]:
    """Return recent comparable prompts and feedback for local novelty steering."""
    safe_limit = max(1, min(limit, 50))
    with _connection() as connection:
        rows = connection.execute(
            """
            SELECT prompt_runs.positive_prompt, prompt_runs.novelty_signature,
                   generation_feedback.score AS user_rating,
                   generation_feedback.reason_codes AS feedback_reasons
            FROM prompt_runs
            LEFT JOIN generation_feedback
              ON generation_feedback.generation_id = prompt_runs.generation_id
            WHERE prompt_runs.model = ? AND prompt_runs.style = ?
              AND prompt_runs.content_rating = ?
              AND prompt_runs.status != 'failed'
            ORDER BY prompt_runs.id DESC
            LIMIT ?
            """,
            (model, style, content_rating, safe_limit),
        ).fetchall()
    return [
        PromptHistoryRecord(
            positive_prompt=str(row["positive_prompt"]),
            novelty_signature=(
                str(row["novelty_signature"])
                if row["novelty_signature"] is not None
                else None
            ),
            user_rating=(
                int(row["user_rating"]) if row["user_rating"] is not None else None
            ),
            feedback_reasons=_decode_feedback_reasons(row["feedback_reasons"]),
        )
        for row in rows
    ]


def finish_prompt_run(
    prompt_run_id: int,
    *,
    status: str,
    generation_id: int | None = None,
    error_message: str | None = None,
) -> None:
    """Link a prompt attempt to its outcome without discarding failed attempts."""
    if status not in {"created", "succeeded", "failed"}:
        raise ValueError("Unsupported prompt-run status.")
    with _connection() as connection:
        connection.execute(
            """
            UPDATE prompt_runs
            SET status = ?, generation_id = ?, error_message = ?
            WHERE id = ?
            """,
            (status, generation_id, error_message, prompt_run_id),
        )


def _decode_feedback_reasons(value: object) -> tuple[str, ...]:
    """Return only recognized feedback codes from persisted JSON."""
    if not value:
        return ()
    try:
        decoded = json.loads(str(value))
    except (TypeError, ValueError):
        return ()
    if not isinstance(decoded, list):
        return ()
    return tuple(
        code for code in (str(item) for item in decoded) if code in FEEDBACK_REASON_CODES
    )


def rate_generation(
    generation_id: int, score: int, reason_codes: tuple[str, ...] = ()
) -> tuple[int, tuple[str, ...]]:
    """Create or replace durable, actionable quality feedback."""
    if score not in range(1, 6):
        raise ValueError("Rating must be between 1 and 5.")
    normalized_reasons = tuple(dict.fromkeys(reason_codes))
    if set(normalized_reasons) - FEEDBACK_REASON_CODES:
        raise ValueError("Unsupported feedback reason.")
    if score <= 2 and not normalized_reasons:
        raise ValueError("Choose at least one reason for a low rating.")
    with _connection() as connection:
        exists = connection.execute(
            "SELECT 1 FROM generations WHERE id = ?", (generation_id,)
        ).fetchone()
        if exists is None:
            raise LookupError("Generation not found.")
        connection.execute(
            """
            INSERT INTO generation_feedback (generation_id, score, reason_codes)
            VALUES (?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                score = excluded.score,
                reason_codes = excluded.reason_codes,
                updated_at = datetime('now','localtime')
            """,
            (
                generation_id,
                score,
                json.dumps(normalized_reasons, separators=(",", ":")),
            ),
        )
    return score, normalized_reasons

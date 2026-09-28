"""Local account authentication with durable users and revocable sessions."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
import sqlite3
import struct
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Iterator

from .database import DATABASE_PATH

PASSWORD_ITERATIONS = 600_000
SESSION_DAYS = 30
TOTP_DIGITS = 6
TOTP_PERIOD = 30
TOTP_ISSUER = "NyxForge"
AUTHENTICATOR_MAX_FAILURES = 5
AUTHENTICATOR_LOCK_SECONDS = 60
RECOVERY_CODE_COUNT = 8
DEFAULT_MAX_ACTIVE_JOBS = 3
MIN_MAX_ACTIVE_JOBS = 1
MAX_MAX_ACTIVE_JOBS = 50
SESSION_WORKSPACES = frozenset({"forgeai", "forgeimg", "forgevid", "forgebat"})
_EMAIL_PATTERN = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


@dataclass(frozen=True, slots=True)
class Account:
    """Safe account fields that may be returned to the browser."""

    id: int
    email: str
    display_name: str
    is_admin: bool
    created_at: str
    authenticator_enabled: bool = False
    max_active_jobs: int = DEFAULT_MAX_ACTIVE_JOBS


class AuthenticationError(ValueError):
    """Raised when supplied credentials cannot be authenticated."""


class AuthenticationRateLimitError(AuthenticationError):
    """Raised while authenticator sign-in is temporarily rate limited."""


@contextmanager
def _auth_connection() -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(DATABASE_PATH, timeout=5.0)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA foreign_keys=ON")
        with connection:
            yield connection
    finally:
        connection.close()


def initialize_auth_database() -> None:
    """Create account and session tables without changing generation ownership."""
    with _auth_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL UNIQUE COLLATE NOCASE,
                display_name TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                is_admin INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_recovery_codes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                code_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                used_at TEXT
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS auth_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS generation_access_tokens (
                generation_id INTEGER PRIMARY KEY REFERENCES generations(id) ON DELETE CASCADE,
                token_hash TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_auth_sessions_expiry ON auth_sessions(expires_at)"
        )
        session_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(auth_sessions)")
        }
        if "active_workspace" not in session_columns:
            connection.execute("ALTER TABLE auth_sessions ADD COLUMN active_workspace TEXT")
        user_columns = {
            row["name"] for row in connection.execute("PRAGMA table_info(users)")
        }
        if "is_admin" not in user_columns:
            connection.execute(
                "ALTER TABLE users ADD COLUMN is_admin INTEGER NOT NULL DEFAULT 0"
            )
        authenticator_columns = {
            "totp_secret": "TEXT",
            "totp_pending_secret": "TEXT",
            "totp_enabled": "INTEGER NOT NULL DEFAULT 0",
            "totp_last_counter": "INTEGER",
            "totp_failed_attempts": "INTEGER NOT NULL DEFAULT 0",
            "totp_locked_until": "TEXT",
            "max_active_jobs": f"INTEGER NOT NULL DEFAULT {DEFAULT_MAX_ACTIVE_JOBS}",
        }
        for column, definition in authenticator_columns.items():
            if column not in user_columns:
                connection.execute(f"ALTER TABLE users ADD COLUMN {column} {definition}")
        # Migrate installations created before roles existed. The oldest local
        # account is the owner of this single-user Forge installation.
        connection.execute(
            """
            UPDATE users SET is_admin = 1
            WHERE id = (SELECT MIN(id) FROM users)
              AND NOT EXISTS (SELECT 1 FROM users WHERE is_admin = 1)
            """
        )


def _normalize_email(email: str) -> str:
    normalized = email.strip().lower()
    if len(normalized) > 254 or not _EMAIL_PATTERN.fullmatch(normalized):
        raise ValueError("Enter a valid email address.")
    return normalized


def _normalize_display_name(display_name: str) -> str:
    normalized = " ".join(display_name.strip().split())
    if not 2 <= len(normalized) <= 50:
        raise ValueError("Display name must be between 2 and 50 characters.")
    return normalized


def _hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PASSWORD_ITERATIONS
    )
    return f"pbkdf2_sha256${PASSWORD_ITERATIONS}${salt.hex()}${digest.hex()}"


def _verify_password(password: str, encoded: str) -> bool:
    try:
        algorithm, iterations, salt_hex, expected_hex = encoded.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            bytes.fromhex(salt_hex),
            int(iterations),
        )
        return hmac.compare_digest(candidate, bytes.fromhex(expected_hex))
    except (TypeError, ValueError):
        return False


def _account_from_row(row: sqlite3.Row) -> Account:
    return Account(
        id=int(row["id"]),
        email=str(row["email"]),
        display_name=str(row["display_name"]),
        is_admin=bool(row["is_admin"]),
        created_at=str(row["created_at"]),
        authenticator_enabled=bool(row["totp_enabled"]),
        max_active_jobs=int(row["max_active_jobs"]),
    )


def update_max_active_jobs(user_id: int, value: int) -> Account:
    """Persist an account's maximum number of queued plus running jobs."""
    normalized = int(value)
    if not MIN_MAX_ACTIVE_JOBS <= normalized <= MAX_MAX_ACTIVE_JOBS:
        raise ValueError(
            f"Concurrent job limit must be between {MIN_MAX_ACTIVE_JOBS} and {MAX_MAX_ACTIVE_JOBS}."
        )
    with _auth_connection() as connection:
        connection.execute(
            "UPDATE users SET max_active_jobs = ? WHERE id = ?",
            (normalized, user_id),
        )
        row = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, totp_enabled,
                   max_active_jobs
            FROM users WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    if row is None:
        raise ValueError("Account not found.")
    return _account_from_row(row)


def registration_is_open() -> bool:
    """Return whether this installation still needs its first admin account."""
    with _auth_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM users WHERE is_admin = 1 LIMIT 1"
        ).fetchone()
    return row is None


def create_account(email: str, display_name: str, password: str) -> Account:
    """Create the installation's first and only self-service admin account."""
    normalized_email = _normalize_email(email)
    normalized_name = _normalize_display_name(display_name)
    if not 10 <= len(password) <= 256:
        raise ValueError("Password must be between 10 and 256 characters.")
    try:
        with _auth_connection() as connection:
            if connection.execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None:
                raise ValueError("The admin account already exists. Sign in instead.")
            cursor = connection.execute(
                """
                INSERT INTO users (email, display_name, password_hash, is_admin)
                VALUES (?, ?, ?, 1)
                """,
                (normalized_email, normalized_name, _hash_password(password)),
            )
            row = connection.execute(
                """
                SELECT id, email, display_name, is_admin, created_at, totp_enabled,
                       max_active_jobs
                FROM users WHERE id = ?
                """,
                (cursor.lastrowid,),
            ).fetchone()
    except sqlite3.IntegrityError as exc:
        raise ValueError("An account with this email already exists.") from exc
    if row is None:
        raise RuntimeError("Account creation did not return a record.")
    return _account_from_row(row)


def authenticate_account(email: str, password: str) -> Account:
    """Validate an email/password pair without revealing which field failed."""
    try:
        normalized_email = _normalize_email(email)
    except ValueError as exc:
        raise AuthenticationError("Email or password is incorrect.") from exc
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, password_hash,
                   totp_enabled, max_active_jobs
            FROM users WHERE email = ?
            """,
            (normalized_email,),
        ).fetchone()
    if row is None or not _verify_password(password, str(row["password_hash"])):
        raise AuthenticationError("Email or password is incorrect.")
    return _account_from_row(row)


def create_session(user_id: int) -> str:
    """Create a random browser session and persist only its SHA-256 digest."""
    token = secrets.token_urlsafe(48)
    now = datetime.now(timezone.utc)
    expires_at = now + timedelta(days=SESSION_DAYS)
    with _auth_connection() as connection:
        connection.execute(
            "DELETE FROM auth_sessions WHERE expires_at <= ?",
            (now.isoformat(),),
        )
        connection.execute(
            """
            INSERT INTO auth_sessions (user_id, token_hash, created_at, expires_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                user_id,
                hashlib.sha256(token.encode("utf-8")).hexdigest(),
                now.isoformat(),
                expires_at.isoformat(),
            ),
        )
    return token


def account_for_session(token: str | None) -> Account | None:
    """Resolve a valid session token to its safe account record."""
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc)
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT users.id, users.email, users.display_name, users.is_admin,
                   users.created_at, users.totp_enabled, users.max_active_jobs,
                   auth_sessions.expires_at
            FROM auth_sessions
            JOIN users ON users.id = auth_sessions.user_id
            WHERE auth_sessions.token_hash = ?
            """,
            (token_hash,),
        ).fetchone()
        if row is None:
            return None
        try:
            expires_at = datetime.fromisoformat(str(row["expires_at"]))
        except ValueError:
            expires_at = now - timedelta(seconds=1)
        if expires_at <= now:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
            )
            return None
    return _account_from_row(row)


def workspace_for_session(token: str | None) -> str | None:
    """Return the creator workspace selected for one valid browser session."""
    if not token:
        return None
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT active_workspace FROM auth_sessions
            WHERE token_hash = ? AND expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()
    if row is None or row["active_workspace"] not in SESSION_WORKSPACES:
        return None
    return str(row["active_workspace"])


def set_session_workspace(token: str | None, workspace: str) -> bool:
    """Set or switch the creator workspace for a valid browser session."""
    normalized = workspace.strip().lower()
    if normalized not in SESSION_WORKSPACES:
        raise ValueError("Choose one of the available workspaces.")
    if not token:
        return False
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    now = datetime.now(timezone.utc).isoformat()
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT active_workspace FROM auth_sessions
            WHERE token_hash = ? AND expires_at > ?
            """,
            (token_hash, now),
        ).fetchone()
        if row is None:
            return False
        connection.execute(
            "UPDATE auth_sessions SET active_workspace = ? WHERE token_hash = ?",
            (normalized, token_hash),
        )
    return True


def authenticator_account() -> Account | None:
    """Return the sole local admin when passwordless authenticator login is enabled."""
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, totp_enabled,
                   max_active_jobs
            FROM users
            WHERE is_admin = 1 AND totp_enabled = 1
            ORDER BY id LIMIT 1
            """
        ).fetchone()
    return _account_from_row(row) if row is not None else None


def _decode_totp_secret(secret: str) -> bytes:
    padded = secret + "=" * ((8 - len(secret) % 8) % 8)
    return base64.b32decode(padded, casefold=True)


def totp_code(secret: str, counter: int, *, digits: int = TOTP_DIGITS) -> str:
    """Create an RFC 4226 HOTP value for the supplied TOTP time counter."""
    digest = hmac.new(
        _decode_totp_secret(secret), struct.pack(">Q", counter), hashlib.sha1
    ).digest()
    offset = digest[-1] & 0x0F
    value = struct.unpack(">I", digest[offset : offset + 4])[0] & 0x7FFFFFFF
    return str(value % (10**digits)).zfill(digits)


def _matching_totp_counter(
    secret: str, code: str, *, timestamp: float | None = None
) -> int | None:
    normalized = code.strip()
    if not re.fullmatch(r"\d{6}", normalized):
        return None
    current = int((time.time() if timestamp is None else timestamp) // TOTP_PERIOD)
    for counter in (current, current - 1, current + 1):
        if hmac.compare_digest(totp_code(secret, counter), normalized):
            return counter
    return None


def _new_totp_secret() -> str:
    return base64.b32encode(secrets.token_bytes(20)).decode("ascii").rstrip("=")


def _normalize_recovery_code(code: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def _recovery_code_hash(user_id: int, code: str) -> str:
    normalized = _normalize_recovery_code(code)
    return hashlib.sha256(f"{user_id}:{normalized}".encode("utf-8")).hexdigest()


def begin_authenticator_setup(user_id: int, password: str) -> tuple[str, Account]:
    """Password-confirm and create a pending secret without replacing a live one."""
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, password_hash,
                   totp_enabled, max_active_jobs
            FROM users WHERE id = ? AND is_admin = 1
            """,
            (user_id,),
        ).fetchone()
        if row is None or not _verify_password(password, str(row["password_hash"])):
            raise AuthenticationError("Password is incorrect.")
        secret = _new_totp_secret()
        connection.execute(
            "UPDATE users SET totp_pending_secret = ? WHERE id = ?",
            (secret, user_id),
        )
    return secret, _account_from_row(row)


def confirm_authenticator_setup(
    user_id: int, code: str, *, timestamp: float | None = None
) -> tuple[Account, list[str]]:
    """Activate a pending secret and mint one-time recovery codes."""
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, totp_enabled,
                   max_active_jobs, totp_pending_secret
            FROM users WHERE id = ? AND is_admin = 1
            """,
            (user_id,),
        ).fetchone()
        if row is None or not row["totp_pending_secret"]:
            raise AuthenticationError("Start authenticator setup first.")
        secret = str(row["totp_pending_secret"])
        counter = _matching_totp_counter(secret, code, timestamp=timestamp)
        if counter is None:
            raise AuthenticationError("The authenticator code is incorrect or expired.")
        connection.execute(
            """
            UPDATE users
            SET totp_secret = ?, totp_pending_secret = NULL, totp_enabled = 1,
                totp_last_counter = ?, totp_failed_attempts = 0,
                totp_locked_until = NULL
            WHERE id = ?
            """,
            (secret, counter, user_id),
        )
        connection.execute("DELETE FROM auth_recovery_codes WHERE user_id = ?", (user_id,))
        recovery_codes = []
        now = datetime.now(timezone.utc).isoformat()
        for _ in range(RECOVERY_CODE_COUNT):
            raw = secrets.token_hex(8).upper()
            formatted = "-".join(raw[index : index + 4] for index in range(0, 16, 4))
            recovery_codes.append(formatted)
            connection.execute(
                """
                INSERT INTO auth_recovery_codes (user_id, code_hash, created_at)
                VALUES (?, ?, ?)
                """,
                (user_id, _recovery_code_hash(user_id, formatted), now),
            )
        updated = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, totp_enabled,
                   max_active_jobs
            FROM users WHERE id = ?
            """,
            (user_id,),
        ).fetchone()
    if updated is None:
        raise RuntimeError("Authenticator activation did not return an account.")
    return _account_from_row(updated), recovery_codes


def authenticate_authenticator(
    code: str, *, timestamp: float | None = None
) -> Account:
    """Authenticate the local admin using a TOTP or unused recovery code."""
    now = datetime.now(timezone.utc)
    invalid = False
    with _auth_connection() as connection:
        row = connection.execute(
            """
            SELECT id, email, display_name, is_admin, created_at, totp_enabled,
                   max_active_jobs, totp_secret, totp_last_counter, totp_failed_attempts,
                   totp_locked_until
            FROM users
            WHERE is_admin = 1 AND totp_enabled = 1
            ORDER BY id LIMIT 1
            """
        ).fetchone()
        if row is None:
            raise AuthenticationError("Authenticator sign-in is not configured.")
        locked_until = None
        if row["totp_locked_until"]:
            try:
                locked_until = datetime.fromisoformat(str(row["totp_locked_until"]))
            except ValueError:
                locked_until = None
        if locked_until is not None and locked_until > now:
            raise AuthenticationRateLimitError(
                "Too many attempts. Wait one minute and try again."
            )

        matched_counter = _matching_totp_counter(
            str(row["totp_secret"]), code, timestamp=timestamp
        )
        last_counter = row["totp_last_counter"]
        totp_valid = matched_counter is not None and (
            last_counter is None or matched_counter > int(last_counter)
        )
        recovery_row = None
        if not totp_valid and len(_normalize_recovery_code(code)) == 16:
            recovery_row = connection.execute(
                """
                SELECT id FROM auth_recovery_codes
                WHERE user_id = ? AND code_hash = ? AND used_at IS NULL
                """,
                (int(row["id"]), _recovery_code_hash(int(row["id"]), code)),
            ).fetchone()

        if not totp_valid and recovery_row is None:
            failures = int(row["totp_failed_attempts"] or 0) + 1
            lock_value = None
            if failures >= AUTHENTICATOR_MAX_FAILURES:
                failures = 0
                lock_value = (now + timedelta(seconds=AUTHENTICATOR_LOCK_SECONDS)).isoformat()
            connection.execute(
                """
                UPDATE users SET totp_failed_attempts = ?, totp_locked_until = ?
                WHERE id = ?
                """,
                (failures, lock_value, int(row["id"])),
            )
            invalid = True
        else:
            if recovery_row is not None:
                connection.execute(
                    "UPDATE auth_recovery_codes SET used_at = ? WHERE id = ? AND used_at IS NULL",
                    (now.isoformat(), int(recovery_row["id"])),
                )
            connection.execute(
                """
                UPDATE users
                SET totp_last_counter = COALESCE(?, totp_last_counter),
                    totp_failed_attempts = 0, totp_locked_until = NULL
                WHERE id = ?
                """,
                (matched_counter if totp_valid else None, int(row["id"])),
            )
    if invalid:
        raise AuthenticationError("The authenticator code is incorrect or expired.")
    return _account_from_row(row)


def revoke_session(token: str | None) -> None:
    """Revoke one session without affecting the user's other devices."""
    if not token:
        return
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _auth_connection() as connection:
        connection.execute(
            "DELETE FROM auth_sessions WHERE token_hash = ?", (token_hash,)
        )


def create_generation_access(generation_id: int) -> str:
    """Create a capability token for one result generated by a guest."""
    token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _auth_connection() as connection:
        connection.execute(
            """
            INSERT INTO generation_access_tokens (generation_id, token_hash, created_at)
            VALUES (?, ?, ?)
            ON CONFLICT(generation_id) DO UPDATE SET
                token_hash = excluded.token_hash,
                created_at = excluded.created_at
            """,
            (generation_id, token_hash, datetime.now(timezone.utc).isoformat()),
        )
    return token


def verify_generation_access(generation_id: int, token: str | None) -> bool:
    """Validate a guest capability without persisting its raw value."""
    if not token:
        return False
    candidate_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _auth_connection() as connection:
        row = connection.execute(
            "SELECT token_hash FROM generation_access_tokens WHERE generation_id = ?",
            (generation_id,),
        ).fetchone()
    return row is not None and hmac.compare_digest(
        candidate_hash, str(row["token_hash"])
    )


def create_job_access() -> tuple[str, str]:
    """Create a raw guest capability and the hash safe to persist with a job."""
    token = secrets.token_urlsafe(32)
    return token, hashlib.sha256(token.encode("utf-8")).hexdigest()


def verify_job_access(job_id: int, token: str | None) -> bool:
    """Validate a guest capability against one durable job."""
    if not token:
        return False
    candidate_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _auth_connection() as connection:
        row = connection.execute(
            "SELECT access_token_hash FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return row is not None and row["access_token_hash"] is not None and hmac.compare_digest(
        candidate_hash, str(row["access_token_hash"])
    )


def verify_job_generation_access(generation_id: int, token: str | None) -> bool:
    """Allow a job capability to access the artifact produced by that job."""
    if not token:
        return False
    candidate_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
    with _auth_connection() as connection:
        rows = connection.execute(
            "SELECT access_token_hash FROM jobs WHERE generation_id = ?",
            (generation_id,),
        ).fetchall()
    return any(
        row["access_token_hash"] is not None
        and hmac.compare_digest(candidate_hash, str(row["access_token_hash"]))
        for row in rows
    )


def account_can_access_job_generation(account_id: int, generation_id: int) -> bool:
    """Return whether an authenticated account submitted the artifact's job."""
    with _auth_connection() as connection:
        row = connection.execute(
            "SELECT 1 FROM jobs WHERE account_id = ? AND generation_id = ? LIMIT 1",
            (account_id, generation_id),
        ).fetchone()
    return row is not None

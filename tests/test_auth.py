"""Tests for durable local accounts and revocable browser sessions."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException, Request
from fastapi.testclient import TestClient

from backend import auth
from backend.main import _admin_account, _require_safe_rating, _require_workspace, app


class AuthenticationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database_path = Path(self.temporary_directory.name) / "auth.db"
        self.path_patch = patch.object(auth, "DATABASE_PATH", self.database_path)
        self.iteration_patch = patch.object(auth, "PASSWORD_ITERATIONS", 10_000)
        self.path_patch.start()
        self.iteration_patch.start()
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.execute(
                "CREATE TABLE generations (id INTEGER PRIMARY KEY)"
            )
            connection.commit()
        auth.initialize_auth_database()

    def tearDown(self) -> None:
        self.iteration_patch.stop()
        self.path_patch.stop()
        self.temporary_directory.cleanup()

    def test_account_password_and_session_lifecycle(self) -> None:
        account = auth.create_account(
            " Creator@Example.com ", "  Image   Creator  ", "long-password"
        )

        authenticated = auth.authenticate_account(
            "creator@example.com", "long-password"
        )
        self.assertEqual(authenticated.id, account.id)
        self.assertEqual(authenticated.email, "creator@example.com")
        self.assertEqual(authenticated.display_name, "Image Creator")
        self.assertTrue(authenticated.is_admin)
        self.assertFalse(auth.registration_is_open())

        token = auth.create_session(account.id)
        self.assertNotIn("long-password", self.database_path.read_bytes().decode("latin1"))
        self.assertEqual(auth.account_for_session(token), authenticated)

        auth.revoke_session(token)
        self.assertIsNone(auth.account_for_session(token))

    def test_duplicate_account_and_bad_password_are_rejected(self) -> None:
        auth.create_account("creator@example.com", "Creator", "long-password")

        with self.assertRaisesRegex(ValueError, "already exists"):
            auth.create_account(
                "CREATOR@example.com", "Other Creator", "other-password"
            )
        with self.assertRaisesRegex(
            auth.AuthenticationError, "Email or password is incorrect"
        ):
            auth.authenticate_account("creator@example.com", "wrong-password")

    def test_queue_limit_is_persisted_with_the_account(self) -> None:
        account = auth.create_account(
            "creator@example.com", "Creator", "long-password"
        )
        self.assertEqual(account.max_active_jobs, 3)

        updated = auth.update_max_active_jobs(account.id, 6)
        self.assertEqual(updated.max_active_jobs, 6)

        token = auth.create_session(account.id)
        restored = auth.account_for_session(token)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.max_active_jobs, 6)

        ceiling = auth.MAX_MAX_ACTIVE_JOBS
        with self.assertRaisesRegex(ValueError, f"between 1 and {ceiling}"):
            auth.update_max_active_jobs(account.id, ceiling + 1)

    def test_login_rotates_an_existing_browser_session(self) -> None:
        client = TestClient(app)
        registered = client.post(
            "/api/auth/register",
            json={
                "display_name": "Creator",
                "email": "creator@example.com",
                "password": "long-password",
            },
        )
        self.assertEqual(registered.status_code, 200)
        self.assertTrue(registered.json()["boot_id"])
        original_token = client.cookies.get("aiqg_session")
        self.assertIsNotNone(original_token)
        self.assertIsNotNone(auth.account_for_session(original_token))

        logged_in = client.post(
            "/api/auth/login",
            json={"email": "creator@example.com", "password": "long-password"},
        )
        self.assertEqual(logged_in.status_code, 200)
        self.assertEqual(logged_in.json()["boot_id"], registered.json()["boot_id"])
        rotated_token = client.cookies.get("aiqg_session")
        self.assertNotEqual(rotated_token, original_token)
        self.assertIsNone(auth.account_for_session(original_token))
        self.assertIsNotNone(auth.account_for_session(rotated_token))

    def test_creator_workspace_is_required_and_switchable_for_one_session(self) -> None:
        client = TestClient(app)
        registered = client.post(
            "/api/auth/register",
            json={
                "display_name": "Creator",
                "email": "creator@example.com",
                "password": "long-password",
            },
        )
        self.assertEqual(registered.status_code, 200)
        self.assertIsNone(registered.json()["active_workspace"])
        token = client.cookies.get("aiqg_session")

        unselected_request = Request(
            {
                "type": "http",
                "headers": [(b"cookie", f"aiqg_session={token}".encode("ascii"))],
            }
        )
        with self.assertRaises(HTTPException) as unselected:
            _require_workspace(unselected_request, "forgeai")
        self.assertEqual(unselected.exception.status_code, 409)

        selected = client.post(
            "/api/auth/workspace", json={"workspace": "forgeimg"}
        )
        self.assertEqual(selected.status_code, 200)
        self.assertEqual(selected.json()["active_workspace"], "forgeimg")
        self.assertEqual(auth.workspace_for_session(token), "forgeimg")
        self.assertEqual(
            client.post(
                "/api/auth/workspace", json={"workspace": "forgeimg"}
            ).status_code,
            200,
        )
        switched = client.post(
            "/api/auth/workspace", json={"workspace": "forgeai"}
        )
        self.assertEqual(switched.status_code, 200)
        self.assertEqual(switched.json()["active_workspace"], "forgeai")
        self.assertEqual(auth.workspace_for_session(token), "forgeai")

        with self.assertRaises(HTTPException) as wrong_workspace:
            _require_workspace(unselected_request, "forgeimg")
        self.assertEqual(wrong_workspace.exception.status_code, 403)
        self.assertIn("Switch to ForgeIMG", str(wrong_workspace.exception.detail))
        self.assertEqual(_require_workspace(unselected_request, "forgeai").id, 1)
        # Shared operation state stays visible while another creator workspace
        # is selected; only batch creation and mutation are workspace-gated.
        with patch("backend.main.recent_batches", return_value=[]):
            self.assertEqual(client.get("/api/batches").status_code, 200)

        logged_in = client.post(
            "/api/auth/login",
            json={"email": "creator@example.com", "password": "long-password"},
        )
        self.assertEqual(logged_in.status_code, 200)
        self.assertIsNone(logged_in.json()["active_workspace"])
        self.assertIsNone(
            auth.workspace_for_session(client.cookies.get("aiqg_session"))
        )

    def test_totp_matches_rfc_6238_sha1_vectors(self) -> None:
        secret = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
        vectors = (
            (59, "94287082"),
            (1_111_111_109, "07081804"),
            (1_111_111_111, "14050471"),
            (1_234_567_890, "89005924"),
            (2_000_000_000, "69279037"),
        )
        for timestamp, expected in vectors:
            with self.subTest(timestamp=timestamp):
                self.assertEqual(
                    auth.totp_code(secret, timestamp // 30, digits=8), expected
                )

    def test_authenticator_enrollment_and_passwordless_login(self) -> None:
        client = TestClient(app)
        registered = client.post(
            "/api/auth/register",
            json={
                "display_name": "Creator",
                "email": "creator@example.com",
                "password": "long-password",
            },
        )
        self.assertEqual(registered.status_code, 200)

        setup = client.post(
            "/api/auth/authenticator/setup", json={"password": "long-password"}
        )
        self.assertEqual(setup.status_code, 200)
        setup_body = setup.json()
        self.assertTrue(setup_body["qr_data_url"].startswith("data:image/png;base64,"))
        self.assertIn("otpauth://totp/NyxForge%3Acreator%40example.com", setup_body["otpauth_uri"])

        enrollment_time = int(auth.time.time())
        enrollment_code = auth.totp_code(
            setup_body["manual_secret"], enrollment_time // auth.TOTP_PERIOD
        )
        with patch.object(auth.time, "time", return_value=enrollment_time):
            confirmed = client.post(
                "/api/auth/authenticator/confirm", json={"code": enrollment_code}
            )
        self.assertEqual(confirmed.status_code, 200)
        self.assertTrue(confirmed.json()["user"]["authenticator_enabled"])
        self.assertEqual(len(confirmed.json()["recovery_codes"]), 8)

        logged_out = client.post("/api/auth/logout")
        self.assertTrue(logged_out.json()["authenticator_available"])
        self.assertEqual(logged_out.json()["authenticator_display_name"], "Creator")

        # A browser that has never signed in here must not learn that an
        # authenticator-enrolled admin exists, nor their display name.
        stranger = TestClient(app)
        body = stranger.get("/api/auth/session").json()
        self.assertFalse(body["authenticator_available"])
        self.assertIsNone(body["authenticator_display_name"])
        self.assertIsNone(body["user"])

        with patch.object(auth.time, "time", return_value=enrollment_time):
            replayed = client.post(
                "/api/auth/authenticator/login", json={"code": enrollment_code}
            )
        self.assertEqual(replayed.status_code, 401)

        login_time = enrollment_time + auth.TOTP_PERIOD
        login_code = auth.totp_code(
            setup_body["manual_secret"], login_time // auth.TOTP_PERIOD
        )
        with patch.object(auth.time, "time", return_value=login_time):
            login = client.post(
                "/api/auth/authenticator/login", json={"code": login_code}
            )
        self.assertEqual(login.status_code, 200)
        self.assertTrue(login.json()["authenticated"])

        client.post("/api/auth/logout")
        recovery_code = confirmed.json()["recovery_codes"][0]
        recovered = client.post(
            "/api/auth/authenticator/login", json={"code": recovery_code}
        )
        self.assertEqual(recovered.status_code, 200)
        client.post("/api/auth/logout")
        reused = client.post(
            "/api/auth/authenticator/login", json={"code": recovery_code}
        )
        self.assertEqual(reused.status_code, 401)

    def test_authenticator_attempts_are_rate_limited(self) -> None:
        account = auth.create_account(
            "creator@example.com", "Creator", "long-password"
        )
        secret, _ = auth.begin_authenticator_setup(account.id, "long-password")
        timestamp = 1_800_000_000
        auth.confirm_authenticator_setup(
            account.id,
            auth.totp_code(secret, timestamp // auth.TOTP_PERIOD),
            timestamp=timestamp,
        )
        for _ in range(auth.AUTHENTICATOR_MAX_FAILURES):
            with self.assertRaises(auth.AuthenticationError):
                auth.authenticate_authenticator("not-a-code", timestamp=timestamp + 30)
        with self.assertRaises(auth.AuthenticationRateLimitError):
            auth.authenticate_authenticator("not-a-code", timestamp=timestamp + 30)

    def test_public_generation_rejects_non_safe_ratings(self) -> None:
        guest_request = Request({"type": "http", "headers": []})
        _require_safe_rating(guest_request, "Safe")
        with self.assertRaises(HTTPException) as raised:
            _require_safe_rating(guest_request, "Unsupported")
        self.assertEqual(raised.exception.status_code, 422)

    def test_guest_image_capability_is_scoped_to_one_generation(self) -> None:
        with closing(sqlite3.connect(self.database_path)) as connection:
            connection.executemany(
                "INSERT INTO generations (id) VALUES (?)", [(41,), (42,)]
            )
            connection.commit()
        token = auth.create_generation_access(41)

        self.assertTrue(auth.verify_generation_access(41, token))
        self.assertFalse(auth.verify_generation_access(42, token))
        self.assertFalse(auth.verify_generation_access(41, "wrong-token"))

    def test_admin_only_endpoints_reject_anonymous_requests(self) -> None:
        client = TestClient(app)
        self.assertEqual(client.get("/api/history").status_code, 403)
        self.assertEqual(client.get("/api/analytics").status_code, 403)
        self.assertEqual(
            client.post(
                "/api/cloud/test",
                json={"provider": "deepseek", "api_key": "not-a-real-key"},
            ).status_code,
            403,
        )


if __name__ == "__main__":
    unittest.main()

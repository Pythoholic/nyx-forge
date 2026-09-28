"""Suite-wide safeguards for local service integrations."""

from __future__ import annotations

from unittest.mock import patch

import pytest


@pytest.fixture(autouse=True)
def prevent_live_backend_release():
    """Endpoint tests must not unload checkpoints from running local Forge instances."""
    with patch("backend.main.release_other_backends"):
        yield

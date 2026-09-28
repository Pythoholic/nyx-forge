"""Compatibility ASGI entry point for NyxForge.

Run with: ``python -m uvicorn app:app --reload``.
"""

from backend.main import app

__all__ = ["app"]

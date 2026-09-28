# NyxForge agent guide

Use this file as the shared repository guidance for Codex, Claude Code, GitHub
Copilot, and other coding agents. The extended onboarding guide and reusable
setup tools live in [`ai-agents/`](ai-agents/README.md).

## Project boundaries

- NyxForge is a local, single-user FastAPI and React application for
  Stable Diffusion WebUI Forge/reForge and optional ComfyUI backends.
- The supported public setup path is Windows 10 or 11. Keep platform-specific
  behavior working in PowerShell and Git Bash.
- Forge, ComfyUI, checkpoints, GPU drivers, and generated media are external
  to this repository. Never download models or accept third-party licenses on
  a user's behalf.
- Keep the app bound to `127.0.0.1`. Do not expose it to a public interface.

## Repository map

- `backend/`: FastAPI APIs, prompt engine, persistence, runners, and services.
- `frontend/`: Vite and React interface.
- `tests/`: Python test suite.
- `scripts/`: maintainer utilities and prompt preflight tooling.
- `ai-agents/`: agent-safe setup and diagnostic tools.
- `run.py`: supported application and backend-manager command entry point.

## Setup and verification

For a local checkout, prefer the idempotent setup helper:

```bash
python ai-agents/setup.py --dev
```

On Windows, use the virtual environment interpreter after setup:

```bash
./.venv/Scripts/python.exe -m pytest
./.venv/Scripts/python.exe scripts/preflight.py
cd frontend && npm run build
```

Run focused tests while iterating, then the full Python suite and frontend
build before handing off a release-facing change. Do not rewrite golden hashes
unless the prompt change is intentional and the new output was reviewed.

## Safety and privacy

- Never inspect, copy, summarize, or commit user media, uploads, `prompts.db`,
  `.env*`, `storage-config.json`, logs, external storage, or API credentials.
- Do not start, stop, restart, or reconfigure running app/model services unless
  the user explicitly asks. Read-only health checks are acceptable.
- Preserve Safe-rating boundaries, unsafe-content safeguards, localhost defaults,
  and path validation. Add regression tests for safety-sensitive changes.
- Do not commit generated images, videos, model weights, virtual environments,
  dependency folders, databases, or machine-specific configuration.

## Change discipline

- Preserve unrelated user changes and keep commits scoped.
- Prefer existing patterns and pinned dependencies; explain any new production
  dependency.
- Update README documentation when setup, commands, supported models, or public
  behavior changes.
- Treat external services as optional during code-only agent sessions. An agent
  can build and test the repository without a GPU backend, but cannot validate
  an actual generation without the user's local Forge or ComfyUI installation.

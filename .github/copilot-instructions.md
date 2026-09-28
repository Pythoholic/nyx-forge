# NyxForge Copilot instructions

Follow `AGENTS.md` for the complete repository guidance and use
`ai-agents/README.md` for the supported onboarding workflow.

- This is a local FastAPI and React application for Forge/reForge and optional
  ComfyUI backends. The public setup path targets Windows 10 or 11.
- Use `python ai-agents/setup.py --dev` for an idempotent development setup and
  `python ai-agents/doctor.py` for read-only diagnostics.
- Run focused tests while editing, followed by the full Python suite, prompt
  preflight for prompt-engine changes, and `npm run build` in `frontend/`.
- Never access or commit personal media, uploads, databases, `.env*`,
  `storage-config.json`, model weights, or external generation storage.
- Never download checkpoints, accept model licenses, expose the app beyond
  `127.0.0.1`, or start/stop existing services without explicit user approval.
- Preserve Safe-rating boundaries and add regression tests for safety-sensitive
  behavior.

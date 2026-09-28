# 🤖 AI agent support

This directory gives coding agents a safe, repeatable way to understand and
prepare NyxForge without mixing agent-specific files into application code.

The small discovery files at the repository root and under `.github/` must stay
in their standard locations so the corresponding tools can find them:

| Assistant | Discovery file | Role |
| --- | --- | --- |
| 🧠 Codex and compatible agents | `AGENTS.md` | Shared project rules and verification commands |
| ✨ Claude Code | `CLAUDE.md` | Imports the shared `AGENTS.md` instructions |
| 🐙 GitHub Copilot | `.github/copilot-instructions.md` | Copilot-wide repository context |
| ☁️ Copilot cloud agent | `.github/workflows/copilot-setup-steps.yml` | Prepares a Windows coding environment |

## 🚀 Agent-assisted setup

An agent may run the following idempotent command from the repository root:

```bash
python ai-agents/setup.py --dev
```

It creates `.venv`, installs the pinned Python dependencies, installs the exact
frontend dependencies from `package-lock.json`, and builds the frontend. It
does **not** start services, download checkpoints, touch generated media, or
change machine-specific backend configuration.

For runtime-only dependencies, omit `--dev`:

```bash
python ai-agents/setup.py
```

Useful options:

```bash
python ai-agents/setup.py --help
python ai-agents/setup.py --dev --skip-frontend
python ai-agents/setup.py --dev --skip-build
```

## 🩺 Environment doctor

Run the read-only diagnostic at any time:

```bash
python ai-agents/doctor.py
```

The doctor checks required tool versions, the virtual environment, installed
packages, frontend dependencies/build output, and localhost service health. A
missing Forge or ComfyUI service is reported as an optional warning because
code development and tests do not require a GPU backend.

## 💬 Starter prompt

Copy this into Codex, Claude Code, or GitHub Copilot:

```text
Set up NyxForge for local development. Follow AGENTS.md, run the environment
doctor first, and use the repository's agent setup helper. Do not download model
checkpoints, access personal generation data, or start, stop, or reconfigure
existing services without asking me. Report anything I still need to install.
```

For a normal user installation, replace `local development` with `local use`;
the agent should then install runtime rather than development dependencies.

## 🔐 Safety boundaries

Agents can safely automate repository dependencies, builds, tests, and
diagnostics. They must leave these user-controlled:

- Forge/reForge, Forge Neo, and ComfyUI installation or upgrades
- GPU drivers and acceleration libraries
- checkpoint selection, downloads, and third-party license acceptance
- cloud-provider API keys and private configuration
- generated images, videos, uploads, prompt history, and external storage
- starting or stopping services that may already be in use

Cloud agents can edit, build, and test the source tree, but they cannot validate
real image or video generation without the user's local GPU stack and models.

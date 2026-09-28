# Contributing to NyxForge

Thank you for helping improve NyxForge.

## Before opening a pull request

1. Read `AGENTS.md` and `RESPONSIBLE_USE.md`.
2. Keep all generation, transformation, API, model, data, documentation, and
   test changes inside the Safe-only public boundary.
3. Do not commit generated media, uploads, credentials, model checkpoints,
   runtime databases, virtual environments, dependency folders, or build output.
4. Add focused tests for behavior changes.
5. Run:

```bash
python -m pytest -q
cd frontend && npm ci && npm run build
```

## Pull requests

Explain the user-visible outcome, important implementation choices, and the
verification you ran. Keep unrelated refactors separate. Use synthetic test
data and Safe screenshots only.

By contributing, you agree that your contribution is licensed under the
Apache License 2.0.

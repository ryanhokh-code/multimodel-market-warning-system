# EWS — Market Early Warning System

Multi-model, multi-market risk overlay. See `docs/ARCHITECTURE.md` for the
full pipeline diagram and per-module design rationale.

## Layout

- `src/ews/` — the package. `pip install -e .` from this directory.
- `config/` — registry YAML (models, markets, calibrated cells).
- `scripts/` — thin entrypoints for the daily DAG and periodic recalibration.
- `tests/` — pytest suite; run `pytest tests/`.
- `artifacts/` — persisted meta-model artifacts and registry snapshots
  (gitignored in practice; kept here as the expected local path).
- `docs/` — architecture documentation.

## Quickstart

```bash
pip install -e .
pytest tests/
python scripts/run_daily.py
```

## Status

Signal implementations under `src/ews/signals/` (`lppls.py`, `wavelet.py`,
`mfdfa.py`) are stubs — wire in the existing debugged prototypes behind the
`WarningSignal` interface in `src/ews/signals/base.py`. Everything else
(registry, orchestration, walk-forward optimizer, meta-model aggregation)
has been tested against synthetic data; see `docs/ARCHITECTURE.md` for
what's verified vs. what's still open.

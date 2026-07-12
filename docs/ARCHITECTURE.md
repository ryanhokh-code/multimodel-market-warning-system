# Early Warning System (EWS) — Framework Architecture

Multi-model, multi-market risk overlay: **K** signal families (LPPLS, wavelet
MODWT, MF-DFA, ...) scored against **N** markets (indices, Barra factor
long-short books, ...) every day, aggregated into a per-market and global
warning level.

This document maps the modules built so far to their place in that pipeline.
Code lives alongside this file; nothing here is meant to stand alone from it.

---

## 1. Pipeline overview

```mermaid
flowchart TB
    subgraph Registry["registry_schema.py + registry_config.yaml"]
        R[("Model / Market / Cell registry<br/>hyperparams, active flags, IC stats")]
    end

    subgraph Offline["Offline / periodic jobs"]
        WF["walkforward_optimizer.py<br/>nested walk-forward CV per cell<br/>+ FDR correction across the grid"]
        MM_TRAIN["meta_model.py (training)<br/>pooled classifier, walk-forward + embargo<br/>+ isotonic calibration"]
    end

    subgraph Daily["Daily orchestration — orchestrator.py"]
        SCORE["Fan-out scoring<br/>K x N cells, process pool<br/>isolated per-cell failure"]
        NORM["Normalization<br/>time-series z-score<br/>cross-sectional z-score"]
        AGG1["Per-market aggregation<br/>IC-weighted composite<br/>(or meta-model, see below)"]
        AGG2["Global aggregation<br/>mean + breadth %"]
        DISC["Discretization<br/>hysteresis-based levels<br/>GREEN -> YELLOW -> ORANGE -> RED"]
    end

    subgraph Interface["signal_interface.py"]
        WS["WarningSignal ABC<br/>fit() / score()<br/>implemented per model family"]
    end

    R -->|hyperparams per cell| SCORE
    WS -->|instantiated per (model, market)| SCORE
    SCORE --> NORM --> AGG1 --> AGG2 --> DISC
    MM_TRAIN -->|calibrated artifact| AGG1
    WF -->|writes back hyperparams, IC, active flag| R
    NORM -.->|persisted history| MM_TRAIN
    NORM -.->|persisted history| WF
```

**Two clocks in this system:**
- **Daily**: `orchestrator.py` reads the registry, never writes it.
- **Periodic** (weekly/monthly per model, see `recalibration_frequency`):
  `walkforward_optimizer.py` and `meta_model.py`'s training path are the
  only things that write to the registry / produce a new model artifact.

---

## 2. Project structure

```
ews/
├── artifacts
│   ├── meta_models            # persisted MetaModelArtifact objects, versioned by trained_on date
│   └── registry_snapshots     # point-in-time copies of registry_config.yaml (audit trail)
├── config
│   └── registry_config.example.yaml
├── docs
│   └── ARCHITECTURE.md
├── scripts
│   ├── run_daily.py           # thin entrypoint -> ews.orchestration.daily_pipeline
│   └── run_walkforward.py     # thin entrypoint -> ews.optimization.walkforward
├── src
│   └── ews
│       ├── aggregation
│       │   ├── __init__.py
│       │   └── meta_model.py          # stage-2 aggregation: pooled classifier -> P(drawdown)
│       ├── data
│       │   └── __init__.py            # DataAccessLayer implementations go here
│       ├── optimization
│       │   ├── __init__.py
│       │   └── walkforward.py         # nested walk-forward CV, FDR correction, registry write-back
│       ├── orchestration
│       │   ├── __init__.py
│       │   └── daily_pipeline.py      # fan-out scoring -> normalize -> aggregate -> discretize
│       ├── registry
│       │   ├── __init__.py
│       │   └── schema.py              # RegistryCell / ModelConfig / MarketConfig (Pydantic)
│       ├── signals
│       │   ├── __init__.py
│       │   ├── base.py                # WarningSignal ABC
│       │   ├── lppls.py               # stub -- wire in existing LPPLS prototype
│       │   ├── mfdfa.py               # stub -- wire in existing MF-DFA prototype
│       │   └── wavelet.py             # stub -- wire in existing MODWT prototype
│       └── __init__.py
├── tests
│   ├── __init__.py
│   ├── test_orchestration.py  # hysteresis boundary behavior
│   ├── test_registry.py       # active_cells() pruning
│   └── test_walkforward.py    # embargo gap, BH correction
├── README.md
└── pyproject.toml
```

**Layout notes:**
- `src/` layout (not a flat package at repo root) so `pip install -e .`
  can't accidentally import from the working directory instead of the
  installed package — a common source of "works on my machine" bugs.
- `signals/base.py` is the only file the other four packages
  (`registry`, `orchestration`, `optimization`, `aggregation`) depend on
  in that direction; nothing in `signals/` imports from them. Adding a
  new model family means adding one file under `signals/` and one entry
  in `config/registry_config.yaml` — no other package needs to change.
- `data/` is currently just the `__init__.py` — this is where
  `DataAccessLayer` (stubbed inline in `daily_pipeline.py` and
  `walkforward.py` today) belongs once it's wired to a real vendor/Barra
  source, so both the daily and periodic jobs can share one
  implementation instead of two.
- `artifacts/` is a local working path, not something to commit — the
  registry snapshot subfolder is what gives you an audit trail of
  exactly which hyperparameters were live on any given date, independent
  of git history on `config/registry_config.yaml`.

---

## 3. Module reference

### `registry_schema.py`
Single source of truth for which `(model, market)` cells exist, whether
they're active, and their current calibrated hyperparameters.

- `ModelConfig` — one entry per signal family (LPPLS, MODWT, MF-DFA, ...):
  class path, default hyperparams, recalibration cadence.
- `MarketConfig` — one entry per market (SPX, NKY, Barra factor books, ...):
  data source, minimum history requirement.
- `RegistryCell` — one `(model_id, market_id)` pairing: hyperparams,
  `last_calibration_date`, `last_validation_ic` / `_pvalue`, `active` flag.
- `RegistryConfig.active_cells()` — the daily pipeline's only entry point
  into the registry; everything inactive (pruned by the validity review)
  is invisible downstream without touching model/market definitions.
- `load_registry()` / `save_registry()` — YAML in, Pydantic-validated out;
  fails fast on a malformed config rather than mid-DAG.

### `registry_config.example.yaml`
Populated example: 3 models × 4 markets = 12 cells, including two already
flagged `active: false` to show the pruning pattern in practice (LPPLS
deactivated on factor long-short books where bubble dynamics don't apply).

### `signal_interface.py`
The `WarningSignal` abstract interface every model family implements, so
the orchestrator treats LPPLS, MODWT confluence, and MF-DFA uniformly.

- `fit(data, as_of)` → `FitResult` (converged flag, fitted params).
- `score(as_of)` → `SignalScore` (raw score, confidence, or a typed
  `FailureReason` — non-convergence, thin history, data quality — instead
  of raising, so one broken cell can't take down the daily matrix build).

### `orchestrator.py`
The daily DAG. Scheduler-agnostic — wrap `run_daily_pipeline()` in
Airflow/Prefect/cron.

1. **Fan-out scoring** — process pool over all active cells; each cell's
   exceptions are caught and converted to a failed `SignalScore` rather
   than crashing the pool.
2. **Normalization** — `NormalizationStore` tracks rolling per-cell
   mean/std for **time-series** z-scores (is today extreme vs. this
   cell's own history); a separate **cross-sectional** z-score answers
   "is this market worse than others, according to model k, today."
3. **Per-market aggregation** — IC-weighted composite by default
   (`aggregate_per_market`); swappable for the trained meta-model without
   changing the function's `market_id -> composite_score` contract.
4. **Global aggregation** — mean across markets + breadth (% of markets
   simultaneously elevated), since correlated stress across markets is
   itself a warning signature.
5. **Discretization** — `discretize_with_hysteresis()`: different enter
   vs. exit thresholds per level, so a score sitting near a boundary
   doesn't flip the alert level back and forth daily. Verified in testing
   that a score between the exit and entry thresholds correctly holds its
   current level rather than dropping immediately.

### `walkforward_optimizer.py`
The job that populates `RegistryCell.hyperparams` / `last_validation_ic`.
Runs on each model's own `recalibration_frequency` — not part of the daily
DAG.

- **Outer walk-forward folds** (`generate_walk_forward_folds`) — expanding
  training window, fixed test window, explicit embargo gap. Verified:
  `test_start == train_end + embargo_days` on every fold, no leakage at
  the boundary.
- **Inner nested CV** (`inner_cv_select_hyperparams`) — purged K-fold
  *within* the outer training window only; the outer test fold is never
  used for both tuning and evaluation.
- **IC decay curve** — each fold scored at multiple horizons (5/21/63
  days), not a single number.
- **Multiple-testing correction** (`benjamini_hochberg`) — applied once
  across every cell tested in a run, not per-model or per-market.
  Verified against a synthetic mix of 5 significant + 15 null p-values —
  recovered exactly the 5.
- **Registry write-back** (`apply_validity_review`) — writes hyperparams
  and IC stats, flips `active` based on FDR-corrected significance, with
  a timestamped audit note.

### `meta_model.py`
Optional replacement for the IC-weighted composite in
`aggregate_per_market`: a trained classifier mapping the K-vector of
normalized scores → P(drawdown event within horizon *h*).

- **Pooled across markets** (not one model per market) — crash events are
  rare; pooling multiplies effective sample size, with `market_id` kept
  as a categorical feature so market-specific base rates are still
  learned.
- **`HistGradientBoostingClassifier`** — handles NaN features natively,
  since a missing cell (LPPLS non-convergence, thin history) is the
  normal daily state, not an edge case.
- **Inverse-frequency sample weighting** for class imbalance instead of
  oversampling, which risks leaking near-duplicate rows across a
  time-series train/test boundary.
- **Nested walk-forward + isotonic calibration** — a trailing slice of
  each training window is held out purely for probability calibration;
  the outer test fold is used only for final OOS evaluation. Verified
  end-to-end on synthetic data (3 models × 3 markets, ~8% missingness,
  ~6% event rate) after fixing a datetime64-vs-`date` comparison bug
  caught during testing.
- **Output is `(probability, confidence)`** — confidence is data
  completeness (`n_contributing / n_total`), so a probability computed
  from 2 of 8 models carries less weight downstream than one from 7 of 8,
  even at an identical point estimate. Verified this behaves
  independently of the probability itself.

---

## 4. What's not built yet

- Meta-model artifact **versioning/persistence** as its own registry
  (analogous to `RegistryCell`, tracking `oos_auc`/`oos_brier` decay over
  time the way `last_validation_ic` does for individual cells).
- Wiring `meta_model.predict_market_composite` into
  `orchestrator.aggregate_per_market` as a live swap.
- System-level backtest of the composite warning level against realized
  drawdown events (precision/recall, lead time distribution) — distinct
  from both the cell-level and meta-model walk-forward validation above.
- Alerting/persistence hooks noted as comments in `run_daily_pipeline`.

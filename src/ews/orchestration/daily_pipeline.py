"""
orchestrator.py

Daily EWS orchestration skeleton. This is scheduler-agnostic -- the
`run_daily_pipeline` function is what you'd wrap in an Airflow/Prefect/
Dagster task, or just cron directly for a v1. The DAG shape is:

    load registry
      -> fan out K x N scoring (parallel, isolated failures)
      -> assemble long-format score matrix
      -> normalize (cross-sectional + time-series)
      -> aggregate to per-market composite (+ meta-model hook)
      -> aggregate to global warning level (+ breadth)
      -> discretize into levels with hysteresis
      -> persist + alert

Design choices worth flagging:
  - Cell failures are isolated: one non-convergent LPPLS fit must not
    block the other K*N - 1 cells. Failures are recorded, not raised.
  - Scoring is parallelized with a process pool since LPPLS multi-start
    fits are CPU-bound; swap for a thread pool / async if your data
    fetch is the bottleneck instead.
  - Normalization state (rolling mean/std per cell) is itself a
    persisted artifact, not recomputed from scratch each day -- see
    NormalizationStore.
"""

from __future__ import annotations

import importlib
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ews.registry.schema import RegistryCell, RegistryConfig, load_registry
from ews.signals.base import FailureReason, SignalScore, WarningSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ews.orchestrator")


# --------------------------------------------------------------------------
# 1. Data access (stub -- wire to your actual data layer)
# --------------------------------------------------------------------------

class DataAccessLayer:
    """Thin wrapper so signals never call vendor APIs directly."""

    def get_price_history(self, data_source: str, as_of: date, lookback_days: int) -> pd.DataFrame:
        raise NotImplementedError("wire to your market data / Barra book source")


# --------------------------------------------------------------------------
# 2. Fan-out scoring
# --------------------------------------------------------------------------

def _instantiate_signal(registry: RegistryConfig, cell: RegistryCell) -> WarningSignal:
    model_cfg = registry.model_by_id(cell.model_id)
    module_path, class_name = model_cfg.class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    hyperparams = {**model_cfg.default_hyperparams, **cell.hyperparams}
    return cls(model_id=cell.model_id, market_id=cell.market_id, hyperparams=hyperparams)


def _score_one_cell(
    registry_path: str, model_id: str, market_id: str, as_of: date
) -> SignalScore:
    """
    Runs in a worker process -- must be self-contained (re-load registry
    rather than pickle the whole RegistryConfig across the process pool).
    Any exception is caught and converted to a failed SignalScore so a
    single bad cell can't crash the pool.
    """
    try:
        registry = load_registry(registry_path)
        cell = registry.cell(model_id, market_id)
        market_cfg = registry.market_by_id(market_id)
        signal = _instantiate_signal(registry, cell)

        dal = DataAccessLayer()
        data = dal.get_price_history(
            market_cfg.data_source, as_of=as_of, lookback_days=signal.min_history_days() + 30
        )

        if len(data) < signal.min_history_days():
            return SignalScore(
                model_id=model_id, market_id=market_id, as_of=as_of,
                raw_score=None, confidence=None,
                failure_reason=FailureReason.INSUFFICIENT_HISTORY,
            )

        fit_result = signal.fit(data, as_of=as_of)
        if not fit_result.converged:
            return SignalScore(
                model_id=model_id, market_id=market_id, as_of=as_of,
                raw_score=None, confidence=None,
                failure_reason=FailureReason.FIT_DID_NOT_CONVERGE,
                diagnostics=fit_result.diagnostics,
            )

        return signal.score(as_of=as_of)

    except Exception:
        logger.exception("Cell scoring failed: model=%s market=%s", model_id, market_id)
        return SignalScore(
            model_id=model_id, market_id=market_id, as_of=as_of,
            raw_score=None, confidence=None,
            failure_reason=FailureReason.DATA_QUALITY,
        )


def build_score_matrix(
    registry_path: str, registry: RegistryConfig, as_of: date, max_workers: int = 8
) -> pd.DataFrame:
    """Returns long-format DataFrame: [model_id, market_id, raw_score, confidence, failure_reason]."""
    cells = registry.active_cells()
    logger.info("Scoring %d active (model, market) cells for %s", len(cells), as_of)

    results: list[SignalScore] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_score_one_cell, registry_path, c.model_id, c.market_id, as_of): c
            for c in cells
        }
        for future in as_completed(futures):
            results.append(future.result())

    df = pd.DataFrame([r.__dict__ for r in results])
    n_failed = df["raw_score"].isna().sum()
    if n_failed:
        logger.warning("%d/%d cells failed to score on %s", n_failed, len(df), as_of)
    return df


# --------------------------------------------------------------------------
# 3. Normalization
# --------------------------------------------------------------------------

class NormalizationStore:
    """
    Persists rolling per-cell mean/std for time-series z-scoring.
    Backing store should be a real DB table in production
    (date, model_id, market_id, rolling_mean, rolling_std); this is an
    in-memory stand-in for the skeleton.
    """

    def __init__(self):
        self._history: dict[tuple[str, str], list[float]] = {}

    def update_and_zscore(self, model_id: str, market_id: str, raw_score: float,
                           window: int = 252) -> float:
        key = (model_id, market_id)
        hist = self._history.setdefault(key, [])
        hist.append(raw_score)
        hist[:] = hist[-window:]
        if len(hist) < 20:  # not enough history for a meaningful z-score yet
            return np.nan
        mu, sigma = np.mean(hist[:-1]), np.std(hist[:-1])
        if sigma == 0:
            return 0.0
        return (raw_score - mu) / sigma


def normalize_matrix(matrix: pd.DataFrame, norm_store: NormalizationStore) -> pd.DataFrame:
    matrix = matrix.copy()

    # time-series z-score: is today extreme relative to this cell's own history
    matrix["ts_zscore"] = matrix.apply(
        lambda r: norm_store.update_and_zscore(r["model_id"], r["market_id"], r["raw_score"])
        if pd.notna(r["raw_score"]) else np.nan,
        axis=1,
    )

    # cross-sectional rank within each model row: is this market worse
    # than others according to model k, today
    matrix["cs_zscore"] = matrix.groupby("model_id")["raw_score"].transform(
        lambda s: (s - s.mean()) / s.std() if s.std() > 0 else 0.0
    )

    return matrix


# --------------------------------------------------------------------------
# 4. Aggregation
# --------------------------------------------------------------------------

@dataclass
class MarketComposite:
    market_id: str
    as_of: date
    composite_score: float
    n_contributing_models: int
    n_failed_models: int


def aggregate_per_market(matrix: pd.DataFrame, registry: RegistryConfig, as_of: date,
                          score_col: str = "ts_zscore") -> list[MarketComposite]:
    """
    IC-weighted composite, matching the weighting pattern used in the
    factor pipeline. Swap in a trained meta-model here later -- keep
    the interface (market_id -> composite_score) the same so downstream
    aggregation/discretization doesn't need to change.
    """
    out = []
    for market_id, grp in matrix.groupby("market_id"):
        weights, scores = [], []
        for _, row in grp.iterrows():
            if pd.isna(row[score_col]):
                continue
            cell = registry.cell(row["model_id"], market_id)
            ic = cell.last_validation_ic if cell and cell.last_validation_ic else 0.0
            w = max(ic, 0.0)  # don't let a negative/noisy IC invert a signal's sign
            weights.append(w)
            scores.append(row[score_col])

        n_failed = grp["raw_score"].isna().sum()
        if not weights or sum(weights) == 0:
            composite = np.nan
        else:
            composite = float(np.average(scores, weights=weights))

        out.append(MarketComposite(
            market_id=market_id, as_of=as_of, composite_score=composite,
            n_contributing_models=len(weights), n_failed_models=int(n_failed),
        ))
    return out


@dataclass
class GlobalWarning:
    as_of: date
    global_score: float
    breadth_pct_elevated: float  # % of markets above a "elevated" z-threshold


def aggregate_global(market_composites: list[MarketComposite],
                      elevated_threshold: float = 1.5) -> GlobalWarning:
    scores = [m.composite_score for m in market_composites if not np.isnan(m.composite_score)]
    if not scores:
        return GlobalWarning(as_of=market_composites[0].as_of, global_score=np.nan, breadth_pct_elevated=np.nan)

    global_score = float(np.mean(scores))  # swap for cap-weight / connectedness-weight
    breadth = float(np.mean([s >= elevated_threshold for s in scores]))
    return GlobalWarning(
        as_of=market_composites[0].as_of, global_score=global_score, breadth_pct_elevated=breadth,
    )


# --------------------------------------------------------------------------
# 5. Discretization with hysteresis
# --------------------------------------------------------------------------

class WarningLevel:
    GREEN, YELLOW, ORANGE, RED = "GREEN", "YELLOW", "ORANGE", "RED"


ENTER_THRESHOLDS = {WarningLevel.YELLOW: 1.0, WarningLevel.ORANGE: 1.75, WarningLevel.RED: 2.5}
EXIT_THRESHOLDS = {WarningLevel.YELLOW: 0.5, WarningLevel.ORANGE: 1.25, WarningLevel.RED: 2.0}


def discretize_with_hysteresis(score: float, previous_level: str) -> str:
    """
    Enter a higher level only above ENTER threshold; only drop back down
    once score falls below the (lower) EXIT threshold for that level.
    Prevents daily flip-flopping around a single cutoff.
    """
    levels = [WarningLevel.GREEN, WarningLevel.YELLOW, WarningLevel.ORANGE, WarningLevel.RED]
    prev_idx = levels.index(previous_level)

    # can we step up?
    for lvl in reversed(levels[1:]):
        if score >= ENTER_THRESHOLDS[lvl] and levels.index(lvl) >= prev_idx:
            return lvl

    # otherwise check whether we've fallen enough to step down
    if prev_idx > 0:
        current_level = levels[prev_idx]
        if score < EXIT_THRESHOLDS.get(current_level, 0):
            return levels[prev_idx - 1]

    return previous_level


# --------------------------------------------------------------------------
# 6. Top-level daily pipeline
# --------------------------------------------------------------------------

def run_daily_pipeline(registry_path: str, as_of: date, norm_store: NormalizationStore,
                        previous_levels: dict[str, str]) -> dict:
    registry = load_registry(registry_path)

    raw_matrix = build_score_matrix(registry_path, registry, as_of)
    norm_matrix = normalize_matrix(raw_matrix, norm_store)

    market_composites = aggregate_per_market(norm_matrix, registry, as_of)
    global_warning = aggregate_global(market_composites)

    levels = {}
    for mc in market_composites:
        prev = previous_levels.get(mc.market_id, WarningLevel.GREEN)
        levels[mc.market_id] = discretize_with_hysteresis(mc.composite_score, prev)

    prev_global = previous_levels.get("__GLOBAL__", WarningLevel.GREEN)
    levels["__GLOBAL__"] = discretize_with_hysteresis(global_warning.global_score, prev_global)

    # persist raw_matrix / norm_matrix / composites / levels to your DB here
    # trigger alerting (email/Slack) on any level upgrade here

    return {
        "as_of": as_of,
        "raw_matrix": raw_matrix,
        "norm_matrix": norm_matrix,
        "market_composites": market_composites,
        "global_warning": global_warning,
        "levels": levels,
    }


if __name__ == "__main__":
    result = run_daily_pipeline(
        registry_path="../../../config/registry_config.example.yaml",
        as_of=date.today(),
        norm_store=NormalizationStore(),
        previous_levels={},
    )
    print(result["levels"])

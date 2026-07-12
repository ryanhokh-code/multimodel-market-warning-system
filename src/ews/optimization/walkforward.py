"""
walkforward_optimizer.py

Populates the registry's per-cell hyperparams / last_validation_ic /
last_validation_ic_pvalue fields (see registry_schema.RegistryCell).

This is the job that runs on each cell's `recalibration_frequency`
cadence (weekly for LPPLS, monthly for MODWT/MF-DFA, etc.) -- NOT part
of the daily scoring DAG. Daily scoring (orchestrator.py) only ever
*reads* the registry; this job is the only thing that *writes* it.

Structure, matching the rigor already applied to the MF-DFA walk-forward
work:
  1. Outer walk-forward folds with an embargo gap between train and test
     (prevents the label lookahead leakage that a plain train/test split
     on serially-correlated financial data invites).
  2. Inner nested CV *within* each outer training window to select
     hyperparams -- so the outer test fold is never used for both
     tuning and evaluation.
  3. IC computed per outer fold at multiple forecast horizons -> decay
     curve, not just a single-horizon number.
  4. Because this job runs across every active (model, market) cell in
     one pass, we are implicitly testing K*N hypotheses simultaneously.
     Benjamini-Hochberg FDR correction is applied across the *whole*
     grid's p-values before any cell is judged significant -- doing the
     correction per-model or per-market would understate the multiple-
     testing problem.
  5. Cells whose corrected significance falls below threshold are
     deactivated (not deleted) -- see `apply_validity_review`.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from ews.registry.schema import RegistryCell, RegistryConfig, load_registry, save_registry
from ews.signals.base import WarningSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ews.walkforward")


# --------------------------------------------------------------------------
# 1. Fold generation
# --------------------------------------------------------------------------

@dataclass
class Fold:
    train_start: date
    train_end: date
    embargo_end: date   # train_end + embargo; test_start begins here
    test_start: date
    test_end: date


def generate_walk_forward_folds(
    history_start: date,
    history_end: date,
    train_window_days: int,
    test_window_days: int,
    embargo_days: int,
    step_days: int | None = None,
) -> list[Fold]:
    """
    Expanding-window walk-forward folds with an embargo gap. The embargo
    matters most for signals with autocorrelated features/labels (e.g.
    forward drawdown labels computed over a multi-day horizon can leak
    into a test fold that starts immediately after training ends) --
    same logic as purged K-fold, applied to a rolling walk-forward
    rather than random folds.
    """
    step_days = step_days or test_window_days
    folds = []
    train_start = history_start
    cursor = train_start + timedelta(days=train_window_days)

    while True:
        train_end = cursor
        embargo_end = train_end + timedelta(days=embargo_days)
        test_start = embargo_end
        test_end = test_start + timedelta(days=test_window_days)
        if test_end > history_end:
            break
        folds.append(Fold(train_start, train_end, embargo_end, test_start, test_end))
        cursor += timedelta(days=step_days)

    return folds


# --------------------------------------------------------------------------
# 2. Labels: forward realized outcome the signal is scored against
# --------------------------------------------------------------------------

def compute_forward_drawdown_label(prices: pd.Series, as_of: date, horizon_days: int) -> float:
    """
    Forward max drawdown over [as_of, as_of + horizon_days], expressed
    as a positive number (larger = worse). A warning signal's raw_score
    should be positively correlated with this -- i.e. IC is computed
    against this label directly, no sign flip needed downstream.
    """
    window = prices.loc[as_of: as_of + timedelta(days=horizon_days)]
    if len(window) < 2:
        return np.nan
    running_max = window.cummax()
    drawdown = (window - running_max) / running_max
    return float(-drawdown.min())  # positive magnitude


# --------------------------------------------------------------------------
# 3. Inner nested CV: hyperparameter search within one outer training window
# --------------------------------------------------------------------------

def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid.keys())
    return [dict(zip(keys, combo)) for combo in product(*grid.values())]


def _instantiate(class_path: str, model_id: str, market_id: str, hyperparams: dict) -> WarningSignal:
    module_path, class_name = class_path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(model_id=model_id, market_id=market_id, hyperparams=hyperparams)


def inner_cv_select_hyperparams(
    class_path: str, model_id: str, market_id: str,
    prices: pd.Series, fold: Fold, hyperparam_grid: dict[str, list],
    n_inner_folds: int, inner_embargo_days: int, horizon_days: int,
) -> dict:
    """
    Purged K-fold *within* fold.train_start..fold.train_end only.
    Never touches fold.test_start..test_end -- that stays held out for
    the outer evaluation. Selects the hyperparam set with the best mean
    inner-fold IC.
    """
    candidates = expand_grid(hyperparam_grid)
    inner_span = (fold.train_end - fold.train_start).days
    inner_fold_len = inner_span // n_inner_folds

    best_params, best_score = None, -np.inf

    for params in candidates:
        inner_ics = []
        for i in range(n_inner_folds - 1):  # last slice reserved as inner test
            inner_train_end = fold.train_start + timedelta(days=inner_fold_len * (i + 1))
            inner_embargo_end = inner_train_end + timedelta(days=inner_embargo_days)
            inner_test_end = inner_embargo_end + timedelta(days=inner_fold_len)
            if inner_test_end > fold.train_end:
                break

            signal = _instantiate(class_path, model_id, market_id, params)
            fit_data = prices.loc[fold.train_start:inner_train_end].to_frame("price")
            fit_result = signal.fit(fit_data, as_of=inner_train_end)
            if not fit_result.converged:
                continue

            scores, labels = [], []
            for d in pd.bdate_range(inner_embargo_end, inner_test_end):
                s = signal.score(as_of=d.date())
                if s.raw_score is None:
                    continue
                lbl = compute_forward_drawdown_label(prices, d.date(), horizon_days)
                if np.isnan(lbl):
                    continue
                scores.append(s.raw_score)
                labels.append(lbl)

            if len(scores) >= 10:
                ic, _ = stats.spearmanr(scores, labels)
                if not np.isnan(ic):
                    inner_ics.append(ic)

        if inner_ics:
            mean_ic = float(np.mean(inner_ics))
            if mean_ic > best_score:
                best_score, best_params = mean_ic, params

    return best_params or hyperparam_grid  # fall back to defaults if nothing converged


# --------------------------------------------------------------------------
# 4. Outer walk-forward evaluation + IC decay curve
# --------------------------------------------------------------------------

@dataclass
class CellWalkForwardResult:
    model_id: str
    market_id: str
    selected_hyperparams: dict
    ic_by_horizon: dict[int, float] = field(default_factory=dict)      # mean OOS IC per horizon
    icir_by_horizon: dict[int, float] = field(default_factory=dict)
    pvalue_by_horizon: dict[int, float] = field(default_factory=dict)  # raw, pre-FDR
    n_oos_observations: int = 0
    primary_horizon: int = 21  # which horizon's stats get written to the registry


def run_cell_walk_forward(
    class_path: str, model_id: str, market_id: str, prices: pd.Series,
    hyperparam_grid: dict[str, list],
    history_start: date, history_end: date,
    train_window_days: int = 750, test_window_days: int = 63,
    embargo_days: int = 21, n_inner_folds: int = 4,
    horizons: tuple[int, ...] = (5, 21, 63),
) -> CellWalkForwardResult:
    folds = generate_walk_forward_folds(
        history_start, history_end, train_window_days, test_window_days, embargo_days
    )
    logger.info("cell=(%s,%s): %d outer folds", model_id, market_id, len(folds))

    oos_ic_by_horizon: dict[int, list[float]] = {h: [] for h in horizons}
    last_selected_params = None

    for fold in folds:
        selected_params = inner_cv_select_hyperparams(
            class_path, model_id, market_id, prices, fold,
            hyperparam_grid, n_inner_folds, embargo_days, horizons[len(horizons) // 2],
        )
        last_selected_params = selected_params

        signal = _instantiate(class_path, model_id, market_id, selected_params)
        fit_data = prices.loc[fold.train_start:fold.train_end].to_frame("price")
        fit_result = signal.fit(fit_data, as_of=fold.train_end)
        if not fit_result.converged:
            continue

        for h in horizons:
            scores, labels = [], []
            for d in pd.bdate_range(fold.test_start, fold.test_end):
                s = signal.score(as_of=d.date())
                if s.raw_score is None:
                    continue
                lbl = compute_forward_drawdown_label(prices, d.date(), h)
                if np.isnan(lbl):
                    continue
                scores.append(s.raw_score)
                labels.append(lbl)
            if len(scores) >= 10:
                ic, _ = stats.spearmanr(scores, labels)
                if not np.isnan(ic):
                    oos_ic_by_horizon[h].append(ic)

    result = CellWalkForwardResult(
        model_id=model_id, market_id=market_id,
        selected_hyperparams=last_selected_params or {},
        primary_horizon=horizons[len(horizons) // 2],
    )
    for h, ics in oos_ic_by_horizon.items():
        if not ics:
            continue
        mean_ic, std_ic = float(np.mean(ics)), float(np.std(ics, ddof=1)) if len(ics) > 1 else np.nan
        icir = mean_ic / std_ic if std_ic and std_ic > 0 else np.nan
        # one-sample t-test on fold-level ICs against zero -- honest about
        # small-n outer folds, this is intentionally conservative
        tstat, pval = stats.ttest_1samp(ics, 0.0) if len(ics) > 1 else (np.nan, 1.0)
        result.ic_by_horizon[h] = mean_ic
        result.icir_by_horizon[h] = icir
        result.pvalue_by_horizon[h] = float(pval)
        result.n_oos_observations += len(ics)

    return result


# --------------------------------------------------------------------------
# 5. Multiple-testing correction across the full K x N grid
# --------------------------------------------------------------------------

def benjamini_hochberg(pvalues: list[float], fdr_level: float = 0.10) -> list[bool]:
    """Returns per-index reject/accept booleans at the given FDR level."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    ranked = np.array(pvalues)[order]
    thresholds = (np.arange(1, n + 1) / n) * fdr_level

    below = ranked <= thresholds
    if not below.any():
        return [False] * n
    max_idx = np.max(np.where(below))  # largest rank still under its threshold

    reject = np.zeros(n, dtype=bool)
    reject[order[: max_idx + 1]] = True
    return reject.tolist()


def apply_validity_review(
    registry: RegistryConfig, results: list[CellWalkForwardResult], fdr_level: float = 0.10
) -> RegistryConfig:
    """
    Writes selected hyperparams + IC stats back to each cell, then
    deactivates cells that don't survive FDR correction across the
    whole grid tested in this run. Cells not included in `results`
    (e.g. skipped this cycle) are left untouched.
    """
    pvals = [r.pvalue_by_horizon.get(r.primary_horizon, 1.0) for r in results]
    significant = benjamini_hochberg(pvals, fdr_level)

    today = date.today()
    for result, is_significant in zip(results, significant):
        cell = registry.cell(result.model_id, result.market_id)
        if cell is None:
            cell = RegistryCell(model_id=result.model_id, market_id=result.market_id)
            registry.cells.append(cell)

        cell.hyperparams = result.selected_hyperparams
        cell.last_calibration_date = today
        cell.last_validation_ic = result.ic_by_horizon.get(result.primary_horizon)
        cell.last_validation_ic_pvalue = result.pvalue_by_horizon.get(result.primary_horizon)

        if not is_significant and cell.active:
            cell.active = False
            cell.notes = (
                f"Deactivated {today} validity review: primary-horizon IC "
                f"p={cell.last_validation_ic_pvalue:.3f} not significant after FDR correction "
                f"(q={fdr_level})."
            )
        elif is_significant and not cell.active:
            cell.active = True
            cell.notes = f"Reactivated {today}: passed FDR-corrected significance review."

    return registry


# --------------------------------------------------------------------------
# 6. Top-level job
# --------------------------------------------------------------------------

class PriceDataAccessLayer:
    def get_full_history(self, data_source: str) -> pd.Series:
        raise NotImplementedError("wire to your market data / Barra book source")


def run_walk_forward_optimization_job(
    registry_path: str,
    cell_hyperparam_grids: dict[tuple[str, str], dict[str, list]],
    history_start: date,
    history_end: date,
) -> RegistryConfig:
    """
    cell_hyperparam_grids: e.g.
        {("lppls_v2", "SPX"): {"window_days": [500, 750, 1000], "n_starts": [20, 40]}, ...}
    Only cells present in this dict are re-optimized in this run -- lets
    you run LPPLS's weekly job and MODWT's monthly job as separate
    invocations of this same function with different cell subsets,
    matching each model's recalibration_frequency.
    """
    registry = load_registry(registry_path)
    dal = PriceDataAccessLayer()
    results = []

    for (model_id, market_id), grid in cell_hyperparam_grids.items():
        model_cfg = registry.model_by_id(model_id)
        market_cfg = registry.market_by_id(market_id)
        prices = dal.get_full_history(market_cfg.data_source)

        result = run_cell_walk_forward(
            class_path=model_cfg.class_path, model_id=model_id, market_id=market_id,
            prices=prices, hyperparam_grid=grid,
            history_start=history_start, history_end=history_end,
        )
        results.append(result)
        logger.info(
            "cell=(%s,%s) primary_horizon=%d IC=%.4f ICIR=%.4f p=%.4f n_oos=%d",
            model_id, market_id, result.primary_horizon,
            result.ic_by_horizon.get(result.primary_horizon, np.nan),
            result.icir_by_horizon.get(result.primary_horizon, np.nan),
            result.pvalue_by_horizon.get(result.primary_horizon, np.nan),
            result.n_oos_observations,
        )

    registry = apply_validity_review(registry, results)
    save_registry(registry, registry_path)
    return registry


if __name__ == "__main__":
    grids = {
        ("lppls_v2", "SPX"): {
            "window_days": [500, 750, 1000],
            "n_starts": [20, 40],
            "damping_min": [0.5], "damping_max": [1.0],
        },
        ("modwt_confluence_v1", "SPX"): {
            "window_days": [512],
            "scales": [["D1", "D2", "D3"], ["D1", "D2", "D3", "D4", "D5"]],
            "confluence_min_bands": [2, 3],
        },
    }
    run_walk_forward_optimization_job(
        registry_path="../../../config/registry_config.example.yaml",
        cell_hyperparam_grids=grids,
        history_start=date(2015, 1, 1),
        history_end=date.today(),
    )

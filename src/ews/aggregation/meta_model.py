"""
meta_model.py

Replaces the IC-weighted composite in orchestrator.aggregate_per_market
with a trained meta-model: K normalized signal scores -> P(drawdown
event within horizon h). This is stage 2 of the aggregation pipeline
described in the plan -- it sits strictly downstream of the daily score
matrix and must never be trained on the same window used to calibrate
the underlying (k,n) cells (see leakage note in walkforward_optimizer.py).

Key design decisions, and why:

  1. POOLED ACROSS MARKETS, not one model per market.
     Crash/drawdown events are rare -- a single market's history gives
     you a handful of positive labels at best, nowhere near enough to
     fit a stable classifier. Pooling across all N markets multiplies
     the effective sample size. `market_id` is kept as a categorical
     feature so the model can still learn market-specific base rates
     and sensitivities rather than forcing one global response curve.

  2. HistGradientBoostingClassifier, not logistic regression or plain
     GBM. It handles NaN features natively -- and NaN is the normal
     state here, not an edge case: LPPLS non-convergence, thin history,
     data issues all produce missing cells in the K-vector on any given
     day. Encoding "this model had nothing to say today" as a number
     (0, or a mean-imputed value) throws away information the model
     could otherwise use; native NaN handling lets it learn from the
     missingness pattern itself.

  3. Class imbalance handled via sample_weight (inverse frequency), not
     oversampling/SMOTE. Oversampling on time-series data risks
     duplicating near-identical rows across a train/test boundary --
     easy to accidentally leak. Sample weighting keeps every row's
     original temporal position intact.

  4. Nested walk-forward, same embargo discipline as the cell-level
     optimizer: within each outer training window, a trailing slice is
     held out purely for probability calibration (isotonic regression),
     so the outer test fold is used only for final OOS evaluation --
     never for calibration or training.

  5. Output is (probability, confidence) as a pair, not probability
     alone. Confidence is a data-completeness measure (how many of the
     K cells actually contributed today) -- a probability computed from
     2 of 8 models should carry less weight in the global aggregation
     than one computed from 7 of 8, even if the point estimate is
     identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.calibration import IsotonicRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from ews.optimization.walkforward import Fold, compute_forward_drawdown_label, generate_walk_forward_folds

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ews.metamodel")


# --------------------------------------------------------------------------
# 1. Feature panel construction
# --------------------------------------------------------------------------

@dataclass
class FeaturePanel:
    X: pd.DataFrame          # index=(date, market_id), columns=model_id normalized scores
    market_ids: pd.Series    # aligned categorical column
    n_contributing: pd.Series  # per-row count of non-NaN model scores (completeness)


def build_feature_panel(norm_matrix_history: pd.DataFrame, active_model_ids: list[str]) -> FeaturePanel:
    """
    norm_matrix_history: long format, columns
        [date, model_id, market_id, ts_zscore]
    (the same shape orchestrator.normalize_matrix produces daily --
    this function assumes you've been persisting that output).

    Pivots to wide: one row per (date, market_id), one column per model.
    Missing cells stay NaN -- deliberately, per design note 2 above.
    """
    wide = norm_matrix_history.pivot_table(
        index=["date", "market_id"], columns="model_id", values="ts_zscore"
    )
    # ensure consistent column set even if some models were inactive on some days
    for m in active_model_ids:
        if m not in wide.columns:
            wide[m] = np.nan
    wide = wide[active_model_ids]

    n_contributing = wide.notna().sum(axis=1)
    market_ids = wide.index.get_level_values("market_id").to_series(index=wide.index)

    return FeaturePanel(X=wide, market_ids=market_ids, n_contributing=n_contributing)


def build_labels(panel: FeaturePanel, prices_by_market: dict[str, pd.Series],
                  horizon_days: int, event_threshold: float) -> pd.Series:
    """
    Binary label: forward max drawdown over horizon_days exceeds
    event_threshold (e.g. 0.15 for a 15% drawdown). Kept as a function
    of the panel's own (date, market_id) index so it stays aligned with
    X without a separate join step that could silently misalign dates.
    """
    labels = {}
    for (as_of, market_id) in panel.X.index:
        prices = prices_by_market.get(market_id)
        if prices is None:
            labels[(as_of, market_id)] = np.nan
            continue
        dd = compute_forward_drawdown_label(prices, as_of, horizon_days)
        labels[(as_of, market_id)] = np.nan if np.isnan(dd) else float(dd >= event_threshold)
    return pd.Series(labels, name="label")


# --------------------------------------------------------------------------
# 2. Model artifact + versioned record (mirrors the signal registry pattern)
# --------------------------------------------------------------------------

@dataclass
class MetaModelArtifact:
    version: str
    trained_on: date
    feature_columns: list[str]
    classifier: HistGradientBoostingClassifier
    calibrator: IsotonicRegression
    training_window: tuple[date, date]
    oos_auc: float
    oos_average_precision: float
    oos_brier: float
    n_oos_events: int
    n_oos_observations: int


# --------------------------------------------------------------------------
# 3. Nested walk-forward training
# --------------------------------------------------------------------------

def _one_hot_market(X: pd.DataFrame, market_ids: pd.Series) -> pd.DataFrame:
    dummies = pd.get_dummies(market_ids, prefix="market")
    return pd.concat([X.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)


def _inverse_frequency_weights(y: pd.Series) -> np.ndarray:
    pos_rate = max(y.mean(), 1e-6)
    neg_rate = max(1 - pos_rate, 1e-6)
    return np.where(y == 1, 1.0 / pos_rate, 1.0 / neg_rate)


def train_meta_model_walk_forward(
    panel: FeaturePanel, labels: pd.Series,
    history_start: date, history_end: date,
    train_window_days: int = 1500, test_window_days: int = 126,
    embargo_days: int = 21, calibration_holdout_days: int = 126,
) -> tuple[MetaModelArtifact, list[dict]]:
    """
    Returns the final artifact (trained on the most recent fold) plus a
    per-fold OOS metrics log -- keep the log; it's what you'd plot as
    the meta-model's own performance-decay tracker, same idea as the
    per-cell IC decay curves upstream.
    """
    dates = sorted(panel.X.index.get_level_values("date").unique())
    folds = generate_walk_forward_folds(
        history_start, history_end, train_window_days, test_window_days, embargo_days
    )
    if not folds:
        raise ValueError("no walk-forward folds fit in the given history range")

    fold_metrics = []
    artifact = None

    panel_dates = panel.X.index.get_level_values("date")
    if not pd.api.types.is_datetime64_any_dtype(panel_dates):
        raise TypeError("panel.X index level 'date' must be datetime64 -- convert upstream")

    for fold in folds:
        train_start_ts = pd.Timestamp(fold.train_start)
        train_end_ts = pd.Timestamp(fold.train_end)
        calib_start_ts = train_end_ts - pd.Timedelta(days=calibration_holdout_days)
        test_start_ts = pd.Timestamp(fold.test_start)
        test_end_ts = pd.Timestamp(fold.test_end)

        train_mask = (panel_dates >= train_start_ts) & (panel_dates < calib_start_ts)
        calib_mask = (panel_dates >= calib_start_ts) & (panel_dates < train_end_ts)
        test_mask = (panel_dates >= test_start_ts) & (panel_dates < test_end_ts)

        X_train = _one_hot_market(panel.X[train_mask], panel.market_ids[train_mask])
        y_train = labels[train_mask]
        valid_train = y_train.notna()
        X_train, y_train = X_train[valid_train.values], y_train[valid_train]
        if y_train.nunique() < 2 or len(y_train) < 50:
            continue  # not enough signal to fit this fold, skip rather than force a degenerate model

        weights = _inverse_frequency_weights(y_train)
        clf = HistGradientBoostingClassifier(max_iter=200, max_depth=4, learning_rate=0.05)
        clf.fit(X_train, y_train, sample_weight=weights)

        X_calib = _one_hot_market(panel.X[calib_mask], panel.market_ids[calib_mask])
        X_calib = X_calib.reindex(columns=X_train.columns, fill_value=0)
        y_calib = labels[calib_mask]
        valid_calib = y_calib.notna()
        raw_calib_probs = clf.predict_proba(X_calib[valid_calib.values])[:, 1]
        calibrator = IsotonicRegression(out_of_bounds="clip")
        if valid_calib.sum() >= 20 and y_calib[valid_calib].nunique() > 1:
            calibrator.fit(raw_calib_probs, y_calib[valid_calib])
        else:
            calibrator.fit([0, 1], [0, 1])  # identity fallback when calib slice is too thin

        X_test = _one_hot_market(panel.X[test_mask], panel.market_ids[test_mask])
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
        y_test = labels[test_mask]
        valid_test = y_test.notna()
        if valid_test.sum() < 20 or y_test[valid_test].nunique() < 2:
            continue

        raw_test_probs = clf.predict_proba(X_test[valid_test.values])[:, 1]
        calibrated_probs = calibrator.predict(raw_test_probs)
        y_true = y_test[valid_test].values

        auc = roc_auc_score(y_true, calibrated_probs)
        ap = average_precision_score(y_true, calibrated_probs)
        brier = brier_score_loss(y_true, calibrated_probs)

        fold_metrics.append({
            "test_start": fold.test_start, "test_end": fold.test_end,
            "auc": auc, "average_precision": ap, "brier": brier,
            "n_events": int(y_true.sum()), "n_obs": int(len(y_true)),
        })
        logger.info(
            "fold test=[%s,%s] AUC=%.3f AP=%.3f Brier=%.4f n_events=%d/%d",
            fold.test_start, fold.test_end, auc, ap, brier, int(y_true.sum()), len(y_true),
        )

        artifact = MetaModelArtifact(
            version=f"metamodel_{fold.test_end.isoformat()}",
            trained_on=fold.train_end,
            feature_columns=list(X_train.columns),
            classifier=clf, calibrator=calibrator,
            training_window=(fold.train_start, fold.train_end),
            oos_auc=auc, oos_average_precision=ap, oos_brier=brier,
            n_oos_events=int(y_true.sum()), n_oos_observations=int(len(y_true)),
        )

    if artifact is None:
        raise RuntimeError("no fold produced a usable model -- check label rarity / history length")

    return artifact, fold_metrics


# --------------------------------------------------------------------------
# 4. Prediction with confidence, for use in the daily pipeline
# --------------------------------------------------------------------------

@dataclass
class MetaModelPrediction:
    market_id: str
    as_of: date
    probability: float       # calibrated P(drawdown event within horizon)
    confidence: float        # data completeness, in [0, 1]
    n_contributing_models: int
    n_total_models: int


def predict_market_composite(
    artifact: MetaModelArtifact, panel_row: pd.Series, market_id: str, as_of: date,
    n_contributing: int, n_total_models: int,
) -> MetaModelPrediction:
    """
    Drop-in replacement for orchestrator.aggregate_per_market's per-row
    logic -- call this once per (as_of, market_id) using today's
    normalized score row instead of the IC-weighted average.
    """
    row = panel_row.to_frame().T
    row = _one_hot_market(row, pd.Series([market_id]))
    row = row.reindex(columns=artifact.feature_columns, fill_value=0)

    raw_prob = artifact.classifier.predict_proba(row)[:, 1][0]
    calibrated_prob = float(artifact.calibrator.predict([raw_prob])[0])
    confidence = n_contributing / n_total_models if n_total_models else 0.0

    return MetaModelPrediction(
        market_id=market_id, as_of=as_of, probability=calibrated_prob,
        confidence=confidence, n_contributing_models=n_contributing, n_total_models=n_total_models,
    )

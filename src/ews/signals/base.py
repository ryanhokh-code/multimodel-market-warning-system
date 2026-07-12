"""
signal_interface.py

Common interface every (model) implementation must satisfy so the
orchestrator can treat LPPLS, MODWT confluence, MF-DFA, etc. uniformly.

Concrete implementations live under signals/<model_id>.py and are loaded
dynamically via the `class_path` field in the registry (see
registry_schema.ModelConfig.class_path).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date
from enum import Enum

import pandas as pd


class FailureReason(str, Enum):
    NONE = "none"
    INSUFFICIENT_HISTORY = "insufficient_history"
    FIT_DID_NOT_CONVERGE = "fit_did_not_converge"
    DATA_QUALITY = "data_quality"
    STALE_CALIBRATION = "stale_calibration"


@dataclass
class FitResult:
    converged: bool
    fitted_params: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)


@dataclass
class SignalScore:
    model_id: str
    market_id: str
    as_of: date
    raw_score: float | None          # None if scoring failed -- see failure_reason
    confidence: float | None         # model's own confidence in [0, 1], if applicable
    failure_reason: FailureReason = FailureReason.NONE
    diagnostics: dict = field(default_factory=dict)


class WarningSignal(ABC):
    """
    One instance = one (model_id, market_id) cell, constructed with the
    hyperparams currently stored for that cell in the registry.
    """

    def __init__(self, model_id: str, market_id: str, hyperparams: dict):
        self.model_id = model_id
        self.market_id = market_id
        self.hyperparams = hyperparams

    @abstractmethod
    def fit(self, data: pd.DataFrame, as_of: date) -> FitResult:
        """
        Fit/calibrate on data up to and including `as_of`. Must be
        strictly causal -- no data after `as_of` may be touched (this is
        where COI-style leakage bugs live; enforce the cutoff at the
        data-slicing boundary, not inside the model math).
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, as_of: date) -> SignalScore:
        """
        Produce today's score from the most recent successful fit.
        Must NOT raise on expected failure modes (non-convergence, thin
        history) -- return a SignalScore with failure_reason set instead,
        so one broken cell can't take down the daily matrix build.
        """
        raise NotImplementedError

    def min_history_days(self) -> int:
        """Override if the model has a hyperparam-dependent requirement."""
        return self.hyperparams.get("window_days", 252)

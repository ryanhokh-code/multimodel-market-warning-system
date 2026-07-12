"""
signals/lppls.py

Wire your existing debugged LPPLS prototype in here behind the
WarningSignal interface. Left as a stub -- the fit/score logic itself
already exists in your prior work; this file's job is just the
adapter, not a reimplementation.
"""

from datetime import date

import pandas as pd

from ews.signals.base import FailureReason, FitResult, SignalScore, WarningSignal


class LPPLSSignal(WarningSignal):
    def fit(self, data: pd.DataFrame, as_of: date) -> FitResult:
        raise NotImplementedError("wire in the existing LPPLS multi-start fit routine")

    def score(self, as_of: date) -> SignalScore:
        raise NotImplementedError("wire in confidence-indicator computation from the fitted params")

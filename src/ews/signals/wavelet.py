"""
signals/wavelet.py

Wire your existing causal-safe MODWT (fixed 512-day window, COI-aware)
multi-band confluence engine in here behind the WarningSignal interface.
"""

from datetime import date

import pandas as pd

from ews.signals.base import FailureReason, FitResult, SignalScore, WarningSignal


class MODWTConfluenceSignal(WarningSignal):
    def fit(self, data: pd.DataFrame, as_of: date) -> FitResult:
        raise NotImplementedError("wire in the existing causal-safe MODWT decomposition")

    def score(self, as_of: date) -> SignalScore:
        raise NotImplementedError("wire in the multi-band confluence alert engine")

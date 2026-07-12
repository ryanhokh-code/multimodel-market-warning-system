"""
signals/mfdfa.py

Wire your existing mfdfa_walkforward_opt.py fit/score logic in here
behind the WarningSignal interface. Recall from prior validation: OOS
IC was weak/sign-inconsistent standalone -- this cell is expected to
contribute as one composite component, not a standalone predictor.
"""

from datetime import date

import pandas as pd

from ews.signals.base import FailureReason, FitResult, SignalScore, WarningSignal


class MFDFASignal(WarningSignal):
    def fit(self, data: pd.DataFrame, as_of: date) -> FitResult:
        raise NotImplementedError("wire in the existing MF-DFA fluctuation-function fit")

    def score(self, as_of: date) -> SignalScore:
        raise NotImplementedError("wire in the multifractal spectrum width scoring")

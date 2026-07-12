"""Tests for ews.optimization.walkforward -- fold embargo + FDR correction."""
from datetime import date

from ews.optimization.walkforward import benjamini_hochberg, generate_walk_forward_folds


def test_folds_respect_embargo_gap():
    folds = generate_walk_forward_folds(
        date(2015, 1, 1), date(2020, 1, 1), 750, 63, 21
    )
    assert all((f.test_start - f.train_end).days == 21 for f in folds)


def test_bh_correction_separates_signal_from_null():
    pvals = [0.001, 0.002, 0.003] + [0.5, 0.7, 0.9]
    reject = benjamini_hochberg(pvals, fdr_level=0.10)
    assert all(reject[:3]) and not any(reject[3:])

"""Tests for ews.orchestration.daily_pipeline -- hysteresis behavior."""
from ews.orchestration.daily_pipeline import WarningLevel, discretize_with_hysteresis


def test_hysteresis_holds_between_exit_and_entry_thresholds():
    level = discretize_with_hysteresis(1.2, WarningLevel.GREEN)
    assert level == WarningLevel.YELLOW
    level = discretize_with_hysteresis(0.6, level)  # below entry, above exit
    assert level == WarningLevel.YELLOW  # should hold, not drop to GREEN

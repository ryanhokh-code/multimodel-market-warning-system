#!/usr/bin/env python3
"""
scripts/run_walkforward.py

Entry point for the periodic per-cell recalibration job. Run this once
per model family, on that model's own recalibration_frequency -- e.g.
a weekly cron for LPPLS cells, a monthly cron for MODWT/MF-DFA cells --
each invocation passing only the cells due for recalibration.
"""

from datetime import date

from ews.optimization.walkforward import run_walk_forward_optimization_job

if __name__ == "__main__":
    grids = {
        # populate per model family; see registry_config.yaml for cell list
    }
    run_walk_forward_optimization_job(
        registry_path="config/registry_config.yaml",
        cell_hyperparam_grids=grids,
        history_start=date(2015, 1, 1),
        history_end=date.today(),
    )

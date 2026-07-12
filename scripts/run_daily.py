#!/usr/bin/env python3
"""
scripts/run_daily.py

Entry point for the daily scoring DAG task. Wrap this call in
Airflow/Prefect/cron -- it's intentionally a thin script so the actual
logic in ews.orchestration.daily_pipeline stays testable in isolation.
"""

from datetime import date

from ews.orchestration.daily_pipeline import NormalizationStore, run_daily_pipeline

if __name__ == "__main__":
    result = run_daily_pipeline(
        registry_path="config/registry_config.yaml",
        as_of=date.today(),
        norm_store=NormalizationStore(),  # swap for a persisted store in production
        previous_levels={},               # load yesterday's levels from your DB
    )
    print(result["levels"])

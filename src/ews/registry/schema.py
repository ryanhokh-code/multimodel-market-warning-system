"""
registry_schema.py

Defines the schema for the EWS model/market registry. The registry is the
single source of truth for:
  - which (model, market) cells are active
  - each cell's calibrated hyperparameters
  - each cell's recalibration cadence and last calibration date
  - each cell's validity status (from periodic significance review)

The registry is persisted as YAML (human-editable, diffable in git) and
loaded/validated through these Pydantic models at runtime. Nothing in the
orchestration layer should read raw YAML directly -- always go through
`load_registry()` so malformed configs fail fast at startup, not mid-DAG.
"""

from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator


class RecalibrationFrequency(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"


class SignalCategory(str, Enum):
    """Broad family, used for correlation-aware aggregation and reporting."""
    BUBBLE_DYNAMICS = "bubble_dynamics"          # LPPLS
    MULTISCALE = "multiscale"                    # wavelet MODWT/CWT
    REGIME_SWITCHING = "regime_switching"         # Markov, GARCH-Jump/Hawkes
    EXTREME_VALUE = "extreme_value"               # EVT
    MICROSTRUCTURE_STABILITY = "microstructure"   # RMT, critical slowing down
    MULTIFRACTAL = "multifractal"                 # MF-DFA
    OPTIONS_IMPLIED = "options_implied"           # RND divergence, skew, VRP
    LIQUIDITY_FUNDING = "liquidity_funding"       # cross-asset funding stress
    POSITIONING = "positioning"                   # COT, 13F, sec lending
    SENTIMENT_NLP = "sentiment_nlp"


class MarketConfig(BaseModel):
    market_id: str                     # e.g. "SPX", "NKY", "FACTOR_MOM_LS"
    display_name: str
    asset_class: str                   # "equity_index", "factor_ls", "fx", ...
    region: Optional[str] = None       # "US", "JP", ... (None for factor books)
    data_source: str                   # key into your data layer / vendor config
    min_history_days: int = 512        # e.g. your fixed MODWT window
    active: bool = True


class ModelConfig(BaseModel):
    model_id: str                      # e.g. "lppls_v2", "modwt_confluence_v1"
    category: SignalCategory
    class_path: str                    # dotted import path to WarningSignal subclass
    default_hyperparams: dict = Field(default_factory=dict)
    recalibration_frequency: RecalibrationFrequency
    requires_min_history_days: int = 252
    active: bool = True


class RegistryCell(BaseModel):
    """
    One (model, market) pairing. This is the unit of independent
    walk-forward optimization discussed in the plan -- hyperparams here
    are written back by the optimizer, not derived live.
    """
    model_id: str
    market_id: str
    hyperparams: dict = Field(default_factory=dict)
    last_calibration_date: Optional[date] = None
    last_validation_ic: Optional[float] = None
    last_validation_ic_pvalue: Optional[float] = None  # post multiple-testing correction
    active: bool = True                # can be flipped off by the validity review
    notes: Optional[str] = None

    @field_validator("last_validation_ic_pvalue")
    @classmethod
    def _check_pvalue_range(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError("p-value must be in [0, 1]")
        return v


class RegistryConfig(BaseModel):
    """Top-level container -- the object loaded from registry_config.yaml."""
    models: list[ModelConfig]
    markets: list[MarketConfig]
    cells: list[RegistryCell]

    # ---- convenience accessors used throughout the orchestrator ----

    def active_cells(self) -> list[RegistryCell]:
        model_active = {m.model_id for m in self.models if m.active}
        market_active = {m.market_id for m in self.markets if m.active}
        return [
            c for c in self.cells
            if c.active and c.model_id in model_active and c.market_id in market_active
        ]

    def model_by_id(self, model_id: str) -> ModelConfig:
        return next(m for m in self.models if m.model_id == model_id)

    def market_by_id(self, market_id: str) -> MarketConfig:
        return next(m for m in self.markets if m.market_id == market_id)

    def cell(self, model_id: str, market_id: str) -> Optional[RegistryCell]:
        return next(
            (c for c in self.cells if c.model_id == model_id and c.market_id == market_id),
            None,
        )


def load_registry(path: str) -> RegistryConfig:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return RegistryConfig(**raw)


def save_registry(registry: RegistryConfig, path: str) -> None:
    """
    Used by the walk-forward optimizer job to write back recalibrated
    hyperparams/IC stats. Keep this atomic in production (write to temp
    file + os.replace) so a crash mid-write can't corrupt the registry
    that tomorrow's DAG run depends on.
    """
    with open(path, "w") as f:
        yaml.safe_dump(
            registry.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
        )

"""Tests for ews.registry.schema -- see prior validation in conversation."""
from ews.registry.schema import load_registry


def test_registry_loads_and_prunes_inactive_cells():
    registry = load_registry("config/registry_config.example.yaml")
    active = {(c.model_id, c.market_id) for c in registry.active_cells()}
    assert ("lppls_v2", "FACTOR_MOM_LS") not in active  # pruned by validity review
    assert ("lppls_v2", "SPX") in active

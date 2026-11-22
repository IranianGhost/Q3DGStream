import importlib
import pytest


def select(snapshot,previous=3,extra=0):
    module=importlib.import_module('research.quic.priority_transport.publio_native')
    return module.select_level([40_000,70_000,130_000,250_000,480_000],snapshot,10.,.25,2.,previous,extra)[0]


def test_stale_viewer_feedback_selects_smallest_model():
    assert select({'at':8.,'buffer_seconds':3.,'committed_Bps':10_000_000})==0


def test_fresh_low_playable_buffer_reduces_model_size():
    assert select({'at':10.,'buffer_seconds':.2,'committed_Bps':10_000_000})==2


def test_capacity_budget_caps_model_even_with_healthy_buffer():
    assert select({'at':10.,'buffer_seconds':3.,'committed_Bps':400_000})==1


def test_recovered_buffer_and_delivery_rate_allow_upward_step():
    assert select({'at':10.,'buffer_seconds':2.,'committed_Bps':10_000_000})==4


def test_additions_are_reserved_before_model_budget():
    assert select({'at':10.,'buffer_seconds':3.,'committed_Bps':1_000_000},extra=100_000)==1


@pytest.mark.parametrize('paced,updates,fps,prebuffer',[(False,48,2,2),(True,8,30,.1),(True,48,2,.1)])
def test_dynamic_schedule_requires_exposure_and_playable_reserve(paced,updates,fps,prebuffer):
    from types import SimpleNamespace
    module=importlib.import_module('research.quic.priority_transport.publio_native')
    args=SimpleNamespace(profile='dynamic',paced=paced,updates=updates,fps=fps,prebuffer=prebuffer)
    with pytest.raises(ValueError):module.validate_schedule(args)


def test_pacing_idle_does_not_force_downshift_on_healthy_network():
    # 2fps at480kB is demand-limited to960kB/s; an active transfer measurement
    # demonstrates spare capacity. Idle time must not masquerade as congestion.
    assert select({'at':10.,'buffer_seconds':2.,'committed_Bps':960_000,
                   'delivery_service_Bps':3_000_000},previous=4)==4

"""Simulated capacity profiles, not claims of validation on physical GPUs."""
import pytest
from pydantic import ValidationError

from red_curtain.budget import GIB, MIN_CACHE, WORKSPACE_RESERVE, allocator_ceiling, resolve_budget
from red_curtain.config import ConfigStore, Settings
from red_curtain.api import LoadRequest


@pytest.mark.parametrize('capacity',[12,16,24,48])
def test_budget_scales_with_available_gpu_memory(capacity):
    total=capacity*GIB
    budget=resolve_budget(total,total-GIB,0,int(8.6*GIB))
    assert budget.limit_bytes==min(int(total*.9),total-GIB-max(int(total*.05),GIB//2))
    assert budget.cache_bytes>=MIN_CACHE
    assert budget.limit_bytes==budget.fixed_bytes+budget.cache_bytes+WORKSPACE_RESERVE
    assert budget.mode=='automatic'


def test_small_gpu_gets_clear_failure_before_compute_allocation():
    with pytest.raises(MemoryError,match='compute weights'):
        resolve_budget(8*GIB,7*GIB,0,int(8.6*GIB))


def test_competing_applications_reduce_budget_without_double_counting_own_memory():
    assert allocator_ceiling(24*GIB,5*GIB,10*GIB)==int(15*GIB)-int(24*GIB*.05)
    assert allocator_ceiling(24*GIB,5*GIB,10*GIB)<allocator_ceiling(24*GIB,20*GIB,0)
    with pytest.raises(MemoryError):
        resolve_budget(24*GIB,6*GIB,0,int(8.6*GIB))


def test_manual_limit_is_a_ceiling_and_can_exceed_old_15_gib_cap():
    budget=resolve_budget(48*GIB,46*GIB,0,int(8.6*GIB),32)
    assert budget.limit_bytes==32*GIB
    assert budget.mode=='manual ceiling'
    smaller=resolve_budget(16*GIB,13*GIB,0,int(8.6*GIB),32)
    assert smaller.limit_bytes<13*GIB
    assert resolve_budget(16*GIB,15*GIB,0,int(8.6*GIB),12).limit_bytes==12*GIB


@pytest.mark.parametrize('limit',[0,-1,float('nan'),float('inf')])
def test_invalid_limits_rejected_by_config_api_and_budget(limit):
    with pytest.raises(ValidationError):Settings(vram_limit_gib=limit)
    with pytest.raises(ValidationError):LoadRequest(path='model',vram_limit_gib=limit)
    with pytest.raises(ValueError):resolve_budget(16*GIB,15*GIB,0,8*GIB,limit)


def test_new_settings_are_automatic_and_saved_manual_values_survive(tmp_path):
    store=ConfigStore(tmp_path)
    assert store.value.vram_limit_gib is None
    store.save(Settings(vram_limit_gib=12))
    assert ConfigStore(tmp_path).value.vram_limit_gib==12
    store.save(Settings(vram_limit_gib=32))
    assert ConfigStore(tmp_path).value.vram_limit_gib==32
    store.save(Settings(vram_limit_gib=None))
    assert ConfigStore(tmp_path).value.vram_limit_gib is None

"""Real quantization/routing equivalence; no synthetic output is presented as speech."""
import pytest
import torch
from contextlib import nullcontext

from red_curtain.fish_omni.moe import MoEExperts
from red_curtain.memory import ExpertCache, OffloadedExperts
from red_curtain.safety import Monitor


def storage():
    torch.manual_seed(17)
    experts = MoEExperts(3,128,128,fp8=True,device='cpu')
    for name, value in experts.named_buffers():
        if 'scale' in name:
            value.fill_(0.05)
        else:
            value.copy_(torch.randn(value.shape).to(value.dtype))
    return experts


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('tokens',[1,5])
def test_offloaded_routing_matches_upstream_dense_reference(device,tokens):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('GPU not available')
    experts = storage()
    cache = ExpertCache(2*3*128*128*4,device=device)
    offloaded = OffloadedExperts(experts,0,cache)
    x = torch.randn(tokens,128)
    ids = torch.tensor([[0,1],[1,2],[2,0],[1,0],[2,1]])[:tokens]
    weights = torch.softmax(torch.randn(tokens,2),-1)
    expected = experts(x,ids,weights)
    with (Monitor() if device.startswith('cuda') else nullcontext()) as monitor:
        actual = offloaded(x.to(device),ids.to(device),weights.to(device))
        torch.testing.assert_close(actual.cpu(),expected,rtol=1e-4,atol=1e-5)
        if monitor:monitor.check()
    if monitor:monitor.check()
    assert cache.bytes <= cache.budget
    assert all(buffer.device.type=='cpu' for buffer in experts.buffers())


def test_cache_evicts_and_reloads_without_altering_weights():
    experts = storage()
    cache = ExpertCache(3*128*128*4,device='cpu')
    first = tuple(x.clone() for x in cache.get(0,0,experts,torch.float32))
    cache.get(0,0,experts,torch.float32)
    assert cache.hits==1
    cache.get(0,1,experts,torch.float32)
    assert cache.bytes <= cache.budget
    for actual, expected in zip(cache.get(0,0,experts,torch.float32),first):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    assert cache.misses==3
    cache.clear()
    assert cache.bytes==0 and not cache.entries


def test_zero_budget_is_bounded_and_checks_cancellation():
    experts = storage()
    cache = ExpertCache(0,device='cpu')
    cache.get(0,0,experts,torch.float32)
    assert cache.bytes==0 and not cache.entries
    def cancel():
        raise InterruptedError('cancelled')
    cache.check = cancel
    with pytest.raises(InterruptedError):
        cache.get(0,0,experts,torch.float32)


def test_budget_shrink_evicts_lru_and_preserves_retained_expert_weights():
    experts=storage()
    size=3*128*128*4
    cache=ExpertCache(size*3,device='cpu')
    for index in range(3):cache.get(0,index,experts,torch.float32)
    expected=tuple(t.clone() for t in cache.get(0,1,experts,torch.float32))
    cache.resize(size)
    assert list(cache.entries)==[(0,1,torch.float32)]
    assert cache.bytes<=cache.budget
    for actual,reference in zip(cache.get(0,1,experts,torch.float32),expected):
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)

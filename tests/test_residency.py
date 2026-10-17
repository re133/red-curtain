"""Real device round-trip on the actual architecture at a tiny test size."""
from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

from red_curtain.fish_omni.backbone import FishOmniBackbone
from red_curtain.fish_omni.config import FishOmniConfig
from red_curtain.fish_omni.generate import ensure_drama3_caches
from red_curtain.fish_omni.model import FishOmniModel
from red_curtain.fish_omni.tower import FishOmniAudioDecoder
from red_curtain.memory import ExpertCache, OffloadedExperts
from red_curtain.residency import StageResidency
from red_curtain.safety import Monitor


@pytest.mark.skipif(not torch.cuda.is_available(), reason='GPU required')
def test_park_resume_preserves_weights_and_never_moves_cpu_experts_to_gpu():
    original=FishOmniConfig.from_dict(json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/drama-3-config.json').read_text()))
    text=replace(original.text,vocab_size=64,hidden_size=128,num_hidden_layers=2,
        layer_types=['linear_attention','full_attention'],num_attention_heads=4,num_key_value_heads=2,
        head_dim=32,linear_num_key_heads=2,linear_num_value_heads=4,linear_key_head_dim=16,
        linear_value_head_dim=16,num_experts=3,num_experts_per_tok=2,
        moe_intermediate_size=128,shared_expert_intermediate_size=128)
    audio=replace(original.audio_decoder,dim=128,intermediate_size=384,n_head=4,n_local_heads=2,head_dim=32,vocab_size=16)
    config=replace(original,text=text,audio_decoder=audio,semantic_start_token_id=16,semantic_end_token_id=31,eos_token_id=2)
    with Monitor() as monitor,torch.inference_mode():
        backbone=FishOmniBackbone(text,fp8_experts=True)
        tower=FishOmniAudioDecoder(config,dtype=torch.bfloat16)
        cache=ExpertCache(1024**2,check=monitor.check)
        for i,layer in enumerate(backbone.model.layers):
            for name,buffer in layer.mlp.experts.named_buffers():
                buffer.fill_(1 if 'scale' in name else 0)
            layer.mlp.experts=OffloadedExperts(layer.mlp.experts,i,cache)
        model=FishOmniModel(config,backbone,tower,None).eval()
        residency=StageResidency(model,cache,check=monitor.check)
        residency.resume()
        expected=model.backbone.embed(torch.tensor([1,2])).cpu()
        ensure_drama3_caches(model,16)
        cache.get(0,0,model.backbone.model.layers[0].mlp.experts.storage,torch.bfloat16)
        assert cache.bytes>0
        assert residency.park()>0
        assert cache.bytes==0 and model.cache_capacity==0
        assert model.backbone.model.layers[0].linear_attn.recurrent_state is None
        assert model.backbone.model.layers[1].self_attn.k_cache is None
        assert all(p.device.type=='cpu' for p in model.parameters())
        residency.resume()
        ensure_drama3_caches(model,16)
        assert model.device.type=='cuda'
        torch.testing.assert_close(model.backbone.embed(torch.tensor([1,2])).cpu(),expected,rtol=0,atol=0)
        assert all(t.device.type=='cpu' for layer in model.backbone.model.layers for t in layer.mlp.experts.storage.buffers())
        monitor.check()
    monitor.check()

import json
from pathlib import Path

import pytest
import torch

from red_curtain.adapter import DramaAdapter, validate_checkpoint
from red_curtain.fish_omni.backbone import FishOmniBackbone, load_backbone_weights
from red_curtain.fish_omni.config import FishOmniConfig

ROOT = Path(__file__).resolve().parents[1]


def test_int4_is_rejected_honestly(tmp_path):
    (tmp_path/'config.json').write_text((ROOT/'tests/fixtures/drama-3-int4-config.json').read_text())
    with pytest.raises(NotImplementedError,match='no validated ROCm loader'):
        validate_checkpoint(tmp_path)


def test_shard_traversal_and_missing_checkpoint_are_rejected(tmp_path):
    (tmp_path/'config.json').write_text((ROOT/'tests/fixtures/drama-3-config.json').read_text())
    for name in ('tokenizer.json','codec.pth'):
        (tmp_path/name).touch()
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'w':'../escape.safetensors'}}))
    with pytest.raises(ValueError,match='unsafe'):
        validate_checkpoint(tmp_path)


def test_missing_model_cannot_transition_to_ready(tmp_path):
    adapter=DramaAdapter()
    with pytest.raises(ValueError):
        adapter.load(tmp_path/'absent')
    assert adapter.state=='unloaded'
    with pytest.raises(RuntimeError,match='not ready'):
        adapter.generate('Hello',tmp_path/'out.wav')
    adapter.unload()
    assert adapter.state=='unloaded'


def test_huggingface_blob_symlink_is_accepted_without_allowing_index_traversal(tmp_path):
    checkpoint=tmp_path/'snapshot'
    checkpoint.mkdir()
    (checkpoint/'config.json').write_text((ROOT/'tests/fixtures/drama-3-config.json').read_text())
    for name in ('tokenizer.json','codec.pth'):
        (checkpoint/name).touch()
    blob=tmp_path/'blob'
    blob.touch()
    (checkpoint/'model.safetensors').symlink_to(blob)
    path,_=validate_checkpoint(checkpoint)
    assert path==checkpoint


def test_loader_rejects_missing_and_duplicate_slots_without_allocating_35b():
    config=FishOmniConfig.from_dict(json.loads((ROOT/'tests/fixtures/drama-3-config.json').read_text()))
    with torch.device('meta'):
        backbone=FishOmniBackbone(config.text,fp8_experts=True)
    with pytest.raises(ValueError,match='were not loaded'):
        load_backbone_weights(backbone,[])
    weight=torch.empty(config.text.vocab_size,config.text.hidden_size,device='meta',dtype=torch.bfloat16)
    with pytest.raises(ValueError,match='filled twice'):
        load_backbone_weights(backbone,[('lm_head.weight',weight),('lm_head.weight',weight)])

"""Compare real original-checkpoint metadata against the reused codec architecture."""
import json
from pathlib import Path

import torch

from red_curtain.audio import make_codec


def test_original_codec_parameter_names_and_shapes_are_covered():
    metadata=json.loads((Path(__file__).resolve().parents[1]/'tests/fixtures/codec-tensor-metadata.json').read_text())
    with torch.device('meta'):
        codec=make_codec()
    state=codec.state_dict()
    runtime={name for name in metadata if name.endswith(('.freqs_cis','.causal_mask'))}
    assert set(state)==set(metadata)-runtime
    for name,tensor in state.items():
        assert list(tensor.shape)==metadata[name]['shape'],name

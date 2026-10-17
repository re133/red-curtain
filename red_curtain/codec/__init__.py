# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored DAC codec modules for Fish Speech S2 Pro.

Adopted from the ``fish-speech`` 0.1.0 PyPI release
(https://pypi.org/project/fish-speech/0.1.0/, Apache-2.0) and
descript-audio-codec (https://github.com/descriptinc/descript-audio-codec,
MIT), trimmed to inference-only so the codec loads without external model
packages. Module attribute paths and weight-norm styles are kept identical
to upstream so ``codec.pth`` checkpoints load unchanged.
"""

from red_curtain.codec.codec import DAC
from red_curtain.codec.rvq import (
    DownsampleResidualVectorQuantize,
)
from red_curtain.codec.transformer import (
    ModelArgs,
    WindowLimitedTransformer,
)

__all__ = [
    "DAC",
    "DownsampleResidualVectorQuantize",
    "ModelArgs",
    "WindowLimitedTransformer",
]

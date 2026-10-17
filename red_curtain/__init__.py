"""Local Drama 3 ROCm inference; experimental until checkpoint validation."""
import os

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('DO_NOT_TRACK','1')
# Supported MIOpen mode: avoid a long convolution solver search for every audio length.
# This changes only per-process algorithm selection, never driver or hardware controls.
os.environ.setdefault('MIOPEN_FIND_MODE','FAST')

"""Strict finite, non-silent PCM output and the upstream DAC checkpoint loader."""
from functools import partial
from pathlib import Path
import os
import math

import numpy as np
import soundfile as sf
import torch


def read_reference(path):
    """Bounded, local reference decoding. Never follows an external URL."""
    from scipy.signal import resample_poly
    path=Path(path)
    if not path.is_file() or path.stat().st_size > 20*1024**2:
        raise ValueError('Reference must be a local audio file up to 20 MiB')
    info=sf.info(path)
    if info.format not in {'WAV','FLAC'} or not 8000<=info.samplerate<=192000:
        raise ValueError('Reference must be WAV or FLAC at 8–192 kHz')
    if not 0.2<=info.duration<=15 or not 1<=info.channels<=2:
        raise ValueError('Reference must be mono/stereo and between 0.2 and 15 seconds')
    samples,rate=sf.read(path,dtype='float32',always_2d=True)
    samples=samples.mean(axis=1)
    if not np.isfinite(samples).all() or np.max(np.abs(samples))>1.01 or np.max(np.abs(samples))<1e-5:
        raise ValueError('Reference contains silence, invalid values or out-of-range samples')
    divisor=math.gcd(rate,44100)
    samples=resample_poly(samples,44100//divisor,rate//divisor).astype(np.float32)
    return torch.from_numpy(samples)[None,None]


def write_wav(path, samples, sample_rate):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Decoder returned empty, malformed, or non-finite audio")
    peak = float(np.abs(samples).max())
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if peak < 1e-5 or rms < 1e-7:
        raise ValueError("Decoder returned silence; refusing a successful generation")
    if peak > 1.01:
        raise ValueError(f"Decoder audio exceeds PCM range (peak={peak}); refusing silent clipping")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial.wav")
    try:
        sf.write(temporary, samples, sample_rate, subtype="PCM_16")
        restored, rate = sf.read(temporary)
        if rate != sample_rate or len(restored) != len(samples) or not np.isfinite(restored).all():
            raise ValueError("WAV round-trip validation failed")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(sample_rate=sample_rate, samples=len(samples), duration_s=len(samples) / sample_rate,
                peak=peak, rms=rms, format="PCM_16", **audio_activity(samples,sample_rate))


def audio_activity(samples, sample_rate):
    """Energy-based trailing-silence warning, not speech recognition or trimming."""
    samples=np.asarray(samples,dtype=np.float32)
    if samples.ndim!=1 or not len(samples) or sample_rate<=0:
        raise ValueError('Expected nonempty mono audio and a positive sample rate')
    window=max(1,round(sample_rate*0.1))
    padded=np.pad(samples,(0,(-len(samples))%window))
    energies=np.sqrt(np.mean(padded.reshape(-1,window).astype(np.float64)**2,axis=1))
    threshold=max(1e-4,float(energies.max())*0.01)
    active=np.flatnonzero(energies>threshold)
    duration=len(samples)/sample_rate
    active_end=min(duration,(int(active[-1])+1)*window/sample_rate) if len(active) else 0.0
    trailing=duration-active_end
    warning=''
    if duration>=5 and trailing>=3 and trailing>=duration*0.5:
        warning=(f'A long quiet tail follows approximately {active_end:.1f} seconds of audio '
                 f'({trailing:.1f} s of trailing silence). Some text may not have been spoken. '
                 'Listen to check completeness and try another seed if needed.')
    return dict(energy_active_end_s=active_end,trailing_silence_s=trailing,
                energy_active_fraction=len(active)/len(energies),audio_warning=warning)


def make_codec():
    from red_curtain.codec.codec import DAC
    from red_curtain.codec.rvq import DownsampleResidualVectorQuantize
    from red_curtain.codec.transformer import ModelArgs, WindowLimitedTransformer
    def transformer():
        return WindowLimitedTransformer(
            causal=True, window_size=128, input_dim=1024,
            config=ModelArgs(block_size=4096, n_layer=8, n_head=16, dim=1024,
                             intermediate_size=3072, head_dim=64))
    quantizer = DownsampleResidualVectorQuantize(
        input_dim=1024, n_codebooks=9, codebook_size=1024, codebook_dim=8,
        quantizer_dropout=0.5, downsample_factor=(2, 2),
        pre_module=transformer(), post_module=transformer(), semantic_codebook_size=4096)
    return DAC(sample_rate=44100, encoder_dim=64, encoder_rates=[2, 4, 8, 8],
               decoder_dim=1536, decoder_rates=[8, 8, 4, 2],
               encoder_transformer_layers=[0, 0, 0, 4], decoder_transformer_layers=[4, 0, 0, 0],
               quantizer=quantizer,
               transformer_general_config=partial(ModelArgs, block_size=8192, head_dim=64))


def load_codec(path, device):
    from red_curtain.codec.transformer import Transformer, precompute_freqs_cis
    with torch.device("meta"):
        codec = make_codec()
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    if any(k.startswith("generator.") for k in state):
        state = {k.removeprefix("generator."): v for k, v in state.items() if k.startswith("generator.")}
    runtime_buffers = {f"{name}.{buffer}" for name, module in codec.named_modules()
                       if isinstance(module, Transformer) for buffer in ("freqs_cis", "causal_mask")}
    state = {name: value for name, value in state.items() if name not in runtime_buffers}
    # Never accept randomly initialized missing decoder weights.
    codec.load_state_dict(state, strict=True, assign=True)
    # These are deterministic runtime buffers, absent from the Fish checkpoint.
    # Rebuild them after meta construction, with bounded context sizes.
    for module in codec.modules():
        if isinstance(module, Transformer):
            cfg = module.config
            module.freqs_cis = precompute_freqs_cis(cfg.block_size,cfg.head_dim,cfg.rope_base)
            module.causal_mask = torch.ones(cfg.block_size,cfg.block_size,dtype=torch.bool).tril_()
    return codec.eval().to(device)

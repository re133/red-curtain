"""Experimental original Drama 3 checkpoint adapter for ROCm.

Uses upstream model math. NVIDIA-only FP8 GEMMs and CUDA graphs are never used.
"""
import gc
import json
import os
import math
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from tokenizers import Tokenizer

from red_curtain.audio import load_codec, write_wav, read_reference
from red_curtain.fish_omni.backbone import FishOmniBackbone, load_backbone_weights, module_bytes
from red_curtain.fish_omni.config import FishOmniConfig
from red_curtain.fish_omni.generate import build_drama3_prompt, generate_drama3
from red_curtain.fish_omni.model import FishOmniModel, iter_checkpoint_tensors, AUDIO_DECODER_PREFIXES
from red_curtain.fish_omni.tower import FishOmniAudioDecoder, load_audio_decoder_weights
from red_curtain.memory import ExpertCache, OffloadedExperts
from red_curtain.residency import StageResidency
from red_curtain.budget import resolve_budget
from red_curtain.gpu import device_memory

GIB = 1024 ** 3


def validate_checkpoint(path):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Model directory does not exist: {path}")
    raw = json.loads((path / "config.json").read_text())
    if raw.get("quantization_config", {}).get("quant_method") == "drama3-int4":
        raise NotImplementedError("INT4 GPTQ-Marlin has no validated ROCm loader in this adapter. Use original Drama 3.")
    config = FishOmniConfig.from_dict(raw)
    if not config.fp8_experts:
        raise NotImplementedError("This bounded offload loader requires the original block-FP8 checkpoint")
    for name in ("tokenizer.json", "codec.pth"):
        if not (path / name).is_file():
            raise ValueError(f"Missing checkpoint file: {name}")
    index = path / "model.safetensors.index.json"
    shards = set(json.loads(index.read_text())["weight_map"].values()) if index.exists() else {"model.safetensors"}
    for shard in shards:
        relative = Path(shard)
        # HF cache snapshots legitimately symlink immutable blobs outside the snapshot.
        # Reject traversal in the index, while allowing such read-only file links.
        target = path / relative
        if relative.is_absolute() or '..' in relative.parts or not target.is_file():
            raise ValueError(f"Missing or unsafe checkpoint shard: {shard}")
    return path, config


class DramaAdapter:
    def __init__(self, vram_limit_gib=None, check=lambda: None, progress=lambda *args: None):
        if vram_limit_gib is not None and (not math.isfinite(vram_limit_gib) or vram_limit_gib<1):
            raise ValueError("VRAM limit must be at least 1 GiB, or automatic")
        self.requested_limit_gib=vram_limit_gib
        self.limit=None
        self.fixed_bytes=0
        self.memory_budget=None
        self.check, self.progress = check, progress
        self.model = self.codec = self.cache = None
        self.residency = None
        self.state = "unloaded"
        self.metrics = {}

    def refresh_budget(self):
        budget=resolve_budget(**device_memory(0),fixed_bytes=self.fixed_bytes,
                              requested_gib=self.requested_limit_gib)
        torch.cuda.set_per_process_memory_fraction(budget.limit_bytes/torch.cuda.get_device_properties(0).total_memory)
        if self.cache is not None:self.cache.resize(budget.cache_bytes)
        self.limit=budget.limit_bytes
        self.memory_budget=budget.to_dict()
        return budget

    @torch.inference_mode()
    def load(self, path):
        if self.state != "unloaded":
            raise RuntimeError(f"Cannot load while {self.state}")
        path, config = validate_checkpoint(path)
        if not torch.version.hip or not torch.cuda.is_available():
            raise RuntimeError("A working ROCm PyTorch installation is required")
        self.check()
        self.state = "loading"
        started = time.perf_counter()
        try:
            torch.set_num_threads(min(8, os.cpu_count() or 1))
            with torch.device("meta"):
                backbone = FishOmniBackbone(config.text, fp8_experts=True, dtype=torch.bfloat16)
                tower = FishOmniAudioDecoder(config, dtype=torch.bfloat16)
            expert_bytes = sum(module_bytes(layer.mlp.experts) for layer in backbone.model.layers)
            if psutil.virtual_memory().available < expert_bytes + 8 * GIB:
                raise MemoryError("Insufficient system RAM for compressed experts plus loading reserve")
            # Codec checkpoint ~1.87 GB; retain a further 1.5 GiB for KV, activations and staging.
            fixed_bytes = module_bytes(backbone) - expert_bytes + module_bytes(tower) + (path/'codec.pth').stat().st_size
            self.fixed_bytes=fixed_bytes
            budget=self.refresh_budget()
            self.cache = ExpertCache(budget.cache_bytes, check=self.check)
            self.progress("loading", "Allocating dense GPU weights and compressed CPU experts")
            backbone.model.embed_tokens.to_empty(device="cuda:0")
            backbone.model.norm.to_empty(device="cuda:0")
            backbone.lm_head.to_empty(device="cuda:0")
            for layer in backbone.model.layers:
                self.check()
                # Allocate the expert subtree on CPU, all other child modules on GPU.
                for name, child in layer.named_children():
                    if name == "mlp":
                        for mlp_name, module in child.named_children():
                            module.to_empty(device="cpu" if mlp_name == "experts" else "cuda:0")
                    else:
                        child.to_empty(device="cuda:0")
            def checked_weights(skip):
                for i, pair in enumerate(iter_checkpoint_tensors(path, skip_prefixes=skip)):
                    if i % 256 == 0:
                        self.check()
                        self.progress("loading", f"Reading checkpoint tensors: {i}")
                    yield pair
            load_backbone_weights(backbone, checked_weights(AUDIO_DECODER_PREFIXES))
            tower.to_empty(device="cuda:0")
            load_audio_decoder_weights(tower, checked_weights(("model.language_model.", "lm_head.weight")))
            for index, layer in enumerate(backbone.model.layers):
                layer.mlp.experts = OffloadedExperts(layer.mlp.experts, index, self.cache)
            self.model = FishOmniModel(config, backbone, tower, Tokenizer.from_file(str(path/'tokenizer.json'))).eval()
            self.residency = StageResidency(self.model, self.cache, check=self.check)
            self.progress("loading", "Loading DAC codec")
            self.codec = load_codec(path/'codec.pth', "cuda:0")
            torch.cuda.synchronize()
            self.check()
            self.metrics = dict(load_seconds=time.perf_counter()-started,
                                cpu_expert_bytes=expert_bytes, fixed_weight_estimate_bytes=fixed_bytes,
                                gpu=torch.cuda.get_device_name(), torch=torch.__version__, hip=torch.version.hip,
                                miopen_find_mode=os.environ.get('MIOPEN_FIND_MODE','default'),
                                loaded_gpu_allocated_bytes=torch.cuda.memory_allocated(),
                                loaded_process_rss_bytes=psutil.Process().memory_info().rss)
            self.state = "ready"
        except BaseException:
            self.unload()
            raise

    @torch.inference_mode()
    def generate(self, text, output, max_new_tokens=256, temperature=0.7, top_p=0.7, top_k=30, seed=42,
                 references=()):
        if self.state != "ready":
            raise RuntimeError(f"Model is not ready: {self.state}")
        if not (1 <= max_new_tokens <= 1024 and 0 < temperature <= 2 and 0 < top_p <= 1 and 1 <= top_k <= 4096):
            raise ValueError("Invalid sampling settings")
        self.state = "generating"
        started = time.perf_counter()
        before = self.cache.stats()
        timings = {}
        def report_progress(stage, message):
            if stage == 'generating':
                timings.setdefault('prefill_seconds',time.perf_counter()-started)
            if stage == 'first_frame':
                timings['first_frame_seconds']=time.perf_counter()-started
            self.progress(stage,message)
        try:
            self.check()
            self.refresh_budget()
            torch.manual_seed(seed)
            torch.cuda.reset_peak_memory_stats()
            if len(references)>4:
                raise ValueError('At most four local voice references per generation')
            if references:
                # A cancelled prior request can leave a full expert cache behind.
                # Reference encoding needs the same workspace reserve as decoding.
                self.residency.park()
            reference_texts, reference_tokens = [], []
            for reference in references:
                speaker=reference.get('speaker',0)
                if not isinstance(speaker,int) or not 0<=speaker<=15:
                    raise ValueError('Reference speaker must be 0–15')
                transcript=reference['text']
                from red_curtain.fish_omni.generate import validate_drama3_text
                validate_drama3_text(transcript,'reference transcript')
                self.check()
                self.progress('reference','Encoding local voice reference')
                audio=read_reference(reference['path']).to('cuda:0')
                tokens,lengths=self.codec.encode(audio,torch.tensor([audio.shape[-1]],device='cuda'))
                reference_tokens.append(tokens[0,:,:int(lengths[0])].cpu())
                reference_texts.append(f'<|speaker:{speaker}|> {transcript}')
            self.progress('preparing','Restoring text model and audio tower to GPU')
            self.residency.resume()
            prompt = build_drama3_prompt(text, self.model.tokenizer, self.model.config,
                                        prompt_texts=reference_texts,prompt_tokens=reference_tokens)
            # This adapter uses Python CPU routing and dynamic transfers: graph capture is incompatible.
            prior = os.environ.get("FISH_DRAMA3_CUDA_GRAPH")
            os.environ["FISH_DRAMA3_CUDA_GRAPH"] = "0"
            try:
                frames, reason = generate_drama3(self.model, prompt,
                    max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p,
                    top_k=top_k, prefill_chunk_size=32, progress=report_progress, check=self.check)
            finally:
                if prior is None:
                    os.environ.pop("FISH_DRAMA3_CUDA_GRAPH", None)
                else:
                    os.environ["FISH_DRAMA3_CUDA_GRAPH"] = prior
            if frames.numel() == 0:
                raise RuntimeError("Drama 3 generated zero audio frames")
            codes = frames[:,1:].T.contiguous()
            if codes.shape[0] != 10 or (codes < 0).any() or (codes[0] >= 4096).any() or (codes[1:] >= 1024).any():
                raise RuntimeError("Drama 3 generated invalid DAC codebook indices")
            semantic_seconds = time.perf_counter() - started
            # Valid acoustic codes survive a later codec failure. Never replace
            # them with fake audio or discard minutes of successful inference.
            output = Path(output)
            output.parent.mkdir(parents=True, exist_ok=True)
            codes_path = output.with_suffix('.codes.npy')
            temporary = output.with_suffix('.codes.partial.npy')
            try:
                np.save(temporary, codes.numpy(), allow_pickle=False)
                temporary.replace(codes_path)
            finally:
                temporary.unlink(missing_ok=True)
            self.check()
            self.progress("decoding", "Moving unused text/audio-tower weights to system RAM")
            offload_start = time.perf_counter()
            offload_bytes = self.residency.park()
            offload_seconds = time.perf_counter() - offload_start
            dac_start_bytes = torch.cuda.memory_allocated()
            self.progress("decoding", f"DAC decoding {codes.shape[1]} frames")
            audio, lengths = self.codec.decode(codes[None].cuda(), torch.tensor([codes.shape[1]], device="cuda"))
            torch.cuda.synchronize()
            samples = audio[0,0,:int(lengths[0])].float().cpu().numpy()
            self.check()
            info = write_wav(output, samples, self.codec.sample_rate)
            self.metrics.update(info, finish_reason=reason, frames=codes.shape[1],
                decode_offload_bytes=offload_bytes, decode_offload_seconds=offload_seconds,
                gpu_allocated_before_dac_bytes=dac_start_bytes,
                semantic_seconds=semantic_seconds, generation_seconds=time.perf_counter()-started,
                peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
                peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
                process_rss_bytes=psutil.Process().memory_info().rss, **self.cache.stats())
            self.metrics.update(timings)
            for counter in ('hits','misses','host_to_device_bytes'):
                self.metrics[counter] -= before[counter]
            self.metrics['first_audio_seconds']=self.metrics['generation_seconds']
            self.metrics['real_time_factor'] = self.metrics['generation_seconds']/info['duration_s']
            return dict(self.metrics)
        finally:
            self.state = "ready"

    def unload(self):
        self.residency = None
        self.model = self.codec = None
        if self.cache:
            self.cache.clear()
        self.cache = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.state = "unloaded"

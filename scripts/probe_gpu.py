#!/usr/bin/env python3
"""Short, bounded real GPU operator checks; no model weights required."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import torch.nn.functional as F
from red_curtain.fish_omni.layers import chunk_gated_delta_rule, recurrent_gated_delta_rule, grouped_attention
from red_curtain.fish_omni.moe import dequantize_fp8_blocks
from red_curtain.safety import Monitor


def main():
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("A working ROCm PyTorch GPU is required")
    torch.manual_seed(42)
    report = dict(torch=torch.__version__, hip=torch.version.hip,
                  gpu=torch.cuda.get_device_name(), architecture=torch.cuda.get_device_properties(0).gcnArchName,
                  operations={})
    with Monitor() as monitor, torch.inference_mode():
        def run(name, operation):
            monitor.check()
            start = time.perf_counter()
            value = operation()
            torch.cuda.synchronize()
            if not torch.isfinite(value).all():
                raise RuntimeError(f"Non-finite output: {name}")
            report['operations'][name] = dict(ok=True, seconds=time.perf_counter()-start,
                                              shape=list(value.shape))
        run("bf16_linear", lambda: F.linear(torch.randn(8,2048,device='cuda',dtype=torch.bfloat16),
                                             torch.randn(1024,2048,device='cuda',dtype=torch.bfloat16)))
        for length in (1, 32):
            run(f"gqa_sdpa_{length}", lambda: grouped_attention(
                torch.randn(1,16,length,256,device='cuda',dtype=torch.bfloat16),
                torch.randn(1,2,32,256,device='cuda',dtype=torch.bfloat16),
                torch.randn(1,2,32,256,device='cuda',dtype=torch.bfloat16),None))
        args = [torch.randn(1,17,4,32) for _ in range(3)]
        args += [-torch.rand(1,17,4), torch.rand(1,17,4)]
        reference, state = recurrent_gated_delta_rule(*args)
        gpu_args = [x.cuda() for x in args]
        result, gpu_state = chunk_gated_delta_rule(*gpu_args)
        torch.testing.assert_close(result.cpu(),reference,rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(gpu_state.cpu(),state,rtol=1e-4,atol=1e-5)
        run("deltanet_chunk_vs_cpu_recurrence",lambda: result)
        run("deltanet_decode",lambda: recurrent_gated_delta_rule(
            *[x[:,:1] for x in gpu_args],initial_state=gpu_state)[0])
        raw = torch.randn(512,2048).to(torch.float8_e4m3fn)
        scales = torch.rand(4,16)
        expected = dequantize_fp8_blocks(raw,scales,torch.bfloat16)
        decoded = dequantize_fp8_blocks(raw.cuda(),scales.cuda(),torch.bfloat16)
        torch.testing.assert_close(decoded.cpu(),expected,rtol=0,atol=0)
        run("fp8_storage_to_bf16",lambda: decoded)
        run("moe_topk_bmm",lambda: torch.bmm(torch.randn(8,1024,2048,device='cuda',dtype=torch.bfloat16),
                                              torch.randn(8,2048,1,device='cuda',dtype=torch.bfloat16)))
        run("dac_conv_transpose",lambda: F.conv_transpose1d(torch.randn(1,32,16,device='cuda'),
                                                           torch.randn(32,16,16,device='cuda'),stride=8))
        monitor.check()
    monitor.check()
    report['monitor'] = monitor.samples
    report['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
    Path('reports').mkdir(exist_ok=True)
    Path('reports/gpu-probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()

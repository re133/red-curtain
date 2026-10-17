#!/usr/bin/env python3
"""Load the real model, generate sequential takes, and measure unload in one process."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from red_curtain.adapter import DramaAdapter
from red_curtain.safety import Monitor


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=Path('models/drama-3'))
    parser.add_argument('--text',default='Hello, this is my first locally generated voice.')
    parser.add_argument('--takes',type=int,default=2)
    parser.add_argument('--max-new-tokens',type=int,default=128)
    parser.add_argument('--vram-limit-gib',type=float,default=None)
    parser.add_argument('--name',default='benchmark')
    args=parser.parse_args()
    if not 1<=args.takes<=4 or not args.name.replace('-','').isalnum():
        parser.error('Use 1–4 takes and a simple output name')
    adapter=None
    report=dict(text=args.text,takes=[],status='failed')
    monitor=Monitor()
    try:
        with monitor:
            adapter=DramaAdapter(args.vram_limit_gib,check=monitor.check,
                progress=lambda s,m:print(f'[{s}] {m}',flush=True))
            adapter.load(args.model)
            for take in range(args.takes):
                target=Path('outputs')/f'{args.name}-{take+1}.wav'
                if target.exists():raise ValueError(f'Output already exists: {target}')
                metrics=adapter.generate(args.text,target,max_new_tokens=args.max_new_tokens,seed=42+take)
                report['takes'].append(dict(output=str(target),**metrics))
                print(json.dumps(report['takes'][-1],indent=2),flush=True)
            report['status']='audio_generated_unreviewed'
    finally:
        if adapter:adapter.unload()
        report['gpu_allocated_after_unload_bytes']=torch.cuda.memory_allocated()
        report['gpu_reserved_after_unload_bytes']=torch.cuda.memory_reserved()
        report['monitor']=monitor.samples
        report['peak_process_rss_bytes']=max((s['process_rss_bytes'] for s in monitor.samples),default=0)
        Path('reports').mkdir(exist_ok=True)
        Path(f'reports/{args.name}.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()

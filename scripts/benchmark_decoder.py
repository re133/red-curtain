#!/usr/bin/env python3
"""Compare supported MIOpen find modes on real, previously generated DAC codes."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--find-mode',choices=['FAST','DYNAMIC_HYBRID'],default='FAST')
parser.add_argument('--codes',type=Path,default=Path('outputs/first-real-1.codes.npy'))
parser.add_argument('--report',type=Path)
args=parser.parse_args()
os.environ['MIOPEN_FIND_MODE']=args.find_mode
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf
import torch
from red_curtain.audio import load_codec
from red_curtain.safety import Monitor

report=dict(find_mode=args.find_mode,runs=[])
with Monitor() as monitor,torch.inference_mode():
    torch.set_num_threads(8)
    codec=load_codec('models/drama-3/codec.pth','cuda:0')
    codes=torch.from_numpy(np.load(args.codes,allow_pickle=False))[None].cuda()
    for extra in (0,1,1):
        monitor.check()
        sequence=torch.cat([codes,codes[:,:,-1:]],dim=-1) if extra else codes
        start=time.perf_counter()
        waveform,lengths=codec.decode(sequence,torch.tensor([sequence.shape[-1]],device='cuda'))
        torch.cuda.synchronize()
        result=dict(frames=sequence.shape[-1],seconds=time.perf_counter()-start)
        if not extra:
            expected,_=sf.read(args.codes.with_suffix('').with_suffix('.wav'),dtype='float32')
            actual=waveform[0,0,:len(expected)].float().cpu().numpy()
            result['max_pcm_difference']=float(np.max(np.abs(expected-actual)))
        report['runs'].append(result)
        if not torch.isfinite(waveform).all():raise ValueError('Non-finite codec output')
    monitor.check()
report['monitor']=list(monitor.samples)
Path('reports').mkdir(exist_ok=True)
(args.report or Path(f'reports/decoder-{args.find_mode.lower()}.json')).write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report['runs'],indent=2))

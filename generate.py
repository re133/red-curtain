#!/usr/bin/env python3
"""Real local Drama 3 inference only. No download or fallback occurs here."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--text', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--vram-limit-gib', type=float, default=None,help='Optional upper limit; default: automatic GPU memory budget')
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-audio',type=Path,action='append',default=[])
    parser.add_argument('--reference-text',action='append',default=[])
    parser.add_argument('--reference-speaker',type=int,action='append',default=[])
    args = parser.parse_args()
    if len(args.reference_audio)!=len(args.reference_text) or (
        args.reference_speaker and len(args.reference_speaker)!=len(args.reference_audio)):
        parser.error('Every reference needs an audio file and transcript; speaker assignments must match')
    references=[dict(path=path,text=text,speaker=args.reference_speaker[i] if args.reference_speaker else i)
                for i,(path,text) in enumerate(zip(args.reference_audio,args.reference_text))]
    from red_curtain.adapter import DramaAdapter
    from red_curtain.safety import Monitor
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}; choose a new filename")
    report = {"status":"failed", "model":str(args.model), "started":time.time()}
    adapter = None
    monitor = Monitor()
    try:
        with monitor:
            adapter = DramaAdapter(args.vram_limit_gib, check=monitor.check,
                progress=lambda stage, message: print(f"[{stage}] {message}", flush=True))
            adapter.load(args.model)
            report.update(adapter.generate(args.text,args.output,
                max_new_tokens=args.max_new_tokens,temperature=args.temperature,
                top_p=args.top_p,top_k=args.top_k,seed=args.seed,references=references))
            report['status'] = 'audio_generated_unreviewed'
    except (Exception, KeyboardInterrupt) as exc:
        report['error'] = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        if adapter:
            adapter.unload()
            import torch
            report['gpu_allocated_after_unload_bytes'] = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        report['monitor'] = monitor.samples
        if monitor.error:
            report.update(status='failed', error=monitor.error)
        if report['status'] == 'failed':
            args.output.unlink(missing_ok=True)
        report['codes_available']=args.output.with_suffix('.codes.npy').is_file()
        report['elapsed_seconds'] = time.time()-report['started']
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_suffix('.metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'monitor'},indent=2))
    return 0 if report['status']=='audio_generated_unreviewed' else 1


if __name__ == '__main__':
    sys.exit(main())

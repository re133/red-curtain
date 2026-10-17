#!/usr/bin/env python3
"""Explicit, pinned, project-local model download. Never invoked by inference."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import os

os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
os.environ.setdefault('DO_NOT_TRACK','1')

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-download',action='store_true',help='Explicitly authorize a roughly 41.5 GB download')
    parser.add_argument('--plan',action='store_true',help='Show metadata only, never download weights')
    parser.add_argument('--destination',type=Path,default=ROOT/'models'/'drama-3')
    args=parser.parse_args()
    from huggingface_hub import HfApi, snapshot_download
    repo='fishaudio/drama-3'
    info=HfApi().model_info(repo,files_metadata=True)
    patterns=['*.safetensors','model.safetensors.index.json','config.json','tokenizer.json',
              'tokenizer_config.json','special_tokens_map.json','LICENSE','README.md','codec.pth','chat_template.jinja']
    import fnmatch
    files=[f for f in info.siblings if any(fnmatch.fnmatch(f.rfilename,p) for p in patterns)]
    required=sum(f.size or 0 for f in files)
    target=args.destination.expanduser().resolve()
    print(json.dumps(dict(repo=repo,revision=info.sha,download_bytes=required,destination=str(target),
                          license='Read the checkpoint LICENSE before use',
                          files=[f.rfilename for f in files]),indent=2),flush=True)
    if args.plan:
        return 0
    if not args.confirm_download:
        parser.error('No weights downloaded. Use --confirm-download only after explicit authorization.')
    if shutil.disk_usage(ROOT).free < required+5*1024**3:
        parser.error('Insufficient free disk space (download plus 5 GiB reserve)')
    target.mkdir(parents=True,exist_ok=True)
    snapshot_download(repo_id=repo,revision=info.sha,local_dir=target,allow_patterns=patterns,max_workers=2)
    (target/'download-provenance.json').write_text(json.dumps(dict(repo=repo,revision=info.sha,
        bytes=required),indent=2)+'\n')
    return 0


if __name__=='__main__':
    sys.exit(main())

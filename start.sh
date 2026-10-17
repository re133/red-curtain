#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-FAST}"
if [[ ! -x .venv/bin/python ]]; then
  printf 'Missing project environment. Run ./setup.sh first.\n' >&2
  exit 1
fi
.venv/bin/python - <<'PY'
import importlib.util
required=['torch','numpy','tokenizers','safetensors','soundfile','loguru','tqdm','fastapi','uvicorn','psutil','scipy','python_multipart']
missing=[name for name in required if importlib.util.find_spec(name) is None]
if missing:raise SystemExit('Missing project dependencies: '+', '.join(missing)+'. Run ./setup.sh')
import torch
if not torch.version.hip or not torch.cuda.is_available():raise SystemExit('ROCm GPU unavailable')
print('GPU:',torch.cuda.get_device_name(),'HIP:',torch.version.hip)
PY
if [[ "${1:-}" == '--probe' ]]; then
  exec .venv/bin/python scripts/probe_gpu.py
fi
mkdir -p logs
if [[ ! -f frontend/dist/index.html ]]; then
  printf 'Missing UI build. Run: cd frontend && npm ci && npm run build\n' >&2
  exit 1
fi
printf '\nRed Curtain: http://127.0.0.1:7860\n'
exec .venv/bin/python -m uvicorn red_curtain.api:create_app --factory --host 127.0.0.1 --port 7860 --log-level info

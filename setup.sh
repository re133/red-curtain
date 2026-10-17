#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
python - <<'PY'
import torch
if not torch.version.hip or not torch.cuda.is_available():
    raise SystemExit('Working system ROCm PyTorch required. No drivers or torch wheels will be installed.')
print('Using installed ROCm torch',torch.__version__,'HIP',torch.version.hip)
PY
if [[ ! -x .venv/bin/python ]]; then
  python -m venv --system-site-packages .venv
fi
.venv/bin/python -m pip install --no-cache-dir -r requirements-dev.txt
if ! command -v npm >/dev/null; then
  printf 'Node.js/npm required for the local UI; no system packages installed.\n' >&2
  exit 1
fi
cd frontend
npm ci --no-audit --no-fund
npm run build
printf '\nProject environment ready. No model weights downloaded.\n'

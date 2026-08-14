#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
# stop anything already serving 8998 (setsid children can outlive the wrapper pid)
if [ -f server.pid ]; then
  kill "$(cat server.pid)" 2>/dev/null || true
fi
fuser -k 8998/tcp 2>/dev/null || true
sleep 1
export CUDA_VISIBLE_DEVICES=1
export PORT=8998
export TTS_MODEL_DIR="${TTS_MODEL_DIR:-/home/gpusvr/hf-models/Multilingual-TTS-1.7B-Base}"
setsid -f ./venv/bin/python app.py > server.log 2>&1 < /dev/null &
echo $! > server.pid
echo "started pid $(cat server.pid) on port $PORT (GPU 1), log: $(pwd)/server.log"

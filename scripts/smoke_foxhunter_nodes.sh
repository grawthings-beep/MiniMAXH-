#!/usr/bin/env bash
set -Eeuo pipefail

cd "${COMFYUI_ROOT:-/opt/ComfyUI}"
python main.py --cpu --listen 127.0.0.1 --port 8191 \
  > /tmp/foxhunter-comfy-smoke.log 2>&1 &
smoke_pid=$!
cleanup() {
  kill "${smoke_pid}" 2>/dev/null || true
  wait "${smoke_pid}" 2>/dev/null || true
}
trap cleanup EXIT
ready=0
for _ in $(seq 1 90); do
  if curl -fsS --max-time 2 http://127.0.0.1:8191/object_info \
    -o /tmp/foxhunter-object-info.json; then
    ready=1
    break
  fi
  if ! kill -0 "${smoke_pid}" 2>/dev/null; then break; fi
  sleep 2
done
if [[ "${ready}" != "1" ]]; then
  tail -n 100 /tmp/foxhunter-comfy-smoke.log
  echo "[foxhunter] ComfyUI failed to start on CPU during image build" >&2
  exit 1
fi
python /opt/minimax-h3/scripts/check_foxhunter_object_info.py \
  --object-info /tmp/foxhunter-object-info.json \
  --workflow /opt/minimax-h3/workflows/foxhunter_seed_hunter_v3.json || {
    tail -n 100 /tmp/foxhunter-comfy-smoke.log
    exit 1
  }

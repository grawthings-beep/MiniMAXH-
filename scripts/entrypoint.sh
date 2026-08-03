#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
MODEL_DIR="${COMFYUI_MODEL_DIR:-${COMFYUI_ROOT}/models}"
MANIFEST="${MODEL_MANIFEST:-${PROJECT_DIR}/manifests/minimax_h3_i2v.json}"

if [[ "${ACCEPT_MINIMAX_H3_LICENSE:-0}" != "1" ]]; then
  echo "MiniMax H3 model download was not started."
  echo "Review the license and set ACCEPT_MINIMAX_H3_LICENSE=1:"
  echo "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
  exit 64
fi

check_deployment_territory() {
  local dc_id="${RUNPOD_DC_ID:-}"
  dc_id="${dc_id^^}"

  case "${dc_id}" in
    US-*|EU-*|UK-*|GB-*|KR-*|KOR-*|*-KR-*|*-KOR-*)
      if [[ "${MINIMAX_H3_SEPARATE_LICENSE:-0}" != "1" ]]; then
        echo "MiniMax H3 model download was not started."
        echo "RunPod data center ${RUNPOD_DC_ID} is in a territory excluded by the community license."
        echo "Choose an eligible data center, or set MINIMAX_H3_SEPARATE_LICENSE=1 only if MiniMax has granted you separate authorization."
        exit 65
      fi
      echo "[license] separate territory authorization declared for ${RUNPOD_DC_ID}"
      ;;
    "")
      if [[ "${MINIMAX_H3_DEPLOYMENT_ALLOWED:-0}" != "1" ]]; then
        echo "MiniMax H3 model download was not started."
        echo "RUNPOD_DC_ID is unavailable, so the deployment territory cannot be checked."
        echo "After confirming the physical compute location is permitted, set MINIMAX_H3_DEPLOYMENT_ALLOWED=1."
        exit 66
      fi
      echo "[license] deployment territory manually confirmed"
      ;;
    *)
      echo "[license] RunPod data center: ${RUNPOD_DC_ID}"
      ;;
  esac
}

check_deployment_territory

mkdir -p "${MODEL_DIR}" "${COMFYUI_ROOT}/input" "${COMFYUI_ROOT}/output" \
  "${COMFYUI_ROOT}/temp" "${COMFYUI_ROOT}/user/default/workflows"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V.json"

python "${SCRIPT_DIR}/preflight.py" \
  --manifest "${MANIFEST}" \
  --model-dir "${MODEL_DIR}" \
  --json-out "${COMFYUI_ROOT}/user/default/minimax_h3_runtime_profile.json"

"${SCRIPT_DIR}/download_models.sh"

cd "${COMFYUI_ROOT}"
read -r -a EXTRA_ARGS <<< "${COMFYUI_ARGS:-}"
echo "[comfyui] http://0.0.0.0:${COMFYUI_PORT:-8188}"
exec python main.py \
  --listen 0.0.0.0 \
  --port "${COMFYUI_PORT:-8188}" \
  "${EXTRA_ARGS[@]}"

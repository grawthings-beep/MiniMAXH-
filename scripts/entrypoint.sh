#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
MODEL_DIR="${COMFYUI_MODEL_DIR:-${COMFYUI_ROOT}/models}"
MANIFEST="${MODEL_MANIFEST:-${PROJECT_DIR}/manifests/minimax_h3_all.json}"

if [[ "${ACCEPT_MINIMAX_H3_LICENSE:-0}" != "1" ]]; then
  echo "MiniMax H3 model download was not started."
  echo "Review the license and set ACCEPT_MINIMAX_H3_LICENSE=1:"
  echo "https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE"
  exit 64
fi

check_licensee_territory() {
  local dc_id="${RUNPOD_DC_ID:-}"
  dc_id="${dc_id^^}"

  if [[ "${MINIMAX_H3_SEPARATE_LICENSE:-0}" == "1" ]]; then
    echo "[license] separate MiniMax authorization declared"
  elif [[ "${MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY:-0}" == "1" ]]; then
    echo "[license] licensee confirmed as based in the Applicable Territory"
  else
    echo "MiniMax H3 model download was not started."
    echo "Confirm that the licensee is based in the Applicable Territory and set MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1."
    echo "If the licensee is based in an Excluded Territory, use MINIMAX_H3_SEPARATE_LICENSE=1 only after MiniMax grants separate authorization."
    exit 65
  fi

  case "${dc_id}" in
    US-*|EU-*|UK-*|GB-*|KR-*|KOR-*|*-KR-*|*-KOR-*)
      echo "[license] WARNING: RunPod data center ${RUNPOD_DC_ID} is physically in an Excluded Territory."
      echo "[license] The official Q&A describes authorization in terms of where the licensee is based; the license does not explicitly define cloud compute location."
      echo "[license] Continuing from the licensee declaration above. Review the current license for your deployment."
      ;;
    "")
      echo "[license] RunPod data center is unavailable; continuing from the licensee declaration above"
      ;;
    *)
      echo "[license] RunPod data center: ${RUNPOD_DC_ID}"
      ;;
  esac
}

check_licensee_territory

mkdir -p "${MODEL_DIR}" "${COMFYUI_ROOT}/input" "${COMFYUI_ROOT}/output" \
  "${COMFYUI_ROOT}/temp" "${COMFYUI_ROOT}/user/default/workflows"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_easycache.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_easycache.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache.json"

PREFLIGHT_ARGS=(
  --manifest "${MANIFEST}"
  --model-dir "${MODEL_DIR}"
  --json-out "${COMFYUI_ROOT}/user/default/minimax_h3_runtime_profile.json"
)
if [[ "${REQUIRE_COMFY_KITCHEN_CUDA:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require-comfy-kitchen-cuda)
fi
python "${SCRIPT_DIR}/preflight.py" "${PREFLIGHT_ARGS[@]}"

"${SCRIPT_DIR}/download_models.sh"

cd "${COMFYUI_ROOT}"
read -r -a EXTRA_ARGS <<< "${COMFYUI_ARGS:-}"
for arg in "${EXTRA_ARGS[@]}"; do
  if [[ "${arg}" == "--fast-disk" ]]; then
    echo "[comfyui] WARNING: --fast-disk can make H3 model offload much slower; use it only when system RAM is insufficient."
  fi
done
echo "[comfyui] http://0.0.0.0:${COMFYUI_PORT:-8188}"
echo "[comfyui] extra args: ${EXTRA_ARGS[*]:-(none)}"
exec python main.py \
  --listen 0.0.0.0 \
  --port "${COMFYUI_PORT:-8188}" \
  "${EXTRA_ARGS[@]}"

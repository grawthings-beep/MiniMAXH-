#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
MODEL_DIR="${COMFYUI_MODEL_DIR:-${COMFYUI_ROOT}/models}"
MANIFEST="${MODEL_MANIFEST:-${PROJECT_DIR}/manifests/minimax_h3_i2v_upscale.json}"
DIRECTOR_ROOT="${MINIMAX_H3_DIRECTOR_ROOT:-${COMFYUI_ROOT}/custom_nodes/ComfyUI_MiniMaxH3_Director}"
STORY_NODE_ROOT="${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard"
AUTO_MOSAIC_MODEL="${MODEL_DIR}/auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"
AUTO_MOSAIC_REQUIRED="${AUTO_MOSAIC_REQUIRED:-1}"
AUTO_MOSAIC_REQUIRED="${AUTO_MOSAIC_REQUIRED,,}"

lora_selected() {
  local requested="${H3_LORA_SELECTION:-all}"
  requested="${requested,,}"
  requested="${requested//[[:space:]]/}"
  requested=",${requested},"
  [[ "${requested}" == ",all," || "${requested}" == *",$1,"* ]]
}

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

if [[ ! -f "${DIRECTOR_ROOT}/nodes/director.py" ]] \
  || [[ ! -f "${STORY_NODE_ROOT}/storyboard.py" ]] \
  || [[ ! -f "${STORY_NODE_ROOT}/exporter.py" ]] \
  || [[ ! -f "${STORY_NODE_ROOT}/mosaic_nodes.py" ]]; then
  echo "[workflow] required ordered-story custom nodes are missing from the image"
  exit 68
fi

if [[ "${AUTO_MOSAIC_REQUIRED}" =~ ^(1|true|yes|on)$ ]] \
  && [[ -z "${CIVITAI_API_TOKEN:-}" ]]; then
  echo "[auto-mosaic] CIVITAI_API_TOKEN is required for the segmentation model"
  exit 70
fi
if ! grep -Fq "MiniMAXH- local modification (2026)" \
  "${DIRECTOR_ROOT}/director/executor_core.py"; then
  echo "[workflow] required Director segments-mode memory patch is missing"
  exit 69
fi
if ! grep -Fq "MiniMAXH- local fix for AIMixer issue #26" \
  "${DIRECTOR_ROOT}/director/h3_motion_context.py"; then
  echo "[workflow] required Director FL2V Motion Context fix is missing"
  exit 72
fi

LORA_REQUIRED="${H3_LORA_REQUIRED:-1}"
LORA_REQUIRED="${LORA_REQUIRED,,}"
if [[ "${LORA_REQUIRED}" =~ ^(1|true|yes|on)$ ]]; then
  if lora_selected "hmmotion_v1" && [[ -z "${HF_TOKEN:-}" ]]; then
    echo "[lora] HF_TOKEN is required for the selected HMMotion V1 LoRA"
    exit 66
  fi
  if lora_selected "hmnsfw_aio_v2" && [[ -z "${CIVITAI_TOKEN:-}" ]]; then
    echo "[lora] CIVITAI_TOKEN is required for the selected Civitai V2 LoRA"
    exit 67
  fi
fi

mkdir -p "${MODEL_DIR}" "${MODEL_DIR}/auto_mosaic" \
  "${COMFYUI_ROOT}/input" "${COMFYUI_ROOT}/output" \
  "${COMFYUI_ROOT}/temp" "${COMFYUI_ROOT}/user/default/workflows"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_easycache.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_upscale.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_2x.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_easycache_upscale.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache_2x.json"
rm -f \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_Selectable_LoRA_2x.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Quality_Selectable_LoRA_2x.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_Selectable_LoRA_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Quality_Selectable_LoRA_2x_AutoMosaic.json" \
  "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x_AutoMosaic.json"

HAS_R2V=0
if python - "${MANIFEST}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
raise SystemExit(0 if any("ref2va" in item["path"] for item in manifest["files"]) else 1)
PY
then
  HAS_R2V=1
  echo "[workflow] REF2VA model selected; installing R2V workflows"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality.json"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_easycache.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache.json"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_upscale.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality_2x.json"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_easycache_upscale.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache_2x.json"
else
  echo "[workflow] I2V-only manifest selected; hiding R2V workflows with missing models"
  rm -f \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality_2x.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache_2x.json"
fi

if [[ "${MINIMAX_H3_ENTRYPOINT_SMOKE:-0}" == "1" ]]; then
  test -d "${MODEL_DIR}/auto_mosaic"
  test -f "${STORY_NODE_ROOT}/mosaic_nodes.py"
  test -f "${PROJECT_DIR}/manifests/auto_mosaic.json"
  test -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_upscale_auto_mosaic.json"
  echo "[smoke] entrypoint contract passed before network/model startup"
  exit 0
fi

PREFLIGHT_ARGS=(
  --manifest "${MANIFEST}"
  --model-dir "${MODEL_DIR}"
  --json-out "${COMFYUI_ROOT}/user/default/minimax_h3_runtime_profile.json"
)
if [[ "${REQUIRE_COMFY_KITCHEN_CUDA:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require-comfy-kitchen-cuda)
fi
python "${SCRIPT_DIR}/preflight.py" "${PREFLIGHT_ARGS[@]}"

python "${SCRIPT_DIR}/download_lora.py" &
HMMOTION_DOWNLOAD_PID=$!
python "${SCRIPT_DIR}/download_civitai_lora.py" &
CIVITAI_DOWNLOAD_PID=$!
python "${SCRIPT_DIR}/download_auto_mosaic.py" &
AUTO_MOSAIC_DOWNLOAD_PID=$!
"${SCRIPT_DIR}/download_models.sh" &
MODEL_DOWNLOAD_PID=$!

DOWNLOAD_FAILED=0
if ! wait "${HMMOTION_DOWNLOAD_PID}"; then
  echo "[download] HMMotion V1 LoRA failed"
  DOWNLOAD_FAILED=1
fi
if ! wait "${CIVITAI_DOWNLOAD_PID}"; then
  echo "[download] Civitai V2 LoRA failed"
  DOWNLOAD_FAILED=1
fi
if ! wait "${AUTO_MOSAIC_DOWNLOAD_PID}"; then
  echo "[download] auto-mosaic segmentation model failed"
  DOWNLOAD_FAILED=1
fi
if [[ "${DOWNLOAD_FAILED}" == "1" ]]; then
  echo "[download] required auxiliary model failed; stopping the base-model download"
  kill "${MODEL_DOWNLOAD_PID}" 2>/dev/null || true
  wait "${MODEL_DOWNLOAD_PID}" 2>/dev/null || true
  exit 1
fi
wait "${MODEL_DOWNLOAD_PID}"

if [[ "${AUTO_MOSAIC_REQUIRED}" =~ ^(1|true|yes|on)$ ]]; then
  if [[ ! -s "${AUTO_MOSAIC_MODEL}" ]]; then
    echo "[auto-mosaic] required model is missing after verified download: ${AUTO_MOSAIC_MODEL}"
    exit 71
  fi
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_auto_mosaic.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_AutoMosaic.json"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_easycache_auto_mosaic.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache_AutoMosaic.json"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_upscale_auto_mosaic.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_2x_AutoMosaic.json"
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_easycache_upscale_auto_mosaic.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Fast_EasyCache_2x_AutoMosaic.json"
  if [[ "${HAS_R2V}" == "1" ]]; then
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_auto_mosaic.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality_AutoMosaic.json"
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_easycache_auto_mosaic.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache_AutoMosaic.json"
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_upscale_auto_mosaic.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Quality_2x_AutoMosaic.json"
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_r2v_easycache_upscale_auto_mosaic.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_R2V_Fast_EasyCache_2x_AutoMosaic.json"
  fi
fi

if lora_selected "hmmotion_v1" \
  && [[ -f "${MODEL_DIR}/loras/hmmotion_minimax-h3_epoch12.safetensors" ]]; then
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_hmmotion_lora_upscale.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x.json"
  if [[ -s "${AUTO_MOSAIC_MODEL}" ]]; then
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_hmmotion_lora_upscale_auto_mosaic.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_HMMotion_LoRA_2x_AutoMosaic.json"
  fi
fi
if lora_selected "hmnsfw_aio_v2" \
  && [[ -f "${MODEL_DIR}/loras/HMNSFW_AIO_V2.safetensors" ]]; then
  cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_selectable_lora_upscale.json" \
    "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_Selectable_LoRA_2x.json"
  if [[ -s "${AUTO_MOSAIC_MODEL}" ]]; then
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_i2v_selectable_lora_upscale_auto_mosaic.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V_Quality_Selectable_LoRA_2x_AutoMosaic.json"
  fi
  if [[ -f "${MODEL_DIR}/upscale_models/RealESRGAN_x2plus.pth" ]]; then
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_story_quality_lora_2x.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Quality_Selectable_LoRA_2x.json"
    cp -f "${PROJECT_DIR}/workflows/minimax_h3_story_easycache_lora_2x.json" \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x.json"
    if [[ -s "${AUTO_MOSAIC_MODEL}" ]]; then
      cp -f "${PROJECT_DIR}/workflows/minimax_h3_story_quality_lora_2x_auto_mosaic.json" \
        "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Quality_Selectable_LoRA_2x_AutoMosaic.json"
      cp -f "${PROJECT_DIR}/workflows/minimax_h3_story_easycache_lora_2x_auto_mosaic.json" \
        "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_Story_Fast_EasyCache_Selectable_LoRA_2x_AutoMosaic.json"
    fi
  else
    echo "[workflow] Storyboard workflows skipped: RealESRGAN_x2plus.pth is unavailable"
  fi
fi

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

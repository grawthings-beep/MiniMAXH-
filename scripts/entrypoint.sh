#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
MODEL_DIR="${COMFYUI_MODEL_DIR:-${COMFYUI_ROOT}/models}"
MANIFEST="${MODEL_MANIFEST:-${PROJECT_DIR}/manifests/minimax_h3_i2v_upscale.json}"
DIRECTOR_ROOT="${MINIMAX_H3_DIRECTOR_ROOT:-${COMFYUI_ROOT}/custom_nodes/ComfyUI_MiniMaxH3_Director}"
FBC_ROOT="${MINIMAX_H3_FBC_ROOT:-${COMFYUI_ROOT}/custom_nodes/ComfyUI-MiniMaxH3-FirstBlockCache}"
STORY_NODE_ROOT="${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard"
AUTO_MOSAIC_MODEL="${MODEL_DIR}/auto_mosaic/ntd11_anime_nsfw_segm_v5.pt"
AUTO_MOSAIC_REQUIRED="${AUTO_MOSAIC_REQUIRED:-1}"
AUTO_MOSAIC_REQUIRED="${AUTO_MOSAIC_REQUIRED,,}"
RUNTIME_VARIANT="${MINIMAX_H3_RUNTIME_VARIANT:-community-cu128}"

echo "[runtime] variant=${RUNTIME_VARIANT}"
if [[ "${RUNTIME_VARIANT}" == "community-cu128" ]] \
  && [[ "${REQUIRE_COMFY_KITCHEN_CUDA:-0}" == "1" ]]; then
  echo "[runtime] overriding legacy REQUIRE_COMFY_KITCHEN_CUDA=1 for the cu128 compatibility image"
  echo "[runtime] WARNING: this does not enable CUDA 13 kernels; RTX 5090/r580+ users should select the fast-cu130 Docker image"
  export REQUIRE_COMFY_KITCHEN_CUDA=0
fi
if [[ "${COMFYUI_ARGS:-}" == "--disable-dynamic-vram --reserve-vram 4" ]]; then
  echo "[runtime] replacing the unsafe fixed-reservation profile with DynamicVRAM headroom"
  export COMFYUI_ARGS="--lowvram --vram-headroom 2"
fi

# One RunPod Civitai secret is enough for both creator-LoRA and mosaic-model
# downloads.  Keep the established API-specific override when supplied.
if [[ -z "${CIVITAI_API_TOKEN:-}" ]] && [[ -n "${CIVITAI_TOKEN:-}" ]]; then
  export CIVITAI_API_TOKEN="${CIVITAI_TOKEN}"
fi

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
  || [[ ! -f "${STORY_NODE_ROOT}/mosaic_nodes.py" ]] \
  || [[ ! -f "${STORY_NODE_ROOT}/turbo_nodes.py" ]] \
  || [[ ! -f "${STORY_NODE_ROOT}/memory_nodes.py" ]]; then
  echo "[workflow] required ordered-story custom nodes are missing from the image"
  exit 68
fi
if [[ ! -f "${FBC_ROOT}/nodes.py" ]]; then
  echo "[workflow] pinned MiniMax H3 FirstBlockCache custom node is missing from the image"
  exit 73
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
  if lora_selected "motion_booster_anime" && [[ -z "${CIVITAI_TOKEN:-}" ]]; then
    echo "[lora] CIVITAI_TOKEN is required for Shake Harder ANIME"
    exit 76
  fi
fi

mkdir -p "${MODEL_DIR}" "${MODEL_DIR}/auto_mosaic" \
  "${COMFYUI_ROOT}/input" "${COMFYUI_ROOT}/output" \
  "${COMFYUI_ROOT}/temp" "${COMFYUI_ROOT}/user/default/workflows"
find "${COMFYUI_ROOT}/user/default/workflows" -maxdepth 1 -type f \
  -name '*MiniMax_H3*.json' -delete
cp -f "${PROJECT_DIR}/workflows/minimax_h3_preset_01_quality.json" \
  "${COMFYUI_ROOT}/user/default/workflows/01_MiniMax_H3_Quality_2x.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_preset_02_fast_fbcache.json" \
  "${COMFYUI_ROOT}/user/default/workflows/02_MiniMax_H3_Fast_FBCache_2x.json"
cp -f "${PROJECT_DIR}/workflows/minimax_h3_preset_03_turbo.json" \
  "${COMFYUI_ROOT}/user/default/workflows/03_MiniMax_H3_Turbo_4_8step_768p_2x.json"
echo "[workflow] installed exactly 3 MiniMax H3 presets: Quality, Fast FBCache, Turbo selectable 4/8-step 768p"

if [[ "${MINIMAX_H3_ENTRYPOINT_SMOKE:-0}" == "1" ]]; then
  test -d "${MODEL_DIR}/auto_mosaic"
  test -f "${STORY_NODE_ROOT}/mosaic_nodes.py"
  test -f "${STORY_NODE_ROOT}/turbo_nodes.py"
  test -f "${STORY_NODE_ROOT}/memory_nodes.py"
  test -f "${PROJECT_DIR}/manifests/auto_mosaic.json"
  test -f "${PROJECT_DIR}/workflows/minimax_h3_preset_01_quality.json"
  test -f "${PROJECT_DIR}/workflows/minimax_h3_preset_02_fast_fbcache.json"
  test -f "${PROJECT_DIR}/workflows/minimax_h3_preset_03_turbo.json"
  test -f "${FBC_ROOT}/nodes.py"
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
python "${SCRIPT_DIR}/download_turbo_lora.py" &
TURBO_DOWNLOAD_PID=$!
python "${SCRIPT_DIR}/download_anime_loras.py" &
ANIME_LORA_DOWNLOAD_PID=$!
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
if ! wait "${TURBO_DOWNLOAD_PID}"; then
  echo "[download] selectable LightX2V Turbo 4/8-step LoRAs failed"
  DOWNLOAD_FAILED=1
fi
if ! wait "${ANIME_LORA_DOWNLOAD_PID}"; then
  echo "[download] selectable anime motion/style LoRAs failed"
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
fi

TURBO_REQUIRED="${H3_TURBO_REQUIRED:-1}"
TURBO_REQUIRED="${TURBO_REQUIRED,,}"
if [[ "${TURBO_REQUIRED}" =~ ^(1|true|yes|on)$ ]]; then
  for TURBO_MODEL in \
    "${MODEL_DIR}/loras/minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors" \
    "${MODEL_DIR}/loras/minimax_h3_fl2v_turbo_4step_v1.2_768p_comfyui_bf16.safetensors"; do
    if [[ ! -s "${TURBO_MODEL}" ]]; then
      echo "[turbo] required verified LoRA is missing: ${TURBO_MODEL}"
      exit 74
    fi
  done
fi

REQUIRED_PRESET_MODELS=("${MODEL_DIR}/upscale_models/RealESRGAN_x2plus.pth")
lora_selected "hmmotion_v1" && REQUIRED_PRESET_MODELS+=(
  "${MODEL_DIR}/loras/hmmotion_minimax-h3_epoch12.safetensors"
)
lora_selected "hmnsfw_aio_v2" && REQUIRED_PRESET_MODELS+=(
  "${MODEL_DIR}/loras/HMNSFW_AIO_V2.safetensors"
)
lora_selected "motion_booster_anime" && REQUIRED_PRESET_MODELS+=(
  "${MODEL_DIR}/loras/H3_Motion_Booster_anime.safetensors"
)
lora_selected "nsfw_anime_v04" && REQUIRED_PRESET_MODELS+=(
  "${MODEL_DIR}/loras/NSFW_ANIME_V7_H3-step00019500.safetensors"
)
for required_path in "${REQUIRED_PRESET_MODELS[@]}"; do
  if [[ ! -s "${required_path}" ]]; then
    echo "[workflow] one of the 3 presets is missing a required model: ${required_path}"
    exit 75
  fi
done

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

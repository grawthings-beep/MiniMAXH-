#!/usr/bin/env bash
# Sourced only after entrypoint.sh validates the license and selects H3_PROFILE=r2v.
set -Eeuo pipefail

echo "[r2v-only] selected checkpoint + pinned INT8 VAE; legacy MODEL_MANIFEST/H3_CHARACTER_R2V/LoRA settings do not add downloads"
echo "[r2v-only] skipping FL2VA, creator LoRAs, FL2VA Turbo LoRAs and the extra-LoRA URL list"

for node_file in __init__.py character_nodes.py refmod_nodes.py web/refmod_images.js refmod_vendor/core.py refmod_vendor/bundle.py refmod_vendor/LICENSE refmod_vendor/REVISION memory_nodes.py mosaic_nodes.py storyboard.py exporter.py turbo_nodes.py; do
  if [[ ! -f "${STORY_NODE_ROOT}/${node_file}" ]]; then
    echo "[r2v-only] required custom node file is missing: ${node_file}"
    exit 77
  fi
done
if [[ "$(cat "${STORY_NODE_ROOT}/refmod_vendor/REVISION")" != "f9462081e28794389b5a6c5067eb327412ad8ee7" ]]; then
  echo "[r2v-only] RefMod runtime revision mismatch; rebuild the image"
  exit 77
fi
if [[ "${AUTO_MOSAIC_REQUIRED}" =~ ^(1|true|yes|on)$ ]] && [[ -z "${CIVITAI_API_TOKEN:-}" ]]; then
  echo "[auto-mosaic] CIVITAI_API_TOKEN is required for the segmentation model"
  exit 70
fi
R2V_MODEL="${H3_R2V_MODEL:-official}"
case "${R2V_MODEL}" in
  official) ;;
  dasiwa-v2|dasiwa-turbo-v2)
    if [[ -z "${CIVITAI_API_TOKEN:-}" ]]; then
      echo "[r2v-only] CIVITAI_API_TOKEN is required for the selected DaSiWa checkpoint"
      exit 70
    fi ;;
  *) echo "[r2v-only] H3_R2V_MODEL must be official, dasiwa-v2 or dasiwa-turbo-v2"; exit 79 ;;
esac
mkdir -p "${MODEL_DIR}/auto_mosaic" "${MODEL_DIR}/refmods" "${COMFYUI_ROOT}/input" "${COMFYUI_ROOT}/output/refmods" \
  "${COMFYUI_ROOT}/temp" "${COMFYUI_ROOT}/user/default/workflows"
# Replace only known shipped presets, not arbitrary user-named workflows.
for shipped in 01_MiniMax_H3_Quality_2x.json 02_MiniMax_H3_Fast_FBCache_2x.json \
  03_MiniMax_H3_Turbo_4_8step_768p_2x.json; do
  rm -f "${COMFYUI_ROOT}/user/default/workflows/${shipped}"
done
python "${SCRIPT_DIR}/prepare_r2v_profile.py" --profile "${R2V_MODEL}" \
  --output-dir "${COMFYUI_ROOT}/user/default"
MANIFEST="${COMFYUI_ROOT}/user/default/minimax_h3_r2v_models.json"
R2V_HF_MANIFEST="${COMFYUI_ROOT}/user/default/minimax_h3_r2v_hf_models.json"
export MODEL_MANIFEST="${MANIFEST}"
export COMFYUI_MODEL_DIR="${MODEL_DIR}"
echo "[workflow] installed one R2VA Full Prompt + RefMod 1-8 images preset (${R2V_MODEL}); reopen 04 from Workflows"
python "${SCRIPT_DIR}/verify_workflow.py" \
  --workflow "${COMFYUI_ROOT}/user/default/workflows/04_MiniMax_H3_Character_R2V_2x.json" \
  --manifest "${MANIFEST}" --mode r2v --expect-upscale --expect-memory-safe-decode \
  --expect-auto-mosaic --auto-mosaic-manifest "${PROJECT_DIR}/manifests/auto_mosaic.json"

read -r -a EXTRA_ARGS <<< "${COMFYUI_ARGS:---lowvram --vram-headroom 2}"
if [[ "${H3_FAST_VAE:-1}" != "0" && "${H3_FAST_VAE:-1}" != "1" ]]; then
  echo "[r2v-only] H3_FAST_VAE must be 0 or 1"
  exit 79
fi
if [[ "${H3_FAST_VAE:-1}" == "1" && " ${EXTRA_ARGS[*]} " != *" fp16_accumulation "* ]]; then
  if [[ " ${EXTRA_ARGS[*]} " == *" --fast "* ]]; then
    # Insert into the existing --fast value list, never after another option's value.
    R2V_ARGS=()
    for arg in "${EXTRA_ARGS[@]}"; do
      R2V_ARGS+=("${arg}")
      if [[ "${arg}" == "--fast" ]]; then R2V_ARGS+=(fp16_accumulation); fi
    done
    EXTRA_ARGS=("${R2V_ARGS[@]}")
  else
    EXTRA_ARGS+=(--fast fp16_accumulation)
  fi
fi
if [[ " ${EXTRA_ARGS[*]} " != *" --fast-disk "* && " ${EXTRA_ARGS[*]} " != *" --disable-fast-disk "* ]]; then
  EXTRA_ARGS+=(--disable-fast-disk)
fi
echo "[r2v-only] extra args: ${EXTRA_ARGS[*]:-(none)}"

if [[ "${MINIMAX_H3_ENTRYPOINT_SMOKE:-0}" == "1" ]]; then
  python "${SCRIPT_DIR}/verify_workflow.py" \
    --workflow "${COMFYUI_ROOT}/user/default/workflows/04_MiniMax_H3_Character_R2V_2x.json" \
    --manifest "${MANIFEST}" --mode r2v --expect-upscale --expect-memory-safe-decode \
    --expect-auto-mosaic --auto-mosaic-manifest "${PROJECT_DIR}/manifests/auto_mosaic.json"
  echo "[smoke] R2VA-only entrypoint contract passed before network/model startup"
  exit 0
fi

PREFLIGHT_ARGS=(--manifest "${MANIFEST}" --model-dir "${MODEL_DIR}"
  --json-out "${COMFYUI_ROOT}/user/default/minimax_h3_runtime_profile.json")
if [[ "${REQUIRE_COMFY_KITCHEN_CUDA:-1}" == "1" ]]; then
  PREFLIGHT_ARGS+=(--require-comfy-kitchen-cuda)
fi
python "${SCRIPT_DIR}/preflight.py" "${PREFLIGHT_ARGS[@]}"

R2V_DOWNLOAD_PIDS=()
cleanup_r2v_downloads() {
  for pid in "${R2V_DOWNLOAD_PIDS[@]}"; do kill "${pid}" 2>/dev/null || true; done
  for pid in "${R2V_DOWNLOAD_PIDS[@]}"; do wait "${pid}" 2>/dev/null || true; done
}
trap cleanup_r2v_downloads EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
python "${SCRIPT_DIR}/download_auto_mosaic.py" &
R2V_DOWNLOAD_PIDS+=("$!")
MODEL_MANIFEST="${R2V_HF_MANIFEST}" "${SCRIPT_DIR}/download_models.sh" &
R2V_DOWNLOAD_PIDS+=("$!")
python "${SCRIPT_DIR}/download_civitai_models.py" --manifest "${MANIFEST}" --root "${MODEL_DIR}" &
R2V_DOWNLOAD_PIDS+=("$!")
for pid in "${R2V_DOWNLOAD_PIDS[@]}"; do
  if ! wait "${pid}"; then
    echo "[r2v-only] required model download failed; ComfyUI will not start"
    exit 1
  fi
done
R2V_DOWNLOAD_PIDS=()
trap - EXIT INT TERM
python "${SCRIPT_DIR}/verify_models.py" --manifest "${MANIFEST}" \
  --root "${MODEL_DIR}" --mode "${MODEL_VERIFY:-size}" --workers "${HF_DOWNLOAD_WORKERS:-4}"
if [[ "${AUTO_MOSAIC_REQUIRED}" =~ ^(1|true|yes|on)$ ]] && [[ ! -s "${AUTO_MOSAIC_MODEL}" ]]; then
  echo "[auto-mosaic] required verified segmentation model is missing"
  exit 71
fi
cd "${COMFYUI_ROOT}"
echo "[comfyui] http://0.0.0.0:${COMFYUI_PORT:-8188} (R2VA only)"
exec python main.py --listen 0.0.0.0 --port "${COMFYUI_PORT:-8188}" "${EXTRA_ARGS[@]}"

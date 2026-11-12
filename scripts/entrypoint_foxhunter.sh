#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/opt/minimax-h3"
COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
MODEL_DIR="${COMFYUI_MODEL_DIR:-${COMFYUI_ROOT}/models}"
MANIFEST="${ROOT}/manifests/foxhunter_models.json"
WORKFLOW="${ROOT}/workflows/foxhunter_seed_hunter_v3.json"

if [[ "${H3_PROFILE:-foxhunter}" != "foxhunter" ]]; then
  echo "[foxhunter] this image only supports H3_PROFILE=foxhunter" >&2
  exit 79
fi
if [[ "${ACCEPT_MINIMAX_H3_LICENSE:-0}" != "1" ]]; then
  echo "[license] Review https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE and set ACCEPT_MINIMAX_H3_LICENSE=1" >&2
  exit 64
fi
if [[ "${MINIMAX_H3_SEPARATE_LICENSE:-0}" != "1" && "${MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY:-0}" != "1" ]]; then
  echo "[license] Confirm the licensee's applicable territory or separate authorization" >&2
  exit 65
fi

for node_dir in ComfyUI-H3-AnimeCadence ComfyUI-Fantastic-MiniMaxH3-PromptBuilder \
  ComfyUI-Pixaroma ComfyUI-H3-Prompt-IDE ComfyUI-MiniMaxH3Mod cg-use-everywhere \
  ComfyUI-VideoHelperSuite rgthree-comfy ComfyUI-KJNodes ComfyUI-VFI foxhunter_compat; do
  test -f "${COMFYUI_ROOT}/custom_nodes/${node_dir}/__init__.py" || {
    echo "[foxhunter] required node pack missing: ${node_dir}" >&2
    exit 77
  }
done
mkdir -p "${MODEL_DIR}" "${COMFYUI_ROOT}/input" "${COMFYUI_ROOT}/output" \
  "${COMFYUI_ROOT}/temp" "${COMFYUI_ROOT}/user/default/workflows"
python "${ROOT}/scripts/verify_foxhunter.py" --workflow "${WORKFLOW}" --manifest "${MANIFEST}"
cp -f "${WORKFLOW}" "${COMFYUI_ROOT}/user/default/workflows/00_FoxFur_Seed_Hunter_v3.json"

if [[ "${MINIMAX_H3_ENTRYPOINT_SMOKE:-0}" == "1" ]]; then
  echo "[smoke] FoxFur entrypoint contract passed; no model download started"
  exit 0
fi

PREFLIGHT=(--manifest "${MANIFEST}" --model-dir "${MODEL_DIR}"
  --json-out "${COMFYUI_ROOT}/user/default/foxhunter_runtime_profile.json")
if [[ "${REQUIRE_COMFY_KITCHEN_CUDA:-0}" == "1" ]]; then
  PREFLIGHT+=(--require-comfy-kitchen-cuda)
fi
python "${ROOT}/scripts/preflight.py" "${PREFLIGHT[@]}"
python "${ROOT}/scripts/download_foxhunter_models.py" --manifest "${MANIFEST}" --root "${MODEL_DIR}"
python "${ROOT}/scripts/verify_models.py" --manifest "${MANIFEST}" --root "${MODEL_DIR}" \
  --mode "${MODEL_VERIFY:-size}" --workers "${HF_DOWNLOAD_WORKERS:-4}"
python "${ROOT}/scripts/extract_foxhunter_rife.py" \
  --archive "${MODEL_DIR}/rife/RIFEv4.26_0921.zip" \
  --target "${MODEL_DIR}/rife/flownet.pkl"

read -r -a EXTRA_ARGS <<< "${COMFYUI_ARGS:---lowvram --vram-headroom 2}"
if [[ " ${EXTRA_ARGS[*]} " != *" --fast-disk "* && " ${EXTRA_ARGS[*]} " != *" --disable-fast-disk "* ]]; then
  EXTRA_ARGS+=(--disable-fast-disk)
fi
cd "${COMFYUI_ROOT}"
echo "[foxhunter] starting ComfyUI at :${COMFYUI_PORT:-8188}; open 00_FoxFur_Seed_Hunter_v3"
exec python main.py --listen 0.0.0.0 --port "${COMFYUI_PORT:-8188}" "${EXTRA_ARGS[@]}"

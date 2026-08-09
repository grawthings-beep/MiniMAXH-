#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${MODEL_MANIFEST:-${PROJECT_DIR}/manifests/minimax_h3_i2v_upscale.json}"
MODEL_DIR="${COMFYUI_MODEL_DIR:-/opt/ComfyUI/models}"
VERIFY_MODE="${MODEL_VERIFY:-size}"
HF_DOWNLOAD_WORKERS="${HF_DOWNLOAD_WORKERS:-4}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-3}"

export HF_XET_CHUNK_CACHE_SIZE_BYTES="${HF_XET_CHUNK_CACHE_SIZE_BYTES:-0}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-30}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export HF_HUB_DISABLE_UPDATE_CHECK="${HF_HUB_DISABLE_UPDATE_CHECK:-1}"
export HF_HOME="${HF_HOME:-/tmp/huggingface}"

configure_xet() {
  local requested="${HF_XET_HIGH_PERFORMANCE:-auto}"
  local normalized="${requested,,}"
  if [[ "${normalized}" == "auto" ]]; then
    local ram_kib
    ram_kib="$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)"
    if ((ram_kib >= 60 * 1024 * 1024)); then
      export HF_XET_HIGH_PERFORMANCE=1
      echo "[download] Xet mode: high-performance (64 GB-class RAM detected)"
    else
      unset HF_XET_HIGH_PERFORMANCE
      echo "[download] Xet mode: adaptive concurrency (less than 64 GB RAM)"
    fi
  elif [[ "${normalized}" =~ ^(1|on|yes|true)$ ]]; then
    export HF_XET_HIGH_PERFORMANCE=1
    echo "[download] Xet mode: high-performance (forced)"
  else
    unset HF_XET_HIGH_PERFORMANCE
    echo "[download] Xet mode: adaptive concurrency (forced)"
  fi
}

configure_xet

readarray -t META < <(python - "${MANIFEST}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
print(manifest["repo_id"])
print(manifest["revision"])
print(manifest["total_bytes"])
PY
)
REPO_ID="${META[0]}"
REVISION="${META[1]}"
TOTAL_BYTES="${META[2]}"
readarray -t FILES < <(python - "${MANIFEST}" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["files"]:
    print(item["path"])
PY
)
readarray -t HF_FILES < <(python - "${MANIFEST}" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["files"]:
    if "source_url" not in item:
        print(item["path"])
PY
)
readarray -t EXTERNAL_RECORDS < <(python - "${MANIFEST}" <<'PY'
import json
import sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["files"]:
    if "source_url" in item:
        print(f'{item["path"]}\t{item["source_url"]}')
PY
)
readarray -t ARIA_RECORDS < <(python - "${MANIFEST}" <<'PY'
import json
import sys
manifest = json.load(open(sys.argv[1], encoding="utf-8"))
for item in manifest["files"]:
    url = item.get("source_url")
    if url is None:
        url = (
            f'https://huggingface.co/{manifest["repo_id"]}/resolve/'
            f'{manifest["revision"]}/{item["path"]}?download=true'
        )
    print(f'{item["path"]}\t{url}')
PY
)

mkdir -p "${MODEL_DIR}" "${HF_HOME}"

verify() {
  python "${SCRIPT_DIR}/verify_models.py" \
    --manifest "${MANIFEST}" \
    --root "${MODEL_DIR}" \
    --mode "${VERIFY_MODE}" \
    --workers "${HF_DOWNLOAD_WORKERS}" \
    "$@"
}

download_with_xet() {
  if ((${#HF_FILES[@]} == 0)); then
    return 0
  fi
  echo "[download] hf_xet with ${HF_DOWNLOAD_WORKERS} file workers"
  hf download "${REPO_ID}" "${HF_FILES[@]}" \
    --revision "${REVISION}" \
    --local-dir "${MODEL_DIR}" \
    --max-workers "${HF_DOWNLOAD_WORKERS}"
}

download_one_with_aria2() {
  local relative="$1"
  local url="$2"
  local destination="${MODEL_DIR}/${relative}"
  local directory
  directory="$(dirname -- "${destination}")"
  mkdir -p "${directory}"
  aria2c \
    --continue=true \
    --max-connection-per-server="${ARIA2_CONNECTIONS_PER_FILE:-16}" \
    --split="${ARIA2_CONNECTIONS_PER_FILE:-16}" \
    --min-split-size=16M \
    --file-allocation=none \
    --max-tries=10 \
    --retry-wait=3 \
    --timeout=120 \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --dir="${directory}" \
    --out="$(basename -- "${relative}")" \
    "${url}"
}

download_records() {
  local label="$1"
  shift
  local records=("$@")
  if ((${#records[@]} == 0)); then
    return 0
  fi
  echo "[download] ${label}: ${#records[@]} file(s) with parallel aria2 ranges"
  local pids=()
  local status=0
  local record relative url
  for record in "${records[@]}"; do
    IFS=$'\t' read -r relative url <<< "${record}"
    download_one_with_aria2 "${relative}" "${url}" &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "${pid}" || status=1
  done
  return "${status}"
}

download_external_files() {
  download_records "external model source" "${EXTERNAL_RECORDS[@]}"
}

download_with_aria2() {
  echo "[download] Xet retries exhausted; using parallel aria2 range fallback"
  download_records "complete manifest fallback" "${ARIA_RECORDS[@]}"
}

start_epoch="$(date +%s)"
echo "[download] MiniMax H3 manifest $(basename -- "${MANIFEST}"): ${TOTAL_BYTES} bytes from ${REPO_ID}@${REVISION}"

for ((attempt = 1; attempt <= DOWNLOAD_RETRIES; attempt++)); do
  echo "[download] Xet attempt ${attempt}/${DOWNLOAD_RETRIES}"
  if download_with_xet && download_external_files && verify; then
    success=1
    break
  fi
  verify --remove-invalid || true
  delay=$((attempt * 5))
  echo "[download] retrying in ${delay}s"
  sleep "${delay}"
done

if [[ "${success:-0}" != "1" ]]; then
  download_with_aria2
  verify
fi

end_epoch="$(date +%s)"
python - "${TOTAL_BYTES}" "$((end_epoch - start_epoch))" <<'PY'
import sys
size = int(sys.argv[1])
seconds = max(1, int(sys.argv[2]))
print(f"[download] ready in {seconds}s; effective throughput {size * 8 / seconds / 1e9:.2f} Gbit/s")
PY

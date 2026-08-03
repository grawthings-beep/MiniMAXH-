FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385

ARG COMFYUI_VERSION=v0.30.0
ARG COMFYUI_COMMIT=b1693ecba9f5b65f8c80ab36b195ab963ec92413

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFYUI_ROOT=/opt/ComfyUI \
    COMFYUI_MODEL_DIR=/opt/ComfyUI/models \
    MODEL_MANIFEST=/opt/minimax-h3/manifests/minimax_h3_i2v.json \
    HF_HOME=/tmp/huggingface \
    HF_XET_HIGH_PERFORMANCE=auto \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ETAG_TIMEOUT=30 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_UPDATE_CHECK=1 \
    HF_DOWNLOAD_WORKERS=4 \
    MODEL_VERIFY=sha256

RUN apt-get update && apt-get install -y --no-install-recommends \
      aria2 \
      curl \
      ffmpeg \
      git \
      libgl1 \
      libglib2.0-0 \
      tini \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --branch "${COMFYUI_VERSION}" --depth 1 \
      https://github.com/Comfy-Org/ComfyUI.git "${COMFYUI_ROOT}" \
    && test "$(git -C "${COMFYUI_ROOT}" rev-parse HEAD)" = "${COMFYUI_COMMIT}" \
    && pip install --no-cache-dir -r "${COMFYUI_ROOT}/requirements.txt" \
    && pip install --no-cache-dir \
      "huggingface_hub==1.26.0" \
      "hf-xet==1.5.2"

COPY . /opt/minimax-h3

RUN chmod +x /opt/minimax-h3/scripts/*.sh /opt/minimax-h3/scripts/*.py \
    && mkdir -p "${COMFYUI_ROOT}/user/default/workflows" \
    && cp /opt/minimax-h3/workflows/minimax_h3_i2v.json \
      "${COMFYUI_ROOT}/user/default/workflows/MiniMax_H3_I2V.json" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v.json \
      --comfyui-root "${COMFYUI_ROOT}"

WORKDIR /opt/ComfyUI
EXPOSE 8188

HEALTHCHECK --interval=15s --timeout=5s --start-period=30m --retries=4 \
  CMD curl -fsS http://127.0.0.1:8188/system_stats >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/minimax-h3/scripts/entrypoint.sh"]

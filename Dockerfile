ARG PYTORCH_IMAGE=pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime@sha256:7b324d212a4450795b49edba9949b7cdc72429148a64e974334bfe5774d51385
FROM ${PYTORCH_IMAGE}

ARG COMFYUI_VERSION=v0.31.0
ARG COMFYUI_COMMIT=43cb4fffc89bba20ab7bd61467a36d0339338dab
ARG MINIMAX_H3_DIRECTOR_COMMIT=a267324a9f88141ff4e4b0e8c1a6ed90b4e45db7
ARG MINIMAX_H3_FBC_COMMIT=725973c3bfd9de6dce249bc93dc5fe27f820df31
ARG MINIMAX_H3_RUNTIME_VARIANT=community-cu128
ARG REQUIRE_COMFY_KITCHEN_CUDA_DEFAULT=0
ARG COMFYUI_ARGS_DEFAULT="--vram-headroom 2"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    COMFYUI_ROOT=/opt/ComfyUI \
    COMFYUI_MODEL_DIR=/opt/ComfyUI/models \
    MINIMAX_H3_DIRECTOR_ROOT=/opt/ComfyUI/custom_nodes/ComfyUI_MiniMaxH3_Director \
    MINIMAX_H3_FBC_ROOT=/opt/ComfyUI/custom_nodes/ComfyUI-MiniMaxH3-FirstBlockCache \
    MODEL_MANIFEST=/opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
    HF_HOME=/tmp/huggingface \
    HF_XET_HIGH_PERFORMANCE=auto \
    HF_XET_CHUNK_CACHE_SIZE_BYTES=0 \
    HF_HUB_DOWNLOAD_TIMEOUT=120 \
    HF_HUB_ETAG_TIMEOUT=30 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_UPDATE_CHECK=1 \
    HF_DOWNLOAD_WORKERS=4 \
    MODEL_VERIFY=size \
    H3_LORA_REQUIRED=1 \
    H3_LORA_SELECTION=all \
    H3_LORA_REPO_ID=uwgm/nikke-civitai-backup \
    H3_LORA_SOURCE_PATH=hmmotion_minimax-h3_epoch12.safetensors \
    H3_LORA_REVISION=main \
    H3_CIVITAI_LORA_URL=https://civitai.red/api/download/models/3206518?fileId=3088013 \
    H3_TURBO_REQUIRED=1 \
    H3_TURBO_REPO_ID=lightx2v/Minimax-h3-Turbo \
    H3_TURBO_SOURCE_PATH=minimax_h3_fl2v_turbo_8step_v1.0_comfyui_bf16.safetensors \
    H3_TURBO_REVISION=05ef678438e84933c406131b59abbf86919b3aac \
    AUTO_MOSAIC_REQUIRED=1 \
    AUTO_MOSAIC_MANIFEST=/opt/minimax-h3/manifests/auto_mosaic.json \
    MINIMAX_H3_RUNTIME_VARIANT=${MINIMAX_H3_RUNTIME_VARIANT} \
    REQUIRE_COMFY_KITCHEN_CUDA=${REQUIRE_COMFY_KITCHEN_CUDA_DEFAULT} \
    COMFYUI_ARGS=${COMFYUI_ARGS_DEFAULT}

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

RUN mkdir -p "${MINIMAX_H3_DIRECTOR_ROOT}" \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" init \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" remote add origin \
      https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director.git \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" fetch --depth 1 origin \
      "${MINIMAX_H3_DIRECTOR_COMMIT}" \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${MINIMAX_H3_DIRECTOR_ROOT}" rev-parse HEAD)" = \
      "${MINIMAX_H3_DIRECTOR_COMMIT}" \
    && pip install --no-cache-dir \
      -r "${MINIMAX_H3_DIRECTOR_ROOT}/requirements.txt"

RUN mkdir -p "${MINIMAX_H3_FBC_ROOT}" \
    && git -C "${MINIMAX_H3_FBC_ROOT}" init \
    && git -C "${MINIMAX_H3_FBC_ROOT}" remote add origin \
      https://github.com/duckyshell/ComfyUI-MiniMaxH3-FirstBlockCache.git \
    && git -C "${MINIMAX_H3_FBC_ROOT}" fetch --depth 1 origin \
      "${MINIMAX_H3_FBC_COMMIT}" \
    && git -C "${MINIMAX_H3_FBC_ROOT}" checkout --detach FETCH_HEAD \
    && test "$(git -C "${MINIMAX_H3_FBC_ROOT}" rev-parse HEAD)" = \
      "${MINIMAX_H3_FBC_COMMIT}" \
    && grep -Fq '"ApplyMiniMaxH3FirstBlockCache"' \
      "${MINIMAX_H3_FBC_ROOT}/nodes.py"

COPY patches/minimax_h3_director_segments_no_concat.patch /tmp/minimax_h3_director_segments_no_concat.patch
COPY patches/minimax_h3_director_fl2v_motion_context.patch /tmp/minimax_h3_director_fl2v_motion_context.patch

RUN git -C "${MINIMAX_H3_DIRECTOR_ROOT}" apply --recount --check \
      /tmp/minimax_h3_director_segments_no_concat.patch \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" apply --recount \
      /tmp/minimax_h3_director_segments_no_concat.patch \
    && python -c "from pathlib import Path; p=Path('${MINIMAX_H3_DIRECTOR_ROOT}/director/h3_motion_context.py'); p.write_bytes(p.read_bytes().replace(b'\\r\\n', b'\\n'))" \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" apply --recount --check \
      /tmp/minimax_h3_director_fl2v_motion_context.patch \
    && git -C "${MINIMAX_H3_DIRECTOR_ROOT}" apply --recount \
      /tmp/minimax_h3_director_fl2v_motion_context.patch \
    && grep -Fq "MiniMAXH- local fix for AIMixer issue #26" \
      "${MINIMAX_H3_DIRECTOR_ROOT}/director/h3_motion_context.py" \
    && rm /tmp/minimax_h3_director_segments_no_concat.patch \
      /tmp/minimax_h3_director_fl2v_motion_context.patch

COPY . /opt/minimax-h3

RUN pip install --no-cache-dir \
      -r /opt/minimax-h3/custom_nodes/minimax_h3_ordered_storyboard/requirements.txt \
    && python -c "import ultralytics; assert ultralytics.__version__ == '8.4.104'" \
    && chmod +x /opt/minimax-h3/scripts/*.sh /opt/minimax-h3/scripts/*.py \
    && cp -R /opt/minimax-h3/custom_nodes/minimax_h3_ordered_storyboard \
      "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
    && mkdir -p /tmp/minimax-h3-story-workflows \
    && python /opt/minimax-h3/scripts/build_story_workflows.py \
      --director-source "${MINIMAX_H3_DIRECTOR_ROOT}/example_workflows/minimax_h3_director_fl2v.json" \
      --output-dir /tmp/minimax-h3-story-workflows \
    && cmp /tmp/minimax-h3-story-workflows/minimax_h3_story_quality_lora_2x.json \
      /opt/minimax-h3/workflows/minimax_h3_story_quality_lora_2x.json \
    && cmp /tmp/minimax-h3-story-workflows/minimax_h3_story_easycache_lora_2x.json \
      /opt/minimax-h3/workflows/minimax_h3_story_easycache_lora_2x.json \
    && cmp /tmp/minimax-h3-story-workflows/minimax_h3_story_quality_lora_2x_auto_mosaic.json \
      /opt/minimax-h3/workflows/minimax_h3_story_quality_lora_2x_auto_mosaic.json \
    && cmp /tmp/minimax-h3-story-workflows/minimax_h3_story_easycache_lora_2x_auto_mosaic.json \
      /opt/minimax-h3/workflows/minimax_h3_story_easycache_lora_2x_auto_mosaic.json \
    && rm -r /tmp/minimax-h3-story-workflows \
    && mkdir -p "${COMFYUI_ROOT}/user/default/workflows" \
    && cp /opt/minimax-h3/workflows/minimax_h3_preset_01_quality.json \
      "${COMFYUI_ROOT}/user/default/workflows/01_MiniMax_H3_Quality_2x.json" \
    && cp /opt/minimax-h3/workflows/minimax_h3_preset_02_fast_fbcache.json \
      "${COMFYUI_ROOT}/user/default/workflows/02_MiniMax_H3_Fast_FBCache_2x.json" \
    && cp /opt/minimax-h3/workflows/minimax_h3_preset_03_turbo_8step.json \
      "${COMFYUI_ROOT}/user/default/workflows/03_MiniMax_H3_Turbo_8step_2x.json" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v.json \
      --mode i2v \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v_easycache.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v.json \
      --mode i2v \
      --expect-easycache \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v_upscale.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v \
      --expect-upscale \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v_hmmotion_lora_upscale.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v \
      --expect-upscale \
      --expect-lora \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v_selectable_lora_upscale.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v \
      --expect-upscale \
      --expect-lora HMNSFW_AIO_V2.safetensors \
      --expect-lora-strength 0.5 \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_story_quality_lora_2x.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode story \
      --expect-upscale \
      --expect-lora HMNSFW_AIO_V2.safetensors \
      --expect-lora-strength 0.5 \
      --comfyui-root "${COMFYUI_ROOT}" \
      --custom-node-root "${MINIMAX_H3_DIRECTOR_ROOT}" \
      --custom-node-root "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_story_easycache_lora_2x.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode story \
      --expect-easycache \
      --expect-upscale \
      --expect-lora HMNSFW_AIO_V2.safetensors \
      --expect-lora-strength 0.5 \
      --comfyui-root "${COMFYUI_ROOT}" \
      --custom-node-root "${MINIMAX_H3_DIRECTOR_ROOT}" \
      --custom-node-root "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_i2v_easycache_upscale.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v \
      --expect-easycache \
      --expect-upscale \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_r2v.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_r2v.json \
      --mode r2v \
      --require-video-reference \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_r2v_easycache.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_r2v.json \
      --mode r2v \
      --expect-easycache \
      --require-video-reference \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_r2v_upscale.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_r2v_upscale.json \
      --mode r2v \
      --expect-upscale \
      --require-video-reference \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_r2v_easycache_upscale.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_r2v_upscale.json \
      --mode r2v \
      --expect-easycache \
      --expect-upscale \
      --require-video-reference \
      --comfyui-root "${COMFYUI_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_auto_mosaic_workflows.py \
      --comfyui-root "${COMFYUI_ROOT}" \
      --director-root "${MINIMAX_H3_DIRECTOR_ROOT}" \
      --local-node-root "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_preset_01_quality.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v --expect-upscale --expect-auto-mosaic \
      --auto-mosaic-manifest /opt/minimax-h3/manifests/auto_mosaic.json \
      --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5 \
      --comfyui-root "${COMFYUI_ROOT}" \
      --custom-node-root "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_preset_02_fast_fbcache.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v --expect-upscale --expect-auto-mosaic --expect-first-block-cache \
      --auto-mosaic-manifest /opt/minimax-h3/manifests/auto_mosaic.json \
      --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.5 \
      --comfyui-root "${COMFYUI_ROOT}" \
      --custom-node-root "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
      --custom-node-root "${MINIMAX_H3_FBC_ROOT}" \
    && python /opt/minimax-h3/scripts/verify_workflow.py \
      --workflow /opt/minimax-h3/workflows/minimax_h3_preset_03_turbo_8step.json \
      --manifest /opt/minimax-h3/manifests/minimax_h3_i2v_upscale.json \
      --mode i2v --expect-upscale --expect-auto-mosaic --expect-turbo \
      --auto-mosaic-manifest /opt/minimax-h3/manifests/auto_mosaic.json \
      --expect-lora HMNSFW_AIO_V2.safetensors --expect-lora-strength 0.0 \
      --comfyui-root "${COMFYUI_ROOT}" \
      --custom-node-root "${COMFYUI_ROOT}/custom_nodes/minimax_h3_ordered_storyboard" \
    && cd "${COMFYUI_ROOT}" \
    && python -c "import sys; sys.path.insert(0, '${COMFYUI_ROOT}/custom_nodes'); import minimax_h3_ordered_storyboard as p; assert 'WanAutoMosaicVideo' in p.NODE_CLASS_MAPPINGS"

RUN ACCEPT_MINIMAX_H3_LICENSE=1 \
    MINIMAX_H3_LICENSEE_IN_APPLICABLE_TERRITORY=1 \
    HF_TOKEN=entrypoint-smoke \
    CIVITAI_TOKEN=entrypoint-smoke \
    CIVITAI_API_TOKEN=entrypoint-smoke \
    MINIMAX_H3_ENTRYPOINT_SMOKE=1 \
    /opt/minimax-h3/scripts/entrypoint.sh

WORKDIR /opt/ComfyUI
EXPOSE 8188

HEALTHCHECK --interval=15s --timeout=5s --start-period=30m --retries=4 \
  CMD curl -fsS http://127.0.0.1:8188/system_stats >/dev/null || exit 1

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/minimax-h3/scripts/entrypoint.sh"]

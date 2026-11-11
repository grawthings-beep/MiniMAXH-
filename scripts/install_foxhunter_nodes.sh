#!/usr/bin/env bash
set -Eeuo pipefail

COMFYUI_ROOT="${COMFYUI_ROOT:-/opt/ComfyUI}"
CUSTOM="${COMFYUI_ROOT}/custom_nodes"
mkdir -p "${CUSTOM}"

install_pinned() {
  local repository="$1" revision="$2" directory="$3"
  local target="${CUSTOM}/${directory}"
  git init -q "${target}"
  git -C "${target}" remote add origin "https://github.com/${repository}.git"
  git -C "${target}" fetch -q --depth 1 origin "${revision}"
  git -C "${target}" checkout -q --detach FETCH_HEAD
  test "$(git -C "${target}" rev-parse HEAD)" = "${revision}"
  echo "[foxhunter] pinned ${repository}@${revision}"
}

install_pinned tazmannner/ComfyUI-H3-AnimeCadence bfc249ed2e356c7354869d1a8a23885345f6794f ComfyUI-H3-AnimeCadence
install_pinned Adudeguyman/ComfyUI-Fantastic-MiniMaxH3-PromptBuilder 22e719fe89336c628383363182a4da3cb58d4f9b ComfyUI-Fantastic-MiniMaxH3-PromptBuilder
install_pinned pixaroma/ComfyUI-Pixaroma a350978508b2105b3107ae735c78b389701e16e1 ComfyUI-Pixaroma
install_pinned ethanfel/ComfyUI-H3-Prompt-IDE c1667471203bc3afd0385e0711230ec968ff3dae ComfyUI-H3-Prompt-IDE
install_pinned Luisacaotica/ComfyUI-MiniMaxH3Mod 517e8c419a8584fbf69fceb487bc39d83e6a6ec8 ComfyUI-MiniMaxH3Mod
install_pinned chrisgoringe/cg-use-everywhere ffc99037eea8eac4d4ad20bb72a12a87740279a9 cg-use-everywhere
install_pinned Kosinkadink/ComfyUI-VideoHelperSuite 4d907bee61e92c2e65af3bd6383a4e4d356126d1 ComfyUI-VideoHelperSuite
install_pinned rgthree/rgthree-comfy bc48c6aaeca87bedbc14245d89f9bc6be37a630e rgthree-comfy
install_pinned kijai/ComfyUI-KJNodes ab72eda7ffad6280422d71ab7d54d3fb87c0ba18 ComfyUI-KJNodes
install_pinned GACLove/ComfyUI-VFI 6176a430f12cd16003f4664c1e3c6af8e96cc3c6 ComfyUI-VFI

# Full KJNodes requires these non-Torch packages. Headless OpenCV also serves
# VideoHelperSuite without replacing the container's working PyTorch stack.
pip install --no-cache-dir -r "${CUSTOM}/ComfyUI-KJNodes/requirements.txt" \
  imageio-ffmpeg
python -m pip check

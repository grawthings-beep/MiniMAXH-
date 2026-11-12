# FoxFurEssence H3 Seed Hunter v3

This is an isolated, first-priority RunPod image for the user-supplied workflow.
It does **not** overwrite the existing stable 04/09 workflows or their Docker
tags. Open `00_FoxFur_Seed_Hunter_v3` from ComfyUI Workflows after the pod starts.

## What is included

- The exact selected [CivitAI Eros Max beta5 file 3185144](https://civitai.com/api/download/models/3294059?fileId=3185144):
  `10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors`.
  It is fetched from the creator's pinned [Hugging Face copy](https://huggingface.co/TenStrip/10Eros-Max/blob/e4753986789e11aa50dd3eea15a774d4ef773a5a/10Eros_Max_h3_TURBO-hybrid_beta5_w4a8_14gb_optimized.safetensors),
  whose SHA256 matches the selected CivitAI file. Its Turbo delta is baked in,
  so the old Silver Turbo LoRA is not stacked. This W4A8 edition is the exact
  file requested, although its creator recommends the separate INT8 edition
  as the beta5 default; treat output quality as experimental until rendered.
- Qwen3-VL NVFP4 encoder, INT8 video VAE, FP32 audio VAE, and
  `MysticXXX_MMH3-V5-ref2va.safetensors` from the requested private
  `uwgm/illustrious` repository. Mystic V5 is enabled at 0.5 strength in the
  visible Pixaroma LoRA loader. The first pass is Euler/simple at 8 steps,
  with video/audio sigma shifts 12/7 per the Eros beta5 creator's guidance.
- LBH 3D latent upscaler v1 BF16, TAE preview weights, and RIFE v4.26 weights.
  LBH moved the same v1 weights to a versioned folder, so the saved widget is
  updated to the **current filename**. Its optional upscale/RIFE nodes remain
  bypassed exactly as they were in the supplied JSON; use its control panel to
  enable them deliberately.
- Pinned AnimeCadence, Fantastic Media Loader, Pixaroma, Prompt IDE, RefMods,
  Use Everywhere, VideoHelperSuite, rgthree, KJNodes, and ComfyUI-VFI. The two
  Easy-Use IDs used by this graph are supplied by a small local compatibility
  package to avoid installing that pack's unrelated inference dependencies.

The default author-character example was replaced by a neutral <Picture 1>
prompt. The first-run duration was reduced from 14 to 5 seconds to avoid an
unintentional long, multi-pass run. The author's local output preview paths
were removed. No model weights or secrets are embedded in the image. All eight
assets total **33,962,725,035 bytes (~31.6 GiB)** and are downloaded at each
ephemeral Pod start with resumable parallel transfers and pinned size/SHA256
validation. Plan for a 150 GB container disk, no persistent volume.

## First run

1. Select the Community CUDA 12.8 image unless the host driver supports CUDA
   13 / r580+. Set `HF_TOKEN` from a RunPod Secret and explicitly confirm the
   MiniMax license/territory flags. `HF_TOKEN` also needs read access to
   `uwgm/illustrious`. The checkpoint is byte-identical to the requested
   CivitAI file but downloaded from its author's Hugging Face repo, so no
   `CIVITAI_API_TOKEN` is needed for this profile.
2. Upload a new image in the Fantastic Media Loader. Replace the neutral
   Prompt IDE text with the desired character/scene. The supplied `lunalovegood.wav`
   audio-upload node is muted; upload your own audio before enabling that branch.
3. Keep the initial 5-second setting and optional latent upscale/RIFE branches
   off for a first render. The saved workflow makes three parallel seed
   candidates, then offers an optional second-pass upscale;
   24/32 GB VRAM is **not** guaranteed. Raise duration or enable optional
   passes one at a time after measuring VRAM and output quality.

The build checks that ComfyUI can register all executable node IDs on CPU and
that the pinned ComfyUI/ComfyKitchen stack recognizes asymmetric W4A8.
That check is **not** a full GPU-generation test; only a live Pod render can
establish image quality and practical VRAM limits.

RunPod examples: `runpod-template.foxhunter-community-cu128.example.json` and
`runpod-template.foxhunter-fast-cu130.example.json`.
